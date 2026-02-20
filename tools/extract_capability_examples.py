#!/usr/bin/env python3
"""
Extract qualitative language examples (capability preservation) from saved per-scene JSON predictions.

This script does NOT run inference. It only reads already-exported *.json files produced by the eval pipeline.
It samples examples with a fixed random seed and prints LaTeX table rows (baseline vs ours).

This is a vendored copy of the extractor so the public release folder is self-contained.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LATEX_ESCAPE_RE = re.compile(r"([\\%$&#_{}~^])")
_SPECIAL_TOKENS_RE = re.compile(
    r"(<s>|</s>|<pad>|<unk>|<ref>|<inref>|<p>|</p>|<geom>)", re.IGNORECASE
)
_ROLE_PREFIX_RE = re.compile(r"^\s*(USER|ASSISTANT)\s*:\s*", re.IGNORECASE)
_ROLE_ANYWHERE_RE = re.compile(r"\b(USER|ASSISTANT)\s*:\s*", re.IGNORECASE)
_META_KEY_CANDIDATES: Tuple[str, ...] = (
    "scene_id",
    "scan_id",
    "scan_name",
    "scene",
    "ann_id",
    "sample_id",
    "qid",
    "question_id",
    "target_inst_id",
    "target_instance_id",
    "instance_id",
    "inst_id",
    "object_id",
    "target_id",
    "gt_inst_ids",
    "gt_target_q",
    "gt_target_qs",
    "pred_target_q",
)


def _latex_escape(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = _LATEX_ESCAPE_RE.sub(r"\\\1", text)
    return re.sub(r"\s+", " ", text)


def _clean_generation(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = _SPECIAL_TOKENS_RE.sub("", text)
    text = _ROLE_PREFIX_RE.sub("", text)
    text = _ROLE_ANYWHERE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cutoff = max(0, max_chars - 3)
    head = text[:cutoff].rstrip()
    if len(head) >= 20 and " " in head:
        head = head[: head.rfind(" ")].rstrip()
    return head + "..."


def _clean_input(text: str) -> str:
    raw = text.replace("\n", " ").strip()
    raw = _SPECIAL_TOKENS_RE.sub("", raw)
    m = re.search(r"\bASSISTANT\s*:\s*", raw, flags=re.IGNORECASE)
    if m:
        raw = raw[: m.start()]
    raw = _ROLE_ANYWHERE_RE.sub("", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _first_int(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, int):
        return int(obj)
    try:
        if isinstance(obj, str):
            s = obj.strip()
            if s and re.fullmatch(r"-?\d+", s):
                return int(s)
            return None
        return int(obj)  # type: ignore[arg-type]
    except Exception:
        pass
    if isinstance(obj, dict):
        for v in obj.values():
            out = _first_int(v)
            if out is not None:
                return out
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out = _first_int(v)
            if out is not None:
                return out
        return None
    return None


def _extract_scene_id(fallback: str, line: Dict[str, Any]) -> str:
    v = line.get("scene_id")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return fallback


def _extract_target_inst_id(line: Dict[str, Any]) -> Optional[int]:
    for k in ("target_inst_id", "target_instance_id", "instance_id", "inst_id", "object_id", "target_id"):
        out = _first_int(line.get(k))
        if out is not None:
            return out
    out = _first_int(line.get("gt_inst_ids"))
    if out is not None:
        return out
    return None


def _extract_meta(line: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for k in _META_KEY_CANDIDATES:
        if k not in line:
            continue
        v = line.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
            continue
        if isinstance(v, list):
            meta[k] = v[:5]
            continue
        if isinstance(v, dict):
            meta[k] = {kk: v[kk] for kk in list(v.keys())[:5]}
            continue
    target_inst_id = _extract_target_inst_id(line)
    if target_inst_id is not None:
        meta["target_inst_id"] = target_inst_id
    return meta


def _iter_prediction_lines(pred_dir: Path) -> Iterable[Tuple[str, Dict[str, Any], Path]]:
    json_paths = sorted(pred_dir.rglob("*.json"))
    for path in json_paths:
        data = _safe_load_json(path)
        if not data:
            continue
        pred_list = data.get("prediction")
        if not isinstance(pred_list, list):
            continue
        for line in pred_list:
            if isinstance(line, dict):
                scene_id = _extract_scene_id(path.stem, line)
                yield scene_id, line, path


def _task_from_type(type_str: str) -> Optional[str]:
    t = type_str.lower()
    if "dialog" in t:
        return "Dialog"
    if "scanqa" in t and "text_only" in t:
        return "ScanQA"
    if "scan2cap" in t:
        return "Scan2Cap"
    if "objdesc" in t:
        return "ObjDesc"
    return None


@dataclass(frozen=True)
class Example:
    task: str
    scene_id: str
    input_text: str
    output_text: str
    gt_text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None


def _key_for_example(ex: Example) -> str:
    return f"{ex.scene_id}|{ex.task}|{ex.input_text}".lower()


def _collect_examples(pred_dir: Path) -> Dict[str, Dict[str, Example]]:
    out: Dict[str, Dict[str, Example]] = {"Dialog": {}, "ScanQA": {}, "Scan2Cap": {}, "ObjDesc": {}}
    for scene_id, line, src in _iter_prediction_lines(pred_dir):
        type_str = str(line.get("type", "") or "")
        task = _task_from_type(type_str)
        if task is None:
            continue
        in_text = str(line.get("input_language", "") or "")
        out_text = str(line.get("output_language", "") or "")
        gt_text = str(line.get("gt", "") or "")
        meta = _extract_meta(line)
        ex = Example(
            task=task,
            scene_id=scene_id,
            input_text=in_text,
            output_text=out_text,
            gt_text=gt_text,
            meta=meta,
            source_path=str(src),
        )
        out[task][_key_for_example(ex)] = ex
    return out


def _sample_keys(keys: Sequence[str], k: int, rng: random.Random) -> List[str]:
    keys = list(keys)
    if not keys or k <= 0:
        return []
    if len(keys) <= k:
        rng.shuffle(keys)
        return keys
    return rng.sample(keys, k=k)


def _export_scenes(
    scene_ids: Sequence[str],
    *,
    export_dir: Path,
    search_roots: Sequence[Path],
    globs: Sequence[str],
    max_files_per_scene: int,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for scene_id in scene_ids:
        if scene_id in seen:
            continue
        seen.add(scene_id)
        files: List[Path] = []
        for root in search_roots:
            for pat in globs:
                g = pat.format(scene_id=scene_id)
                files.extend(sorted(root.glob(g)))
        files = [p for p in files if p.is_file()]
        if not files:
            print(f"# [warn] no scene files found for {scene_id}")
            continue
        files = files[: max(1, int(max_files_per_scene))]
        scene_out = export_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = scene_out / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
        print(f"# exported {scene_id}: {len(files)} file(s)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample qualitative capability-preservation examples from saved per-scene JSON predictions."
    )
    ap.add_argument("--baseline-pred-dir", type=Path, required=True)
    ap.add_argument("--ours-pred-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k-per-task", type=int, default=1)
    ap.add_argument("--max-chars-input", type=int, default=140)
    ap.add_argument("--max-chars-output", type=int, default=160)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--export-scenes-dir", type=Path, default=None)
    ap.add_argument("--scene-search-root", type=Path, action="append", default=[])
    ap.add_argument("--scene-glob", type=str, action="append", default=[])
    ap.add_argument("--max-files-per-scene", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = _collect_examples(args.baseline_pred_dir)
    ours = _collect_examples(args.ours_pred_dir)

    print(f"# baseline_pred_dir={args.baseline_pred_dir}")
    print(f"# ours_pred_dir={args.ours_pred_dir}")
    print(f"# seed={args.seed}, k_per_task={args.k_per_task}")
    print("# Paste the following rows into Table `tab:cap_preserve_examples` (Appendix).")
    print()

    sampled_scene_ids: List[str] = []
    for task in ["Dialog", "ScanQA", "Scan2Cap", "ObjDesc"]:
        common = sorted(set(base[task].keys()) & set(ours[task].keys()))
        picked = _sample_keys(common, args.k_per_task, rng)
        if not picked:
            continue
        for key in picked:
            b_ex = base[task][key]
            o_ex = ours[task][key]
            sampled_scene_ids.append(b_ex.scene_id)

            in_text = _latex_escape(_truncate(_clean_input(b_ex.input_text), args.max_chars_input))
            b_out = _latex_escape(_truncate(_clean_generation(b_ex.output_text), args.max_chars_output))
            o_out = _latex_escape(_truncate(_clean_generation(o_ex.output_text), args.max_chars_output))

            if not args.no_meta:
                b_tid = b_ex.meta.get("target_inst_id")
                o_tid = o_ex.meta.get("target_inst_id")
                parts = [f"task={task}", f"scene_id={b_ex.scene_id}"]
                if b_tid is not None or o_tid is not None:
                    parts.append(f"target_inst_id(b/o)={b_tid}/{o_tid}")
                for k in ("ann_id", "sample_id", "question_id", "qid"):
                    if k in b_ex.meta:
                        parts.append(f"{k}={b_ex.meta.get(k)}")
                if b_ex.source_path:
                    parts.append(f"src={Path(b_ex.source_path).name}")
                print("% " + ", ".join(parts))

            print(f"{task} & {in_text} & {b_out} & {o_out} \\\\")

    if args.export_scenes_dir is not None:
        globs = args.scene_glob or ["{scene_id}/{scene_id}_vh_clean_2.ply", "**/{scene_id}_vh_clean_2.ply"]
        if not args.scene_search_root:
            print("# [warn] --export-scenes-dir is set but no --scene-search-root provided; skipping export.")
            return
        _export_scenes(
            sampled_scene_ids,
            export_dir=args.export_scenes_dir,
            search_roots=args.scene_search_root,
            globs=globs,
            max_files_per_scene=args.max_files_per_scene,
        )


if __name__ == "__main__":
    main()
