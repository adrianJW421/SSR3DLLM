from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GroundingStepsKey:
    scene_id: str
    target_gt_id: int
    raw_text: str


def _norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())

_END_PUNCT_RE = re.compile(r"[\\s\\.,;:!?]+$")
_INNER_PUNCT_RE = re.compile(r"[\\.,;:!?]+")


def _norm_text_strip_end_punct(text: str) -> str:
    """
    A more forgiving normalization for matching precomputed grounding_steps
    across slightly different dataset preprocessing versions (e.g., with/without
    trailing period added by a punctuation-normalization script).
    """
    s = _norm_text(text)
    s = _END_PUNCT_RE.sub("", s)
    return s


def _norm_text_strip_punct(text: str) -> str:
    """
    Even more forgiving normalization: strips common punctuation anywhere in the
    string. This helps when different preprocessing versions rewrite sentence
    punctuation (e.g., '.' -> ';').
    """
    s = _norm_text(text)
    s = _INNER_PUNCT_RE.sub(" ", s)
    s = " ".join(s.strip().split())
    return s


_CACHE: Dict[Tuple[str, str], Dict[GroundingStepsKey, List[str]]] = {}
_LABEL_SET_CACHE: Dict[str, set[str]] = {}

def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in {"0", "false", "no", "off"}

_STEP_NORM_RE = re.compile(r"[^a-z0-9 ]+")

def _normalize_step_phrase(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for prefix in ("a ", "an ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    s = s.replace("&", " ")
    s = _STEP_NORM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _singularize_last_word(phrase: str) -> str:
    w = phrase.split()
    if not w:
        return phrase
    last = w[-1]
    cand: List[str] = [last]
    if last.endswith("ies") and len(last) > 3:
        cand.append(last[:-3] + "y")
    if last.endswith("es") and len(last) > 2:
        cand.append(last[:-2])
    if last.endswith("s") and len(last) > 1 and not last.endswith("ss"):
        cand.append(last[:-1])
    for c in cand[1:]:
        if c and c != last:
            w2 = w[:-1] + [c]
            return " ".join(w2)
    return phrase

def _load_label_set(name: str) -> set[str]:
    name = str(name).strip().lower()
    if not name:
        return set()
    if name in _LABEL_SET_CACHE:
        return _LABEL_SET_CACHE[name]
    if name == "scannet200":
        from baseline.dataset.datasets.scannet200.scannet200_constants import CLASS_LABELS_200
        labels = {str(x).strip().lower() for x in CLASS_LABELS_200}
        _LABEL_SET_CACHE[name] = labels
        return labels
    if name in {"scannet20", "scannet18", "scannet"}:
        from baseline.dataset.datasets.scannet200.scannet200_constants import CLASS_LABELS_20
        labels = {str(x).strip().lower() for x in CLASS_LABELS_20}
        _LABEL_SET_CACHE[name] = labels
        return labels
    raise ValueError(f"Unknown label set: {name}")

def _match_step_phrase(phrase: str, labels: set[str], *, relaxed: bool) -> bool:
    p = _normalize_step_phrase(phrase)
    if not p:
        return False
    if p in labels:
        return True
    if not relaxed:
        return False
    p2 = _singularize_last_word(p)
    if p2 in labels:
        return True
    # Keep synonyms intentionally small and stable across runs.
    synonyms = {
        "television": "tv",
        "tv monitor": "tv",
        "trash bin": "trash can",
        "garbage can": "trash can",
        "sofa": "couch",
    }
    if p in synonyms and synonyms[p] in labels:
        return True
    if p2 in synonyms and synonyms[p2] in labels:
        return True
    return False


def _json_path_for_split(split: str) -> str:
    split = str(split).strip().lower()
    if split in {"train", "training"}:
        return (
            os.environ.get("SSR3DLLM_GROUNDING_STEPS_JSON_TRAIN", "").strip()
            or os.environ.get("SSR3DLLM_GROUNDING_STEPS_JSON", "").strip()
        )
    return (
        os.environ.get("SSR3DLLM_GROUNDING_STEPS_JSON_EVAL", "").strip()
        or os.environ.get("SSR3DLLM_GROUNDING_STEPS_JSON", "").strip()
    )

def _iter_step_items(path: Path):
    """
    Yield dict items from:
      - list-JSON (.json)
      - JSONL (.jsonl)
      - a directory containing shard files (prefers *.jsonl, excluding *cache*)
    """
    if path.is_dir():
        jsonl_files = sorted(
            [
                p
                for p in path.glob("*.jsonl")
                if p.is_file() and "cache" not in p.name.lower()
            ]
        )
        json_files = sorted(
            [
                p
                for p in path.glob("*.json")
                if p.is_file() and "cache" not in p.name.lower()
            ]
        )
        files = jsonl_files if jsonl_files else json_files
        for fp in files:
            yield from _iter_step_items(fp)
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    # default: list-json
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list):
        raise ValueError(f"Expected list JSON at {path}")
    for obj in items:
        if isinstance(obj, dict):
            yield obj


def load_grounding_steps_map(split: str) -> Dict[GroundingStepsKey, List[str]]:
    """
    Loads a JSON list of items with at least:
      - scene_id (str)
      - target_object_gt_id (int)
      - question (str) OR distill_text (str)
      - referential_order (list[str])

    And builds a lookup keyed by (scene_id, target_gt_id, raw_text_norm).
    This is used to attach teacher-forced referential order to ScanRefer/M3DRef
    grounding samples during Step4 listener finetuning.
    """
    split_key = str(split).strip().lower() or "eval"
    path = _json_path_for_split(split_key)

    # Optional: filter out ScanRefer/M3DRef samples whose step phrases do NOT map
    # to a fixed label space (e.g. ScanNet200). This intentionally trades off
    # coverage for quality to stabilize listener training.
    #
    # Env:
    #   - SSR3DLLM_GROUNDING_STEPS_FILTER_LABEL_SET=scannet200|scannet20|...
    #   - SSR3DLLM_GROUNDING_STEPS_FILTER_MATCH_MODE=relaxed|strict (default: relaxed)
    #   - SSR3DLLM_GROUNDING_STEPS_FILTER_REQUIRE_ALL=1/0 (default: 1)
    #   - SSR3DLLM_GROUNDING_STEPS_FILTER_LOG=1/0 (default: 0)
    filter_label_set = str(os.environ.get("SSR3DLLM_GROUNDING_STEPS_FILTER_LABEL_SET", "")).strip().lower()
    filter_match_mode = str(os.environ.get("SSR3DLLM_GROUNDING_STEPS_FILTER_MATCH_MODE", "relaxed")).strip().lower()
    filter_relaxed = filter_match_mode != "strict"
    filter_require_all = _env_truthy("SSR3DLLM_GROUNDING_STEPS_FILTER_REQUIRE_ALL", default=True)
    filter_log = _env_truthy("SSR3DLLM_GROUNDING_STEPS_FILTER_LOG", default=False)

    cache_key = (
        split_key,
        path,
        filter_label_set or "",
        "relaxed" if filter_relaxed else "strict",
        "all" if filter_require_all else "any",
    )
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not path:
        _CACHE[cache_key] = {}
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SSR3DLLM grounding steps JSON not found: {path}")

    out: Dict[GroundingStepsKey, List[str]] = {}
    labels: set[str] = set()
    if filter_label_set:
        labels = _load_label_set(filter_label_set)
    filtered_total = 0
    kept_total = 0
    for item in _iter_step_items(p):
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id", "")).strip()
        if not scene_id:
            continue
        tgt = item.get("target_object_gt_id", None)
        try:
            target_gt_id = int(tgt)
        except Exception:
            continue
        raw_text_raw = item.get("question", None) or item.get("distill_text", None) or item.get("utterance", None) or ""
        raw_text = _norm_text(str(raw_text_raw))
        raw_text_alt = _norm_text_strip_end_punct(str(raw_text_raw))
        raw_text_alt2 = _norm_text_strip_punct(str(raw_text_raw))
        if not raw_text and not raw_text_alt:
            continue
        ro = item.get("referential_order", None)
        if not isinstance(ro, list):
            continue
        order: List[str] = []
        for x in ro:
            s = str(x).strip()
            if s:
                order.append(s)
        if not order:
            continue
        if labels:
            ok_any = False
            ok_all = True
            for s in order:
                ok = _match_step_phrase(s, labels, relaxed=filter_relaxed)
                ok_any = ok_any or ok
                ok_all = ok_all and ok
                if filter_require_all and (not ok):
                    break
            keep = ok_all if filter_require_all else ok_any
            if not keep:
                filtered_total += 1
                continue
        kept_total += 1
        if raw_text:
            out[GroundingStepsKey(scene_id=scene_id, target_gt_id=target_gt_id, raw_text=raw_text)] = order
        if raw_text_alt and raw_text_alt != raw_text:
            out.setdefault(
                GroundingStepsKey(scene_id=scene_id, target_gt_id=target_gt_id, raw_text=raw_text_alt), order
            )
        if raw_text_alt2 and raw_text_alt2 not in {raw_text, raw_text_alt}:
            out.setdefault(
                GroundingStepsKey(scene_id=scene_id, target_gt_id=target_gt_id, raw_text=raw_text_alt2), order
            )

    if labels and filter_log and os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            print(
                f"[grounding_steps][filter] split={split_key} label_set={filter_label_set} "
                f"match_mode={'relaxed' if filter_relaxed else 'strict'} "
                f"require={'all' if filter_require_all else 'any'} kept={kept_total} filtered={filtered_total}",
                flush=True,
            )
        except Exception:
            pass

    if not out and _env_truthy("SSR3DLLM_GROUNDING_STEPS_STRICT", default=False) and path:
        raise ValueError(
            f"Loaded grounding steps JSON but found 0 valid entries (strict=1). "
            f"Please check that {path} contains items with "
            f"scene_id/target_object_gt_id/question(or distill_text)/referential_order."
        )

    _CACHE[cache_key] = out
    return out


def lookup_referential_order(
    *,
    split: str,
    scene_id: str,
    target_gt_id: int,
    raw_text: str,
) -> Optional[List[str]]:
    m = load_grounding_steps_map(split)
    if not m:
        return None
    key = GroundingStepsKey(
        scene_id=str(scene_id).strip(),
        target_gt_id=int(target_gt_id),
        raw_text=_norm_text(raw_text),
    )
    out = m.get(key, None)
    if out is not None:
        return out

    key2 = GroundingStepsKey(
        scene_id=str(scene_id).strip(),
        target_gt_id=int(target_gt_id),
        raw_text=_norm_text_strip_end_punct(raw_text),
    )
    out = m.get(key2, None)
    if out is not None:
        return out

    key3 = GroundingStepsKey(
        scene_id=str(scene_id).strip(),
        target_gt_id=int(target_gt_id),
        raw_text=_norm_text_strip_punct(raw_text),
    )
    return m.get(key3, None)
