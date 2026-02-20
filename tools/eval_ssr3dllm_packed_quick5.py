#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick 5-sample eval for packed SSR3DLLM ckpt.")
    p.add_argument("--checkpoint", required=True, help="Packed SSR3DLLM ckpt path")
    p.add_argument("--profile", default="503", help="Listener profile in packed ckpt: 503|519|main|ub")
    p.add_argument("--num-samples", type=int, default=5, help="Samples per grounding dataset")
    p.add_argument("--datasets", default="nr3d,sr3d", help="Comma-separated: nr3d,sr3d")
    p.add_argument("--nr3d-train-csv", required=True, help="Path to nr3d train csv")
    p.add_argument("--sr3d-train-csv", required=True, help="Path to sr3d train csv")
    p.add_argument("--scannet-file", required=True, help="ReferIt3D ScanNet pkl path")
    p.add_argument("--bert-path", required=True, help="BERT path for listener runtime")
    p.add_argument("--scannet-processed-root", required=True, help="Processed ScanNet root")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--output-json", default="", help="Optional JSON output path")
    p.add_argument("--llm-max-new-tokens", type=int, default=128)
    return p.parse_args()


def _resolve_test_csv(train_csv: Path) -> Path:
    p = str(train_csv)
    if "train" in p:
        p = p.replace("train", "test")
    p = re.sub(r"_\d+\.\d+", "", p)
    return Path(p)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _pick_str(row: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return ""


def _pick_int(row: Dict[str, str], keys: List[str]) -> Optional[int]:
    for key in keys:
        val = str(row.get(key, "")).strip()
        if not val:
            continue
        try:
            return int(val)
        except Exception:
            continue
    return None


def _extract_pred_qid(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, list) and first:
            try:
                return int(first[0])
            except Exception:
                return None
        try:
            return int(first)
        except Exception:
            return None
    return None


def _resolve_gt_qid(gt_to_query_map: Dict[Any, Any], target_id: int) -> Optional[int]:
    for cand in (target_id, target_id + 1, target_id - 1):
        qid = gt_to_query_map.get(cand, None)
        if qid is None:
            qid = gt_to_query_map.get(str(cand), None)
        if qid is not None:
            try:
                return int(qid)
            except Exception:
                continue
    return None


def _build_api(args: argparse.Namespace):
    from baseline.api.baseline_interface import BaselineModelAPI

    config_root = Path("baseline/core/conf")
    cfg = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split_type": args.split,
        "scannet_processed_root": str(Path(args.scannet_processed_root).resolve()),
        "config_path": str(config_root),
        "data_config": str(config_root / "data/indoor_dialog.yaml"),
        "model_config": str(config_root / "model/mask3d_lang.yaml"),
        "trainer_config": str(config_root / "trainer/trainer50.yaml"),
        "llm_config": str(config_root / "llm/tiny_vicuna_len512_bs4.json"),
        "llm_data_config": str(config_root / "llm/det10.json"),
        "topk_per_image": 750,
    }
    return BaselineModelAPI(cfg)


@torch.no_grad()
def _eval_grounding_dataset(
    *,
    api: Any,
    dataset_name: str,
    test_csv: Path,
    num_samples: int,
    llm_max_new_tokens: int,
) -> Dict[str, Any]:
    rows = _read_csv_rows(test_csv)
    total = 0
    correct = 0
    skipped = 0
    details: List[Dict[str, Any]] = []

    for row in rows:
        if total >= int(num_samples):
            break
        scene_id = _pick_str(row, ["scan_id", "scene_id", "scanid", "sceneid", "scan"])
        utterance = _pick_str(row, ["utterance", "description", "text", "query", "sentence"])
        target_id = _pick_int(row, ["target_id", "target_instance_id", "targetid", "instance_id", "target"])
        if not scene_id or not utterance or target_id is None:
            skipped += 1
            continue

        scene_pack = api.get_best_match_for_scene(scene_id)
        if not isinstance(scene_pack, dict):
            skipped += 1
            continue
        gt_map = scene_pack.get("gt_to_query_map", None)
        object_queries = scene_pack.get("object_queries", None)
        if not isinstance(gt_map, dict) or not torch.is_tensor(object_queries):
            skipped += 1
            continue

        gt_qid = _resolve_gt_qid(gt_map, int(target_id))
        if gt_qid is None:
            skipped += 1
            continue

        oq = object_queries.float()
        oq_norm = F.normalize(oq, p=2, dim=1)
        prompt = f"<geom> {utterance.strip()}"
        out = api.llama_model.evaluate(
            input_text_list=[prompt],
            batch_instance_queries_hidden_state=[oq],
            batch_instance_queries_normalized_embed=[oq_norm],
            batch_eval_types=["chat"],
            use_mini_batch=False,
            max_new_tokens=int(llm_max_new_tokens),
            text_only_output=False,
        )
        pred_qid = _extract_pred_qid(out[0].get("grounding_result", None)) if isinstance(out, list) and out else None
        is_correct = pred_qid is not None and int(pred_qid) == int(gt_qid)

        total += 1
        if is_correct:
            correct += 1

        details.append(
            {
                "scene_id": scene_id,
                "utterance": utterance,
                "target_id": int(target_id),
                "gt_qid": int(gt_qid),
                "pred_qid": None if pred_qid is None else int(pred_qid),
                "correct": bool(is_correct),
            }
        )

    acc = float(correct) / float(total) if total > 0 else 0.0
    return {
        "dataset": dataset_name,
        "test_csv": str(test_csv),
        "num_samples_requested": int(num_samples),
        "num_samples_used": int(total),
        "num_correct": int(correct),
        "accuracy": acc,
        "skipped_rows": int(skipped),
        "samples": details,
    }


@torch.no_grad()
def _eval_language(
    *,
    api: Any,
    scene_ids: List[str],
    num_samples: int,
    llm_max_new_tokens: int,
) -> Dict[str, Any]:
    prompts = [
        "Describe this scene briefly.",
        "List the most important objects in this room.",
        "What object would you use to sit down here?",
        "What object would you use to place a laptop?",
        "Give one concise safety observation about this room.",
    ]
    samples: List[Dict[str, Any]] = []
    non_empty = 0

    for idx in range(min(int(num_samples), len(prompts))):
        scene_id = scene_ids[idx % len(scene_ids)] if scene_ids else ""
        if not scene_id:
            break
        scene_pack = api.get_scene_data_for_verification(scene_id)
        if not isinstance(scene_pack, dict):
            continue
        object_queries = scene_pack.get("object_queries", None)
        if not torch.is_tensor(object_queries):
            continue
        oq = object_queries.float()
        oq_norm = F.normalize(oq, p=2, dim=1)
        prompt = prompts[idx]
        out = api.llama_model.evaluate(
            input_text_list=[prompt],
            batch_instance_queries_hidden_state=[oq],
            batch_instance_queries_normalized_embed=[oq_norm],
            batch_eval_types=["chat"],
            use_mini_batch=False,
            max_new_tokens=int(llm_max_new_tokens),
            text_only_output=False,
        )
        answer = ""
        if isinstance(out, list) and out:
            answer = str(out[0].get("output_language", "")).strip()
        if answer:
            non_empty += 1
        samples.append(
            {
                "scene_id": scene_id,
                "question": prompt,
                "answer": answer,
                "non_empty": bool(answer),
            }
        )

    return {
        "num_questions": int(len(samples)),
        "num_non_empty_answers": int(non_empty),
        "non_empty_rate": (float(non_empty) / float(len(samples))) if samples else 0.0,
        "samples": samples,
    }


def main() -> None:
    args = _parse_args()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    nr3d_train = Path(args.nr3d_train_csv).expanduser().resolve()
    sr3d_train = Path(args.sr3d_train_csv).expanduser().resolve()
    nr3d_test = _resolve_test_csv(nr3d_train)
    sr3d_test = _resolve_test_csv(sr3d_train)
    if not nr3d_test.is_file():
        raise FileNotFoundError(f"nr3d test csv not found: {nr3d_test}")
    if not sr3d_test.is_file():
        raise FileNotFoundError(f"sr3d test csv not found: {sr3d_test}")

    # Force <geom> routing from packed checkpoint.
    os.environ["SSR3DLLM_ROUTE_GEOM_VIGOR"] = "1"
    os.environ["SSR3DLLM_REFERIT3D_LISTENER_CKPT"] = str(ckpt)
    os.environ["SSR3DLLM_REFERIT3D_LISTENER_BERT"] = str(Path(args.bert_path).expanduser().resolve())
    os.environ["SSR3DLLM_REFERIT_SCANNET_FILE"] = str(Path(args.scannet_file).expanduser().resolve())
    os.environ["SCANNET_PKL"] = str(Path(args.scannet_file).expanduser().resolve())
    os.environ["SSR3DLLM_VIGOR_PROFILE"] = str(args.profile).strip()
    os.environ.setdefault("VIGOR_USE_PRED_BOX_INFO", "1")
    os.environ.setdefault("VIGOR_PRED_CLASS_MASK_MODE", "all_ones")

    api = _build_api(args)

    run_sets = [x.strip().lower() for x in str(args.datasets).split(",") if x.strip()]
    grounding: Dict[str, Any] = {}
    scene_ids_for_language: List[str] = []

    if "nr3d" in run_sets:
        out = _eval_grounding_dataset(
            api=api,
            dataset_name="nr3d",
            test_csv=nr3d_test,
            num_samples=int(args.num_samples),
            llm_max_new_tokens=int(args.llm_max_new_tokens),
        )
        grounding["nr3d"] = out
        scene_ids_for_language.extend([s["scene_id"] for s in out.get("samples", []) if s.get("scene_id")])
    if "sr3d" in run_sets:
        out = _eval_grounding_dataset(
            api=api,
            dataset_name="sr3d",
            test_csv=sr3d_test,
            num_samples=int(args.num_samples),
            llm_max_new_tokens=int(args.llm_max_new_tokens),
        )
        grounding["sr3d"] = out
        scene_ids_for_language.extend([s["scene_id"] for s in out.get("samples", []) if s.get("scene_id")])

    if not scene_ids_for_language:
        scene_ids_for_language = sorted(list(api.scene_index.keys()))[: max(1, int(args.num_samples))]
    unique_scenes = []
    seen = set()
    for sid in scene_ids_for_language:
        if sid in seen:
            continue
        seen.add(sid)
        unique_scenes.append(sid)

    language = _eval_language(
        api=api,
        scene_ids=unique_scenes[: max(1, int(args.num_samples))],
        num_samples=int(args.num_samples),
        llm_max_new_tokens=int(args.llm_max_new_tokens),
    )

    summary = {
        "checkpoint": str(ckpt),
        "profile": str(args.profile),
        "num_samples": int(args.num_samples),
        "grounding": grounding,
        "language": language,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
