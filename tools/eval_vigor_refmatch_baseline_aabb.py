#!/usr/bin/env python3
"""
Evaluate Grounded3D-LLM baseline (<ref> -> match Mask3D queries) on Vigor CSV datasets,
while using exported Mask3D features (*.pt) to compute AABB-based IoU metrics.

Goal: align evaluation protocol across ScanRefer / M3DRef / Nr3D / Sr3D with the same:
- ReferIt3D preprocessed ScanNet pkl (axis-aligned frame)
- Vigor CSV loaders & filtering
- Mask3D exported features (object_queries + pred_aabb + gt_to_query_map)

This script does NOT use Vigor's listener scoring head; it uses the baseline LLM <ref> matching.

Example:
  python tools/eval_vigor_refmatch_baseline_aabb.py \
    --scannet-file data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl \
    --referit3d-file third_party/Vigor/referit3d/data/csv_data/scanrefer_train_vigor_single_latest.csv \
    --mask3d-feature-root data/MASK3D_FEATS_TRAIN_predbox_qnorm \
    --mask3d-feature-root-test data/MASK3D_FEATS_TEST_predbox_qnorm \
    --g3dllm-ckpt data/grounded3dllm_ckpts/step3/last-epoch.ckpt \
    --g3dllm-scannet-root data/SCANNET200_ROOT \
    --out-dir outputs/ssr3dllm/refmatch_baseline_qnorm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _force_vigor_referit3d(repo_root: Path) -> None:
    vigor_root = repo_root / "third_party" / "Vigor"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(vigor_root))


def _aabb_to_corners(aabb6: np.ndarray) -> np.ndarray:
    mn = aabb6[0:3].astype(np.float32)
    mx = aabb6[3:6].astype(np.float32)
    return np.asarray(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )


def _safe_int(x: Any, default: Optional[int] = -1) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class AvgMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return float(self.total / max(self.count, 1))


def _stable_softmax(x: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = float(temperature) if float(temperature) > 0 else 1.0
    x = x / t
    if x.size == 0:
        return x
    m = np.max(x)
    y = np.exp(x - m)
    s = float(np.sum(y))
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return y / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scannet-file", required=True)
    ap.add_argument("--referit3d-file", required=True, help="Vigor *train* CSV; loader auto-reads the corresponding test CSV.")
    ap.add_argument("--mask3d-feature-root", required=True, help="Train split Mask3D exported features dir.")
    ap.add_argument("--mask3d-feature-root-test", required=True, help="Test/val split Mask3D exported features dir.")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--g3dllm-ckpt", required=True, help="Grounded3D-LLM Step3 lightning ckpt (has LLM + adapters).")
    ap.add_argument("--g3dllm-scannet-root", required=True, help="processed ScanNet200 root (required by baseline init).")
    ap.add_argument("--config-root", default="baseline/core/conf")
    ap.add_argument("--g3dllm-data-config", default="baseline/core/conf/data/indoor_dialog.yaml")
    ap.add_argument("--g3dllm-model-config", default="baseline/core/conf/model/mask3d_lang.yaml")
    ap.add_argument("--g3dllm-trainer-config", default="baseline/core/conf/trainer/trainer50.yaml")
    ap.add_argument("--g3dllm-llm-config", default="baseline/core/conf/llm/tiny_vicuna_len512.json")
    ap.add_argument("--g3dllm-llm-data-config", default="baseline/core/conf/llm/det10.json")

    # Match common Vigor eval knobs
    ap.add_argument("--mentions-target-class-only", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=40)
    ap.add_argument("--order-len", type=int, default=4)
    ap.add_argument("--points-per-object", type=int, default=1024)
    ap.add_argument("--unit-sphere-norm", type=int, default=1)
    ap.add_argument("--max-test-objects", type=int, default=256)
    ap.add_argument("--max-distractors", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument(
        "--scene-ids",
        type=str,
        default="",
        help="Optional comma-separated scene ids to evaluate (e.g., 'scene0011_00').",
    )

    # Baseline matching knobs
    ap.add_argument("--topk", type=int, default=1, help="Top-k queries to extract from baseline matcher.")
    ap.add_argument("--llm-cache", default="", help="Optional jsonl cache to avoid repeated generation.")
    ap.add_argument(
        "--print-generation-first",
        type=int,
        default=0,
        help="Print the baseline LLM generation for the first N examples (debug).",
    )
    ap.add_argument(
        "--wrap-grounding-question",
        type=int,
        default=1,
        help="Wrap utterance with a grounding question template (match training distribution).",
    )
    ap.add_argument(
        "--debug-gt-rank-first",
        type=int,
        default=0,
        help="For the first N examples, compute the rank/score of q_star among all queries (no cache).",
    )
    ap.add_argument(
        "--seeds",
        default="2020",
        help="Comma-separated evaluation seeds (affects Vigor test context sampling).",
    )
    ap.add_argument(
        "--restrict-to-context",
        type=int,
        default=0,
        help="If 1, evaluate query matching on the Vigor sampled candidate set (context objects) instead of all Mask3D queries.",
    )
    ap.add_argument(
        "--export-context-probs",
        type=int,
        default=0,
        help="If 1, export softmax-normalized top-1 prob among context candidates (used for confident-wrong diagnostics).",
    )
    ap.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for context probability export (only used when --export-context-probs=1).",
    )
    ap.add_argument(
        "--cache-all-scores",
        type=int,
        default=0,
        help="If 1, cache full ranked query scores in llm_cache.jsonl (larger file, speeds multi-seed runs).",
    )

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _force_vigor_referit3d(repo_root)

    from referit3d.in_out.neural_net_oriented import (  # type: ignore
        load_scan_related_data,
        load_referential_data,
        compute_auxiliary_data,
    )
    from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders  # type: ignore
    from referit3d.in_out.cuboid import iou_3d  # type: ignore
    from baseline.api.baseline_interface import BaselineModelAPI  # type: ignore

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vigor relies on env for several data-path choices.
    os.environ.setdefault("VIGOR_USE_ALL_OBJECTS", os.environ.get("VIGOR_USE_ALL_OBJECTS", "1"))
    os.environ.setdefault("VIGOR_ALLOW_UNIQUE_TEST", os.environ.get("VIGOR_ALLOW_UNIQUE_TEST", "1"))
    os.environ.setdefault("VIGOR_USE_PRED_BOX_INFO", os.environ.get("VIGOR_USE_PRED_BOX_INFO", "1"))

    # Build data loaders using Vigor's canonical CSV parsing + filtering.
    scannet_file = str(args.scannet_file)
    referit_csv = str(args.referit3d_file)
    all_scans, scans_split, class_to_idx = load_scan_related_data(scannet_file)

    # Minimal args namespace for Vigor helpers.
    class _Args:
        # Be robust to Vigor utility functions accessing optional args fields.
        # If a field is missing, treat it as "not provided" (None/False-y).
        def __getattr__(self, name: str):
            return None

    a = _Args()
    a.mentions_target_class_only = bool(int(args.mentions_target_class_only))
    a.vocab_file = None
    a.min_word_freq = 3
    a.max_seq_len = int(args.max_seq_len)
    a.augment_with_sr3d = None
    a.s_vs_n_weight = None
    a.n_workers = int(args.n_workers)
    a.max_distractors = int(args.max_distractors)
    a.max_test_objects = int(args.max_test_objects)
    a.unit_sphere_norm = bool(int(args.unit_sphere_norm))
    a.points_per_object = int(args.points_per_object)
    a.mode = "evaluate"
    a.lang_multilabel = True
    a.multilabel_pretraining = True
    a.cascading = True
    a.order_len = int(args.order_len)
    a.batch_size = int(args.batch_size)
    a.random_seed = 2020
    a.mask3d_feature_root = str(args.mask3d_feature_root)
    a.mask3d_feature_root_test = str(args.mask3d_feature_root_test)

    referit_data = load_referential_data(a, referit_csv, scans_split)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans, a)
    def _make_test_loader(seed: int):
        a.random_seed = int(seed)
        data_loaders = make_data_loaders(a, referit_data, vocab, class_to_idx, all_scans, mean_rgb)
        return data_loaders["test"]

    # Baseline model API: provides <ref> generation + query matching.
    api_cfg = {
        "checkpoint": str(args.g3dllm_ckpt),
        "scannet_processed_root": str(args.g3dllm_scannet_root),
        "config_path": str(args.config_root),
        "data_config": str(args.g3dllm_data_config),
        "model_config": str(args.g3dllm_model_config),
        "trainer_config": str(args.g3dllm_trainer_config),
        "llm_config": str(args.g3dllm_llm_config),
        "llm_data_config": str(args.g3dllm_llm_data_config),
        "split_type": "validation",
    }
    api = BaselineModelAPI(api_cfg)

    # LLM cache: key = (scan_id, tokens_string)
    llm_cache_path = Path(args.llm_cache) if args.llm_cache else (out_dir / "llm_cache.jsonl")
    llm_cache: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(llm_cache_path):
        k = row.get("key")
        if isinstance(k, str) and k:
            llm_cache[k] = row

    feat_test_root = Path(args.mask3d_feature_root_test)
    feat_cache: Dict[str, Dict[str, Any]] = {}

    def _load_feat(scene_id: str) -> Dict[str, Any]:
        if scene_id in feat_cache:
            return feat_cache[scene_id]
        p = feat_test_root / f"{scene_id}.pt"
        d = torch.load(p, map_location="cpu")
        if not isinstance(d, dict):
            raise RuntimeError(f"Unexpected feature type for {p}: {type(d).__name__}")
        feat_cache[scene_id] = d
        return d

    max_examples = int(args.max_examples)
    topk = max(1, int(args.topk))
    print_generation_first = max(0, int(args.print_generation_first))
    wrap_grounding_question = bool(int(args.wrap_grounding_question))
    debug_gt_rank_first = max(0, int(args.debug_gt_rank_first))
    scene_id_filter = {s.strip() for s in str(args.scene_ids).split(",") if s.strip()} or None
    restrict_to_context = bool(int(args.restrict_to_context))
    export_context_probs = bool(int(args.export_context_probs))
    cache_all_scores = bool(int(args.cache_all_scores))

    seeds: List[int] = []
    for s in str(args.seeds).split(","):
        s = s.strip()
        if not s:
            continue
        try:
            seeds.append(int(s))
        except Exception:
            continue
    if not seeds:
        seeds = [2020]

    grounding_q_template = os.environ.get(
        "REFERIT_GROUNDING_Q_TEMPLATE",
        "Can you help me find what's described here: {grounding_text}?",
    )
    def _build_prompt(utterance: str) -> str:
        u = str(utterance).strip()
        if not wrap_grounding_question:
            return u
        # Avoid awkward punctuation like ".?" when the raw utterance is a sentence.
        u = u.rstrip().rstrip(".").rstrip(";").rstrip(",").rstrip("!").rstrip("?").strip()
        # Avoid double-wrapping if upstream already provides a question-like prompt.
        u_low = u.lower()
        if "described here:" in u_low or u_low.startswith("can you ") or u_low.startswith("could you ") or u.endswith("?"):
            return u
        try:
            return grounding_q_template.format(grounding_text=u)
        except Exception:
            return f"Can you help me find what's described here: {u}?"

    # Per-example dump (optional, but useful).
    pred_rows_path = out_dir / "pred_rows.jsonl"
    if pred_rows_path.exists():
        pred_rows_path.unlink()

    matcher = getattr(api, "get_topk_query_ids_for_prompt_light", None)
    if matcher is None:
        matcher = getattr(getattr(api, "interface", None), "get_topk_query_ids_for_prompt_light", None)
    if matcher is None:
        raise AttributeError(
            "Neither BaselineModelAPI nor its `.interface` exposes `get_topk_query_ids_for_prompt_light`."
        )

    def _map_obj_to_q(obj_id: int, gt_map: Any) -> Optional[int]:
        try:
            if isinstance(gt_map, dict):
                if obj_id in gt_map:
                    return _safe_int(gt_map[obj_id], default=None)
                if (obj_id + 1) in gt_map:
                    return _safe_int(gt_map[obj_id + 1], default=None)
        except Exception:
            return None
        return None

    all_seed_metrics: Dict[str, Any] = {"seeds": seeds, "per_seed": {}}

    for seed_idx, seed in enumerate(seeds):
        # Metrics (per seed)
        n_total = 0
        query_acc_all = AvgMeter()
        q_star_found = AvgMeter()
        q_hat_found = AvgMeter()
        q_hat_in_range = AvgMeter()
        iou_all = AvgMeter()
        acc25_all = AvgMeter()
        acc50_all = AvgMeter()
        iou_oracle = AvgMeter()
        acc25_oracle = AvgMeter()
        acc50_oracle = AvgMeter()
        acc25_u = AvgMeter()
        acc50_u = AvgMeter()
        acc25_m = AvgMeter()
        acc50_m = AvgMeter()
        iou_correct = AvgMeter()
        acc25_correct = AvgMeter()
        acc50_correct = AvgMeter()
        iou_wrong = AvgMeter()
        acc25_wrong = AvgMeter()
        acc50_wrong = AvgMeter()

        # Context-restricted metrics (optional)
        query_acc_ctx = AvgMeter()
        q_star_in_ctx = AvgMeter()
        q_hat_ctx_found = AvgMeter()
        iou_ctx = AvgMeter()
        acc25_ctx = AvgMeter()
        acc50_ctx = AvgMeter()
        confident_wrong_ctx = AvgMeter()

        loader = _make_test_loader(seed)
        for batch_idx, batch in enumerate(loader):
            if max_examples > 0 and n_total >= max_examples:
                break

            scan_ids = batch.get("scan_id")
            tokens = batch.get("tokens")
            utterances = batch.get("utterance", None)
            stimulus_ids = batch.get("stimulus_id", None)
            target_pos = batch.get("target_pos")
            gt_box_corners = batch.get("gt_box_corners")  # [B,max_ctx,8,3]
            inst_ids = batch.get("instance_ids")          # [B,max_ctx]
            obj_mask = batch.get("obj_mask")              # [B,max_ctx,1]
            cls_labels = batch.get("class_labels")        # [B,max_ctx]
            ori_order_len = batch.get("ori_order_len", None)
            context_size_t = batch.get("context_size", None)

            if not isinstance(scan_ids, (list, tuple)) or not isinstance(tokens, (list, tuple)):
                raise RuntimeError("Unexpected collate layout: expected list scan_id/tokens.")
            if utterances is not None and not isinstance(utterances, (list, tuple)):
                utterances = None
            if stimulus_ids is not None and not isinstance(stimulus_ids, (list, tuple)):
                stimulus_ids = None
            if not torch.is_tensor(target_pos) or not torch.is_tensor(gt_box_corners) or not torch.is_tensor(inst_ids):
                raise RuntimeError("Missing required tensors in batch (target_pos/gt_box_corners/instance_ids).")

            B = int(target_pos.shape[0])
            for b in range(B):
                if max_examples > 0 and n_total >= max_examples:
                    break
                scene_id = str(scan_ids[b])
                if scene_id_filter is not None and scene_id not in scene_id_filter:
                    continue

                if utterances is not None:
                    utt = str(utterances[b])
                else:
                    t = tokens[b]
                    if isinstance(t, str):
                        utt = t
                    elif isinstance(t, (list, tuple)):
                        utt = " ".join(str(x) for x in t)
                    else:
                        utt = str(t)
                prompt = _build_prompt(utt)
                stimulus_id = str(stimulus_ids[b]) if stimulus_ids is not None else None

                tpos = int(target_pos[b].item())
                gt_c = gt_box_corners[b, tpos].detach().cpu().numpy().astype(np.float32)

                feat = _load_feat(scene_id)
                object_queries = feat.get("object_queries", None)
                match_queries = feat.get("queries_normalized_embed", None)
                pred_aabb = feat.get("pred_aabb", None)
                gt_map = feat.get("gt_to_query_map", None) or {}
                if not torch.is_tensor(object_queries):
                    raise RuntimeError(f"Feature file missing object_queries: {scene_id}")
                object_queries_t = object_queries.detach().cpu()
                match_queries_t: Optional[torch.Tensor] = None
                if torch.is_tensor(match_queries):
                    match_queries_t = match_queries.detach().cpu()
                pred_aabb_np = pred_aabb.detach().cpu().numpy() if torch.is_tensor(pred_aabb) else None

                obj_id = int(inst_ids[b, tpos].item())
                q_star = _map_obj_to_q(obj_id, gt_map)
                hit_raw = bool(isinstance(gt_map, dict) and obj_id in gt_map and q_star is not None)
                hit_plus1 = bool(isinstance(gt_map, dict) and (obj_id + 1) in gt_map and q_star is not None and not hit_raw)

                # Determine oracle L and context size from the loader (preferred for consistency with Vigor).
                L_oracle = None
                if torch.is_tensor(ori_order_len):
                    try:
                        L_oracle = int(ori_order_len[b].item())
                    except Exception:
                        L_oracle = None
                ctx_size = None
                if torch.is_tensor(context_size_t):
                    try:
                        ctx_size = int(context_size_t[b].item())
                    except Exception:
                        ctx_size = None

                # Cache key must include the actual prompt to avoid mixing wrapped/unwrapped runs.
                key = f"{scene_id}|{prompt}"
                cached = llm_cache.get(key)
                scores_all = None
                topk_q = None

                # Debug mode: compute full ranking for q_star without polluting cache.
                want_all_scores = bool(export_context_probs or restrict_to_context)
                need_debug_all = bool(debug_gt_rank_first and n_total < debug_gt_rank_first)
                if need_debug_all:
                    want_all_scores = True

                if want_all_scores and cached and isinstance(cached.get("scores_all"), list) and cached.get("scores_all"):
                    scores_all = cached["scores_all"]
                elif cached and isinstance(cached.get("topk"), list) and cached.get("topk") and not want_all_scores:
                    topk_q = cached["topk"]

                if want_all_scores and scores_all is None:
                    if pred_aabb_np is None:
                        scores_all = []
                    else:
                        k_all = int(pred_aabb_np.shape[0])
                        pairs = matcher(
                            prompt=prompt,
                            object_queries=object_queries_t.to(getattr(api, "device", "cpu")),
                            match_queries=(
                                match_queries_t.to(getattr(api, "device", "cpu"))
                                if match_queries_t is not None
                                else None
                            ),
                            k=k_all,
                            print_generation=bool(print_generation_first and n_total < print_generation_first),
                        )
                        scores_all = [{"qidx": int(q), "score": float(s)} for q, s in (pairs or [])]
                    # Keep full scores in memory for multi-seed reuse; persist to disk only if requested.
                    mem_row = {"key": key, "scene_id": scene_id, "utt": utt, "prompt": prompt}
                    mem_row["scores_all"] = scores_all
                    mem_row["topk"] = (scores_all[:topk] if scores_all else [])
                    disk_row = dict(mem_row)
                    if not cache_all_scores:
                        disk_row.pop("scores_all", None)
                    _append_jsonl(llm_cache_path, disk_row)
                    llm_cache[key] = mem_row

                if topk_q is None:
                    if scores_all is not None and scores_all:
                        topk_q = scores_all[:topk]
                    else:
                        # Fall back to top-k only mode (smaller cache).
                        if cached and isinstance(cached.get("topk"), list) and cached.get("topk"):
                            topk_q = cached["topk"]
                        else:
                            pairs = matcher(
                                prompt=prompt,
                                object_queries=object_queries_t.to(getattr(api, "device", "cpu")),
                                match_queries=(
                                    match_queries_t.to(getattr(api, "device", "cpu"))
                                    if match_queries_t is not None
                                    else None
                                ),
                                k=topk,
                                print_generation=bool(print_generation_first and n_total < print_generation_first),
                            )
                            topk_q = [{"qidx": int(q), "score": float(s)} for q, s in (pairs or [])]
                            _append_jsonl(llm_cache_path, {"key": key, "scene_id": scene_id, "utt": utt, "prompt": prompt, "topk": topk_q})
                            llm_cache[key] = {"key": key, "topk": topk_q}

                top1_score = float(topk_q[0]["score"]) if topk_q else None
                q_hat = int(topk_q[0]["qidx"]) if topk_q else -1

                gt_rank = None
                gt_score = None
                if scores_all is not None and q_star is not None:
                    for rnk, d in enumerate(scores_all, start=1):
                        if int(d.get("qidx", -1)) == int(q_star):
                            gt_rank = int(rnk)
                            gt_score = float(d.get("score"))
                            break

                is_correct = int(q_star is not None and q_hat == int(q_star))
                query_acc_all.update(float(is_correct), 1)
                q_star_found.update(float(q_star is not None), 1)
                q_hat_found.update(float(q_hat >= 0), 1)
                q_hat_in_range.update(
                    float(
                        pred_aabb_np is not None
                        and q_hat >= 0
                        and q_hat < int(pred_aabb_np.shape[0])
                    ),
                    1,
                )

                # Compute IoU for the all-queries prediction.
                iou = 0.0
                if pred_aabb_np is not None and 0 <= q_hat < int(pred_aabb_np.shape[0]):
                    a6 = np.asarray(pred_aabb_np[q_hat], dtype=np.float32).reshape(6)
                    if np.isfinite(a6).all() and (a6[3:6] > a6[0:3]).all():
                        pred_c = _aabb_to_corners(a6)
                        v = iou_3d(pred_c, gt_c)
                        iou = float(v[0] if isinstance(v, tuple) else v)
                iou_all.update(iou, 1)
                acc25_all.update(float(iou >= 0.25), 1)
                acc50_all.update(float(iou >= 0.50), 1)

                # Oracle IoU for diagnosis: only meaningful when q_star exists and points to a valid query.
                iou_star = None
                if pred_aabb_np is not None and q_star is not None and 0 <= int(q_star) < int(pred_aabb_np.shape[0]):
                    a6s = np.asarray(pred_aabb_np[int(q_star)], dtype=np.float32).reshape(6)
                    if np.isfinite(a6s).all() and (a6s[3:6] > a6s[0:3]).all():
                        pred_cs = _aabb_to_corners(a6s)
                        vv = iou_3d(pred_cs, gt_c)
                        iou_star = float(vv[0] if isinstance(vv, tuple) else vv)
                if iou_star is not None:
                    iou_oracle.update(float(iou_star), 1)
                    acc25_oracle.update(float(iou_star >= 0.25), 1)
                    acc50_oracle.update(float(iou_star >= 0.50), 1)

                if is_correct:
                    iou_correct.update(iou, 1)
                    acc25_correct.update(float(iou >= 0.25), 1)
                    acc50_correct.update(float(iou >= 0.50), 1)
                else:
                    iou_wrong.update(iou, 1)
                    acc25_wrong.update(float(iou >= 0.25), 1)
                    acc50_wrong.update(float(iou >= 0.50), 1)

                # Unique/Multiple split (same-class count in valid context)
                tag = None
                try:
                    if torch.is_tensor(cls_labels):
                        tgt_cls = int(cls_labels[b, tpos].item())
                        if torch.is_tensor(obj_mask):
                            valid = (obj_mask[b, :, 0] > 0.5).detach().cpu().numpy().astype(bool)
                        else:
                            valid = np.ones((int(cls_labels.shape[1]),), dtype=bool)
                        cls_row = cls_labels[b].detach().cpu().numpy()
                        same = (cls_row == tgt_cls) & valid & (cls_row >= 0)
                        n_same = int(same.astype(np.int64).sum())
                        tag = "unique" if n_same <= 1 else "multiple"
                except Exception:
                    tag = None
                if tag == "unique":
                    acc25_u.update(float(iou >= 0.25), 1)
                    acc50_u.update(float(iou >= 0.50), 1)
                elif tag == "multiple":
                    acc25_m.update(float(iou >= 0.25), 1)
                    acc50_m.update(float(iou >= 0.50), 1)

                # Context-restricted prediction (align with candidate set), optionally exporting softmax probs.
                ctx_qids: List[int] = []
                if torch.is_tensor(obj_mask):
                    valid_ctx = (obj_mask[b, :, 0] > 0.5).detach().cpu().numpy().astype(bool)
                    ctx_obj_ids = inst_ids[b].detach().cpu().numpy().astype(np.int64)[valid_ctx].tolist()
                    for oid in ctx_obj_ids:
                        q = _map_obj_to_q(int(oid), gt_map)
                        if q is not None:
                            ctx_qids.append(int(q))
                # De-dup for scoring
                ctx_qids = list(dict.fromkeys(ctx_qids))
                ctx_top1_prob = None
                ctx_top1_score = None
                ctx_gt_rank = None
                ctx_gt_score = None
                q_hat_ctx = -1

                if ctx_qids and scores_all is not None:
                    scores_dict = {int(d.get("qidx", -1)): float(d.get("score")) for d in scores_all if "qidx" in d and "score" in d}
                    ctx_scores = np.asarray([scores_dict.get(int(q), -1e9) for q in ctx_qids], dtype=np.float64)
                    if ctx_scores.size:
                        best_i = int(np.argmax(ctx_scores))
                        q_hat_ctx = int(ctx_qids[best_i])
                        ctx_top1_score = float(ctx_scores[best_i])
                        if export_context_probs:
                            probs = _stable_softmax(ctx_scores, temperature=float(args.softmax_temperature))
                            ctx_top1_prob = float(probs[best_i]) if probs.size else None
                        if q_star is not None:
                            # Rank within context (1=best)
                            order = np.argsort(-ctx_scores)
                            for rnk, idx in enumerate(order.tolist(), start=1):
                                if int(ctx_qids[int(idx)]) == int(q_star):
                                    ctx_gt_rank = int(rnk)
                                    ctx_gt_score = float(ctx_scores[int(idx)])
                                    break

                is_correct_ctx = int(q_star is not None and q_hat_ctx >= 0 and q_hat_ctx == int(q_star))
                if restrict_to_context:
                    query_acc_ctx.update(float(is_correct_ctx), 1)
                if q_star is not None and ctx_qids:
                    q_star_in_ctx.update(float(int(q_star) in set(ctx_qids)), 1)
                else:
                    q_star_in_ctx.update(0.0, 1)
                q_hat_ctx_found.update(float(q_hat_ctx >= 0), 1)

                # Context IoU (if ctx prediction exists)
                iou_c = 0.0
                if pred_aabb_np is not None and 0 <= q_hat_ctx < int(pred_aabb_np.shape[0]):
                    a6c = np.asarray(pred_aabb_np[q_hat_ctx], dtype=np.float32).reshape(6)
                    if np.isfinite(a6c).all() and (a6c[3:6] > a6c[0:3]).all():
                        pred_cc = _aabb_to_corners(a6c)
                        vv = iou_3d(pred_cc, gt_c)
                        iou_c = float(vv[0] if isinstance(vv, tuple) else vv)
                iou_ctx.update(iou_c, 1)
                acc25_ctx.update(float(iou_c >= 0.25), 1)
                acc50_ctx.update(float(iou_c >= 0.50), 1)
                if export_context_probs and ctx_top1_prob is not None:
                    cw = float((is_correct_ctx < 0.5) and (float(ctx_top1_prob) >= 0.9))
                    confident_wrong_ctx.update(cw, 1)

                _append_jsonl(
                    pred_rows_path,
                    {
                        "seed": int(seed),
                        "scene_id": scene_id,
                        "stimulus_id": stimulus_id,
                        "utterance": utt,
                        "prompt": prompt,
                        "target_pos": tpos,
                        "target_object_id": obj_id,
                        "context_size": ctx_size,
                        "ori_order_len": L_oracle,
                        "q_star": q_star,
                        "q_hat": q_hat,
                        "is_correct": is_correct,
                        "iou": float(iou),
                        "iou_oracle": float(iou_star) if iou_star is not None else None,
                        "tag": tag,
                        "top1_score": top1_score,
                        "gt_rank": gt_rank,
                        "gt_score": gt_score,
                        "gt_map_n": int(len(gt_map)) if isinstance(gt_map, dict) else None,
                        "q_star_hit_raw": bool(hit_raw),
                        "q_star_hit_plus1": bool(hit_plus1),
                        "q_star_found": bool(q_star is not None),
                        "q_hat_found": bool(q_hat >= 0),
                        "q_hat_in_range": bool(
                            pred_aabb_np is not None
                            and q_hat >= 0
                            and q_hat < int(pred_aabb_np.shape[0])
                        ),
                        "n_queries": int(pred_aabb_np.shape[0]) if pred_aabb_np is not None else None,
                        "ctx_qids_n": int(len(ctx_qids)),
                        "ctx_q_hat": int(q_hat_ctx),
                        "ctx_is_correct": int(is_correct_ctx),
                        "ctx_top1_score": ctx_top1_score,
                        "ctx_top1_prob": ctx_top1_prob,
                        "ctx_gt_rank": ctx_gt_rank,
                        "ctx_gt_score": ctx_gt_score,
                    },
                )

                n_total += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"[g3dllm_refmatch] seed={seed} processed={n_total}", flush=True)

        # Summarize per-seed
        print("")
        print(f"[g3dllm_refmatch] ===== seed={seed} done (n={n_total}) =====")
        print(f"Query-Accuracy(all): {query_acc_all.avg:.4f}")
        if restrict_to_context:
            print(f"Query-Accuracy(context): {query_acc_ctx.avg:.4f}")
        if export_context_probs and confident_wrong_ctx.count:
            print(f"Confident-Wrong(context, prob>=0.9): {confident_wrong_ctx.avg:.4f}")
        print(f"Query-GTMap-Coverage(q_star_found): {q_star_found.avg:.4f}")
        print(f"Query-Prediction-Rate(q_hat_found): {q_hat_found.avg:.4f}")
        print(f"Query-Prediction-InRange(q_hat_in_range): {q_hat_in_range.avg:.4f}")
        print(f"BBox-Acc@IoU(all,0.25/0.50): {acc25_all.avg:.4f} / {acc50_all.avg:.4f}")
        print(f"BBox-Acc@IoU(ctx,0.25/0.50): {acc25_ctx.avg:.4f} / {acc50_ctx.avg:.4f}")
        if iou_oracle.count:
            print(
                f"BBox-Acc@IoU Oracle(0.25/0.50): {acc25_oracle.avg:.4f} / {acc50_oracle.avg:.4f} "
                f"(mean_iou={iou_oracle.avg:.4f}, n={iou_oracle.count})"
            )
        if acc25_u.count > 0 or acc25_m.count > 0:
            print(f"BBox-Acc@IoU Unique(0.25/0.50): {acc25_u.avg:.4f} / {acc50_u.avg:.4f}")
            print(f"BBox-Acc@IoU Multiple(0.25/0.50): {acc25_m.avg:.4f} / {acc50_m.avg:.4f}")
        print(f"BBox-Mean-IoU(all): {iou_all.avg:.4f}")

        metrics_seed = {
            "seed": int(seed),
            "n": int(n_total),
            "query_accuracy_all": query_acc_all.avg,
            "query_accuracy_ctx": query_acc_ctx.avg if restrict_to_context else None,
            "confident_wrong_ctx": confident_wrong_ctx.avg if export_context_probs and confident_wrong_ctx.count else None,
            "q_star_found_rate": q_star_found.avg,
            "q_star_in_ctx_rate": q_star_in_ctx.avg,
            "q_hat_found_rate": q_hat_found.avg,
            "q_hat_in_range_rate": q_hat_in_range.avg,
            "q_hat_ctx_found_rate": q_hat_ctx_found.avg,
            "bbox_acc_iou_0.25_all": acc25_all.avg,
            "bbox_acc_iou_0.50_all": acc50_all.avg,
            "bbox_mean_iou_all": iou_all.avg,
            "bbox_acc_iou_0.25_ctx": acc25_ctx.avg,
            "bbox_acc_iou_0.50_ctx": acc50_ctx.avg,
            "bbox_mean_iou_ctx": iou_ctx.avg,
            "bbox_oracle_acc_iou_0.25": acc25_oracle.avg if acc25_oracle.count else None,
            "bbox_oracle_acc_iou_0.50": acc50_oracle.avg if acc50_oracle.count else None,
            "bbox_oracle_mean_iou": iou_oracle.avg if iou_oracle.count else None,
            "bbox_oracle_n": iou_oracle.count if iou_oracle.count else 0,
            "bbox_acc_iou_0.25_unique": acc25_u.avg if acc25_u.count else None,
            "bbox_acc_iou_0.50_unique": acc50_u.avg if acc50_u.count else None,
            "bbox_acc_iou_0.25_multiple": acc25_m.avg if acc25_m.count else None,
            "bbox_acc_iou_0.50_multiple": acc50_m.avg if acc50_m.count else None,
        }
        (out_dir / f"metrics_seed{seed}.json").write_text(json.dumps(metrics_seed, indent=2), encoding="utf-8")
        all_seed_metrics["per_seed"][str(seed)] = metrics_seed

    print(f"[g3dllm_refmatch] saved: {pred_rows_path}")
    print(f"[g3dllm_refmatch] llm_cache: {llm_cache_path}")

    # Aggregate across seeds (mean/std over per-seed metrics where defined)
    def _mean_std(key: str) -> Dict[str, Any]:
        vals: List[float] = []
        for s in seeds:
            v = all_seed_metrics["per_seed"].get(str(s), {}).get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return {"mean": None, "std": None}
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    all_seed_metrics["mean_std"] = {
        "query_accuracy_all": _mean_std("query_accuracy_all"),
        "query_accuracy_ctx": _mean_std("query_accuracy_ctx"),
        "confident_wrong_ctx": _mean_std("confident_wrong_ctx"),
        "bbox_acc_iou_0.25_all": _mean_std("bbox_acc_iou_0.25_all"),
        "bbox_acc_iou_0.50_all": _mean_std("bbox_acc_iou_0.50_all"),
        "bbox_acc_iou_0.25_ctx": _mean_std("bbox_acc_iou_0.25_ctx"),
        "bbox_acc_iou_0.50_ctx": _mean_std("bbox_acc_iou_0.50_ctx"),
    }
    all_seed_metrics["config"] = {
        "restrict_to_context": bool(restrict_to_context),
        "export_context_probs": bool(export_context_probs),
        "softmax_temperature": float(args.softmax_temperature),
        "confident_thr": 0.9,
    }
    (out_dir / "metrics.json").write_text(json.dumps(all_seed_metrics, indent=2), encoding="utf-8")
    print(f"[g3dllm_refmatch] metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
