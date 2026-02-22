#!/usr/bin/env python3
"""
Export per-scene Mask3D features for release evaluation scripts.

Output format (`<scene_id>.pt`) includes:
- object_queries
- sampled_coords
- gt_to_query_map
- pred_aabb
- pred_box_info
- pred_classes / pred_class_names / pred_scores
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


def _parse_scene_ids(raw: str) -> Optional[List[str]]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return items or None


def _load_label_name_map(repo_root: Path) -> Dict[int, str]:
    label_file = repo_root / "label_database.yaml"
    if not label_file.exists():
        return {}
    try:
        data = yaml.safe_load(label_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        out: Dict[int, str] = {}
        for k, v in data.items():
            try:
                kid = int(k)
            except Exception:
                continue
            if isinstance(v, dict) and "name" in v:
                out[kid] = str(v["name"])
        return out
    except Exception:
        return {}


def _compute_aabb(points_xyz: torch.Tensor, masks: torch.Tensor, thresh: float) -> torch.Tensor:
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3 or masks.ndim != 2:
        return torch.zeros((0, 6), dtype=torch.float32)
    if points_xyz.shape[0] != masks.shape[0]:
        return torch.zeros((masks.shape[1], 6), dtype=torch.float32)

    q = int(masks.shape[1])
    out = torch.zeros((q, 6), dtype=torch.float32)
    bin_mask = masks > float(thresh)
    xyz = points_xyz[:, :3].float()

    for qi in range(q):
        m = bin_mask[:, qi]
        if not bool(m.any()):
            continue
        pts = xyz[m]
        mn = pts.min(dim=0).values
        mx = pts.max(dim=0).values
        out[qi, 0:3] = mn
        out[qi, 3:6] = mx
    return out


def _aabb_to_box_info(aabb: torch.Tensor) -> torch.Tensor:
    if aabb.ndim != 2 or aabb.shape[1] != 6:
        return torch.zeros((0, 4), dtype=torch.float32)
    mn = aabb[:, 0:3]
    mx = aabb[:, 3:6]
    center = (mn + mx) * 0.5
    size = (mx - mn).clamp(min=0.0)
    volume = size[:, 0] * size[:, 1] * size[:, 2]
    out = torch.zeros((aabb.shape[0], 4), dtype=torch.float32)
    out[:, 0:3] = center
    out[:, 3] = volume
    return out


def _build_api_config(args: argparse.Namespace, repo_root: Path) -> Dict:
    scannet_root = args.scannet_root or str(repo_root / "data/SCANNET200_ROOT")
    cfg = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "split_type": args.data_split,
        "scannet_processed_root": scannet_root,
        "config_path": "baseline/core/conf",
        "data_config": args.data_config,
        "model_config": args.model_config,
        "trainer_config": args.trainer_config,
        "llm_config": args.llm_config,
        "llm_data_config": args.llm_data_config,
        "experiment_name": "mask3d_feature_export_release",
        "project_name": "scannet200",
        "topk_per_image": 750,
    }
    if args.data_split == "train":
        # For train export we ask the interface to instantiate train-side data.
        cfg["extra_overrides"] = {"general": {"train_mode": True}}
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Step-2 checkpoint path.")
    p.add_argument("--data-split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--output-dir", required=True, help="Directory to save per-scene *.pt files.")
    p.add_argument("--data-config", default="baseline/core/conf/data/indoor_dialog.yaml")
    p.add_argument("--model-config", default="baseline/core/conf/model/mask3d_lang.yaml")
    p.add_argument("--trainer-config", default="baseline/core/conf/trainer/trainer50.yaml")
    p.add_argument("--llm-config", default="baseline/core/conf/llm/nollm.json")
    p.add_argument("--llm-data-config", default="baseline/core/conf/llm/det10.json")
    p.add_argument("--scannet-root", default=None, help="Processed ScanNet200 root (contains train/validation/test).")
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--max-scenes", type=int, default=0, help="0 means no limit.")
    p.add_argument("--scene-ids", type=str, default="", help="Optional comma-separated scene ids.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from baseline.api.baseline_interface import BaselineModelAPI

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name = _load_label_name_map(repo_root)
    api = BaselineModelAPI(_build_api_config(args, repo_root))

    scene_ids = sorted(api.interface.scene_index.keys())
    scene_filter = _parse_scene_ids(args.scene_ids)
    if scene_filter is not None:
        scene_filter_set = set(scene_filter)
        scene_ids = [sid for sid in scene_ids if sid in scene_filter_set]
    if args.max_scenes and args.max_scenes > 0:
        scene_ids = scene_ids[: int(args.max_scenes)]

    dataset_mode = getattr(api.interface.dataset, "mode", "unknown")
    print(f"[export_mask3d_features] split={args.data_split} dataset_mode={dataset_mode} scenes={len(scene_ids)}")
    print(f"[export_mask3d_features] output_dir={out_dir}")

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for sid in scene_ids:
        out_file = out_dir / f"{sid}.pt"
        if out_file.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            forward = api._run_full_forward(sid)  # release export path; shared with eval runtime.
            if forward is None:
                n_fail += 1
                print(f"[WARN] skip scene={sid}: no forward output")
                continue

            points = torch.as_tensor(forward["points"], dtype=torch.float32)[:, :3].cpu()
            masks = torch.as_tensor(forward["pred_masks_full_res"], dtype=torch.float32).cpu()
            aabb = _compute_aabb(points, masks, args.mask_thresh)
            box_info = _aabb_to_box_info(aabb)

            pred_classes = torch.as_tensor(forward.get("pred_classes", torch.empty(0, dtype=torch.long))).cpu().long()
            pred_scores = torch.as_tensor(forward.get("pred_scores", torch.empty(0, dtype=torch.float32))).cpu().float()
            class_names = [id_to_name.get(int(x.item()), "unknown") for x in pred_classes]

            payload = {
                "scene_id": sid,
                "object_queries": torch.as_tensor(forward["object_queries"]).cpu().float(),
                "sampled_coords": torch.as_tensor(forward["sampled_coords"]).cpu().float(),
                "gt_to_query_map": {int(k): int(v) for k, v in (forward.get("gt_to_query_map") or {}).items()},
                "pred_aabb": aabb.cpu().float(),
                "pred_box_info": box_info.cpu().float(),
                "pred_classes": pred_classes,
                "pred_class_names": class_names,
                "pred_scores": pred_scores,
            }

            qne = forward.get("queries_normalized_embed")
            if torch.is_tensor(qne):
                payload["queries_normalized_embed"] = qne.cpu().float()

            torch.save(payload, out_file)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"[WARN] scene={sid} failed: {exc}")

    print(
        "[export_mask3d_features] done "
        f"ok={n_ok} skip={n_skip} fail={n_fail} total={len(scene_ids)}"
    )


if __name__ == "__main__":
    main()
