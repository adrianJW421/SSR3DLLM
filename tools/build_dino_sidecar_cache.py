#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_dict(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value)} from {path}")
    return value


def _float_2d(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be 2D, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    return tensor


def _center_size_to_aabb(locs: torch.Tensor, name: str) -> torch.Tensor:
    if locs.ndim != 2 or locs.shape[1] < 6:
        raise RuntimeError(f"{name} locs must be [N,6], got {tuple(locs.shape)}")
    size = locs[:, 3:6]
    if not torch.all(size > 0):
        raise RuntimeError(f"{name} locs contain non-positive box size")
    half = size / 2.0
    return torch.cat([locs[:, 0:3] - half, locs[:, 0:3] + half], dim=1)


def _qcond_aabb(value: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    aabb = _float_2d(value, name)
    if aabb.shape[1] != 6:
        raise RuntimeError(f"{name} must be [N,6], got {tuple(aabb.shape)}")
    valid = torch.all(aabb[:, 3:6] > aabb[:, 0:3], dim=1)
    return aabb, valid


def _pairwise_iou(aabb_a: torch.Tensor, aabb_b: torch.Tensor) -> torch.Tensor:
    min_a = aabb_a[:, None, 0:3]
    max_a = aabb_a[:, None, 3:6]
    min_b = aabb_b[None, :, 0:3]
    max_b = aabb_b[None, :, 3:6]
    inter_min = torch.maximum(min_a, min_b)
    inter_max = torch.minimum(max_a, max_b)
    inter = torch.clamp(inter_max - inter_min, min=0.0).prod(dim=-1)
    vol_a = torch.clamp(max_a - min_a, min=0.0).prod(dim=-1)
    vol_b = torch.clamp(max_b - min_b, min=0.0).prod(dim=-1)
    union = vol_a + vol_b - inter
    if not torch.all(union > 0):
        raise RuntimeError("AABB IoU union has non-positive entries")
    return inter / union


def _item_key(scene_id: str, obj_id: int) -> str:
    return f"{scene_id}_{int(obj_id):02d}"


def _obj_ids(raw: Any, count: int, scene_id: str) -> list[int]:
    if raw is None:
        return list(range(count))
    values = raw.detach().cpu().tolist() if torch.is_tensor(raw) else list(raw)
    if len(values) < count:
        raise RuntimeError(f"{scene_id}: obj_ids shorter than locs")
    return [int(x) for x in values[:count]]


def _load_source_features(paths: list[Path]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for path in paths:
        data = _load_dict(path, f"source feature file {path}")
        overlap = set(merged).intersection(data)
        if overlap:
            raise RuntimeError(f"duplicate source feature key: {sorted(overlap)[0]}")
        for key, value in data.items():
            feat = torch.as_tensor(value, dtype=torch.float32)
            if feat.ndim != 1:
                raise RuntimeError(f"{key}: source DINO feature must be 1D, got {tuple(feat.shape)}")
            merged[str(key)] = feat
    if not merged:
        raise RuntimeError("no source DINO features loaded")
    return merged


class SourceIndex:
    def __init__(self, feature_paths: list[Path], attribute_paths: list[Path], scene_ids: set[str]) -> None:
        features = _load_source_features(feature_paths)
        self.scenes: dict[str, dict[str, Any]] = {}
        for path in attribute_paths:
            data = _load_dict(path, f"source attribute file {path}")
            for scene_id_raw, record in data.items():
                scene_id = str(scene_id_raw)
                if scene_id not in scene_ids:
                    continue
                if scene_id in self.scenes:
                    raise RuntimeError(f"duplicate source scene: {scene_id}")
                if not isinstance(record, dict) or "locs" not in record:
                    raise RuntimeError(f"{scene_id}: source attribute record must contain locs")
                locs = _float_2d(record["locs"], f"{scene_id}.locs")
                obj_ids = _obj_ids(record.get("obj_ids"), int(locs.shape[0]), scene_id)
                rows: list[int] = []
                ids: list[int] = []
                feats: list[torch.Tensor] = []
                for row, obj_id in enumerate(obj_ids):
                    key = _item_key(scene_id, obj_id)
                    if key not in features:
                        continue
                    rows.append(row)
                    ids.append(obj_id)
                    feats.append(features[key])
                if not feats:
                    raise RuntimeError(f"{scene_id}: no source DINO features matched attributes")
                selected_locs = locs[torch.as_tensor(rows, dtype=torch.long)]
                self.scenes[scene_id] = {
                    "aabb": _center_size_to_aabb(selected_locs, scene_id),
                    "obj_ids": torch.as_tensor(ids, dtype=torch.long),
                    "features": torch.stack(feats, dim=0),
                    "attribute_file": str(path),
                    "num_attribute_objects": int(locs.shape[0]),
                    "num_feature_objects": int(len(feats)),
                }
        missing = sorted(scene_ids.difference(self.scenes))
        if missing:
            raise RuntimeError(f"source attributes/features missing scenes, first={missing[0]}, count={len(missing)}")

    def get(self, scene_id: str) -> dict[str, Any]:
        return self.scenes[scene_id]


def _scene_id_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 3 or not parts[1].startswith("scene"):
        raise RuntimeError(f"cannot parse scene_id from sample filename: {path.name}")
    return f"{parts[1]}_{parts[2]}"


def _sample_paths(root: Path, max_samples: int, num_shards: int, shard_index: int) -> list[Path]:
    sample_root = root / "samples"
    if not sample_root.is_dir():
        raise FileNotFoundError(f"sample directory not found: {sample_root}")
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    paths = [p for i, p in enumerate(sorted(sample_root.glob("*.pt"))) if i % num_shards == shard_index]
    if max_samples > 0:
        paths = paths[:max_samples]
    if not paths:
        raise RuntimeError(f"no qcond sample files selected under {sample_root}")
    return paths


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"n": 0.0}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "ge_0.25": float((arr >= 0.25).mean()),
        "ge_0.50": float((arr >= 0.50).mean()),
        "ge_0.75": float((arr >= 0.75).mean()),
    }


def _normalize_gt_map(value: Any, path: Path) -> dict[int, int]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: qcond sample must contain gt_to_query_map dict")
    return {int(k): int(v) for k, v in value.items()}


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, action="append", required=True)
    parser.add_argument("--source-attributes", type=Path, action="append", required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = _sample_paths(args.cache_root, args.max_samples, args.num_shards, args.shard_index)
    scene_ids = {_scene_id_from_path(path) for path in samples}
    source = SourceIndex(args.source_features, args.source_attributes, scene_ids)
    output_samples = args.output_root / "samples"
    output_samples.mkdir(parents=True, exist_ok=True)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    manifest_path = args.output_root / f"manifest.shard{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    summary_path = args.output_root / f"summary.shard{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    ious: list[float] = []
    written = 0
    reused = 0
    invalid_rows = 0
    total_rows = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, sample_path in enumerate(samples):
            out_path = output_samples / sample_path.name
            if out_path.is_file() and not args.overwrite:
                reused += 1
                manifest.write(json.dumps({"status": "reused", "path": str(out_path)}, ensure_ascii=False) + "\n")
                continue
            sample = _load_dict(sample_path, f"qcond sample {sample_path}")
            scene_id = str(sample["scene_id"])
            qcond_aabb, qcond_valid = _qcond_aabb(sample["pred_aabb"], f"{sample_path}.pred_aabb")
            src = source.get(scene_id)
            matched_features = torch.zeros((qcond_aabb.shape[0], src["features"].shape[1]), dtype=out_dtype)
            matched_obj_ids = torch.full((qcond_aabb.shape[0],), -1, dtype=torch.long)
            matched_iou = torch.zeros((qcond_aabb.shape[0],), dtype=torch.float32)
            if bool(qcond_valid.any().item()):
                rows = torch.nonzero(qcond_valid, as_tuple=False).flatten()
                iou = _pairwise_iou(qcond_aabb[rows], src["aabb"])
                best_iou, best_idx = torch.max(iou, dim=1)
                matched_features[rows] = src["features"][best_idx].to(out_dtype)
                matched_obj_ids[rows] = src["obj_ids"][best_idx]
                matched_iou[rows] = best_iou.float()
                ious.extend(float(x) for x in best_iou.tolist())
            payload = {
                "scene_id": scene_id,
                "sample_uid": str(sample.get("sample_uid", sample_path.stem)),
                "gt_to_query_map": _normalize_gt_map(sample.get("gt_to_query_map"), sample_path),
                "proposal_dino_features": matched_features,
                "proposal_dino_valid_mask": qcond_valid,
                "proposal_dino_match_iou": matched_iou,
                "proposal_dino_match_obj_ids": matched_obj_ids,
                "source_feature_files": [str(p) for p in args.source_features],
                "source_attribute_file": src["attribute_file"],
            }
            _save(out_path, payload)
            written += 1
            total_rows += int(qcond_aabb.shape[0])
            invalid_rows += int((~qcond_valid).sum().item())
            manifest.write(
                json.dumps(
                    {
                        "status": "written",
                        "index": idx,
                        "source": str(sample_path),
                        "path": str(out_path),
                        "scene_id": scene_id,
                        "valid_proposals": int(qcond_valid.sum().item()),
                        "mean_match_iou": float(matched_iou[qcond_valid].mean().item()) if bool(qcond_valid.any().item()) else 0.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "cache_root": str(args.cache_root),
        "output_root": str(args.output_root),
        "source_features": [str(p) for p in args.source_features],
        "source_attributes": [str(p) for p in args.source_attributes],
        "written": written,
        "reused": reused,
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "match_iou_valid": _stats(ious),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
