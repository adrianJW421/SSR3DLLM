#!/usr/bin/env python3
"""
Probe the alignment between Vigor referit3d instances and exported Mask3D features.

It loads one batch from the Vigor dataloader, then prints:
  - mask3d feature path and shapes
  - first few instance_ids -> query_idx mapping via gt_to_query_map
  - centers vs mask3d coords with L2 distance

Example:
CUDA_VISIBLE_DEVICES=0 python third_party/Vigor/tools/probe_mask3d_mapping.py \
  --scannet-pkl data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl \
  --referit-csv third_party/Vigor/referit3d/data/csv_data/sr3d_train_LLM_step4_485_0.05.csv \
  --mask3d-feats data/MASK3D_FEATS_TRAIN
"""
import os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]  # release repo root
sys.path.insert(0, str(repo_root))  # prefer local packages
sys.path.insert(0, str(repo_root / "src"))
import argparse
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet-pkl", required=True, help="scannet keep_all_points_with_global_scan_alignment.pkl")
    parser.add_argument("--referit-csv", required=True, help="sr3d/nr3d csv")
    parser.add_argument("--mask3d-feats", required=True, help="dir with scene*.pt (object_queries, gt_to_query_map)")
    parser.add_argument("--mask3d-dim", type=int, default=128, help="dim of object_queries")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default="/tmp/vigor_probe", help="dummy log dir required by parser")
    parser.add_argument("--scene-ids", type=str, default=None,
                        help="comma-separated scene ids to probe; if set, will filter dataloader to these scenes only")
    args = parser.parse_args()

    from third_party.Vigor.referit3d.in_out.arguments import parse_arguments
    from third_party.Vigor.referit3d.in_out.neural_net_oriented import (
        load_scan_related_data,
        load_referential_data,
        compute_auxiliary_data,
    )
    from third_party.Vigor.referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders

    # Minimal args for loader; log_dir is mandatory for this parser.
    vigor_args = parse_arguments(
        [
            "-scannet-file",
            args.scannet_pkl,
            "-referit3D-file",
            args.referit_csv,
            "--batch-size",
            str(args.batch_size),
            "--n-workers",
            str(args.num_workers),
            "--max-train-epochs",
            "1",
            "--log-dir",
            args.log_dir,
        ]
    )
    vigor_args.mask3d_feature_root = args.mask3d_feats
    vigor_args.mask3d_feature_dim = args.mask3d_dim

    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(vigor_args.scannet_file)
    referit_data = load_referential_data(vigor_args, vigor_args.referit3D_file, scans_split)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, vigor_args)
    loaders = make_data_loaders(vigor_args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)

    # Optionally filter to specific scene ids
    if args.scene_ids:
        target_sids = set([s.strip() for s in args.scene_ids.split(",") if s.strip()])
    else:
        target_sids = None

    batch = None
    for b in loaders["train"]:
        sid = b["scan_id"][0]
        if target_sids is None or sid in target_sids:
            batch = b
            break
    if batch is None:
        raise RuntimeError("No batch found matching requested scene_ids.")
    scan_id = batch["scan_id"][0]
    instance_ids = np.array(batch["instance_ids"][0])  # [max_context]
    center_coors = batch["center_coors"][0].numpy()  # [max_context,3]
    obj_mask = batch["obj_mask"][0].squeeze(-1).numpy().astype(bool)

    mask3d_path = batch["mask3d_feature_path"][0]
    feat = torch.load(mask3d_path, map_location="cpu")
    gt_map = feat.get("gt_to_query_map", {}) or {}
    coords = feat.get("sampled_coords", None)
    coords = coords.numpy() if coords is not None else None

    print(f"=== scan_id: {scan_id}")
    print(f"mask3d feature path: {mask3d_path}")
    print(f"gt_to_query_map size: {len(gt_map)}")
    print(f"object_queries shape: {feat['object_queries'].shape}")
    print(f"sampled_coords shape: {None if coords is None else coords.shape}")

    valid_indices = np.where(obj_mask)[0][:10]
    for idx in valid_indices:
        inst_id = int(instance_ids[idx])
        qidx = gt_map.get(inst_id, None)
        c = center_coors[idx]
        if qidx is None:
            print(f"[NO MAP] inst={inst_id} center={c}")
        else:
            mc = coords[qidx] if coords is not None else None
            dist = np.linalg.norm(c - mc) if mc is not None else None
            print(f"[MAP] inst={inst_id} -> query={qidx}, center={c}, mask3d_coord={mc}, l2={dist}")


if __name__ == "__main__":
    main()
