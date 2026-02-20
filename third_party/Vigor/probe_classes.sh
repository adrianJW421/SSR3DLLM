#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT

export PROBE_SCANNET_PKL="${PROBE_SCANNET_PKL:-${REPO_ROOT}/data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl}"
export PROBE_SR3D_CSV="${PROBE_SR3D_CSV:-${REPO_ROOT}/third_party/Vigor/referit3d/data/csv_data/sr3d_train_LLM_step4_485_0.05.csv}"
export PROBE_VIGOR_FEAT_ROOT="${PROBE_VIGOR_FEAT_ROOT:-${REPO_ROOT}/data/MASK3D_FEATS_TRAIN}"
export PROBE_LABEL_DB_YAML="${PROBE_LABEL_DB_YAML:-${REPO_ROOT}/data/SCANNET200_ROOT/label_database.yaml}"

python - <<'PY'
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

scannet_pkl = Path(os.environ["PROBE_SCANNET_PKL"]).resolve()
sr3d_csv = Path(os.environ["PROBE_SR3D_CSV"]).resolve()
vigor_feat_root = Path(os.environ["PROBE_VIGOR_FEAT_ROOT"]).resolve()
label_db_yaml = Path(os.environ["PROBE_LABEL_DB_YAML"]).resolve()

from third_party.Vigor.referit3d.in_out.arguments import parse_arguments
from third_party.Vigor.referit3d.in_out.neural_net_oriented import (
    compute_auxiliary_data,
    load_referential_data,
    load_scan_related_data,
)
from third_party.Vigor.referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders

args = parse_arguments(
    [
        "-scannet-file",
        str(scannet_pkl),
        "-referit3D-file",
        str(sr3d_csv),
        "--batch-size",
        "1",
        "--n-workers",
        "2",
        "--max-train-epochs",
        "1",
        "--log-dir",
        "/tmp/vigor_cls_probe",
    ]
)

all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
referit_data = load_referential_data(args, args.referit3D_file, scans_split)
mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
batch = next(iter(loaders["train"]))

scene_id = batch["scan_id"][0]
instance_ids = batch["instance_ids"][0].numpy()
class_labels = batch["class_labels"][0].numpy()
obj_mask = batch["obj_mask"][0].squeeze(-1).numpy().astype(bool)

print(f"=== scene_id: {scene_id}")
print("Vigor class_to_idx size (incl pad):", len(class_to_idx))

idx2class = {v: k for k, v in class_to_idx.items()}

feat_path = vigor_feat_root / f"{scene_id}.pt"
feat = torch.load(feat_path, map_location="cpu")
gt_inst_classes = feat.get("gt_instance_classes", {})

with label_db_yaml.open("r", encoding="utf-8") as f:
    label_db = yaml.safe_load(f)

id2name = {}
classes_dict = label_db.get("classes", {}) or label_db.get("labels", {})
for k, v in classes_dict.items():
    try:
        cid = int(k)
    except Exception:
        continue
    name = v.get("name") or v.get("raw_name") or str(cid)
    id2name[cid] = name

print("---- sample objects ----")
count = 0
for j in range(len(instance_ids)):
    if not obj_mask[j]:
        continue
    inst_id = int(instance_ids[j])
    vig_idx = int(class_labels[j])
    vig_name = idx2class.get(vig_idx, f"idx_{vig_idx}")
    scannet_id = gt_inst_classes.get(inst_id, None)
    scannet_name = id2name.get(scannet_id, f"id_{scannet_id}") if scannet_id is not None else "None"
    print(f"ctx[{j}] inst_id={inst_id:3d} | Vigor='{vig_name}' | ScanNet200='{scannet_name}'")
    count += 1
    if count >= 10:
        break
PY
