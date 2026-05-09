#!/usr/bin/env bash
set -euo pipefail

# Machine-specific paths for the public release.
#
# Recommended workflow:
#   cp configs/paths.example.sh configs/paths.sh
#   edit configs/paths.sh for your machine
#
# `configs/paths.sh` is NOT meant to be committed.
#
# Default values below point to the repository-local scaffold under `data/`.
# Put real files/directories there, or override these env vars to external paths.

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_THIS_DIR}/.." && pwd)}"

# Output root for logs/checkpoints/exports.
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"

# -------------------- Shared / datasets --------------------
# ScanNet200 processed root (contains train/ validation/ with *.npy, etc).
export SCANNET200_ROOT="${SCANNET200_ROOT:-${REPO_ROOT}/data/SCANNET200_ROOT}"

# Raw ScanNet scans root (contains sceneXXXX_YY/sceneXXXX_YY.txt with axisAlignment).
# Required by dataset axis-align path in evaluation.
export SSR3DLLM_RAWSCANNET_SCANS_DIR="${SSR3DLLM_RAWSCANNET_SCANS_DIR:-${REPO_ROOT}/data/rawscannet/scans}"

# ReferIt3D ScanNet pkl (axis alignment matrices; used by Vigor and feature exports).
export REFERIT_SCANNET_FILE="${REFERIT_SCANNET_FILE:-${REPO_ROOT}/data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl}"

# BERT teacher (bert-base-uncased) directory (HuggingFace format).
export BERT_PATH="${BERT_PATH:-${REPO_ROOT}/data/BERT_PATH/bert-base-uncased}"

# Tiny-Vicuna-1B directory (HuggingFace format).
export LLM_PATH="${LLM_PATH:-${REPO_ROOT}/data/LLM_PATH/Tiny-Vicuna-1B}"

# -------------------- Mask3D feature export --------------------
# Step-2 Mask3D checkpoint (.ckpt) used to export per-scene instance features for Vigor.
export STEP2_CKPT="${STEP2_CKPT:-${REPO_ROOT}/data/grounded3dllm_ckpts/step2/last-epoch.ckpt}"

# -------------------- SSR3DLLM (SSR3D-LLM) training/eval --------------------
# ReferIt3D train CSVs (Vigor format).
# NOTE: the public release does not ship these CSVs; download/build them and point to the files here.
export NR3D_TRAIN_CSV="${NR3D_TRAIN_CSV:-${REPO_ROOT}/data/NR3D_TRAIN_CSV/nr3d_train_LLM_step4_485.csv}"
export SR3D_TRAIN_CSV="${SR3D_TRAIN_CSV:-${REPO_ROOT}/data/SR3D_TRAIN_CSV/sr3d_train_LLM_step4_485.csv}"
export SCANREFER_TRAIN_CSV="${SCANREFER_TRAIN_CSV:-${REPO_ROOT}/data/SCANREFER_TRAIN_CSV/scanrefer_train_vigor_chain.csv}"
export MULTI3DREF_TRAIN_CSV="${MULTI3DREF_TRAIN_CSV:-${REPO_ROOT}/data/MULTI3DREF_TRAIN_CSV/multi3dref_train_vigor_chain.csv}"

# Pre-exported per-scene Mask3D features for ReferIt3D listener teacher training/eval (ViGOR-based).
export MASK3D_FEATS_TRAIN="${MASK3D_FEATS_TRAIN:-${REPO_ROOT}/data/MASK3D_FEATS_TRAIN}"
export MASK3D_FEATS_TEST="${MASK3D_FEATS_TEST:-${REPO_ROOT}/data/MASK3D_FEATS_TEST}"
export SCANREFER_MASK3D_FEATS_TRAIN="${SCANREFER_MASK3D_FEATS_TRAIN:-${REPO_ROOT}/data/SCANREFER_MASK3D_FEATS_TRAIN}"
export SCANREFER_MASK3D_FEATS_TEST="${SCANREFER_MASK3D_FEATS_TEST:-${REPO_ROOT}/data/SCANREFER_MASK3D_FEATS_TEST}"
export MULTI3DREF_MASK3D_FEATS_TRAIN="${MULTI3DREF_MASK3D_FEATS_TRAIN:-${REPO_ROOT}/data/MULTI3DREF_MASK3D_FEATS_TRAIN}"
export MULTI3DREF_MASK3D_FEATS_TEST="${MULTI3DREF_MASK3D_FEATS_TEST:-${REPO_ROOT}/data/MULTI3DREF_MASK3D_FEATS_TEST}"
export FEAT_DIM="${FEAT_DIM:-128}"
export SCANREFER_QCOND_CACHE_ROOT="${SCANREFER_QCOND_CACHE_ROOT:-${SCANREFER_MASK3D_FEATS_TRAIN}}"
export MULTI3DREF_QCOND_CACHE_ROOT="${MULTI3DREF_QCOND_CACHE_ROOT:-${MULTI3DREF_MASK3D_FEATS_TRAIN}}"
export NR3D_QCOND_CACHE_ROOT="${NR3D_QCOND_CACHE_ROOT:-${MASK3D_FEATS_TRAIN}}"
export SCANREFER_DINO_SAMPLE_CACHE_ROOT="${SCANREFER_DINO_SAMPLE_CACHE_ROOT:-}"
export MULTI3DREF_DINO_SAMPLE_CACHE_ROOT="${MULTI3DREF_DINO_SAMPLE_CACHE_ROOT:-}"
export NR3D_DINO_SAMPLE_CACHE_ROOT="${NR3D_DINO_SAMPLE_CACHE_ROOT:-}"
export DINO_SOURCE_FEATURES="${DINO_SOURCE_FEATURES:-}"
export DINO_SOURCE_ATTRIBUTES="${DINO_SOURCE_ATTRIBUTES:-}"
export DINO_ALPHA="${DINO_ALPHA:-2.0}"
export DINO_FEATURE_DIM="${DINO_FEATURE_DIM:-1024}"

# Frozen listener-teacher init checkpoint (trained via customized `third_party/Vigor`).
export LISTENER_INIT_CKPT_BERT="${LISTENER_INIT_CKPT_BERT:-${REPO_ROOT}/data/LISTENER_INIT_CKPT_BERT/best_model.pth}"

# Optional: stageC-pred checkpoint for direct evaluation without re-training.
export LLAMA_STEPSLOT_EVAL_CKPT="${LLAMA_STEPSLOT_EVAL_CKPT:-${REPO_ROOT}/data/LLAMA_STEPSLOT_EVAL_CKPT/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth}"
export SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT="${SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT:-${REPO_ROOT}/data/SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT/best_model.pth}"
export MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT="${MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT:-${REPO_ROOT}/data/MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT/best_model.pth}"

# Optional: UB listener checkpoint (single release directory).
export LLAMA_STEPSLOT_EVAL_CKPT_UB="${LLAMA_STEPSLOT_EVAL_CKPT_UB:-${REPO_ROOT}/data/LLAMA_STEPSLOT_EVAL_CKPT_UB/best_model.pth}"
# Backward-compatible alias: 519 profile now defaults to UB checkpoint path.
export LLAMA_STEPSLOT_EVAL_CKPT_519="${LLAMA_STEPSLOT_EVAL_CKPT_519:-${LLAMA_STEPSLOT_EVAL_CKPT_UB}}"

# Optional: unified SSR3DLLM checkpoint for language + "<geom>" routed inference.
export SSR3DLLM_UNIFIED_CKPT="${SSR3DLLM_UNIFIED_CKPT:-${REPO_ROOT}/data/grounded3dllm_ckpts/step3/last-epoch.ckpt}"

# Optional: packed single-file releases (language backbone + bundled listener profiles).
export SSR3DLLM_PACKED_CKPT="${SSR3DLLM_PACKED_CKPT:-${REPO_ROOT}/data/SSR3DLLM_CKPT/SSR3DLLM.ckpt}"
export SSR3DLLM_PACKED_UB_CKPT="${SSR3DLLM_PACKED_UB_CKPT:-${REPO_ROOT}/data/SSR3DLLM_CKPT/SSR3DLLM_UB.ckpt}"
