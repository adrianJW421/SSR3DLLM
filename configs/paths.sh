#!/usr/bin/env bash
set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_THIS_DIR}/.." && pwd)}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export SCANNET200_ROOT="${SCANNET200_ROOT:-${REPO_ROOT}/data/SCANNET200_ROOT}"
export REFERIT_SCANNET_FILE="${REFERIT_SCANNET_FILE:-${REPO_ROOT}/data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl}"
export BERT_PATH="${BERT_PATH:-${REPO_ROOT}/data/BERT_PATH/bert-base-uncased}"
export LLM_PATH="${LLM_PATH:-${REPO_ROOT}/data/LLM_PATH/Tiny-Vicuna-1B}"
export STEP2_CKPT="${STEP2_CKPT:-${REPO_ROOT}/data/grounded3dllm_ckpts/step2/last-epoch.ckpt}"
export NR3D_TRAIN_CSV="${NR3D_TRAIN_CSV:-${REPO_ROOT}/data/NR3D_TRAIN_CSV/nr3d_train_LLM_step4_485.csv}"
export SR3D_TRAIN_CSV="${SR3D_TRAIN_CSV:-${REPO_ROOT}/data/SR3D_TRAIN_CSV/sr3d_train_LLM_step4_485.csv}"
export MASK3D_FEATS_TRAIN="${MASK3D_FEATS_TRAIN:-${REPO_ROOT}/data/MASK3D_FEATS_TRAIN}"
export MASK3D_FEATS_TEST="${MASK3D_FEATS_TEST:-${REPO_ROOT}/data/MASK3D_FEATS_TEST}"
export FEAT_DIM="${FEAT_DIM:-128}"
export LISTENER_INIT_CKPT_BERT="${LISTENER_INIT_CKPT_BERT:-${REPO_ROOT}/data/LISTENER_INIT_CKPT_BERT/best_model.pth}"
export LLAMA_STEPSLOT_EVAL_CKPT="${LLAMA_STEPSLOT_EVAL_CKPT:-${REPO_ROOT}/data/LLAMA_STEPSLOT_EVAL_CKPT/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth}"

# 51.9 / multipass listener ckpt.
# If you do not have a dedicated 519 model yet, fallback to 503 to keep scripts runnable.
export LLAMA_STEPSLOT_EVAL_CKPT_519="${LLAMA_STEPSLOT_EVAL_CKPT_519:-${REPO_ROOT}/data/LLAMA_STEPSLOT_EVAL_CKPT_UB/best_model.pth}"
if [[ ! -f "${LLAMA_STEPSLOT_EVAL_CKPT_519}" ]]; then
  export LLAMA_STEPSLOT_EVAL_CKPT_519="${LLAMA_STEPSLOT_EVAL_CKPT}"
fi

# Unified (non-packed) SSR3DLLM base checkpoint (Step3).
export SSR3DLLM_UNIFIED_CKPT="${SSR3DLLM_UNIFIED_CKPT:-${REPO_ROOT}/data/grounded3dllm_ckpts/step3/last-epoch.ckpt}"

# Packed single-file checkpoints.
export SSR3DLLM_PACKED_CKPT="${SSR3DLLM_PACKED_CKPT:-${REPO_ROOT}/data/SSR3DLLM_CKPT/SSR3DLLM.ckpt}"
export SSR3DLLM_PACKED_UB_CKPT="${SSR3DLLM_PACKED_UB_CKPT:-${REPO_ROOT}/data/SSR3DLLM_CKPT/SSR3DLLM_UB.ckpt}"
