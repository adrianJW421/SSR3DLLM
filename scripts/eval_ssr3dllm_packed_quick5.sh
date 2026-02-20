#!/usr/bin/env bash
set -euo pipefail

# Quick packed-ckpt validation:
# - grounding smoke eval on nr3d/sr3d (5 samples each by default)
# - language QA smoke eval (5 prompts)
#
# Usage:
#   bash scripts/eval_ssr3dllm_packed_quick5.sh
#   MODEL_VARIANT=main bash scripts/eval_ssr3dllm_packed_quick5.sh
#   MODEL_VARIANT=ub NUM_SAMPLES=5 bash scripts/eval_ssr3dllm_packed_quick5.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL_VARIANT="${MODEL_VARIANT:-both}" # both|main|ub
NUM_SAMPLES="${NUM_SAMPLES:-5}"
DATASETS="${DATASETS:-nr3d,sr3d}"
SPLIT="${SPLIT:-validation}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-128}"

PACKED_MAIN="${PACKED_MAIN:-${SSR3DLLM_PACKED_CKPT:-${DATA_ROOT:-${REPO_ROOT}/data}/SSR3DLLM_CKPT/SSR3DLLM.ckpt}}"
PACKED_UB="${PACKED_UB:-${SSR3DLLM_PACKED_UB_CKPT:-${DATA_ROOT:-${REPO_ROOT}/data}/SSR3DLLM_CKPT/SSR3DLLM_UB.ckpt}}"

_need_file "${NR3D_TRAIN_CSV}"
_need_file "${SR3D_TRAIN_CSV}"
_need_file "${REFERIT_SCANNET_FILE}"
_need_dir "${BERT_PATH}"
_need_dir "${SCANNET200_ROOT}"

LOG_ROOT="${OUTPUT_ROOT}/ssr3dllm/packed_quick5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"

run_one() {
  local tag="$1"
  local ckpt="$2"
  local profile="$3"
  _need_file "${ckpt}"
  local out_json="${LOG_ROOT}/${tag}.json"
  local out_log="${LOG_ROOT}/${tag}.log"

  echo "[packed_quick5] ===== ${tag} ====="
  echo "[packed_quick5] ckpt=${ckpt}"
  echo "[packed_quick5] profile=${profile} num_samples=${NUM_SAMPLES} datasets=${DATASETS}"

  python "${REPO_ROOT}/tools/eval_ssr3dllm_packed_quick5.py" \
    --checkpoint "${ckpt}" \
    --profile "${profile}" \
    --num-samples "${NUM_SAMPLES}" \
    --datasets "${DATASETS}" \
    --nr3d-train-csv "${NR3D_TRAIN_CSV}" \
    --sr3d-train-csv "${SR3D_TRAIN_CSV}" \
    --scannet-file "${REFERIT_SCANNET_FILE}" \
    --bert-path "${BERT_PATH}" \
    --scannet-processed-root "${SCANNET200_ROOT}" \
    --split "${SPLIT}" \
    --llm-max-new-tokens "${LLM_MAX_NEW_TOKENS}" \
    --output-json "${out_json}" \
    2>&1 | tee "${out_log}"
}

case "${MODEL_VARIANT}" in
  both)
    run_one "SSR3DLLM" "${PACKED_MAIN}" "503"
    run_one "SSR3DLLM_UB" "${PACKED_UB}" "519"
    ;;
  main)
    run_one "SSR3DLLM" "${PACKED_MAIN}" "503"
    ;;
  ub)
    run_one "SSR3DLLM_UB" "${PACKED_UB}" "519"
    ;;
  *)
    echo "[FATAL] MODEL_VARIANT must be one of: both|main|ub" >&2
    exit 2
    ;;
esac

echo "[packed_quick5] done. logs=${LOG_ROOT}"
