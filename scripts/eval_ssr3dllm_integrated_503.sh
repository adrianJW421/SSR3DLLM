#!/usr/bin/env bash
set -euo pipefail

# Integrated SSR3DLLM evaluation (503 protocol) using a single ckpt:
# - loads packed/integrated ckpt through BaselineModelAPI
# - routes "<geom>" samples to listener runtime via unified model
# - reports Reference-Accuracy on NR3D/SR3D
#
# Usage:
#   bash scripts/eval_ssr3dllm_integrated_503.sh
#   CKPT=/path/to/SSR3DLLM.ckpt bash scripts/eval_ssr3dllm_integrated_503.sh
#   DATASETS=nr3d MAX_SAMPLES=5 bash scripts/eval_ssr3dllm_integrated_503.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_ssr3dllm_integrated_503.sh
  CKPT=/path/to/SSR3DLLM.ckpt bash scripts/eval_ssr3dllm_integrated_503.sh
  DATASETS=nr3d MAX_SAMPLES=5 bash scripts/eval_ssr3dllm_integrated_503.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CKPT="${CKPT:-${SSR3DLLM_PACKED_CKPT:-${DATA_ROOT:-${REPO_ROOT}/data}/SSR3DLLM_CKPT/SSR3DLLM.ckpt}}"
PROFILE="${PROFILE:-503}"
DATASETS="${DATASETS:-nr3d,sr3d}"
MAX_SAMPLES="${MAX_SAMPLES:-0}" # 0 means full split
BATCH_SIZE="${BATCH_SIZE:-16}"
N_WORKERS="${N_WORKERS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SPLIT="${SPLIT:-validation}"

_need_file "${CKPT}"
_need_file "${NR3D_TRAIN_CSV}"
_need_file "${SR3D_TRAIN_CSV}"
_need_file "${REFERIT_SCANNET_FILE}"
_need_dir "${BERT_PATH}"
_need_dir "${MASK3D_FEATS_TRAIN}"
_need_dir "${MASK3D_FEATS_TEST}"
_need_dir "${SCANNET200_ROOT}"

LOG_ROOT="${OUTPUT_ROOT}/ssr3dllm/integrated_503_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/eval.log"
JSON_FILE="${LOG_ROOT}/summary.json"

echo "[eval_ssr3dllm_integrated_503] ckpt=${CKPT}"
echo "[eval_ssr3dllm_integrated_503] profile=${PROFILE} datasets=${DATASETS} max_samples=${MAX_SAMPLES}"
echo "[eval_ssr3dllm_integrated_503] log_root=${LOG_ROOT}"

python "${REPO_ROOT}/tools/eval_ssr3dllm_integrated_503.py" \
  --checkpoint "${CKPT}" \
  --profile "${PROFILE}" \
  --datasets "${DATASETS}" \
  --nr3d-train-csv "${NR3D_TRAIN_CSV}" \
  --sr3d-train-csv "${SR3D_TRAIN_CSV}" \
  --scannet-file "${REFERIT_SCANNET_FILE}" \
  --bert-path "${BERT_PATH}" \
  --mask3d-feature-root "${MASK3D_FEATS_TRAIN}" \
  --mask3d-feature-root-test "${MASK3D_FEATS_TEST}" \
  --scannet-processed-root "${SCANNET200_ROOT}" \
  --split "${SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --n-workers "${N_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-json "${JSON_FILE}" \
  2>&1 | tee "${LOG_FILE}"

echo "[eval_ssr3dllm_integrated_503] done."
echo "[eval_ssr3dllm_integrated_503] summary=${JSON_FILE}"
echo "[eval_ssr3dllm_integrated_503] log=${LOG_FILE}"
