#!/usr/bin/env bash
set -euo pipefail

# Verify packed single-ckpt path with the *exact* proven 503 evaluation chain:
# 1) eval-adapt packed SSR3DLLM bundle -> Vigor wrapper ckpt
# 2) call original script: eval_llama_stepslot_varlen_chain_onepass_pred.sh
#
# Usage:
#   bash scripts/eval_ssr3dllm_packed_503_exact.sh
#   CKPT=/path/to/SSR3DLLM.ckpt bash scripts/eval_ssr3dllm_packed_503_exact.sh
#   PROFILE=503 bash scripts/eval_ssr3dllm_packed_503_exact.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_ssr3dllm_packed_503_exact.sh
  CKPT=/path/to/SSR3DLLM.ckpt bash scripts/eval_ssr3dllm_packed_503_exact.sh
  PROFILE=503 bash scripts/eval_ssr3dllm_packed_503_exact.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CKPT="${CKPT:-${SSR3DLLM_PACKED_CKPT:-${DATA_ROOT:-${REPO_ROOT}/data}/SSR3DLLM_CKPT/SSR3DLLM.ckpt}}"
PROFILE="${PROFILE:-503}" # 503|519|main|ub

_need_file "${CKPT}"
_need_file "${NR3D_TRAIN_CSV}"
_need_file "${SR3D_TRAIN_CSV}"
_need_file "${REFERIT_SCANNET_FILE}"
_need_dir "${BERT_PATH}"
_need_dir "${MASK3D_FEATS_TRAIN}"
_need_dir "${MASK3D_FEATS_TEST}"
_need_dir "${LLM_PATH}"
_need_file "${LISTENER_INIT_CKPT_BERT}"

RUN_ROOT="${OUTPUT_ROOT}/ssr3dllm/packed_exact_503_$(date +%Y%m%d_%H%M%S)"
ADAPT_DIR="${RUN_ROOT}/eval_adapt"
mkdir -p "${ADAPT_DIR}"

ADAPT_WRAPPER="${ADAPT_DIR}/llama_stepslot_${PROFILE}_from_bundle.pth"
ADAPT_LISTENER="${ADAPT_DIR}/listener_${PROFILE}_from_bundle.pth"
ADAPT_REPORT="${ADAPT_DIR}/eval_adapt_report.json"
RUN_LOG="${RUN_ROOT}/eval.log"

echo "[eval_ssr3dllm_packed_503_exact] packed_ckpt=${CKPT}"
echo "[eval_ssr3dllm_packed_503_exact] profile=${PROFILE}"
echo "[eval_ssr3dllm_packed_503_exact] run_root=${RUN_ROOT}"

python "${REPO_ROOT}/tools/eval_adapt_ssr3dllm_ckpt.py" \
  --checkpoint "${CKPT}" \
  --profile "${PROFILE}" \
  --out-wrapper "${ADAPT_WRAPPER}" \
  --out-listener "${ADAPT_LISTENER}" \
  | tee "${ADAPT_REPORT}"

echo "[eval_ssr3dllm_packed_503_exact] eval_adapt_wrapper=${ADAPT_WRAPPER}"
echo "[eval_ssr3dllm_packed_503_exact] running exact 503 script..."

# Important: eval-adapted ckpt stores the packed adapter subset, so strict-loading must be disabled.
VIGOR_STRICT_LOAD="${VIGOR_STRICT_LOAD:-0}" \
VIGOR_VERBOSE_LOAD="${VIGOR_VERBOSE_LOAD:-1}" \
LLAMA_STEPSLOT_EVAL_CKPT="${ADAPT_WRAPPER}" \
bash "${REPO_ROOT}/scripts/eval_llama_stepslot_varlen_chain_onepass_pred.sh" \
  2>&1 | tee "${RUN_LOG}"

echo "[eval_ssr3dllm_packed_503_exact] done."
echo "[eval_ssr3dllm_packed_503_exact] eval_adapt_report=${ADAPT_REPORT}"
echo "[eval_ssr3dllm_packed_503_exact] eval_log=${RUN_LOG}"
