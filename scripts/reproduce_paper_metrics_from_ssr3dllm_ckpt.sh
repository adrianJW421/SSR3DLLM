#!/usr/bin/env bash
set -euo pipefail

# One-click paper-metric reproduction from a single packed SSR3DLLM checkpoint.
# Internally this script builds eval-adapt checkpoint views, then calls the
# existing reproducible evaluation pipeline.
#
# Usage:
#   bash scripts/reproduce_paper_metrics_from_ssr3dllm_ckpt.sh
#   CKPT=/path/to/SSR3DLLM.ckpt bash scripts/reproduce_paper_metrics_from_ssr3dllm_ckpt.sh
#   STAGES="p1_eval_503,p1_eval_519,p2_eval_ours" bash scripts/reproduce_paper_metrics_from_ssr3dllm_ckpt.sh
#   INCLUDE_P2_BASELINE=1 bash scripts/reproduce_paper_metrics_from_ssr3dllm_ckpt.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_paper_metrics_from_ssr3dllm_ckpt.sh
  CKPT=/path/to/SSR3DLLM.ckpt bash scripts/reproduce_paper_metrics_from_ssr3dllm_ckpt.sh

Options via env:
  STAGES               default: p1_eval_503,p1_eval_519,p2_eval_ours
  INCLUDE_P2_BASELINE  default: 0 (set 1 to additionally run p2_eval_baseline)
  PROFILE_503          default: 503
  PROFILE_519          default: 519
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CKPT="${CKPT:-${SSR3DLLM_PACKED_CKPT:-${DATA_ROOT:-${REPO_ROOT}/data}/SSR3DLLM_CKPT/SSR3DLLM.ckpt}}"
PROFILE_503="${PROFILE_503:-503}"
PROFILE_519="${PROFILE_519:-519}"
INCLUDE_P2_BASELINE="${INCLUDE_P2_BASELINE:-0}"
STAGES_DEFAULT="p1_eval_503,p1_eval_519,p2_eval_ours"
STAGES="${STAGES:-${STAGES_DEFAULT}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${INCLUDE_P2_BASELINE}" == "1" ]]; then
  case ",${STAGES}," in
    *,p2_eval_baseline,*) ;;
    *) STAGES="${STAGES},p2_eval_baseline" ;;
  esac
fi

_need_file "${CKPT}"

ADAPT_ROOT="${OUTPUT_ROOT}/ssr3dllm/eval_adapt_${RUN_TAG}"
mkdir -p "${ADAPT_ROOT}"

ADAPT_503_WRAPPER="${ADAPT_ROOT}/ssr3dllm_eval_adapt_503.pth"
ADAPT_503_LISTENER="${ADAPT_ROOT}/ssr3dllm_eval_adapt_listener_503.pth"
ADAPT_503_REPORT="${ADAPT_ROOT}/eval_adapt_503_report.json"

ADAPT_519_WRAPPER="${ADAPT_ROOT}/ssr3dllm_eval_adapt_519.pth"
ADAPT_519_LISTENER="${ADAPT_ROOT}/ssr3dllm_eval_adapt_listener_519.pth"
ADAPT_519_REPORT="${ADAPT_ROOT}/eval_adapt_519_report.json"

echo "[reproduce_from_ssr3dllm] checkpoint=${CKPT}"
echo "[reproduce_from_ssr3dllm] stages=${STAGES}"
echo "[reproduce_from_ssr3dllm] run_tag=${RUN_TAG}"
echo "[reproduce_from_ssr3dllm] adapt_root=${ADAPT_ROOT}"

python "${REPO_ROOT}/tools/eval_adapt_ssr3dllm_ckpt.py" \
  --checkpoint "${CKPT}" \
  --profile "${PROFILE_503}" \
  --out-wrapper "${ADAPT_503_WRAPPER}" \
  --out-listener "${ADAPT_503_LISTENER}" \
  | tee "${ADAPT_503_REPORT}"

python "${REPO_ROOT}/tools/eval_adapt_ssr3dllm_ckpt.py" \
  --checkpoint "${CKPT}" \
  --profile "${PROFILE_519}" \
  --out-wrapper "${ADAPT_519_WRAPPER}" \
  --out-listener "${ADAPT_519_LISTENER}" \
  | tee "${ADAPT_519_REPORT}"

echo "[reproduce_from_ssr3dllm] adapted_503=${ADAPT_503_WRAPPER}"
echo "[reproduce_from_ssr3dllm] adapted_519=${ADAPT_519_WRAPPER}"
echo "[reproduce_from_ssr3dllm] running reproduce script..."

STAGES="${STAGES}" \
RUN_TAG="${RUN_TAG}" \
P1_EVAL_CKPT="${ADAPT_503_WRAPPER}" \
EVAL_519_CKPT="${ADAPT_519_WRAPPER}" \
P2_OURS_CKPT="${ADAPT_503_WRAPPER}" \
VIGOR_STRICT_LOAD="${VIGOR_STRICT_LOAD:-0}" \
VIGOR_VERBOSE_LOAD="${VIGOR_VERBOSE_LOAD:-1}" \
bash "${REPO_ROOT}/scripts/reproduce_paper_metrics_from_data.sh"

echo "[reproduce_from_ssr3dllm] done."
echo "[reproduce_from_ssr3dllm] eval_adapt_503_report=${ADAPT_503_REPORT}"
echo "[reproduce_from_ssr3dllm] eval_adapt_519_report=${ADAPT_519_REPORT}"
