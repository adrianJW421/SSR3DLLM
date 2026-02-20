#!/usr/bin/env bash
set -euo pipefail

# Paper-equivalent dialog demo from a single packed `SSR3DLLM.ckpt`.
# It reuses the appendix protocol pipeline:
#   eval_adapt -> --mode test -> eval_llm -> extract_capability_examples
# Then prints one Dialog row as the demo output.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_ts() { date +%Y%m%d_%H%M%S; }

RUN_TAG="dialog_demo_$(_ts)"
LOG_DIR="${OUTPUT_ROOT}/ssr3dllm/${RUN_TAG}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run.log"

# Keep defaults paper-compatible; caller can override externally.
export NUM_GPUS="${NUM_GPUS:-1}"
export SSR3DLLM_LIMIT_TEST_BATCHES="${SSR3DLLM_LIMIT_TEST_BATCHES:-1.0}"

echo "[dialog_demo] NUM_GPUS=${NUM_GPUS}" | tee -a "${RUN_LOG}"
echo "[dialog_demo] SSR3DLLM_LIMIT_TEST_BATCHES=${SSR3DLLM_LIMIT_TEST_BATCHES}" | tee -a "${RUN_LOG}"
echo "[dialog_demo] CKPT=${CKPT:-${SSR3DLLM_PACKED_CKPT:-<unset>}}" | tee -a "${RUN_LOG}"
echo "[dialog_demo] BASELINE_CKPT=${BASELINE_CKPT:-${SSR3DLLM_UNIFIED_CKPT:-<unset>}}" | tee -a "${RUN_LOG}"

set +e
bash "${REPO_ROOT}/scripts/eval_ssr3dllm_appendix_cap_examples_from_ckpt.sh" 2>&1 | tee -a "${RUN_LOG}"
rc=${PIPESTATUS[0]}
set -e
if [[ "${rc}" != "0" ]]; then
  echo "[FATAL] appendix pipeline failed (rc=${rc}). log=${RUN_LOG}" >&2
  exit "${rc}"
fi

OUT_ROOT="$(grep -Eo "\\[cap_examples\\] DONE out_root=.*" "${RUN_LOG}" | tail -n 1 | sed -E 's/.*out_root=//')"
if [[ -z "${OUT_ROOT}" ]]; then
  echo "[FATAL] cannot parse out_root from run log: ${RUN_LOG}" >&2
  exit 2
fi

ROWS_FILE="${OUT_ROOT}/cap_examples_rows_seed1.txt"
_need_file "${ROWS_FILE}"

DIALOG_LINE="$(grep -E '^Dialog &' "${ROWS_FILE}" | head -n 1 || true)"
if [[ -z "${DIALOG_LINE}" ]]; then
  echo "[FATAL] no Dialog row found in ${ROWS_FILE}" >&2
  exit 2
fi

python - "${DIALOG_LINE}" <<'PY'
import sys

line = sys.argv[1].strip()
parts = [p.strip() for p in line.split("&")]
if len(parts) < 4:
    raise SystemExit(f"[FATAL] unexpected Dialog row format: {line}")

task = parts[0]
question = parts[1]
baseline = parts[2]
ours = "&".join(parts[3:]).rstrip("\\").strip()

print("[dialog_demo] ===== Paper-Protocol Dialog Demo =====")
print(f"[dialog_demo] task      : {task}")
print(f"[dialog_demo] question  : {question}")
print(f"[dialog_demo] baseline  : {baseline}")
print(f"[dialog_demo] ours      : {ours}")
print("[dialog_demo] =====================================")
PY

echo "[dialog_demo] rows_file=${ROWS_FILE}"
echo "[dialog_demo] out_root=${OUT_ROOT}"
echo "[dialog_demo] log=${RUN_LOG}"
