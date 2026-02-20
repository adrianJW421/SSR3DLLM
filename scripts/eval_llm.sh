#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_llm.sh /path/to/saved/step3_mask3d_lang_4GPUS/<TIMESTAMP>/" >&2
  exit 2
fi

export PYTHONPATH="./"

# Allow disabling detection evaluation (which requires ScanNet200 instance_gt txt files).
TEST_DETECTION="${SSR3DLLM_EVAL_DETECTION:-true}"

python -m models.metrics.evaluate_LLM \
  --directory_path="$1" \
  --statistics=true \
  --test_scanrefer=true \
  --test_m3drefer=true \
  --test_lan=true \
  --test_detection="${TEST_DETECTION}"
