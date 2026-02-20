#!/usr/bin/env bash
set -euo pipefail

# Human-friendly entrypoint alias.
# Delegates to the original release script for backward compatibility.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/eval_ssr3dllm_appendix_cap_examples_from_ckpt.sh" "$@"
