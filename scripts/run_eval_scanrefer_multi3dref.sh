#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_ssr3dllm_scanrefer_multi3dref.sh" "$@"
