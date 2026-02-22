#!/usr/bin/env bash
set -euo pipefail

# Human-friendly entrypoint alias.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/export_mask3d_feats_predbox_fullresfix.sh" "$@"
