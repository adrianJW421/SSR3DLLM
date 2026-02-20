#!/usr/bin/env bash
set -euo pipefail

# Quick path sanity check for the public release.
#
# Run:
#   bash scripts/check_paths.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "[check_paths] REPO_ROOT=${REPO_ROOT}"
echo "[check_paths] OUTPUT_ROOT=${OUTPUT_ROOT}"

_need_dir "${SCANNET200_ROOT:-}"
_need_file "${REFERIT_SCANNET_FILE:-}"
_need_dir "${BERT_PATH:-}"
_need_dir "${LLM_PATH:-}"

echo "[check_paths] OK: SCANNET200_ROOT"
echo "[check_paths] OK: REFERIT_SCANNET_FILE"
echo "[check_paths] OK: BERT_PATH"
echo "[check_paths] OK: LLM_PATH"

# Optional checks for SSR3DLLM pipeline.
if [[ -n "${NR3D_TRAIN_CSV:-}" ]]; then _check_vigor_train_test_csv_pair "${NR3D_TRAIN_CSV}" "NR3D"; fi
if [[ -n "${SR3D_TRAIN_CSV:-}" ]]; then _check_vigor_train_test_csv_pair "${SR3D_TRAIN_CSV}" "SR3D"; fi
if [[ -n "${MASK3D_FEATS_TRAIN:-}" ]]; then _need_dir "${MASK3D_FEATS_TRAIN}"; echo "[check_paths] OK: MASK3D_FEATS_TRAIN"; fi
if [[ -n "${MASK3D_FEATS_TEST:-}" ]]; then _need_dir "${MASK3D_FEATS_TEST}"; echo "[check_paths] OK: MASK3D_FEATS_TEST"; fi
if [[ -n "${LISTENER_INIT_CKPT_BERT:-}" ]]; then _need_file "${LISTENER_INIT_CKPT_BERT}"; echo "[check_paths] OK: LISTENER_INIT_CKPT_BERT"; fi
if [[ -n "${STEP2_CKPT:-}" ]]; then _need_file "${STEP2_CKPT}"; echo "[check_paths] OK: STEP2_CKPT"; fi
if [[ -n "${LLAMA_STEPSLOT_EVAL_CKPT:-}" ]]; then _need_file "${LLAMA_STEPSLOT_EVAL_CKPT}"; echo "[check_paths] OK: LLAMA_STEPSLOT_EVAL_CKPT"; fi
if [[ -n "${LLAMA_STEPSLOT_EVAL_CKPT_UB:-}" ]]; then _need_file "${LLAMA_STEPSLOT_EVAL_CKPT_UB}"; echo "[check_paths] OK: LLAMA_STEPSLOT_EVAL_CKPT_UB"; fi
if [[ -n "${LLAMA_STEPSLOT_EVAL_CKPT_519:-}" ]]; then _need_file "${LLAMA_STEPSLOT_EVAL_CKPT_519}"; echo "[check_paths] OK: LLAMA_STEPSLOT_EVAL_CKPT_519"; fi
if [[ -n "${SSR3DLLM_UNIFIED_CKPT:-}" ]]; then _need_file "${SSR3DLLM_UNIFIED_CKPT}"; echo "[check_paths] OK: SSR3DLLM_UNIFIED_CKPT"; fi
if [[ -n "${SSR3DLLM_PACKED_CKPT:-}" && -f "${SSR3DLLM_PACKED_CKPT}" ]]; then echo "[check_paths] OK: SSR3DLLM_PACKED_CKPT"; fi
if [[ -n "${SSR3DLLM_PACKED_UB_CKPT:-}" && -f "${SSR3DLLM_PACKED_UB_CKPT}" ]]; then echo "[check_paths] OK: SSR3DLLM_PACKED_UB_CKPT"; fi

echo "[check_paths] Done."
