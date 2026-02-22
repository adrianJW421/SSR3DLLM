#!/usr/bin/env bash
set -euo pipefail

# Export per-scene Mask3D features (`*.pt`) for release eval scripts.
#
# Usage:
#   bash scripts/export_mask3d_feats_predbox_fullresfix.sh validation
#   bash scripts/export_mask3d_feats_predbox_fullresfix.sh train
#
# Optional knobs:
#   MAX_SCENES=10 OUTPUT_DIR=/tmp/mask3d_feats_val bash scripts/export_mask3d_feats_predbox_fullresfix.sh validation

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "[FATAL] Please run via: bash ${BASH_SOURCE[0]}" >&2
  return 1
fi

SPLIT="${1:-validation}" # train | validation | test
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "validation" && "${SPLIT}" != "test" ]]; then
  echo "[FATAL] split must be one of: train, validation, test (got: ${SPLIT})" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

export GROUNDED3DLLM_FULLRES_MASK_FIX=1

STEP2_CKPT="${STEP2_CKPT:-}"
SCANNET200_ROOT="${SCANNET200_ROOT:-}"

DATA_CONFIG="${DATA_CONFIG:-baseline/core/conf/data/indoor_dialog.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-baseline/core/conf/model/mask3d_lang.yaml}"
TRAINER_CONFIG="${TRAINER_CONFIG:-baseline/core/conf/trainer/trainer50.yaml}"
LLM_CONFIG="${LLM_CONFIG:-baseline/core/conf/llm/nollm.json}"
LLM_DATA_CONFIG="${LLM_DATA_CONFIG:-baseline/core/conf/llm/det10.json}"

MASK_THRESH="${MASK_THRESH:-0.5}"
MAX_SCENES="${MAX_SCENES:-0}" # 0 means no limit
SCENE_IDS="${SCENE_IDS:-}"    # optional comma-separated scene ids
OVERWRITE="${OVERWRITE:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "${OUTPUT_DIR}" ]]; then
  RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
  OUTPUT_DIR="${OUTPUT_ROOT}/mask3d_feats_${SPLIT}_predbox_fullresfix_${RUN_TAG}"
fi

echo "[export_mask3d_feats] split=${SPLIT}"
echo "[export_mask3d_feats] STEP2_CKPT=${STEP2_CKPT}"
echo "[export_mask3d_feats] SCANNET200_ROOT=${SCANNET200_ROOT}"
echo "[export_mask3d_feats] OUTPUT_DIR=${OUTPUT_DIR}"

_need_file "${STEP2_CKPT}"
_need_dir "${SCANNET200_ROOT}"
mkdir -p "${OUTPUT_DIR}"

extra=()
if [[ "${MAX_SCENES}" != "0" ]]; then
  extra+=(--max-scenes "${MAX_SCENES}")
fi
if [[ -n "${SCENE_IDS}" ]]; then
  extra+=(--scene-ids "${SCENE_IDS}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  extra+=(--overwrite)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${REPO_ROOT}/tools/export_mask3d_features.py" \
  --checkpoint "${STEP2_CKPT}" \
  --data-split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scannet-root "${SCANNET200_ROOT}" \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --trainer-config "${TRAINER_CONFIG}" \
  --llm-config "${LLM_CONFIG}" \
  --llm-data-config "${LLM_DATA_CONFIG}" \
  --mask-thresh "${MASK_THRESH}" \
  "${extra[@]}"
