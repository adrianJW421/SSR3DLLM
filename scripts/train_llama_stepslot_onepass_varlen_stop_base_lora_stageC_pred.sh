#!/usr/bin/env bash
set -euo pipefail

# SSR3D-LLM (SSR3DLLM) mainline training entry (per release experiment log):
# Stage C-pred: implicit chain prediction prompt (no step phrases).
#
# Adapted from:
#   final_scripts/train_llama_stepslot_onepass_varlen_stop_base_lora_stageC_pred.sh
# but reads paths from `configs/paths.sh`.
#
# Run:
#   bash scripts/train_llama_stepslot_onepass_varlen_stop_base_lora_stageC_pred.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# -------------------- PATHS (from configs/paths.sh) --------------------
SCANNET_PKL="${REFERIT_SCANNET_FILE:-}"
BERT_PATH="${BERT_PATH:-}"

NR3D_TRAIN_CSV="${NR3D_TRAIN_CSV:-}"
SR3D_TRAIN_CSV="${SR3D_TRAIN_CSV:-}"

MASK3D_FEATS_TRAIN="${MASK3D_FEATS_TRAIN:-}"
MASK3D_FEATS_TEST="${MASK3D_FEATS_TEST:-}"
FEAT_DIM="${FEAT_DIM:-128}"

LISTENER_INIT_CKPT_BERT="${LISTENER_INIT_CKPT_BERT:-}"
LLM_PATH="${LLM_PATH:-${REPO_ROOT}/data/LLM_PATH/Tiny-Vicuna-1B}"

RESUME_CKPT="${RESUME_CKPT:-${OUTPUT_ROOT}/ssr3dllm/mask3d_vigor_llama_step_slot/llama_stepslot_onepass_varlen_mask_base_lora_latest_best.pth}"

_need_file "${SCANNET_PKL}"
_need_dir "${BERT_PATH}"
_need_file "${NR3D_TRAIN_CSV}"
_need_file "${SR3D_TRAIN_CSV}"
_need_dir "${MASK3D_FEATS_TRAIN}"
_need_dir "${MASK3D_FEATS_TEST}"
_need_file "${LISTENER_INIT_CKPT_BERT}"
_need_dir "${LLM_PATH}"
_need_file "${RESUME_CKPT}"

# -------------------- RUNTIME / HPARAMS --------------------
GPU_LIST="${GPU_LIST:-0}"
ORDER_LEN="${ORDER_LEN:-4}"
MAX_DISTRACTORS="${MAX_DISTRACTORS:-51}"
MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS:-88}"
SVSN_WEIGHT="${SVSN_WEIGHT:-0.25}"
N_WORKERS="${N_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

LLM_MAX_LEN="${LLM_MAX_LEN:-128}"
LLM_MEM_TOKENS="${LLM_MEM_TOKENS:-16}"

MAX_EPOCHS="${MAX_EPOCHS:-50}"
INIT_LR="${INIT_LR:-1e-4}"

DISTILL_W="${DISTILL_W:-1.0}"
DISTILL_GLOBAL_W="${DISTILL_GLOBAL_W:-1.0}"

# -------------------- Design toggles --------------------
export VIGOR_LLM_STEPSLOT_ONEPASS="1"
export VIGOR_LLM_ONEPASS_INPUT_MODE="pred"

export VIGOR_VARLEN_CHAIN="1"
export VIGOR_VARLEN_EARLY_STOP="0"
export VIGOR_ADAPTIVE_HALT="0"
export VIGOR_STOP_TOKEN="${VIGOR_STOP_TOKEN:-<STOP>}"

export VIGOR_STOP_EMBED_W="${VIGOR_STOP_EMBED_W:-0.05}"
export VIGOR_STOP_EMBED_REPLACE="${VIGOR_STOP_EMBED_REPLACE:-1}"
export VIGOR_VARLEN_ONEPASS_TRUNC="${VIGOR_VARLEN_ONEPASS_TRUNC:-0}"

export VIGOR_USE_PRED_BOX_INFO="1"
export VIGOR_STEP_MARKERS="1"
export VIGOR_STEP_SLOT_ONLY="1"
export VIGOR_FREEZE_BERT_EXCEPT_STEP="1"
export VIGOR_TEXT_CLS_SCANNET200="0"
export VIGOR_PRED_CLASS_MASK_MODE="all_ones"

# -------------------- One-pass LoRA --------------------
export VIGOR_LLM_LORA="${VIGOR_LLM_LORA:-1}"
export VIGOR_LLM_LORA_R="${VIGOR_LLM_LORA_R:-8}"
export VIGOR_LLM_LORA_ALPHA="${VIGOR_LLM_LORA_ALPHA:-16}"
export VIGOR_LLM_LORA_DROPOUT="${VIGOR_LLM_LORA_DROPOUT:-0.0}"
export VIGOR_LLM_LORA_LAST_N="${VIGOR_LLM_LORA_LAST_N:-4}"
export VIGOR_LLM_LORA_TARGETS="${VIGOR_LLM_LORA_TARGETS:-q_proj,v_proj}"

export VIGOR_LLM_MODEL_PATH="${LLM_PATH}"
export VIGOR_LLM_MAX_LEN="${LLM_MAX_LEN}"
export VIGOR_LLM_MEM_TOKENS="${LLM_MEM_TOKENS}"
export VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"

export VIGOR_LLM_DISTILL_W="${DISTILL_W}"
export VIGOR_LLM_GLOBAL_DISTILL_W="${DISTILL_GLOBAL_W}"
export VIGOR_LLM_DISTILL_TYPE="cos"
export VIGOR_LLM_GLOBAL_DISTILL_TYPE="cos"

export VIGOR_LLM_STEPSLOT_TRAIN_LISTENER="0"
export VIGOR_LLM_STEPSLOT_TRAIN_LISTENER_PARTS=""

export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${RESUME_CKPT}"
export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="0"

export VIGOR_LISTENER_INIT_CKPT="${LISTENER_INIT_CKPT_BERT}"

LOGROOT="${OUTPUT_ROOT}/ssr3dllm/mask3d_vigor_llama_step_slot"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGROOT}/llama_stepslot_onepass_varlen_mask_stageC_pred_${STAMP}"
mkdir -p "${LOGDIR}"

echo "[train_stageC_pred] resume=${RESUME_CKPT}"
echo "[train_stageC_pred] onepass_input_mode=${VIGOR_LLM_ONEPASS_INPUT_MODE} varlen=${VIGOR_VARLEN_CHAIN}"
echo "[train_stageC_pred] distill_w=${VIGOR_LLM_DISTILL_W} global_w=${VIGOR_LLM_GLOBAL_DISTILL_W}"
echo "[train_stageC_pred] lora=${VIGOR_LLM_LORA} r=${VIGOR_LLM_LORA_R} alpha=${VIGOR_LLM_LORA_ALPHA} last_n=${VIGOR_LLM_LORA_LAST_N} targets=${VIGOR_LLM_LORA_TARGETS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}" \
python third_party/Vigor/referit3d/scripts/train_referit3d_llama_stepslot.py \
  -scannet-file "${SCANNET_PKL}" \
  -referit3D-file "${NR3D_TRAIN_CSV}" \
  --mode "train" \
  --augment-with-sr3d "${SR3D_TRAIN_CSV}" \
  --s-vs-n-weight "${SVSN_WEIGHT}" \
  --use-scannet200-obj-cls True \
  --mask3d-feature-root "${MASK3D_FEATS_TRAIN}" \
  --mask3d-feature-root-test "${MASK3D_FEATS_TEST}" \
  --mask3d-feature-dim "${FEAT_DIM}" \
  --max-distractors "${MAX_DISTRACTORS}" \
  --max-test-objects "${MAX_TEST_OBJECTS}" \
  --unit-sphere-norm True \
  --bert-pretrain-path "${BERT_PATH}" \
  --view_number 4 \
  --rotate_number 4 \
  --encoder-layer-num 3 \
  --decoder-layer-num 4 \
  --decoder-nhead-num 8 \
  --label-lang-sup True \
  --multilabel-pretraining True \
  --lang-multilabel True \
  --cascading True \
  --order-len "${ORDER_LEN}" \
  --disable-text-loss True \
  --lang-cls-alpha 0.0 \
  --init-lr "${INIT_LR}" \
  --max-train-epochs "${MAX_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-workers "${N_WORKERS}" \
  --n-gpus 1 \
  --gpu "${GPU_LIST}" \
  --log-dir "${LOGDIR}" \
  2>&1 | tee "${LOGDIR}/train.log"

CKPT="$(find "${LOGDIR}" -path "*/checkpoints/best_model.pth" -print | LC_ALL=C sort | tail -n 1 || true)"
if [[ -f "${CKPT}" ]]; then
  LATEST="${LOGROOT}/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth"
  cp -f "${CKPT}" "${LATEST}"
  echo "[train_stageC_pred] wrote ${LATEST} <- ${CKPT}"
else
  echo "[train_stageC_pred][WARN] best_model.pth not found under ${LOGDIR}" >&2
fi

echo "[train_stageC_pred] Done."
