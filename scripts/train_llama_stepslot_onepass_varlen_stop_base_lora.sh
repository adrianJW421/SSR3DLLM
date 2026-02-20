#!/usr/bin/env bash
set -euo pipefail

# SSR3D-LLM (SSR3DLLM) mainline training entry (per release experiment log):
# one-pass step-slot + freeze-gated variable-length mask + LoRA (base training).
#
# This script is adapted from:
#   final_scripts/train_llama_stepslot_onepass_varlen_stop_base_lora.sh
# but removes hard-coded server paths and reads them from `configs/paths.sh`.
#
# Run:
#   bash scripts/train_llama_stepslot_onepass_varlen_stop_base_lora.sh

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

_need_file "${SCANNET_PKL}"
_need_dir "${BERT_PATH}"
_need_file "${NR3D_TRAIN_CSV}"
_need_file "${SR3D_TRAIN_CSV}"
_need_dir "${MASK3D_FEATS_TRAIN}"
_need_dir "${MASK3D_FEATS_TEST}"
_need_file "${LISTENER_INIT_CKPT_BERT}"
_need_dir "${LLM_PATH}"

# -------------------- RUNTIME / HPARAMS --------------------
GPU_LIST="${GPU_LIST:-0}"
ORDER_LEN="${ORDER_LEN:-4}"
MAX_DISTRACTORS="${MAX_DISTRACTORS:-51}"
MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS:-88}"
SVSN_WEIGHT="${SVSN_WEIGHT:-0.25}"
N_WORKERS="${N_WORKERS:-8}"

BATCH_SIZE_A="${BATCH_SIZE_A:-64}"
BATCH_SIZE_B="${BATCH_SIZE_B:-32}"

LLM_MAX_LEN="${LLM_MAX_LEN:-128}"
LLM_MEM_TOKENS="${LLM_MEM_TOKENS:-16}"

MAX_EPOCHS_A="${MAX_EPOCHS_A:-20}"
LR_A="${LR_A:-1e-4}"
MAX_EPOCHS_B="${MAX_EPOCHS_B:-50}"
LR_B="${LR_B:-5e-5}"

# -------------------- Latest design toggles --------------------
export VIGOR_LLM_STEPSLOT_ONEPASS="1"
export VIGOR_VARLEN_CHAIN="1"
export VIGOR_VARLEN_EARLY_STOP="0"
export VIGOR_ADAPTIVE_HALT="0"
export VIGOR_STOP_TOKEN="${VIGOR_STOP_TOKEN:-<STOP>}"

# STOP embedding regularization (optional but recommended for interpretability).
export VIGOR_STOP_EMBED_W="${VIGOR_STOP_EMBED_W:-0.05}"
export VIGOR_STOP_EMBED_REPLACE="${VIGOR_STOP_EMBED_REPLACE:-1}"
export VIGOR_VARLEN_ONEPASS_TRUNC="${VIGOR_VARLEN_ONEPASS_TRUNC:-0}"

export VIGOR_USE_PRED_BOX_INFO="1"
export VIGOR_STEP_MARKERS="1"
export VIGOR_STEP_SLOT_ONLY="1"
export VIGOR_FREEZE_BERT_EXCEPT_STEP="1"
export VIGOR_TEXT_CLS_SCANNET200="0"
export VIGOR_PRED_CLASS_MASK_MODE="all_ones"

# -------------------- One-pass LoRA (stable + minimal) --------------------
export VIGOR_LLM_LORA="1"
export VIGOR_LLM_LORA_R="${VIGOR_LLM_LORA_R:-8}"
export VIGOR_LLM_LORA_ALPHA="${VIGOR_LLM_LORA_ALPHA:-16}"
export VIGOR_LLM_LORA_DROPOUT="${VIGOR_LLM_LORA_DROPOUT:-0.0}"
export VIGOR_LLM_LORA_LAST_N="${VIGOR_LLM_LORA_LAST_N:-4}"
export VIGOR_LLM_LORA_TARGETS="${VIGOR_LLM_LORA_TARGETS:-q_proj,v_proj}"

export VIGOR_LLM_MODEL_PATH="${LLM_PATH}"
export VIGOR_LLM_MAX_LEN="${LLM_MAX_LEN}"
export VIGOR_LLM_MEM_TOKENS="${LLM_MEM_TOKENS}"
export VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"

export VIGOR_LISTENER_INIT_CKPT="${LISTENER_INIT_CKPT_BERT}"

LOGROOT="${OUTPUT_ROOT}/ssr3dllm/mask3d_vigor_llama_step_slot"
STAMP="$(date +%Y%m%d_%H%M%S)"

_run_train() {
  local phase="$1"   # A|B
  local logdir="$2"
  local lr="$3"
  local epochs="$4"
  local resume_ckpt="$5"  # empty for scratch

  mkdir -p "${logdir}"

  if [[ -n "${resume_ckpt}" ]]; then
    export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${resume_ckpt}"
    export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="0"
  else
    export VIGOR_LLM_STEPSLOT_RESUME_CKPT=""
    export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="0"
  fi

  local disable_text_loss
  local lang_cls_alpha
  local batch_size

  if [[ "${phase}" == "A" ]]; then
    export VIGOR_LLM_DISTILL_W="1.0"
    export VIGOR_LLM_GLOBAL_DISTILL_W="1.0"
    export VIGOR_LLM_DISTILL_TYPE="cos"
    export VIGOR_LLM_GLOBAL_DISTILL_TYPE="cos"
    disable_text_loss="True"
    lang_cls_alpha="0.0"
    export VIGOR_LLM_STEPSLOT_TRAIN_LISTENER="0"
    export VIGOR_LLM_STEPSLOT_TRAIN_LISTENER_PARTS=""
    export VIGOR_LISTENER_LR=""
    export VIGOR_LLM_LR=""
    batch_size="${BATCH_SIZE_A}"
  else
    export VIGOR_LLM_DISTILL_W="0.0"
    export VIGOR_LLM_GLOBAL_DISTILL_W="0.0"
    export VIGOR_LLM_DISTILL_TYPE="cos"
    export VIGOR_LLM_GLOBAL_DISTILL_TYPE="cos"
    disable_text_loss="False"
    lang_cls_alpha="0.05"
    export VIGOR_LLM_STEPSLOT_TRAIN_LISTENER="1"
    export VIGOR_LLM_STEPSLOT_TRAIN_LISTENER_PARTS=""
    export VIGOR_FREEZE_BERT_EXCEPT_STEP="0"
    export VIGOR_LISTENER_LR="1e-5"
    export VIGOR_LLM_LR="5e-5"
    batch_size="${BATCH_SIZE_B}"
  fi

  echo "[train_onepass_base_lora] phase=${phase} logdir=${logdir} lr=${lr} epochs=${epochs} resume=${resume_ckpt}"

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
    --disable-text-loss "${disable_text_loss}" \
    --lang-cls-alpha "${lang_cls_alpha}" \
    --init-lr "${lr}" \
    --max-train-epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --n-workers "${N_WORKERS}" \
    --n-gpus 1 \
    --gpu "${GPU_LIST}" \
    --log-dir "${logdir}" \
    2>&1 | tee "${logdir}/train.log"
}

LOG_A="${LOGROOT}/llama_stepslot_onepass_varlen_mask_base_lora_A_${STAMP}"
_run_train "A" "${LOG_A}" "${LR_A}" "${MAX_EPOCHS_A}" ""

CKPT_A="$(find "${LOG_A}" -path "*/checkpoints/best_model.pth" -print | LC_ALL=C sort | tail -n 1 || true)"
if [[ ! -f "${CKPT_A}" ]]; then
  echo "[train_onepass_base_lora][FATAL] Phase-A best_model.pth not found under ${LOG_A}" >&2
  exit 2
fi

LOG_B="${LOGROOT}/llama_stepslot_onepass_varlen_mask_base_lora_B_${STAMP}"
_run_train "B" "${LOG_B}" "${LR_B}" "${MAX_EPOCHS_B}" "${CKPT_A}"

CKPT_B="$(find "${LOG_B}" -path "*/checkpoints/best_model.pth" -print | LC_ALL=C sort | tail -n 1 || true)"
if [[ -f "${CKPT_B}" ]]; then
  LATEST="${LOGROOT}/llama_stepslot_onepass_varlen_mask_base_lora_latest_best.pth"
  cp -f "${CKPT_B}" "${LATEST}"
  echo "[train_onepass_base_lora] wrote ${LATEST} <- ${CKPT_B}"
else
  echo "[train_onepass_base_lora][WARN] Phase-B best_model.pth not found under ${LOG_B}" >&2
fi

echo "[train_onepass_base_lora] Done."
