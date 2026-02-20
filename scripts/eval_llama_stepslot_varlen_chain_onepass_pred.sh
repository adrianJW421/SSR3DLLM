#!/usr/bin/env bash
set -euo pipefail

# SSR3D-LLM (SSR3DLLM) mainline evaluation entry (per release experiment log):
# one-pass + implicit chain prompt (Stage C-pred), evaluate with varlen mask on/off.
#
# Adapted from:
#   final_scripts/eval_llama_stepslot_varlen_chain_onepass_pred.sh
# but reads paths from `configs/paths.sh` and writes outputs under `outputs/`.
#
# Run:
#   bash scripts/eval_llama_stepslot_varlen_chain_onepass_pred.sh

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

LLAMA_STEPSLOT_EVAL_CKPT="${LLAMA_STEPSLOT_EVAL_CKPT:-${OUTPUT_ROOT}/ssr3dllm/mask3d_vigor_llama_step_slot/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth}"
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
_need_file "${LLAMA_STEPSLOT_EVAL_CKPT}"

# -------------------- RUNTIME HYPERPARAMETERS --------------------
GPU_LIST="${GPU_LIST:-0}"
ORDER_LEN="${ORDER_LEN:-4}"
MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS:-88}"
MAX_DISTRACTORS="${MAX_DISTRACTORS:-51}"
N_WORKERS="${N_WORKERS:-8}"
LLM_MAX_LEN="${LLM_MAX_LEN:-128}"
LLM_MEM_TOKENS="${LLM_MEM_TOKENS:-16}"

# -------------------- INTERNAL --------------------
export VIGOR_USE_PRED_BOX_INFO="1"
export VIGOR_STEP_MARKERS="1"
export VIGOR_STEP_SLOT_ONLY="1"
export VIGOR_FREEZE_BERT_EXCEPT_STEP="1"
export VIGOR_TEXT_CLS_SCANNET200="0"
export VIGOR_PRED_CLASS_MASK_MODE="all_ones"
export VIGOR_ADAPTIVE_HALT="0"

export VIGOR_STOP_TOKEN="${VIGOR_STOP_TOKEN:-<STOP>}"
export VIGOR_VARLEN_ONEPASS_TRUNC="${VIGOR_VARLEN_ONEPASS_TRUNC:-0}"
export VIGOR_STOP_EMBED_W="${VIGOR_STOP_EMBED_W:-0.0}"
export VIGOR_STOP_EMBED_REPLACE="${VIGOR_STOP_EMBED_REPLACE:-1}"

export VIGOR_LLM_MODEL_PATH="${LLM_PATH}"
export VIGOR_LLM_MAX_LEN="${LLM_MAX_LEN}"
export VIGOR_LLM_MEM_TOKENS="${LLM_MEM_TOKENS}"
export VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"

export VIGOR_LLM_STEPSLOT_ONEPASS="1"
export VIGOR_LLM_ONEPASS_INPUT_MODE="pred"
export VIGOR_LLM_DISTILL_W="0"
export VIGOR_LLM_GLOBAL_DISTILL_W="0"

export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${LLAMA_STEPSLOT_EVAL_CKPT}"
export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="0"
export VIGOR_LISTENER_INIT_CKPT="${LISTENER_INIT_CKPT_BERT}"

# Many StageC-pred checkpoints are trained with LoRA enabled.
# Auto-infer LoRA settings from the resume checkpoint to avoid mismatch.
export VIGOR_LLM_LORA_AUTO="${VIGOR_LLM_LORA_AUTO:-1}"

LOGDIR_BASE="${OUTPUT_ROOT}/ssr3dllm/llama_stepslot_onepass_pred_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR_BASE}"

_run_split() {
  local tag="$1"
  local train_csv="$2"
  local varlen="$3"
  local out_dir="${LOGDIR_BASE}/${tag}/varlen_${varlen}"
  mkdir -p "${out_dir}"

  echo "[onepass_pred_eval] ===== ${tag} varlen=${varlen} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}" \
  VIGOR_VARLEN_CHAIN="${varlen}" \
  python third_party/Vigor/referit3d/scripts/train_referit3d_llama_stepslot.py \
    -scannet-file "${SCANNET_PKL}" \
    -referit3D-file "${train_csv}" \
    --mode evaluate \
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
    --batch-size 64 \
    --n-workers "${N_WORKERS}" \
    --n-gpus 1 \
    --gpu "${GPU_LIST}" \
    --log-dir "${out_dir}" \
    2>&1 | tee "${out_dir}/eval.log"

  grep -n "Reference-Accuracy:" "${out_dir}/eval.log" || true
}

_run_split "nr3d" "${NR3D_TRAIN_CSV}" "0"
_run_split "sr3d" "${SR3D_TRAIN_CSV}" "0"
_run_split "nr3d" "${NR3D_TRAIN_CSV}" "1"
_run_split "sr3d" "${SR3D_TRAIN_CSV}" "1"

echo "[onepass_pred_eval] Done. Logs under: ${LOGDIR_BASE}"
