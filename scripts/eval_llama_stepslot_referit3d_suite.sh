#!/usr/bin/env bash
set -euo pipefail

# Historical "51.9" protocol evaluator:
# 1) combined_trainlog (nr3d + sr3d augmentation, s_vs_n_weight=0.25)
# 2) nr3d split
# 3) sr3d split
#
# Paths are loaded from configs/paths.sh via scripts/_common.sh.
# This script expects a 519-compatible listener wrapper checkpoint.
# In this release, LLAMA_STEPSLOT_EVAL_CKPT_519 defaults to the UB checkpoint path.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SCANNET_PKL="${REFERIT_SCANNET_FILE:-}"
BERT_PATH="${BERT_PATH:-}"
NR3D_CSV="${NR3D_TRAIN_CSV:-}"
SR3D_CSV="${SR3D_TRAIN_CSV:-}"
MASK3D_FEATS_TRAIN="${MASK3D_FEATS_TRAIN:-}"
MASK3D_FEATS_TEST="${MASK3D_FEATS_TEST:-}"
FEAT_DIM="${FEAT_DIM:-128}"
VIGOR_WRAPPER_CKPT="${VIGOR_WRAPPER_CKPT:-${LLAMA_STEPSLOT_EVAL_CKPT_519:-${LLAMA_STEPSLOT_EVAL_CKPT_UB:-${REPO_ROOT}/data/LLAMA_STEPSLOT_EVAL_CKPT_UB/best_model.pth}}}"
LISTENER_INIT_CKPT_BERT="${LISTENER_INIT_CKPT_BERT:-}"
LLM_PATH="${LLM_PATH:-${REPO_ROOT}/data/LLM_PATH/Tiny-Vicuna-1B}"

_need_file "${SCANNET_PKL}"
_need_dir "${BERT_PATH}"
_check_vigor_train_test_csv_pair "${NR3D_CSV}" "NR3D"
_check_vigor_train_test_csv_pair "${SR3D_CSV}" "SR3D"
_need_dir "${MASK3D_FEATS_TRAIN}"
_need_dir "${MASK3D_FEATS_TEST}"
_need_file "${VIGOR_WRAPPER_CKPT}"
_need_file "${LISTENER_INIT_CKPT_BERT}"
_need_dir "${LLM_PATH}"

GPU_LIST="${GPU_LIST:-0}"
ORDER_LEN="${ORDER_LEN:-4}"
MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS:-88}"
MAX_DISTRACTORS="${MAX_DISTRACTORS:-51}"
N_WORKERS="${N_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Match historical defaults used by the 51.9 protocol.
VIGOR_LLM_MAX_LEN="${VIGOR_LLM_MAX_LEN:-64}"
VIGOR_LLM_MEM_TOKENS="${VIGOR_LLM_MEM_TOKENS:-16}"
VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"
VIGOR_LLM_STEPSLOT_ONEPASS="${VIGOR_LLM_STEPSLOT_ONEPASS:-0}"
SVSN_WEIGHT="${SVSN_WEIGHT:-0.25}"

export VIGOR_USE_PRED_BOX_INFO="${VIGOR_USE_PRED_BOX_INFO:-1}"
export VIGOR_PRED_CLASS_MASK_MODE="${VIGOR_PRED_CLASS_MASK_MODE:-all_ones}"
export VIGOR_STEP_MARKERS="${VIGOR_STEP_MARKERS:-1}"

export VIGOR_LLM_MODEL_PATH="${LLM_PATH}"
export VIGOR_LLM_MAX_LEN
export VIGOR_LLM_MEM_TOKENS
export VIGOR_LLM_USE_BF16
export VIGOR_LLM_STEPSLOT_ONEPASS

# Kept for parity with historical script; not used in eval forward pass.
export VIGOR_LLM_DISTILL_W="${VIGOR_LLM_DISTILL_W:-1.0}"
export VIGOR_LLM_DISTILL_TYPE="${VIGOR_LLM_DISTILL_TYPE:-cos}"
export VIGOR_LLM_GLOBAL_DISTILL_W="${VIGOR_LLM_GLOBAL_DISTILL_W:-1.0}"
export VIGOR_LLM_GLOBAL_DISTILL_TYPE="${VIGOR_LLM_GLOBAL_DISTILL_TYPE:-cos}"

export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${VIGOR_WRAPPER_CKPT}"
export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT=0
export VIGOR_LISTENER_INIT_CKPT="${LISTENER_INIT_CKPT_BERT}"

LOGDIR_BASE="${OUTPUT_ROOT}/ssr3dllm/llama_stepslot_eval_suite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR_BASE}"

echo "[llama_stepslot_eval_suite] CKPT=${VIGOR_WRAPPER_CKPT}"
echo "[llama_stepslot_eval_suite] LOGDIR_BASE=${LOGDIR_BASE}"
echo "[llama_stepslot_eval_suite] onepass=${VIGOR_LLM_STEPSLOT_ONEPASS} llm_max_len=${VIGOR_LLM_MAX_LEN} llm_mem_tokens=${VIGOR_LLM_MEM_TOKENS}"

_run() {
  local name="$1"
  shift
  local out_dir="${LOGDIR_BASE}/${name}"
  mkdir -p "${out_dir}"

  echo "[llama_stepslot_eval_suite] ===== ${name} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}" \
  python third_party/Vigor/referit3d/scripts/train_referit3d_llama_stepslot.py \
    -scannet-file "${SCANNET_PKL}" \
    "$@" \
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
    --obj-cls-alpha 0.5 \
    --batch-size "${BATCH_SIZE}" \
    --n-workers "${N_WORKERS}" \
    --n-gpus 1 \
    --gpu "${GPU_LIST}" \
    --log-dir "${out_dir}" \
    2>&1 | tee "${out_dir}/eval.log"

  grep -n "Reference-Accuracy:" "${out_dir}/eval.log" || true
}

# (1) Historical trainlog-style combined evaluation (~0.519)
_run "combined_trainlog" -referit3D-file "${NR3D_CSV}" --augment-with-sr3d "${SR3D_CSV}" --s-vs-n-weight "${SVSN_WEIGHT}"

# (2) Split evaluation
_run "nr3d" -referit3D-file "${NR3D_CSV}"
_run "sr3d" -referit3D-file "${SR3D_CSV}"

echo "[llama_stepslot_eval_suite] Done. Logs under: ${LOGDIR_BASE}"
