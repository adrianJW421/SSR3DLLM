#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SCANNET_PKL="${REFERIT_SCANNET_FILE:-}"
BERT_DIR="${BERT_PATH:-}"
LLM_DIR="${LLM_PATH:-}"
LISTENER_CKPT="${LISTENER_INIT_CKPT_BERT:-}"
FEAT_DIM="${FEAT_DIM:-128}"

SCANREFER_TRAIN_CSV="${SCANREFER_TRAIN_CSV:-}"
SCANREFER_MASK3D_FEATS_TRAIN="${SCANREFER_MASK3D_FEATS_TRAIN:-}"
SCANREFER_MASK3D_FEATS_TEST="${SCANREFER_MASK3D_FEATS_TEST:-}"
SCANREFER_DINO_SAMPLE_CACHE_ROOT="${SCANREFER_DINO_SAMPLE_CACHE_ROOT:-}"
SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT="${SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT:-}"

MULTI3DREF_TRAIN_CSV="${MULTI3DREF_TRAIN_CSV:-}"
MULTI3DREF_MASK3D_FEATS_TRAIN="${MULTI3DREF_MASK3D_FEATS_TRAIN:-}"
MULTI3DREF_MASK3D_FEATS_TEST="${MULTI3DREF_MASK3D_FEATS_TEST:-}"
MULTI3DREF_DINO_SAMPLE_CACHE_ROOT="${MULTI3DREF_DINO_SAMPLE_CACHE_ROOT:-}"
MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT="${MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT:-}"

_need_file "${SCANNET_PKL}"
_need_dir "${BERT_DIR}"
_need_dir "${LLM_DIR}"
_need_file "${LISTENER_CKPT}"

DATASETS="${DATASETS:-scanrefer,multi3dref}"
DATASETS="${DATASETS// /,}"
if [[ "${DATASETS}" == "all" ]]; then
  DATASETS="scanrefer,multi3dref"
fi

GPU_LIST="${GPU_LIST:-0}"
ORDER_LEN="${ORDER_LEN:-4}"
MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS:-256}"
MAX_DISTRACTORS="${MAX_DISTRACTORS:-256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_WORKERS="${N_WORKERS:-8}"
LLM_MAX_LEN="${LLM_MAX_LEN:-128}"
LLM_MEM_TOKENS="${LLM_MEM_TOKENS:-16}"
DINO_ALPHA="${DINO_ALPHA:-2.0}"
DINO_FEATURE_DIM="${DINO_FEATURE_DIM:-1024}"
RUN_NAME="${RUN_NAME:-ssr3dllm_scanrefer_multi3dref_eval_$(date +%Y%m%d_%H%M%S)}"
LOGDIR_BASE="${OUTPUT_ROOT}/ssr3dllm/${RUN_NAME}"

mkdir -p "${LOGDIR_BASE}"

export VIGOR_USE_PRED_BOX_INFO="${VIGOR_USE_PRED_BOX_INFO:-1}"
export SSR3DLLM_VIGOR_INMEMORY_BOX_MULTIVIEW="${SSR3DLLM_VIGOR_INMEMORY_BOX_MULTIVIEW:-1}"
export VIGOR_STEP_MARKERS="${VIGOR_STEP_MARKERS:-1}"
export VIGOR_STEP_SLOT_ONLY="${VIGOR_STEP_SLOT_ONLY:-1}"
export VIGOR_FREEZE_BERT_EXCEPT_STEP="${VIGOR_FREEZE_BERT_EXCEPT_STEP:-1}"
export VIGOR_TEXT_CLS_SCANNET200="${VIGOR_TEXT_CLS_SCANNET200:-0}"
export VIGOR_PRED_CLASS_MASK_MODE="${VIGOR_PRED_CLASS_MASK_MODE:-normal}"
export VIGOR_ADAPTIVE_HALT="${VIGOR_ADAPTIVE_HALT:-0}"
export VIGOR_STOP_TOKEN="${VIGOR_STOP_TOKEN:-<STOP>}"
export VIGOR_VARLEN_ONEPASS_TRUNC="${VIGOR_VARLEN_ONEPASS_TRUNC:-0}"
export VIGOR_STOP_EMBED_W="${VIGOR_STOP_EMBED_W:-0.0}"
export VIGOR_STOP_EMBED_REPLACE="${VIGOR_STOP_EMBED_REPLACE:-1}"
export VIGOR_LLM_MODEL_PATH="${LLM_DIR}"
export VIGOR_LLM_MAX_LEN="${LLM_MAX_LEN}"
export VIGOR_LLM_MEM_TOKENS="${LLM_MEM_TOKENS}"
export VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"
export VIGOR_LLM_STEPSLOT_ONEPASS="${VIGOR_LLM_STEPSLOT_ONEPASS:-1}"
export VIGOR_LLM_ONEPASS_INPUT_MODE="${VIGOR_LLM_ONEPASS_INPUT_MODE:-pred}"
export VIGOR_LLM_DISTILL_W="${VIGOR_LLM_DISTILL_W:-0}"
export VIGOR_LLM_GLOBAL_DISTILL_W="${VIGOR_LLM_GLOBAL_DISTILL_W:-0}"
export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="${VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT:-0}"
export VIGOR_LISTENER_INIT_CKPT="${LISTENER_CKPT}"
export VIGOR_LLM_LORA_AUTO="${VIGOR_LLM_LORA_AUTO:-1}"
export VIGOR_VARLEN_CHAIN="${VIGOR_VARLEN_CHAIN:-1}"
export VIGOR_VARLEN_MASK_SOURCE="${VIGOR_VARLEN_MASK_SOURCE:-oracle}"
export VIGOR_SKIP_ANALYZE="${VIGOR_SKIP_ANALYZE:-1}"

_contains_dataset() {
  [[ ",${DATASETS// /}," == *",$1,"* ]]
}

_run_dataset() {
  local tag="$1"
  local train_csv="$2"
  local feats_train="$3"
  local feats_test="$4"
  local ckpt="$5"
  local dino_root="$6"
  local out_dir="${LOGDIR_BASE}/${tag}"

  _check_vigor_train_test_csv_pair "${train_csv}" "${tag}"
  _need_dir "${feats_train}"
  _need_dir "${feats_test}"
  _need_file "${ckpt}"

  mkdir -p "${out_dir}"
  export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${ckpt}"
  if [[ -n "${dino_root}" ]]; then
    _need_dir "${dino_root}"
    export VIGOR_MASK3D_DINO_SAMPLE_CACHE_ROOT="${dino_root}"
    export VIGOR_MASK3D_DINO_ENABLE=1
  else
    unset VIGOR_MASK3D_DINO_SAMPLE_CACHE_ROOT
    export VIGOR_MASK3D_DINO_ENABLE=0
  fi
  export VIGOR_MASK3D_DINO_ALPHA="${DINO_ALPHA}"
  export VIGOR_MASK3D_DINO_FEATURE_DIM="${DINO_FEATURE_DIM}"

  echo "[ssr3dllm_eval] ${tag}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}" \
  python third_party/Vigor/referit3d/scripts/train_referit3d_llama_stepslot.py \
    -scannet-file "${SCANNET_PKL}" \
    -referit3D-file "${train_csv}" \
    --mode evaluate \
    --use-scannet200-obj-cls True \
    --mask3d-feature-root "${feats_train}" \
    --mask3d-feature-root-test "${feats_test}" \
    --mask3d-feature-dim "${FEAT_DIM}" \
    --max-distractors "${MAX_DISTRACTORS}" \
    --max-test-objects "${MAX_TEST_OBJECTS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --unit-sphere-norm True \
    --bert-pretrain-path "${BERT_DIR}" \
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

  grep -E "Reference-Accuracy|QueryAcc|BBox-Acc@IoU|BBox-Mean-IoU|TargetBox-Acc@IoU|F1@IoU|\\[Vigor\\]\\[speed\\]" "${out_dir}/eval.log" > "${out_dir}/metrics_summary.txt" || true
  cat "${out_dir}/metrics_summary.txt"
}

if _contains_dataset "scanrefer"; then
  _run_dataset "scanrefer" "${SCANREFER_TRAIN_CSV}" "${SCANREFER_MASK3D_FEATS_TRAIN}" "${SCANREFER_MASK3D_FEATS_TEST}" "${SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT}" "${SCANREFER_DINO_SAMPLE_CACHE_ROOT}"
fi

if _contains_dataset "multi3dref"; then
  _run_dataset "multi3dref" "${MULTI3DREF_TRAIN_CSV}" "${MULTI3DREF_MASK3D_FEATS_TRAIN}" "${MULTI3DREF_MASK3D_FEATS_TEST}" "${MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT}" "${MULTI3DREF_DINO_SAMPLE_CACHE_ROOT}"
fi

echo "[ssr3dllm_eval] logs: ${LOGDIR_BASE}"
