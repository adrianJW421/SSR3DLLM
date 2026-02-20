#!/usr/bin/env bash
set -euo pipefail

# Reproduce paper-facing metrics from pre-staged assets under `data/`.
#
# Coverage:
# 1) P1 train chain (optional): base LoRA -> stageC_pred
# 2) P1 main eval (50.3-style): varlen=0/1 on NR3D/SR3D
# 3) Main-table 51.9-style eval: combined trainlog protocol (+ nr3d/sr3d split)
# 4) P2 eval:
#    - ours (step-slot proxy; QueryAcc + Reference-Accuracy in logs)
#    - baseline refmatch qnorm (requires eval_vigor_refmatch_baseline_aabb.py tool)
#
# Usage:
#   cd final_github_release
#   bash scripts/reproduce_paper_metrics_from_data.sh
#
# Optional:
#   STAGES="p1_eval_503,p1_eval_519,p2_eval_ours,p2_eval_baseline" bash scripts/reproduce_paper_metrics_from_data.sh
#   STAGES="p1_train,p1_eval_503" bash scripts/reproduce_paper_metrics_from_data.sh
#
# Stage names:
#   - p1_train
#   - p1_eval_503
#   - p1_eval_519
#   - p2_eval_ours
#   - p2_eval_baseline

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${RELEASE_ROOT}}"
DATA_ROOT="${RELEASE_ROOT}/data"

STAGES="${STAGES:-all}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/ssr3dllm/reproduce_paper_${RUN_TAG}"
SUMMARY_FILE="${RUN_ROOT}/summary.txt"

GPU_LIST="${GPU_LIST:-0}"
N_WORKERS="${N_WORKERS:-8}"
ORDER_LEN="${ORDER_LEN:-4}"
MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS:-88}"
MAX_DISTRACTORS="${MAX_DISTRACTORS:-51}"

P1_EVAL_CKPT="${P1_EVAL_CKPT:-${LLAMA_STEPSLOT_EVAL_CKPT:-${DATA_ROOT}/LLAMA_STEPSLOT_EVAL_CKPT/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth}}"

# 51.9 protocol ckpt: in this release, default to UB listener checkpoint path.
EVAL_519_CKPT="${EVAL_519_CKPT:-${LLAMA_STEPSLOT_EVAL_CKPT_519:-${LLAMA_STEPSLOT_EVAL_CKPT_UB:-${DATA_ROOT}/LLAMA_STEPSLOT_EVAL_CKPT_UB/best_model.pth}}}"
EVAL_519_ONEPASS="${EVAL_519_ONEPASS:-0}"
EVAL_519_SVSN_WEIGHT="${EVAL_519_SVSN_WEIGHT:-0.25}"
EVAL_519_BATCH_SIZE="${EVAL_519_BATCH_SIZE:-64}"
EVAL_519_LLM_MAX_LEN="${EVAL_519_LLM_MAX_LEN:-64}"
EVAL_519_LLM_MEM_TOKENS="${EVAL_519_LLM_MEM_TOKENS:-16}"

# P2 ours
P2_OURS_CKPT="${P2_OURS_CKPT:-${P1_EVAL_CKPT}}"
P2_OURS_BATCH_SIZE="${P2_OURS_BATCH_SIZE:-64}"
P2_OURS_RANDOM_SEED="${P2_OURS_RANDOM_SEED:-2020}"
P2_OURS_LLM_MAX_LEN="${P2_OURS_LLM_MAX_LEN:-64}"
P2_OURS_LLM_MEM_TOKENS="${P2_OURS_LLM_MEM_TOKENS:-16}"
P2_OURS_ONEPASS="${P2_OURS_ONEPASS:-1}"

# P2 baseline (qnorm)
MASK3D_FEATS_TRAIN_QNORM="${MASK3D_FEATS_TRAIN_QNORM:-${DATA_ROOT}/MASK3D_FEATS_TRAIN_predbox_qnorm}"
MASK3D_FEATS_TEST_QNORM="${MASK3D_FEATS_TEST_QNORM:-${DATA_ROOT}/MASK3D_FEATS_TEST_predbox_qnorm}"
G3DLLM_CKPT="${G3DLLM_CKPT:-${DATA_ROOT}/grounded3dllm_ckpts/step3/last-epoch.ckpt}"
G3DLLM_SCANNET_ROOT="${G3DLLM_SCANNET_ROOT:-${SCANNET200_ROOT:-${DATA_ROOT}/SCANNET200_ROOT}}"
P2_BASELINE_SEEDS="${P2_BASELINE_SEEDS:-2020,1,10,20,100}"

# Some environments cannot write ~/.matplotlib; avoid noisy warnings.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/.mplconfig}"

log() { echo "[reproduce_paper] $*"; }
warn() { echo "[reproduce_paper][WARN] $*" >&2; }
die() { echo "[reproduce_paper][FATAL] $*" >&2; exit 2; }

mkdir -p "${RUN_ROOT}"
: > "${SUMMARY_FILE}"

append_summary() {
  echo "$*" | tee -a "${SUMMARY_FILE}"
}

normalize_stages() {
  local s
  s="$(echo "${STAGES}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  echo "${s}"
}

STAGES_NORM="$(normalize_stages)"

stage_enabled() {
  local name="$1"
  if [[ "${STAGES_NORM}" == "all" ]]; then
    return 0
  fi
  if [[ ",${STAGES_NORM}," == *",${name},"* ]]; then
    return 0
  fi
  return 1
}

pick_first_existing_file() {
  local p
  for p in "$@"; do
    if [[ -n "${p}" && -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

extract_refacc() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo ""
    return 0
  fi
  grep -Eo 'Reference-Accuracy:[[:space:]]*[0-9.]+' "${log_file}" | head -n 1 | awk '{print $2}'
}

extract_queryacc() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo ""
    return 0
  fi
  grep -Eo 'QueryAcc:[[:space:]]*[0-9.]+' "${log_file}" | head -n 1 | awk '{print $2}'
}

run_logged() {
  local stage_name="$1"
  shift
  local stage_log="${RUN_ROOT}/${stage_name}.log"
  log "run stage=${stage_name}"
  (
    set -x
    "$@"
  ) 2>&1 | tee "${stage_log}"
}

require_llama_paths() {
  _need_file "${REFERIT_SCANNET_FILE:-}" || exit 2
  _need_dir "${BERT_PATH:-}" || exit 2
  _check_vigor_train_test_csv_pair "${NR3D_TRAIN_CSV:-}" "NR3D" || exit 2
  _check_vigor_train_test_csv_pair "${SR3D_TRAIN_CSV:-}" "SR3D" || exit 2
  _need_dir "${MASK3D_FEATS_TRAIN:-}" || exit 2
  _need_dir "${MASK3D_FEATS_TEST:-}" || exit 2
  _need_file "${LISTENER_INIT_CKPT_BERT:-}" || exit 2
  _need_dir "${LLM_PATH:-}" || exit 2
}

append_summary "# reproduce_paper_metrics_from_data"
append_summary "run_tag=${RUN_TAG}"
append_summary "run_root=${RUN_ROOT}"
append_summary "stages=${STAGES}"
append_summary "gpu_list=${GPU_LIST}"

if stage_enabled "p1_train"; then
  log "checking paths for p1_train"
  require_llama_paths
  log "stage p1_train: base lora + stageC_pred"
  run_logged "p1_train_base" env \
    GPU_LIST="${GPU_LIST}" \
    N_WORKERS="${N_WORKERS}" \
    ORDER_LEN="${ORDER_LEN}" \
    MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS}" \
    MAX_DISTRACTORS="${MAX_DISTRACTORS}" \
    bash "${RELEASE_ROOT}/scripts/train_llama_stepslot_onepass_varlen_stop_base_lora.sh"

  run_logged "p1_train_stagec_pred" env \
    GPU_LIST="${GPU_LIST}" \
    N_WORKERS="${N_WORKERS}" \
    ORDER_LEN="${ORDER_LEN}" \
    MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS}" \
    MAX_DISTRACTORS="${MAX_DISTRACTORS}" \
    bash "${RELEASE_ROOT}/scripts/train_llama_stepslot_onepass_varlen_stop_base_lora_stageC_pred.sh"

  TRAINED_STAGEC_CKPT="${OUTPUT_ROOT}/ssr3dllm/mask3d_vigor_llama_step_slot/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth"
  if [[ -f "${TRAINED_STAGEC_CKPT}" ]]; then
    P1_EVAL_CKPT="${TRAINED_STAGEC_CKPT}"
    if [[ -z "${P2_OURS_CKPT:-}" || ! -f "${P2_OURS_CKPT:-}" ]]; then
      P2_OURS_CKPT="${TRAINED_STAGEC_CKPT}"
    fi
  else
    warn "stageC latest ckpt not found at ${TRAINED_STAGEC_CKPT}"
  fi
  append_summary "p1_train=done"
fi

if stage_enabled "p1_eval_503"; then
  log "checking paths for p1_eval_503"
  require_llama_paths
  _need_file "${P1_EVAL_CKPT}" || exit 2
  log "stage p1_eval_503: eval varlen chain on NR3D/SR3D with ckpt=${P1_EVAL_CKPT}"

  run_logged "p1_eval_503" env \
    GPU_LIST="${GPU_LIST}" \
    N_WORKERS="${N_WORKERS}" \
    ORDER_LEN="${ORDER_LEN}" \
    MAX_TEST_OBJECTS="${MAX_TEST_OBJECTS}" \
    MAX_DISTRACTORS="${MAX_DISTRACTORS}" \
    LLAMA_STEPSLOT_EVAL_CKPT="${P1_EVAL_CKPT}" \
    bash "${RELEASE_ROOT}/scripts/eval_llama_stepslot_varlen_chain_onepass_pred.sh"

  P1_EVAL_DIR="$(ls -td "${OUTPUT_ROOT}"/ssr3dllm/llama_stepslot_onepass_pred_eval_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${P1_EVAL_DIR}" ]]; then
    die "cannot find p1 eval output dir under ${OUTPUT_ROOT}/ssr3dllm"
  fi

  P1_NR3D_V0="$(extract_refacc "${P1_EVAL_DIR}/nr3d/varlen_0/eval.log")"
  P1_SR3D_V0="$(extract_refacc "${P1_EVAL_DIR}/sr3d/varlen_0/eval.log")"
  P1_NR3D_V1="$(extract_refacc "${P1_EVAL_DIR}/nr3d/varlen_1/eval.log")"
  P1_SR3D_V1="$(extract_refacc "${P1_EVAL_DIR}/sr3d/varlen_1/eval.log")"

  append_summary "p1_eval_503.dir=${P1_EVAL_DIR}"
  append_summary "p1_eval_503.refacc.nr3d.varlen0=${P1_NR3D_V0}"
  append_summary "p1_eval_503.refacc.sr3d.varlen0=${P1_SR3D_V0}"
  append_summary "p1_eval_503.refacc.nr3d.varlen1=${P1_NR3D_V1}"
  append_summary "p1_eval_503.refacc.sr3d.varlen1=${P1_SR3D_V1}"
fi

run_eval_519_split() {
  local tag="$1"
  local train_csv="$2"
  local out_dir="$3"
  local with_sr3d="$4"

  mkdir -p "${out_dir}"

  local cmd=(
    python third_party/Vigor/referit3d/scripts/train_referit3d_llama_stepslot.py
    -scannet-file "${REFERIT_SCANNET_FILE}"
    -referit3D-file "${train_csv}"
    --mode evaluate
    --use-scannet200-obj-cls True
    --mask3d-feature-root "${MASK3D_FEATS_TRAIN}"
    --mask3d-feature-root-test "${MASK3D_FEATS_TEST}"
    --mask3d-feature-dim "${FEAT_DIM:-128}"
    --max-distractors "${MAX_DISTRACTORS}"
    --max-test-objects "${MAX_TEST_OBJECTS}"
    --unit-sphere-norm True
    --bert-pretrain-path "${BERT_PATH}"
    --view_number 4
    --rotate_number 4
    --encoder-layer-num 3
    --decoder-layer-num 4
    --decoder-nhead-num 8
    --label-lang-sup True
    --multilabel-pretraining True
    --lang-multilabel True
    --cascading True
    --order-len "${ORDER_LEN}"
    --disable-text-loss True
    --lang-cls-alpha 0.0
    --obj-cls-alpha 0.5
    --batch-size "${EVAL_519_BATCH_SIZE}"
    --n-workers "${N_WORKERS}"
    --n-gpus 1
    --gpu "${GPU_LIST}"
    --log-dir "${out_dir}"
  )

  if [[ "${with_sr3d}" == "1" ]]; then
    cmd+=(--augment-with-sr3d "${SR3D_TRAIN_CSV}" --s-vs-n-weight "${EVAL_519_SVSN_WEIGHT}")
  fi

  (
    set -x
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"
    export VIGOR_USE_PRED_BOX_INFO="1"
    export VIGOR_PRED_CLASS_MASK_MODE="all_ones"
    export VIGOR_STEP_MARKERS="1"
    export VIGOR_LLM_MODEL_PATH="${LLM_PATH}"
    export VIGOR_LLM_MAX_LEN="${EVAL_519_LLM_MAX_LEN}"
    export VIGOR_LLM_MEM_TOKENS="${EVAL_519_LLM_MEM_TOKENS}"
    export VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"
    export VIGOR_LLM_STEPSLOT_ONEPASS="${EVAL_519_ONEPASS}"
    export VIGOR_LLM_DISTILL_W="${VIGOR_LLM_DISTILL_W:-1.0}"
    export VIGOR_LLM_DISTILL_TYPE="${VIGOR_LLM_DISTILL_TYPE:-cos}"
    export VIGOR_LLM_GLOBAL_DISTILL_W="${VIGOR_LLM_GLOBAL_DISTILL_W:-1.0}"
    export VIGOR_LLM_GLOBAL_DISTILL_TYPE="${VIGOR_LLM_GLOBAL_DISTILL_TYPE:-cos}"
    export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${EVAL_519_CKPT}"
    export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="0"
    export VIGOR_LISTENER_INIT_CKPT="${LISTENER_INIT_CKPT_BERT}"
    export VIGOR_LLM_LORA_AUTO="${VIGOR_LLM_LORA_AUTO:-1}"
    "${cmd[@]}"
  ) 2>&1 | tee "${out_dir}/eval.log"

  local refacc
  refacc="$(extract_refacc "${out_dir}/eval.log")"
  append_summary "p1_eval_519.refacc.${tag}=${refacc}"
}

if stage_enabled "p1_eval_519"; then
  log "checking paths for p1_eval_519"
  require_llama_paths
  _need_file "${EVAL_519_CKPT}" || exit 2
  log "stage p1_eval_519: combined trainlog protocol with ckpt=${EVAL_519_CKPT}"

  EVAL_519_DIR="${RUN_ROOT}/p1_eval_519"
  mkdir -p "${EVAL_519_DIR}"

  run_eval_519_split "combined_trainlog" "${NR3D_TRAIN_CSV}" "${EVAL_519_DIR}/combined_trainlog" "1"
  run_eval_519_split "nr3d" "${NR3D_TRAIN_CSV}" "${EVAL_519_DIR}/nr3d" "0"
  run_eval_519_split "sr3d" "${SR3D_TRAIN_CSV}" "${EVAL_519_DIR}/sr3d" "0"

  append_summary "p1_eval_519.dir=${EVAL_519_DIR}"
fi

run_p2_ours_split() {
  local tag="$1"
  local train_csv="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"

  (
    set -x
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"
    export VIGOR_CONTEXT_MODE="sampled"
    export VIGOR_USE_ALL_OBJECTS="0"
    export VIGOR_ALLOW_UNIQUE_TEST="1"
    export VIGOR_QUERYACC_PROXY="1"
    export VIGOR_USE_PRED_BOX_INFO="1"
    export VIGOR_PRED_CLASS_MASK_MODE="all_ones"
    export VIGOR_STEP_MARKERS="1"
    export VIGOR_LLM_MODEL_PATH="${LLM_PATH}"
    export VIGOR_LLM_MAX_LEN="${P2_OURS_LLM_MAX_LEN}"
    export VIGOR_LLM_MEM_TOKENS="${P2_OURS_LLM_MEM_TOKENS}"
    export VIGOR_LLM_USE_BF16="${VIGOR_LLM_USE_BF16:-1}"
    export VIGOR_LLM_STEPSLOT_ONEPASS="${P2_OURS_ONEPASS}"
    export VIGOR_LLM_ONEPASS_INPUT_MODE="pred"
    export VIGOR_LLM_DISTILL_W="0"
    export VIGOR_LLM_GLOBAL_DISTILL_W="0"
    export VIGOR_LLM_LORA_AUTO="${VIGOR_LLM_LORA_AUTO:-1}"
    export VIGOR_VARLEN_CHAIN="1"
    export VIGOR_VARLEN_MASK_SOURCE="oracle"
    export VIGOR_STOP_TOKEN="${VIGOR_STOP_TOKEN:-<STOP>}"
    export VIGOR_STOP_EMBED_REPLACE="${VIGOR_STOP_EMBED_REPLACE:-1}"
    export VIGOR_LLM_STEPSLOT_RESUME_CKPT="${P2_OURS_CKPT}"
    export VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT="0"
    export VIGOR_LISTENER_INIT_CKPT="${LISTENER_INIT_CKPT_BERT}"

    python third_party/Vigor/referit3d/scripts/train_referit3d_llama_stepslot.py \
      -scannet-file "${REFERIT_SCANNET_FILE}" \
      -referit3D-file "${train_csv}" \
      --mode evaluate \
      --use-scannet200-obj-cls True \
      --mask3d-feature-root "${MASK3D_FEATS_TRAIN}" \
      --mask3d-feature-root-test "${MASK3D_FEATS_TEST}" \
      --mask3d-feature-dim "${FEAT_DIM:-128}" \
      --max-distractors "${MAX_DISTRACTORS}" \
      --max-test-objects "${MAX_TEST_OBJECTS}" \
      --max-seq-len 40 \
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
      --batch-size "${P2_OURS_BATCH_SIZE}" \
      --n-workers "${N_WORKERS}" \
      --n-gpus 1 \
      --gpu "${GPU_LIST}" \
      --random-seed "${P2_OURS_RANDOM_SEED}" \
      --log-dir "${out_dir}"
  ) 2>&1 | tee "${out_dir}/eval.log"

  local refacc queryacc
  refacc="$(extract_refacc "${out_dir}/eval.log")"
  queryacc="$(extract_queryacc "${out_dir}/eval.log")"
  append_summary "p2_eval_ours.refacc.${tag}=${refacc}"
  append_summary "p2_eval_ours.queryacc.${tag}=${queryacc}"
}

if stage_enabled "p2_eval_ours"; then
  log "checking paths for p2_eval_ours"
  require_llama_paths
  _need_file "${P2_OURS_CKPT}" || exit 2
  log "stage p2_eval_ours: proxy eval (ours) with ckpt=${P2_OURS_CKPT}"
  P2_OURS_DIR="${RUN_ROOT}/p2_eval_ours"
  mkdir -p "${P2_OURS_DIR}"
  run_p2_ours_split "nr3d" "${NR3D_TRAIN_CSV}" "${P2_OURS_DIR}/nr3d"
  run_p2_ours_split "sr3d" "${SR3D_TRAIN_CSV}" "${P2_OURS_DIR}/sr3d"
  append_summary "p2_eval_ours.dir=${P2_OURS_DIR}"
fi

if stage_enabled "p2_eval_baseline"; then
  _need_file "${REFERIT_SCANNET_FILE}" || exit 2
  _need_file "${NR3D_TRAIN_CSV}" || exit 2
  _need_file "${SR3D_TRAIN_CSV}" || exit 2
  _need_dir "${MASK3D_FEATS_TRAIN_QNORM}" || exit 2
  _need_dir "${MASK3D_FEATS_TEST_QNORM}" || exit 2
  _need_file "${G3DLLM_CKPT}" || exit 2
  _need_dir "${G3DLLM_SCANNET_ROOT}" || exit 2

  P2_BASELINE_TOOL="${P2_BASELINE_TOOL:-}"
  if [[ -z "${P2_BASELINE_TOOL}" ]]; then
    P2_BASELINE_TOOL="$(pick_first_existing_file \
      "${RELEASE_ROOT}/tools/eval_vigor_refmatch_baseline_aabb.py" \
      "${PROJECT_ROOT}/tools/eval_vigor_refmatch_baseline_aabb.py" || true)"
  fi
  [[ -n "${P2_BASELINE_TOOL}" ]] || die "cannot find eval_vigor_refmatch_baseline_aabb.py; set P2_BASELINE_TOOL=..."

  BASELINE_CONFIG_ROOT="${BASELINE_CONFIG_ROOT:-${RELEASE_ROOT}/baseline/core/conf}"
  BASELINE_G3_DATA_CFG="${BASELINE_G3_DATA_CFG:-${RELEASE_ROOT}/baseline/core/conf/data/indoor_dialog.yaml}"
  BASELINE_G3_MODEL_CFG="${BASELINE_G3_MODEL_CFG:-${RELEASE_ROOT}/baseline/core/conf/model/mask3d_lang.yaml}"
  BASELINE_G3_TRAINER_CFG="${BASELINE_G3_TRAINER_CFG:-${RELEASE_ROOT}/baseline/core/conf/trainer/trainer50.yaml}"
  BASELINE_G3_LLM_CFG="${BASELINE_G3_LLM_CFG:-${RELEASE_ROOT}/baseline/core/conf/llm/tiny_vicuna_len512.json}"
  BASELINE_G3_LLM_DATA_CFG="${BASELINE_G3_LLM_DATA_CFG:-${RELEASE_ROOT}/baseline/core/conf/llm/det10.json}"

  P2_BASE_DIR="${RUN_ROOT}/p2_eval_baseline_qnorm"
  mkdir -p "${P2_BASE_DIR}"

  run_p2_baseline_split() {
    local tag="$1"
    local csv="$2"
    local out_dir="$3"
    mkdir -p "${out_dir}"

    (
      cd "${PROJECT_ROOT}"
      set -x
      export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
      export VIGOR_CONTEXT_MODE="sampled"
      export VIGOR_ALLOW_UNIQUE_TEST="1"
      export VIGOR_USE_ALL_OBJECTS="0"
      export REFERIT_REF_FEATURE_SOURCE="ref"
      export SSR3DLLM_ADD_GEOM_TOKEN="0"
      export SSR3DLLM_STEP_TOKENS="0"
      export SSR3DLLM_ENABLE_STOP_TOKEN="0"
      python "${P2_BASELINE_TOOL}" \
        --scannet-file "${REFERIT_SCANNET_FILE}" \
        --referit3d-file "${csv}" \
        --mask3d-feature-root "${MASK3D_FEATS_TRAIN_QNORM}" \
        --mask3d-feature-root-test "${MASK3D_FEATS_TEST_QNORM}" \
        --g3dllm-ckpt "${G3DLLM_CKPT}" \
        --g3dllm-scannet-root "${G3DLLM_SCANNET_ROOT}" \
        --config-root "${BASELINE_CONFIG_ROOT}" \
        --g3dllm-data-config "${BASELINE_G3_DATA_CFG}" \
        --g3dllm-model-config "${BASELINE_G3_MODEL_CFG}" \
        --g3dllm-trainer-config "${BASELINE_G3_TRAINER_CFG}" \
        --g3dllm-llm-config "${BASELINE_G3_LLM_CFG}" \
        --g3dllm-llm-data-config "${BASELINE_G3_LLM_DATA_CFG}" \
        --seeds "${P2_BASELINE_SEEDS}" \
        --restrict-to-context 1 \
        --export-context-probs 1 \
        --cache-all-scores 1 \
        --softmax-temperature 1.0 \
        --max-examples 0 \
        --print-generation-first 0 \
        --wrap-grounding-question 1 \
        --debug-gt-rank-first 0 \
        --out-dir "${out_dir}"
    ) 2>&1 | tee "${out_dir}/eval.log"

    local parsed
    parsed="$(python - "${out_dir}/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("NA")
    raise SystemExit(0)

d = json.loads(p.read_text(encoding="utf-8"))
ms = d.get("mean_std", {}) if isinstance(d, dict) else {}

v = ms.get("query_accuracy_ctx")
if isinstance(v, dict):
    m = v.get("mean")
    if isinstance(m, (int, float)):
        print(f"{m:.4f}")
        raise SystemExit(0)
v = ms.get("query_accuracy_all")
if isinstance(v, dict):
    m = v.get("mean")
    if isinstance(m, (int, float)):
        print(f"{m:.4f}")
        raise SystemExit(0)
print("NA")
PY
)"
    append_summary "p2_eval_baseline.queryacc.${tag}=${parsed}"
  }

  log "stage p2_eval_baseline: qnorm baseline proxy"
  run_p2_baseline_split "nr3d" "${NR3D_TRAIN_CSV}" "${P2_BASE_DIR}/nr3d"
  run_p2_baseline_split "sr3d" "${SR3D_TRAIN_CSV}" "${P2_BASE_DIR}/sr3d"
  append_summary "p2_eval_baseline.dir=${P2_BASE_DIR}"
fi

append_summary "done=1"
log "done. summary=${SUMMARY_FILE}"
