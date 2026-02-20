#!/usr/bin/env bash
set -euo pipefail

# Unified SSR3DLLM eval entry:
# - Reproduce paper metrics (50.3 / 51.9) using the existing release scripts.
# - Run single-question inference where:
#   - "<geom>" prompts route to ReferIt3D listener runtime.
#   - normal prompts use the language branch.
# - If --checkpoint points to packed SSR3DLLM.ckpt, ask-mode auto-runs eval_adapt
#   and uses:
#   - language view ckpt for language branch
#   - listener view ckpt for <geom> routing
#
# Usage:
#   bash scripts/eval_ssr3dllm_unified.sh repro --profile 503
#   bash scripts/eval_ssr3dllm_unified.sh repro --profile 519
#   bash scripts/eval_ssr3dllm_unified.sh repro --profile both
#
#   bash scripts/eval_ssr3dllm_unified.sh ask \
#     --scene-id scene0000_00 \
#     --question "Describe this scene."
#
#   bash scripts/eval_ssr3dllm_unified.sh ask \
#     --scene-id scene0000_00 \
#     --question "<geom> the chair next to the table" \
#     --geom-profile 519

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_resolve_ckpt_path() {
  local preferred="$1"
  local fallback="$2"
  local label="$3"

  if [[ -n "${preferred}" && -f "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  if [[ -n "${preferred}" && ! -f "${preferred}" && -n "${fallback}" && -f "${fallback}" ]]; then
    echo "[WARN] ${label} path not found, fallback to repo-local data path." >&2
    echo "[WARN] ${label} missing: ${preferred}" >&2
    echo "[WARN] ${label} using  : ${fallback}" >&2
    echo "${fallback}"
    return 0
  fi

  if [[ -n "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  echo "${fallback}"
}

# Release default: always prefer repo-local data root.
REPO_DATA_ROOT="${REPO_ROOT}/data"
DATA_ROOT_LOCAL="${RELEASE_DATA_ROOT:-${REPO_DATA_ROOT}}"
LLAMA_STEPSLOT_EVAL_CKPT="$(_resolve_ckpt_path "${LLAMA_STEPSLOT_EVAL_CKPT:-}" "${DATA_ROOT_LOCAL}/LLAMA_STEPSLOT_EVAL_CKPT/llama_stepslot_onepass_varlen_mask_stageC_pred_latest_best.pth" "LLAMA_STEPSLOT_EVAL_CKPT")"
LLAMA_STEPSLOT_EVAL_CKPT_519="$(_resolve_ckpt_path "${LLAMA_STEPSLOT_EVAL_CKPT_519:-${LLAMA_STEPSLOT_EVAL_CKPT_UB:-}}" "${DATA_ROOT_LOCAL}/LLAMA_STEPSLOT_EVAL_CKPT_UB/best_model.pth" "LLAMA_STEPSLOT_EVAL_CKPT_519")"
SSR3DLLM_UNIFIED_CKPT="$(_resolve_ckpt_path "${SSR3DLLM_UNIFIED_CKPT:-}" "${DATA_ROOT_LOCAL}/grounded3dllm_ckpts/step3/last-epoch.ckpt" "SSR3DLLM_UNIFIED_CKPT")"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_ssr3dllm_unified.sh repro [--profile 503|519|both]

  bash scripts/eval_ssr3dllm_unified.sh ask \
    --scene-id <scene_id> \
    --question <text> \
    [--prompt-profile raw|appendix|paper] \
    [--geom-profile 503|519] \
    [--checkpoint <step3_ckpt>] \
    [--split train|validation|test] \
    [--max-new-tokens <int>] \
    [--decode-profile raw|paper] \
    [--top-p <float>] \
    [--repetition-penalty <float>] \
    [--length-penalty <float>]

Env:
  UNIFIED_AUTO_EVAL_ADAPT=1|0    default: 1
  UNIFIED_FORCE_EVAL_ADAPT=1|0   default: 0
EOF
}

mode="${1:-}"
if [[ -z "${mode}" ]]; then
  usage
  exit 2
fi
if [[ "${mode}" == "-h" || "${mode}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

run_repro() {
  local profile="$1"
  case "${profile}" in
    503)
      echo "[ssr3dllm_unified] repro profile=503 (onepass varlen chain)"
      LLAMA_STEPSLOT_EVAL_CKPT="${LLAMA_STEPSLOT_EVAL_CKPT}" \
      bash "${REPO_ROOT}/scripts/eval_llama_stepslot_varlen_chain_onepass_pred.sh"
      ;;
    519)
      echo "[ssr3dllm_unified] repro profile=519 (historical multipass trainlog protocol)"
      VIGOR_WRAPPER_CKPT="${LLAMA_STEPSLOT_EVAL_CKPT_519}" \
      bash "${REPO_ROOT}/scripts/eval_llama_stepslot_referit3d_suite.sh"
      ;;
    both)
      run_repro "503"
      run_repro "519"
      ;;
    *)
      echo "[FATAL] unsupported --profile: ${profile}" >&2
      exit 2
      ;;
  esac
}

_is_packed_ssr3dllm_ckpt() {
  local ckpt="$1"
  python - "${ckpt}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu")
    ok = isinstance(payload, dict) and isinstance(payload.get("ssr3dllm_bundle"), dict)
except Exception:
    ok = False
print("1" if ok else "0")
PY
}

_ckpt_cache_tag() {
  local ckpt="$1"
  local profile="$2"
  python - "${ckpt}" "${profile}" <<'PY'
import hashlib
import pathlib
import sys

p = pathlib.Path(sys.argv[1]).resolve()
profile = str(sys.argv[2]).strip()
st = p.stat()
raw = f"{p}|{st.st_size}|{int(st.st_mtime)}|{profile}"
print(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12])
PY
}

_auto_eval_adapt_for_ask() {
  local ckpt="$1"
  local profile="$2"
  local tag
  tag="$(_ckpt_cache_tag "${ckpt}" "${profile}")"
  local cache_root="${OUTPUT_ROOT}/ssr3dllm/eval_adapt_ask_cache"
  local base_name
  base_name="$(basename "${ckpt}")"
  base_name="${base_name%.*}"
  mkdir -p "${cache_root}"

  local out_wrapper="${cache_root}/${base_name}_${tag}_wrapper_${profile}.pth"
  local out_listener="${cache_root}/${base_name}_${tag}_listener_${profile}.pth"
  local out_language="${cache_root}/${base_name}_${tag}_language.ckpt"
  local out_report="${cache_root}/${base_name}_${tag}_report_${profile}.json"
  local force="${UNIFIED_FORCE_EVAL_ADAPT:-0}"

  if [[ "${force}" == "1" || ! -f "${out_wrapper}" || ! -f "${out_listener}" || ! -f "${out_language}" ]]; then
    echo "[ssr3dllm_unified] auto eval_adapt for packed ckpt (profile=${profile})" >&2
    python "${REPO_ROOT}/tools/eval_adapt_ssr3dllm_ckpt.py" \
      --checkpoint "${ckpt}" \
      --profile "${profile}" \
      --out-wrapper "${out_wrapper}" \
      --out-listener "${out_listener}" \
      --out-language "${out_language}" \
      | tee "${out_report}" >&2
  else
    echo "[ssr3dllm_unified] reuse eval_adapt cache (profile=${profile})" >&2
  fi

  echo "${out_language}|${out_listener}|${out_wrapper}|${out_report}"
}

if [[ "${mode}" == "repro" ]]; then
  profile="both"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        profile="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "[FATAL] unknown arg for repro: $1" >&2
        usage
        exit 2
        ;;
    esac
  done

  _need_file "${LLAMA_STEPSLOT_EVAL_CKPT}"
  _need_file "${LLAMA_STEPSLOT_EVAL_CKPT_519}"
  run_repro "${profile}"
  exit 0
fi

if [[ "${mode}" == "ask" ]]; then
  scene_id=""
  question=""
  prompt_profile="paper"
  geom_profile="503"
  checkpoint="${SSR3DLLM_UNIFIED_CKPT}"
  split="validation"
  # Align defaults with `LLama3d.evaluate(...)` to match paper eval behavior.
  max_new_tokens="150"
  top_p="1.0"
  repetition_penalty="1.2"
  length_penalty="1.0"
  decode_profile="paper"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scene-id)
        scene_id="${2:-}"
        shift 2
        ;;
      --question)
        question="${2:-}"
        shift 2
        ;;
      --prompt-profile)
        prompt_profile="${2:-}"
        shift 2
        ;;
      --geom-profile)
        geom_profile="${2:-}"
        shift 2
        ;;
      --checkpoint)
        checkpoint="${2:-}"
        shift 2
        ;;
      --split)
        split="${2:-}"
        shift 2
        ;;
      --max-new-tokens)
        max_new_tokens="${2:-}"
        shift 2
        ;;
      --decode-profile)
        decode_profile="${2:-}"
        shift 2
        ;;
      --top-p)
        top_p="${2:-}"
        shift 2
        ;;
      --repetition-penalty)
        repetition_penalty="${2:-}"
        shift 2
        ;;
      --length-penalty)
        length_penalty="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "[FATAL] unknown arg for ask: $1" >&2
        usage
        exit 2
        ;;
    esac
  done

  [[ -n "${scene_id}" ]] || { echo "[FATAL] --scene-id is required for ask mode" >&2; exit 2; }
  [[ -n "${question}" ]] || { echo "[FATAL] --question is required for ask mode" >&2; exit 2; }
  case "${prompt_profile}" in
    raw|appendix|paper) ;;
    *) echo "[FATAL] --prompt-profile must be raw|appendix|paper" >&2; exit 2 ;;
  esac

  local_listener_ckpt=""
  case "${geom_profile}" in
    503) local_listener_ckpt="${LLAMA_STEPSLOT_EVAL_CKPT}" ;;
    519) local_listener_ckpt="${LLAMA_STEPSLOT_EVAL_CKPT_519}" ;;
    *) echo "[FATAL] --geom-profile must be 503|519" >&2; exit 2 ;;
  esac

  auto_eval_adapt="${UNIFIED_AUTO_EVAL_ADAPT:-1}"
  if [[ "${auto_eval_adapt}" == "1" ]]; then
    packed_flag="$(_is_packed_ssr3dllm_ckpt "${checkpoint}")"
    if [[ "${packed_flag}" == "1" ]]; then
      adapt_triplet="$(_auto_eval_adapt_for_ask "${checkpoint}" "${geom_profile}")"
      IFS='|' read -r checkpoint local_listener_ckpt adapted_wrapper_ckpt adapt_report_path <<< "${adapt_triplet}"
      if [[ -z "${checkpoint}" || -z "${local_listener_ckpt}" || -z "${adapted_wrapper_ckpt}" || -z "${adapt_report_path}" ]]; then
        echo "[FATAL] eval_adapt parsing failed: ${adapt_triplet}" >&2
        exit 2
      fi
      echo "[ssr3dllm_unified] packed ckpt detected: language_ckpt=${checkpoint}"
      echo "[ssr3dllm_unified] packed ckpt detected: listener_ckpt=${local_listener_ckpt}"
      echo "[ssr3dllm_unified] packed ckpt detected: wrapper_ckpt=${adapted_wrapper_ckpt}"
      echo "[ssr3dllm_unified] packed ckpt detected: report=${adapt_report_path}"
    fi
  fi

  _need_file "${checkpoint}"
  _need_file "${local_listener_ckpt}"
  _need_file "${REFERIT_SCANNET_FILE}"
  _need_dir "${SCANNET200_ROOT}"
  _need_dir "${BERT_PATH}"
  case "${split}" in
    train) _need_file "${SCANNET200_ROOT}/train_database.yaml" ;;
    validation) _need_file "${SCANNET200_ROOT}/validation_database.yaml" ;;
    test) _need_file "${SCANNET200_ROOT}/test_database.yaml" ;;
    *) echo "[FATAL] unsupported split: ${split}" >&2; exit 2 ;;
  esac

  echo "[ssr3dllm_unified] ask mode"
  echo "[ssr3dllm_unified] checkpoint=${checkpoint}"
  echo "[ssr3dllm_unified] scene_id=${scene_id}"
  echo "[ssr3dllm_unified] prompt_profile=${prompt_profile}"
  echo "[ssr3dllm_unified] geom_profile=${geom_profile} listener_ckpt=${local_listener_ckpt}"

  has_geom="0"
  if [[ "${question}" == *"<geom>"* ]]; then
    has_geom="1"
  fi

  route_use_llm_order="0"
  geom_lora_enable="0"
  geom_stepslot_ckpt=""
  if [[ "${geom_profile}" == "503" && "${has_geom}" == "1" ]]; then
    route_use_llm_order="1"
    geom_lora_enable="1"
    geom_stepslot_ckpt="${local_listener_ckpt}"
  fi

  SSR3DLLM_ROUTE_GEOM_VIGOR="1" \
  SSR3DLLM_ROUTE_GEOM_USE_LLM_ORDER="${route_use_llm_order}" \
  SSR3DLLM_GEOM_LORA_ENABLE="${geom_lora_enable}" \
  SSR3DLLM_GEOM_STEPSLOT_CKPT="${geom_stepslot_ckpt}" \
  SSR3DLLM_GEOM_LORA_ADAPTER_NAME="${SSR3DLLM_GEOM_LORA_ADAPTER_NAME:-geom503}" \
  SSR3DLLM_STEP_TOKENS="${SSR3DLLM_STEP_TOKENS:-1}" \
  SSR3DLLM_STEP_ORDER_LEN="${SSR3DLLM_STEP_ORDER_LEN:-4}" \
  SSR3DLLM_ENABLE_STOP_TOKEN="${SSR3DLLM_ENABLE_STOP_TOKEN:-1}" \
  SSR3DLLM_STOP_TOKEN="${SSR3DLLM_STOP_TOKEN:-<STOP>}" \
  SSR3DLLM_LLM_STEPSLOT_MAX_LEN="${SSR3DLLM_LLM_STEPSLOT_MAX_LEN:-128}" \
  SSR3DLLM_REFERIT3D_LISTENER_CKPT="${local_listener_ckpt}" \
  SSR3DLLM_REFERIT3D_LISTENER_BERT="${BERT_PATH}" \
  SSR3DLLM_REFERIT_SCANNET_FILE="${REFERIT_SCANNET_FILE}" \
  SCANNET_PKL="${REFERIT_SCANNET_FILE}" \
  python "${REPO_ROOT}/tools/ssr3dllm_unified_eval.py" \
    --checkpoint "${checkpoint}" \
    --scene-id "${scene_id}" \
    --question "${question}" \
    --prompt-profile "${prompt_profile}" \
    --decode-profile "${decode_profile}" \
    --split "${split}" \
    --scannet-processed-root "${SCANNET200_ROOT}" \
    --max-new-tokens "${max_new_tokens}" \
    --top-p "${top_p}" \
    --repetition-penalty "${repetition_penalty}" \
    --length-penalty "${length_penalty}"
  exit 0
fi

echo "[FATAL] unsupported mode: ${mode}" >&2
usage
exit 2
