#!/usr/bin/env bash
set -euo pipefail

# One-click appendix qualitative examples (capability preservation) for the public release.
#
# Inputs:
# - baseline ckpt: a plain Grounded3D-LLM ckpt (typically Step3)
# - ours ckpt: a packed single-file `SSR3DLLM.ckpt` bundle (language + listener profiles)
#
# Pipeline:
# 1) `eval_adapt` the packed ckpt into a "language-view" `.ckpt` (same weights, just a load schema view).
# 2) Run the paper-equivalent test protocol (`train/step3_train_ssr3dllm_geom_entry.py --mode test`)
#    for both baseline and ours, dumping per-scene `*.json` predictions.
# 3) Sample qualitative rows (seed=1) via `tools/extract_capability_examples.py`.
#
# Notes:
# - This script is for appendix-style qualitative reproduction, not the single-question `ask` demo.
# - Default runs full test (`SSR3DLLM_LIMIT_TEST_BATCHES=1.0`). Override for a quick check if needed.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_ts() { date +%Y%m%d_%H%M%S; }

_resolve_abs_path() {
  local p="$1"
  python - "${p}" <<'PY'
from pathlib import Path
import sys

print(str(Path(sys.argv[1]).expanduser().resolve()))
PY
}

_bridge_scannet200_root() {
  local src_root="${SCANNET200_ROOT:-}"
  if [[ -z "${src_root}" ]]; then
    echo "[FATAL] SCANNET200_ROOT is empty; please configure configs/paths.sh." >&2
    exit 2
  fi
  _need_dir "${src_root}"

  local src_abs
  src_abs="$(_resolve_abs_path "${src_root}")"
  local link_parent="${REPO_ROOT}/data/processed"
  local link_path="${link_parent}/scannet200"
  mkdir -p "${link_parent}"

  if [[ -L "${link_path}" || ! -e "${link_path}" ]]; then
    ln -sfn "${src_abs}" "${link_path}"
  elif [[ -d "${link_path}" ]]; then
    local dst_abs
    dst_abs="$(_resolve_abs_path "${link_path}")"
    if [[ "${dst_abs}" != "${src_abs}" ]]; then
      echo "[FATAL] ${link_path} already exists but points to a different directory." >&2
      echo "        existing: ${dst_abs}" >&2
      echo "        expected: ${src_abs}" >&2
      echo "        remove it or update SCANNET200_ROOT before rerun." >&2
      exit 2
    fi
  else
    echo "[FATAL] ${link_path} exists and is not a directory/symlink." >&2
    exit 2
  fi

  echo "[cap_examples] bridge_scannet200: ${link_path} -> ${src_abs}"
}

_preflight_protocol_data_paths() {
  local ds_root="${REPO_ROOT}/data/processed/scannet200"
  _need_dir "${ds_root}"
  _need_file "${ds_root}/validation_database.yaml"
  _need_file "${ds_root}/label_database.yaml"
  _need_file "${ds_root}/color_mean_std.yaml"
  _need_dir "${ds_root}/validation"
  _need_dir "${ds_root}/instance_gt/validation"
  echo "[cap_examples] preflight_ok: ${ds_root}"
}

_resolve_raw_scans_dir() {
  local candidates=()

  if [[ -n "${SSR3DLLM_RAWSCANNET_SCANS_DIR:-}" ]]; then
    candidates+=("${SSR3DLLM_RAWSCANNET_SCANS_DIR}")
  fi
  if [[ -n "${SCANNET_SCANS_ROOT:-}" ]]; then
    candidates+=("${SCANNET_SCANS_ROOT}")
  fi
  candidates+=("${REPO_ROOT}/data/rawscannet/scans")

  if [[ -n "${SCANNET200_ROOT:-}" ]]; then
    local scannet200_abs
    scannet200_abs="$(_resolve_abs_path "${SCANNET200_ROOT}")"
    local maybe_root
    maybe_root="$(dirname "$(dirname "${scannet200_abs}")")"
    candidates+=("${maybe_root}/scannet/scans")
    candidates+=("${maybe_root}/scans")
  fi

  local c=""
  for c in "${candidates[@]}"; do
    if [[ -n "${c}" && -d "${c}" ]]; then
      export SSR3DLLM_RAWSCANNET_SCANS_DIR="${c}"
      echo "[cap_examples] raw_scans_dir=${SSR3DLLM_RAWSCANNET_SCANS_DIR}"
      return 0
    fi
  done

  echo "[FATAL] Cannot resolve raw ScanNet scans directory." >&2
  echo "        Please set SSR3DLLM_RAWSCANNET_SCANS_DIR=/abs/path/to/scannet/scans" >&2
  echo "        Tried candidates:" >&2
  for c in "${candidates[@]}"; do
    [[ -n "${c}" ]] && echo "          - ${c}" >&2
  done
  exit 2
}

_preflight_axis_align_files() {
  local ds_root="${REPO_ROOT}/data/processed/scannet200"
  local val_db="${ds_root}/validation_database.yaml"
  _need_file "${val_db}"
  _need_dir "${SSR3DLLM_RAWSCANNET_SCANS_DIR}"

  python - "${val_db}" "${SSR3DLLM_RAWSCANNET_SCANS_DIR}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

val_db = Path(sys.argv[1])
scans_root = Path(sys.argv[2])
text = val_db.read_text(encoding="utf-8", errors="ignore")

# Extract scene ids from `instance_gt_filepath` first; fallback to generic `filepath`.
ids: list[str] = []
for pat in [r"instance_gt_filepath:\s*([^\n]+)", r"filepath:\s*([^\n]+)"]:
    for m in re.finditer(pat, text):
        raw = m.group(1).strip().strip("'\"")
        sid = Path(raw).stem
        if sid and sid.startswith("scene"):
            ids.append(sid)
    if ids:
        break

uniq: list[str] = []
seen = set()
for sid in ids:
    if sid in seen:
        continue
    seen.add(sid)
    uniq.append(sid)
    if len(uniq) >= 20:
        break

if not uniq:
    raise SystemExit("[FATAL] axis-align preflight: cannot parse scene ids from validation_database.yaml")

missing: list[str] = []
for sid in uniq:
    p = scans_root / sid / f"{sid}.txt"
    if not p.exists():
        missing.append(str(p))

if missing:
    msg = (
        "[FATAL] axis-align preflight failed: missing raw ScanNet scene txt for sampled validation scenes.\n"
        f"  scans_root={scans_root}\n"
        f"  checked={len(uniq)} missing={len(missing)}\n"
        "  first_missing:\n"
        + "\n".join(f"    - {x}" for x in missing[:5])
    )
    raise SystemExit(msg)

print(f"[cap_examples] axis_align_preflight_ok scans_root={scans_root} checked={len(uniq)}")
PY
}

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

_ckpt_cache_tag() {
  local ckpt="$1"
  python - "${ckpt}" <<'PY'
import hashlib
import pathlib
import sys

p = pathlib.Path(sys.argv[1]).resolve()
st = p.stat()
raw = f"{p}|{st.st_size}|{int(st.st_mtime)}"
print(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12])
PY
}

_locate_pred_dir() {
  local save_dir="$1"
  python - "${save_dir}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
json_files = list(root.rglob("*.json"))

def looks_like(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and isinstance(data.get("prediction"), list)
    except Exception:
        return False

pred_files = [p for p in json_files if looks_like(p)]
if not pred_files:
    raise SystemExit(2)
parent = Counter(p.parent for p in pred_files).most_common(1)[0][0]
print(str(parent))
PY
}

_run_test_protocol() {
  local name="$1"
  local ckpt="$2"
  local out_root="$3"

  local save_root="${out_root}/runs"
  local save_dir="${save_root}/${name}"
  mkdir -p "${save_dir}"

  # IMPORTANT:
  # This function is used inside command substitution:
  #   pred_dir="$(_run_test_protocol ...)"
  # Keep logs on stderr so stdout only contains the final `pred_dir` line.
  echo "[cap_examples] run=${name} ckpt=${ckpt}" >&2
  echo "[cap_examples] save_dir=${save_dir}" >&2
  echo "[cap_examples] SSR3DLLM_LIMIT_TEST_BATCHES=${SSR3DLLM_LIMIT_TEST_BATCHES:-1.0}" >&2

  NUM_GPUS="${NUM_GPUS:-1}"
  if [[ "${NUM_GPUS}" != "1" ]]; then
    echo "[FATAL] This eval must run with NUM_GPUS=1 (got ${NUM_GPUS})." >&2
    exit 2
  fi

  # Force protocol-stable configs (avoid inherited envs changing task mixture).
  export DATA_CONFIG="${DATA_CONFIG:-baseline/core/conf/data/indoor_dialog.yaml}"
  export MODEL_CONFIG="${MODEL_CONFIG:-baseline/core/conf/model/mask3d_lang.yaml}"
  export TRAINER_CONFIG="${TRAINER_CONFIG:-baseline/core/conf/trainer/trainer50.yaml}"
  export LLM_CONFIG="${LLM_CONFIG:-baseline/core/conf/llm/tiny_vicuna_len512.json}"
  export LLM_DATA_CONFIG="${LLM_DATA_CONFIG:-baseline/core/conf/llm/det10.json}"

  export SSR3DLLM_TEST_CKPT="${ckpt}"
  export EXPNAME="${name}"
  export SSR3DLLM_SAVE_DIR_ROOT="${save_root}"
  export SSR3DLLM_LIMIT_TEST_BATCHES="${SSR3DLLM_LIMIT_TEST_BATCHES:-1.0}"
  export SSR3DLLM_EVAL_DETECTION="${SSR3DLLM_EVAL_DETECTION:-false}"
  export SSR3DLLM_GEOM_ONLY="${SSR3DLLM_GEOM_ONLY:-0}"
  export REL3D_ENABLE_RELATION_QA="${REL3D_ENABLE_RELATION_QA:-0}"
  export SSR3DLLM_GROUNDING_ADD_GEOM_TOKEN="${SSR3DLLM_GROUNDING_ADD_GEOM_TOKEN:-1}"
  export SSR3DLLM_GROUNDING_GEOM_PREFIXES="${SSR3DLLM_GROUNDING_GEOM_PREFIXES:-scanrefer,m3dref}"

  set +e
  python "${REPO_ROOT}/train/step3_train_ssr3dllm_geom_entry.py" --mode test 2>&1 | tee "${save_dir}/test.log" >&2
  local rc="${PIPESTATUS[0]}"
  set -e
  if [[ "${rc}" != "0" ]]; then
    echo "[FATAL] test failed for ${name} (rc=${rc})" >&2
    tail -200 "${save_dir}/test.log" >&2 || true
    exit "${rc}"
  fi

  local pred_dir
  pred_dir="$(_locate_pred_dir "${save_dir}")" || {
    echo "[FATAL] cannot locate per-scene prediction jsons under: ${save_dir}" >&2
    find "${save_dir}" -maxdepth 3 -type f -name "*.json" | head -n 50 >&2 || true
    exit 2
  }
  echo "[cap_examples] pred_dir=${pred_dir}" >&2

  set +e
  bash "${REPO_ROOT}/scripts/eval_llm.sh" "${pred_dir}" 2>&1 | tee "${save_dir}/eval_llm.log" >&2
  rc="${PIPESTATUS[0]}"
  set -e
  if [[ "${rc}" != "0" ]]; then
    echo "[FATAL] eval_llm failed for ${name} (rc=${rc})" >&2
    tail -200 "${save_dir}/eval_llm.log" >&2 || true
    exit "${rc}"
  fi

  echo "${pred_dir}"
}

REPO_DATA_ROOT="${REPO_ROOT}/data"
BASELINE_REPO_DEFAULT="${REPO_DATA_ROOT}/grounded3dllm_ckpts/step3/last-epoch.ckpt"
OURS_PACKED_REPO_DEFAULT="${REPO_DATA_ROOT}/SSR3DLLM_CKPT/SSR3DLLM.ckpt"

BASELINE_CKPT="$(_resolve_ckpt_path "${BASELINE_CKPT:-${SSR3DLLM_UNIFIED_CKPT:-}}" "${BASELINE_REPO_DEFAULT}" "BASELINE_CKPT")"
OURS_PACKED_CKPT="$(_resolve_ckpt_path "${CKPT:-${SSR3DLLM_PACKED_CKPT:-}}" "${OURS_PACKED_REPO_DEFAULT}" "CKPT")"
PROFILE="${PROFILE:-503}"

_need_file "${BASELINE_CKPT}"
_need_file "${OURS_PACKED_CKPT}"

_bridge_scannet200_root
_preflight_protocol_data_paths
_resolve_raw_scans_dir
_preflight_axis_align_files

out_root="${OUTPUT_ROOT}/ssr3dllm/appendix_cap_examples_$(_ts)"
mkdir -p "${out_root}"
echo "[cap_examples] out_root=${out_root}"

# eval_adapt cache (language-view ckpt)
tag="$(_ckpt_cache_tag "${OURS_PACKED_CKPT}")"
cache_dir="${out_root}/eval_adapt_cache"
mkdir -p "${cache_dir}"
base_name="$(basename "${OURS_PACKED_CKPT}")"
base_name="${base_name%.*}"

out_wrapper="${cache_dir}/${base_name}_${tag}_wrapper_${PROFILE}.pth"
out_listener="${cache_dir}/${base_name}_${tag}_listener_${PROFILE}.pth"
out_language="${cache_dir}/${base_name}_${tag}_language.ckpt"
out_report="${cache_dir}/${base_name}_${tag}_report_${PROFILE}.json"

python "${REPO_ROOT}/tools/eval_adapt_ssr3dllm_ckpt.py" \
  --checkpoint "${OURS_PACKED_CKPT}" \
  --profile "${PROFILE}" \
  --out-wrapper "${out_wrapper}" \
  --out-listener "${out_listener}" \
  --out-language "${out_language}" \
  | tee "${out_report}" >/dev/null

echo "[cap_examples] eval_adapt language_ckpt=${out_language}"

baseline_pred="$(_run_test_protocol "baseline_step3" "${BASELINE_CKPT}" "${out_root}" | tail -n 1)"
ours_pred="$(_run_test_protocol "ours_ssr3dllm_language_view" "${out_language}" "${out_root}" | tail -n 1)"

if [[ ! -d "${baseline_pred}" ]]; then
  echo "[FATAL] baseline_pred_dir is not a directory: ${baseline_pred}" >&2
  exit 2
fi
if [[ ! -d "${ours_pred}" ]]; then
  echo "[FATAL] ours_pred_dir is not a directory: ${ours_pred}" >&2
  exit 2
fi

echo "[cap_examples] baseline_pred_dir=${baseline_pred}"
echo "[cap_examples] ours_pred_dir=${ours_pred}"

rows_out="${out_root}/cap_examples_rows_seed1.txt"
scenes_out="${out_root}/cap_examples_scenes_seed1"

extract_args=()
if [[ -n "${SCANNET_SCANS_ROOT:-}" ]]; then
  extract_args+=(--export-scenes-dir "${scenes_out}")
  extract_args+=(--scene-search-root "${SCANNET_SCANS_ROOT}")
fi

python "${REPO_ROOT}/tools/extract_capability_examples.py" \
  --baseline-pred-dir "${baseline_pred}" \
  --ours-pred-dir "${ours_pred}" \
  --seed 1 \
  --k-per-task 1 \
  "${extract_args[@]}" \
  | tee "${rows_out}" >/dev/null

echo "[cap_examples] rows=${rows_out}"
if [[ -n "${SCANNET_SCANS_ROOT:-}" ]]; then
  echo "[cap_examples] scenes_dir=${scenes_out}"
else
  echo "[cap_examples] scenes_dir=(skipped; set SCANNET_SCANS_ROOT to export ply assets)"
fi
echo "[cap_examples] DONE out_root=${out_root}"
