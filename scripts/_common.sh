#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

_load_paths() {
  if [[ -f "${REPO_ROOT}/configs/paths.sh" ]]; then
    # User-provided machine-specific paths (recommended; not committed).
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/configs/paths.sh"
    return 0
  fi
  if [[ -f "${REPO_ROOT}/configs/paths.example.sh" ]]; then
    echo "[WARN] configs/paths.sh not found; using configs/paths.example.sh placeholders." >&2
    echo "       Copy configs/paths.example.sh -> configs/paths.sh and edit paths for your machine." >&2
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/configs/paths.example.sh"
    return 0
  fi
  echo "[FATAL] Missing configs/paths.sh and configs/paths.example.sh" >&2
  return 2
}

_need_file() { [[ -f "$1" ]] || { echo "[FATAL] file not found: $1" >&2; return 2; }; }
_need_dir() { [[ -d "$1" ]] || { echo "[FATAL] dir not found: $1" >&2; return 2; }; }

_derive_vigor_test_csv() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import re
import sys

p = sys.argv[1]
if "train" in p:
    t = re.sub(r'_\d+\.\d+', '', p.replace("train", "test"))
elif "test" in p:
    t = p
else:
    t = ""
print(t)
PY
}

_check_vigor_train_test_csv_pair() {
  local train_csv="$1"
  local tag="${2:-referit3d}"

  _need_file "${train_csv}" || return 2

  local test_csv
  test_csv="$(_derive_vigor_test_csv "${train_csv}")"
  if [[ -z "${test_csv}" ]]; then
    echo "[FATAL] ${tag}: cannot derive matching test csv from: ${train_csv}" >&2
    return 2
  fi
  _need_file "${test_csv}" || {
    echo "[FATAL] ${tag}: expected matching test csv missing: ${test_csv}" >&2
    echo "        keep train/test csv together (e.g., *_train_*.csv and *_test_*.csv)." >&2
    return 2
  }

  # Fast sanity check: ensure test csv has rows, and (if split file exists) test scan_ids overlap val split.
  python - "${train_csv}" "${test_csv}" "${REPO_ROOT}" "${tag}" <<'PY'
import csv
import sys
from pathlib import Path

train_csv = Path(sys.argv[1])
test_csv = Path(sys.argv[2])
repo_root = Path(sys.argv[3])
tag = sys.argv[4]

def read_scan_ids(path: Path):
    ids = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "scan_id" not in (r.fieldnames or []):
            raise RuntimeError(f"{path} has no 'scan_id' column")
        for row in r:
            sid = str(row.get("scan_id", "")).strip()
            if sid:
                ids.append(sid)
    return ids

train_ids = read_scan_ids(train_csv)
test_ids = read_scan_ids(test_csv)
if len(test_ids) == 0:
    raise RuntimeError(f"{tag}: test csv has 0 rows: {test_csv}")

val_file = repo_root / "third_party" / "Vigor" / "referit3d" / "data" / "scannet" / "splits" / "official" / "v2" / "scannetv2_val.txt"
if val_file.exists():
    val_scans = {ln.strip() for ln in val_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    overlap = len(set(test_ids) & val_scans)
    if overlap == 0:
        raise RuntimeError(
            f"{tag}: no test scan_id overlaps official val split. "
            f"csv may be train-only or wrong protocol: {test_csv}"
        )
print(f"[check_paths] OK: {tag} train/test csv pair ({len(train_ids)} / {len(test_ids)} rows)")
PY
}

_load_paths

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
mkdir -p "${OUTPUT_ROOT}"

# Best-effort default for raw ScanNet scans (axisAlignment txt files).
# Allow user override via `SSR3DLLM_RAWSCANNET_SCANS_DIR`; otherwise fall back to
# `SCANNET_SCANS_ROOT` or the release-local scaffold path.
if [[ -z "${SSR3DLLM_RAWSCANNET_SCANS_DIR:-}" ]]; then
  if [[ -n "${SCANNET_SCANS_ROOT:-}" ]]; then
    export SSR3DLLM_RAWSCANNET_SCANS_DIR="${SCANNET_SCANS_ROOT}"
  else
    export SSR3DLLM_RAWSCANNET_SCANS_DIR="${REPO_ROOT}/data/rawscannet/scans"
  fi
fi
