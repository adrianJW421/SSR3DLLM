#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DATASET="${DATASET:-scanrefer}"
SPLITS="${SPLITS:-train,test}"
DINO_SOURCE_FEATURES="${DINO_SOURCE_FEATURES:-}"
DINO_SOURCE_ATTRIBUTES="${DINO_SOURCE_ATTRIBUTES:-}"
DINO_DTYPE="${DINO_DTYPE:-float16}"
DINO_NUM_SHARDS="${DINO_NUM_SHARDS:-1}"
DINO_SHARD_INDEX="${DINO_SHARD_INDEX:-0}"
DINO_MAX_SAMPLES="${DINO_MAX_SAMPLES:-0}"
DINO_OVERWRITE="${DINO_OVERWRITE:-0}"

if [[ -z "${DINO_SOURCE_FEATURES}" ]]; then
  echo "[build_dino_sidecar][FATAL] set DINO_SOURCE_FEATURES" >&2
  exit 2
fi
if [[ -z "${DINO_SOURCE_ATTRIBUTES}" ]]; then
  echo "[build_dino_sidecar][FATAL] set DINO_SOURCE_ATTRIBUTES" >&2
  exit 2
fi

case "${DATASET}" in
  scanrefer)
    QCOND_CACHE_ROOT="${SCANREFER_QCOND_CACHE_ROOT:-${SCANREFER_MASK3D_FEATS_TRAIN:-}}"
    DINO_OUTPUT_ROOT="${SCANREFER_DINO_SAMPLE_CACHE_ROOT:-${REPO_ROOT}/data/SCANREFER_DINO_SIDECAR_CACHE}"
    ;;
  multi3dref)
    QCOND_CACHE_ROOT="${MULTI3DREF_QCOND_CACHE_ROOT:-${MULTI3DREF_MASK3D_FEATS_TRAIN:-}}"
    DINO_OUTPUT_ROOT="${MULTI3DREF_DINO_SAMPLE_CACHE_ROOT:-${REPO_ROOT}/data/MULTI3DREF_DINO_SIDECAR_CACHE}"
    ;;
  nr3d)
    QCOND_CACHE_ROOT="${NR3D_QCOND_CACHE_ROOT:-${MASK3D_FEATS_TRAIN:-}}"
    DINO_OUTPUT_ROOT="${NR3D_DINO_SAMPLE_CACHE_ROOT:-${REPO_ROOT}/data/NR3D_DINO_SIDECAR_CACHE}"
    ;;
  *)
    echo "[build_dino_sidecar][FATAL] unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

_need_dir "${QCOND_CACHE_ROOT}"

IFS=':' read -r -a FEATURE_ARGS <<< "${DINO_SOURCE_FEATURES}"
IFS=':' read -r -a ATTRIBUTE_ARGS <<< "${DINO_SOURCE_ATTRIBUTES}"

COMMON_ARGS=()
for p in "${FEATURE_ARGS[@]}"; do
  _need_file "${p}"
  COMMON_ARGS+=(--source-features "${p}")
done
for p in "${ATTRIBUTE_ARGS[@]}"; do
  _need_file "${p}"
  COMMON_ARGS+=(--source-attributes "${p}")
done
if [[ "${DINO_OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

IFS=',' read -r -a SPLIT_ARGS <<< "${SPLITS// /,}"
for split in "${SPLIT_ARGS[@]}"; do
  split="${split}"
  [[ -n "${split}" ]] || continue
  cache_split_root="${QCOND_CACHE_ROOT}/${split}"
  output_split_root="${DINO_OUTPUT_ROOT}/${split}"
  _need_dir "${cache_split_root}/samples"
  mkdir -p "${output_split_root}"
  python tools/build_dino_sidecar_cache.py \
    --cache-root "${cache_split_root}" \
    --output-root "${output_split_root}" \
    --dtype "${DINO_DTYPE}" \
    --num-shards "${DINO_NUM_SHARDS}" \
    --shard-index "${DINO_SHARD_INDEX}" \
    --max-samples "${DINO_MAX_SAMPLES}" \
    "${COMMON_ARGS[@]}"
done

echo "[build_dino_sidecar] output=${DINO_OUTPUT_ROOT}"
