#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${RUN_NAME:-ssr3dllm_readout_benchmark_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-scanrefer,multi3dref}"
SPEED_WARMUP="${SPEED_WARMUP:-20}"
SPEED_MAX_BATCHES="${SPEED_MAX_BATCHES:-500}"

VIGOR_SPEED_BENCH=1 \
VIGOR_SPEED_WARMUP="${SPEED_WARMUP}" \
VIGOR_SPEED_MAX_BATCHES="${SPEED_MAX_BATCHES}" \
VIGOR_SPEED_EARLY_STOP=1 \
VIGOR_SPEED_SYNC_CUDA="${VIGOR_SPEED_SYNC_CUDA:-1}" \
VIGOR_SKIP_ANALYZE=1 \
DATASETS="${DATASETS}" \
RUN_NAME="${RUN_NAME}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
N_WORKERS="${N_WORKERS:-0}" \
"${SCRIPT_DIR}/eval_ssr3dllm_scanrefer_multi3dref.sh"

source "${SCRIPT_DIR}/_common.sh"
LOGDIR_BASE="${OUTPUT_ROOT}/ssr3dllm/${RUN_NAME}"

python - "${LOGDIR_BASE}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
pat = re.compile(r"\[Vigor\]\[speed\].*measured_batches=(\d+).*measured_samples=(\d+).*ms_per_batch=([0-9.]+).*ms_per_sample=([0-9.]+)")
for log in sorted(root.glob("*/eval.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    m = pat.search(text)
    if not m:
        continue
    batches = int(m.group(1))
    samples = int(m.group(2))
    ms_batch = float(m.group(3))
    ms_sample = float(m.group(4))
    rows.append({
        "dataset": log.parent.name,
        "measured_batches": batches,
        "measured_samples": samples,
        "ms_per_batch": ms_batch,
        "latency_s_per_query": ms_sample / 1000.0,
        "queries_per_second": 1000.0 / ms_sample if ms_sample > 0 else 0.0,
        "log": str(log),
    })
out_json = root / "readout_benchmark_summary.json"
out_md = root / "readout_benchmark_summary.md"
out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
lines = ["| Dataset | Latency (s/query) | Throughput (query/s) | Samples |", "|---|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['dataset']} | {r['latency_s_per_query']:.4f} | {r['queries_per_second']:.3f} | {r['measured_samples']} |")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"))
print(f"[ssr3dllm_benchmark] summary: {out_json}")
PY
