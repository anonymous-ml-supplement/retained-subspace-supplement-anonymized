#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# One-seed capped MRPC smoke run for verifying the environment and output files.
SEEDS_CSV="${SEEDS_CSV:-0}"
TOPKS_CSV="${TOPKS_CSV:-1,2}"
TASK="${TASK:-mrpc}"
RANK="${RANK:-16}"

mkdir -p "${OUT_ROOT}/quick_start/${TASK}" "${LOG_ROOT}/quick_start"

"${PYTHON_BIN}" -m retained_subspace.glue_experiment \
  --glue-task "${TASK}" \
  --seeds "${SEEDS_CSV}" \
  --lora-r "${RANK}" \
  --topk-values "${TOPKS_CSV}" \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --max-test-samples 128 \
  --epochs 1 \
  --early-stop-patience 1 \
  --report-methods "mag,v5" \
  --plot-methods "mag,v5" \
  --enable-greedy-selector 0 \
  --output-dir "${OUT_ROOT}/quick_start/${TASK}" \
  2>&1 | tee "${LOG_ROOT}/quick_start/${TASK}.log"

"${PYTHON_BIN}" tools/collect_results.py \
  --input-dir "${OUT_ROOT}/quick_start" \
  --output-dir "${OUT_ROOT}/quick_start/tables"
