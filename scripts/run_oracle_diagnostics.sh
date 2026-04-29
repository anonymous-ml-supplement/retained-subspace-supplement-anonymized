#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3}"
TOPKS_CSV="${TOPKS_CSV:-1,2,4}"
RANK="${RANK:-16}"
GLUE_TASKS="${GLUE_TASKS:-mnli qnli mrpc}"
VISION_TASKS="${VISION_TASKS:-cifar100}"
ORACLE_MAX_K="${ORACLE_MAX_K:-4}"
ORACLE_MAX_SEEDS="${ORACLE_MAX_SEEDS:-4}"

mkdir -p "${OUT_ROOT}/oracle" "${LOG_ROOT}/oracle"

for task in ${GLUE_TASKS}; do
  mkdir -p "${OUT_ROOT}/oracle/${task}"
  "${PYTHON_BIN}" -m retained_subspace.glue_experiment \
    --glue-task "${task}" \
    --seeds "${SEEDS_CSV}" \
    --lora-r "${RANK}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "mag,v5" \
    --plot-methods "mag,v5" \
    --enable-greedy-selector 0 \
    --enable-exact-subset-oracle 1 \
    --oracle-max-k "${ORACLE_MAX_K}" \
    --oracle-max-seeds "${ORACLE_MAX_SEEDS}" \
    --output-dir "${OUT_ROOT}/oracle/${task}" \
    2>&1 | tee "${LOG_ROOT}/oracle/${task}.log"
done

for task in ${VISION_TASKS}; do
  mkdir -p "${OUT_ROOT}/oracle/${task}"
  "${PYTHON_BIN}" -m retained_subspace.vision_experiment \
    --dataset-name "${task}" \
    --seeds "${SEEDS_CSV}" \
    --lora-r "${RANK}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "mag,v5" \
    --plot-methods "mag,v5" \
    --enable-greedy-selector 0 \
    --enable-exact-subset-oracle 1 \
    --oracle-max-k "${ORACLE_MAX_K}" \
    --oracle-max-seeds "${ORACLE_MAX_SEEDS}" \
    --output-dir "${OUT_ROOT}/oracle/${task}" \
    2>&1 | tee "${LOG_ROOT}/oracle/${task}.log"
done
