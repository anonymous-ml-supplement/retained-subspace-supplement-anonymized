#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3}"
TOPKS_CSV="${TOPKS_CSV:-1,2,4}"
RANK="${RANK:-16}"
GLUE_TASKS="${GLUE_TASKS:-mnli qnli mrpc}"
VISION_TASKS="${VISION_TASKS:-cifar10 cifar100}"
REPORT_METHODS="${REPORT_METHODS:-random,mag,grad,curv,v1,v2,v5,greedy_v5}"
PLOT_METHODS="${PLOT_METHODS:-mag,v1,v2,v5,greedy_v5}"

mkdir -p "${OUT_ROOT}/main" "${LOG_ROOT}/main"

for task in ${GLUE_TASKS}; do
  mkdir -p "${OUT_ROOT}/main/${task}"
  "${PYTHON_BIN}" -m retained_subspace.glue_experiment \
    --glue-task "${task}" \
    --seeds "${SEEDS_CSV}" \
    --lora-r "${RANK}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "${REPORT_METHODS}" \
    --plot-methods "${PLOT_METHODS}" \
    --enable-exact-subset-oracle 0 \
    --output-dir "${OUT_ROOT}/main/${task}" \
    2>&1 | tee "${LOG_ROOT}/main/${task}.log"
done

for task in ${VISION_TASKS}; do
  mkdir -p "${OUT_ROOT}/main/${task}"
  "${PYTHON_BIN}" -m retained_subspace.vision_experiment \
    --dataset-name "${task}" \
    --seeds "${SEEDS_CSV}" \
    --lora-r "${RANK}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "${REPORT_METHODS}" \
    --plot-methods "${PLOT_METHODS}" \
    --enable-exact-subset-oracle 0 \
    --output-dir "${OUT_ROOT}/main/${task}" \
    2>&1 | tee "${LOG_ROOT}/main/${task}.log"
done
