#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3}"
TOPKS_CSV="${TOPKS_CSV:-1,2,4}"
RANK="${RANK:-16}"
AUDIT_TASKS="${AUDIT_TASKS:-mrpc cola cifar10 cifar100}"
SPLIT_SEEDS="${SPLIT_SEEDS:-3407 2025 777}"
REPORT_METHODS="${REPORT_METHODS:-random,mag,grad,curv,v1,v2,v5,greedy_v5}"
PLOT_METHODS="${PLOT_METHODS:-mag,v1,v2,v5,greedy_v5}"

mkdir -p "${OUT_ROOT}/audits" "${LOG_ROOT}/audits"

for split_seed in ${SPLIT_SEEDS}; do
  for task in ${AUDIT_TASKS}; do
    mkdir -p "${OUT_ROOT}/audits/${task}_split${split_seed}"
    if [[ "${task}" == cifar* ]]; then
      "${PYTHON_BIN}" -m retained_subspace.vision_experiment \
        --dataset-name "${task}" \
        --data-split-seed "${split_seed}" \
        --seeds "${SEEDS_CSV}" \
        --lora-r "${RANK}" \
        --topk-values "${TOPKS_CSV}" \
        --report-methods "${REPORT_METHODS}" \
        --plot-methods "${PLOT_METHODS}" \
        --output-dir "${OUT_ROOT}/audits/${task}_split${split_seed}" \
        2>&1 | tee "${LOG_ROOT}/audits/${task}_split${split_seed}.log"
    else
      "${PYTHON_BIN}" -m retained_subspace.glue_experiment \
        --glue-task "${task}" \
        --dataset-split-seed "${split_seed}" \
        --seeds "${SEEDS_CSV}" \
        --lora-r "${RANK}" \
        --topk-values "${TOPKS_CSV}" \
        --report-methods "${REPORT_METHODS}" \
        --plot-methods "${PLOT_METHODS}" \
        --output-dir "${OUT_ROOT}/audits/${task}_split${split_seed}" \
        2>&1 | tee "${LOG_ROOT}/audits/${task}_split${split_seed}.log"
    fi
  done
done
