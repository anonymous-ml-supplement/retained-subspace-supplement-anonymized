#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3}"
TOPKS_CSV="${TOPKS_CSV:-1,2,4}"
RUN_GSM8K_QWEN="${RUN_GSM8K_QWEN:-1}"
RUN_GSM8K_LARGER_POOL="${RUN_GSM8K_LARGER_POOL:-1}"
RUN_HUMANEVALPLUS="${RUN_HUMANEVALPLUS:-0}"

mkdir -p "${OUT_ROOT}/llm_sanity" "${LOG_ROOT}/llm_sanity"

if [[ "${RUN_GSM8K_QWEN}" == "1" ]]; then
  "${PYTHON_BIN}" llm_sanity/gsm8k_qwen_math7b_experiment.py \
    --model-name "Qwen/Qwen2.5-Math-7B" \
    --seeds "${SEEDS_CSV}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "mag,v5" \
    --plot-methods "mag,v5" \
    --enable-greedy-selector 0 \
    --lora-r 16 \
    --lora-alpha 32 \
    --js-temperature "${JS_TEMPERATURE:-2.0}" \
    --output-dir "${OUT_ROOT}/llm_sanity/gsm8k_qwen_math_7b_r16" \
    2>&1 | tee "${LOG_ROOT}/llm_sanity/gsm8k_qwen_math_7b_r16.log"
fi

if [[ "${RUN_GSM8K_LARGER_POOL}" == "1" ]]; then
  IFS=',' read -r -a seed_array <<< "${SEEDS_CSV}"
  for rank in ${LARGER_POOL_RANKS:-32 64}; do
    for seed in "${seed_array[@]}"; do
      mkdir -p "${OUT_ROOT}/llm_sanity/gsm8k_mistral7b_r${rank}_seed${seed}"
      "${PYTHON_BIN}" llm_sanity/gsm8k_mistral_larger_pool_experiment.py \
        --model-name "mistralai/Mistral-7B-Instruct-v0.2" \
        --seed "${seed}" \
        --lora-r "${rank}" \
        --lora-alpha "$((rank * 2))" \
        --topk-values "${TOPKS_CSV}" \
        --output-dir "${OUT_ROOT}/llm_sanity/gsm8k_mistral7b_r${rank}_seed${seed}" \
        2>&1 | tee "${LOG_ROOT}/llm_sanity/gsm8k_mistral7b_r${rank}_seed${seed}.log"
    done
  done
fi

if [[ "${RUN_HUMANEVALPLUS}" == "1" ]]; then
  "${PYTHON_BIN}" llm_sanity/humanevalplus_qwen_coder7b_launcher.py \
    --script-path "${HUMANEVALPLUS_RUNNER:-llm_sanity/humanevalplus_qwen_coder7b_experiment.py}" \
    --output-dir "${OUT_ROOT}/llm_sanity/humanevalplus_qwen_coder_7b" \
    --seeds "${HUMANEVALPLUS_SEEDS:-0,1,2,3}" \
    --topk-values "${TOPKS_CSV}" \
    2>&1 | tee "${LOG_ROOT}/llm_sanity/humanevalplus_qwen_coder_7b.log"
fi
