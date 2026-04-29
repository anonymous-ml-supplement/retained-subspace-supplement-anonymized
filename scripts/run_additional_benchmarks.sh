#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3}"
TOPKS_CSV="${TOPKS_CSV:-1,2,4}"
RANK="${RANK:-16}"
GLUE_ADDITIONAL_TASKS="${GLUE_ADDITIONAL_TASKS:-sst2 cola qqp}"
VISION_ADDITIONAL_TASKS="${VISION_ADDITIONAL_TASKS:-food101 eurosat}"
REPORT_METHODS="${REPORT_METHODS:-random,mag,grad,curv,v1,v2,v5,greedy_v5}"
PLOT_METHODS="${PLOT_METHODS:-mag,v1,v2,v5,greedy_v5}"

mkdir -p "${OUT_ROOT}/additional" "${LOG_ROOT}/additional"

for task in ${GLUE_ADDITIONAL_TASKS}; do
  mkdir -p "${OUT_ROOT}/additional/${task}"
  "${PYTHON_BIN}" -m retained_subspace.glue_experiment \
    --glue-task "${task}" \
    --seeds "${SEEDS_CSV}" \
    --lora-r "${RANK}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "${REPORT_METHODS}" \
    --plot-methods "${PLOT_METHODS}" \
    --enable-exact-subset-oracle 0 \
    --output-dir "${OUT_ROOT}/additional/${task}" \
    2>&1 | tee "${LOG_ROOT}/additional/${task}.log"
done

for task in ${VISION_ADDITIONAL_TASKS}; do
  mkdir -p "${OUT_ROOT}/additional/${task}"
  case "${task}" in
    eurosat|eurosat-rgb)
      MODEL_NAME="${EUROSAT_MODEL_NAME:-openai/clip-vit-large-patch14}"
      TARGET_MODULES="${EUROSAT_TARGET_MODULES:-q_proj,v_proj}"
      MAX_TRAIN="${EUROSAT_MAX_TRAIN_SAMPLES:-0}"
      MAX_VAL="${EUROSAT_MAX_VAL_SAMPLES:-0}"
      MAX_TEST="${EUROSAT_MAX_TEST_SAMPLES:-0}"
      ;;
    food101|food-101)
      MODEL_NAME="${FOOD101_MODEL_NAME:-google/vit-base-patch16-224}"
      TARGET_MODULES="${FOOD101_TARGET_MODULES:-query,value}"
      MAX_TRAIN="${FOOD101_MAX_TRAIN_SAMPLES:-0}"
      MAX_VAL="${FOOD101_MAX_VAL_SAMPLES:-0}"
      MAX_TEST="${FOOD101_MAX_TEST_SAMPLES:-0}"
      ;;
    *)
      MODEL_NAME="${VISION_MODEL_NAME:-google/vit-base-patch16-224}"
      TARGET_MODULES="${VISION_TARGET_MODULES:-query,value}"
      MAX_TRAIN="${VISION_MAX_TRAIN_SAMPLES:-0}"
      MAX_VAL="${VISION_MAX_VAL_SAMPLES:-0}"
      MAX_TEST="${VISION_MAX_TEST_SAMPLES:-0}"
      ;;
  esac
  "${PYTHON_BIN}" -m retained_subspace.vision_experiment \
    --dataset-name "${task}" \
    --model-name "${MODEL_NAME}" \
    --target-modules "${TARGET_MODULES}" \
    --seeds "${SEEDS_CSV}" \
    --lora-r "${RANK}" \
    --topk-values "${TOPKS_CSV}" \
    --report-methods "${REPORT_METHODS}" \
    --plot-methods "${PLOT_METHODS}" \
    --enable-exact-subset-oracle 0 \
    --max-train-samples "${MAX_TRAIN}" \
    --max-val-samples "${MAX_VAL}" \
    --max-test-samples "${MAX_TEST}" \
    --output-dir "${OUT_ROOT}/additional/${task}" \
    2>&1 | tee "${LOG_ROOT}/additional/${task}.log"
done
