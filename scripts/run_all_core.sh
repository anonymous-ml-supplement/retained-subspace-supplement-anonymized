#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

bash scripts/run_main_text_and_vision.sh

if [[ "${RUN_ORACLE:-0}" == "1" ]]; then
  bash scripts/run_oracle_diagnostics.sh
fi

if [[ "${RUN_APPENDIX_AUDITS:-0}" == "1" ]]; then
  bash scripts/run_appendix_audits.sh
fi

if [[ "${RUN_ADDITIONAL_BENCHMARKS:-0}" == "1" ]]; then
  bash scripts/run_additional_benchmarks.sh
fi

if [[ "${RUN_LLM_SANITY:-0}" == "1" ]]; then
  bash scripts/run_llm_sanity.sh
fi

bash scripts/collect_tables.sh
