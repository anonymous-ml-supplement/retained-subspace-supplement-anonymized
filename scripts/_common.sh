#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/model_cache/huggingface}"
export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${HF_HOME}" "${OUT_ROOT}" "${LOG_ROOT}"
