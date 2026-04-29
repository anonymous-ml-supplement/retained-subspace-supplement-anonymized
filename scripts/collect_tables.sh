#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"${PYTHON_BIN}" tools/collect_results.py \
  --input-dir "${OUT_ROOT}" \
  --output-dir "${OUT_ROOT}/tables"
