#!/usr/bin/env python3
"""
Launch the Qwen2.5-Coder-7B HumanEval+ pass@1 sanity experiment.

The launcher forwards the fixed paper-facing 7B HumanEval+ arguments to
humanevalplus_qwen_coder7b_experiment.py. The reported metric is EvalPlus
HumanEval+ pass@1; syntax-valid rate is kept only as a diagnostic output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-path",
        type=Path,
        default=Path("llm_sanity/humanevalplus_qwen_coder7b_experiment.py"),
        help="Path to the reorganized HumanEval+ pass@1 runner.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_humanevalplus_qwen_coder_7b"),
        help="Clean output directory for 7B seeds 0--3.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3",
        help="Comma-separated seeds to run. Default: 0,1,2,3.",
    )
    parser.add_argument(
        "--topk-values",
        type=str,
        default="1,2,4",
        help="Comma-separated retained budgets. Default: 1,2,4.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory. If omitted, runner default is used.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--include-greedy",
        action="store_true",
        help="Also evaluate greedy_v5. Not needed for the main mag vs v5 table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = args.script_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser()

    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find HumanEval+ runner: {script_path}")

    methods = "mag,v5,greedy_v5" if args.include_greedy else "mag,v5"
    enable_greedy = "1" if args.include_greedy else "0"

    # Match the fixed 7B HumanEval+ benchmark setting used for the paper-facing pass@1 check.
    cmd = [
        args.python_bin,
        "-u",
        str(script_path),
        "--model-name", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--target-modules", "q_proj,v_proj",
        "--seeds", args.seeds,
        "--topk-values", args.topk_values,
        "--report-methods", methods,
        "--plot-methods", methods,
        "--enable-greedy-selector", enable_greedy,
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--lora-dropout", "0.05",
        "--epochs", "1",
        "--early-stop-patience", "1",
        "--batch-size", "2",
        "--eval-batch-size", "2",
        "--grad-accum-steps", "8",
        "--learning-rate", "2e-4",
        "--weight-decay", "0.0001",
        "--warmup-ratio", "0.06",
        "--max-length", "768",
        "--max-prompt-length", "512",
        "--max-new-tokens", "192",
        "--max-train-samples", "1024",
        "--max-val-samples", "128",
        "--max-test-samples", "164",
        "--dataset-split-seed", "3407",
        "--val-ratio", "0.05",
        "--lambda-v1", "0.8",
        "--lambda-v2", "0.4",
        "--alpha-v4", "0.6",
        "--beta-v4", "0.8",
        "--gamma-v4", "0.2",
        "--alpha-v5", "1.2",
        "--beta-v5", "0.9",
        "--gamma-v5", "0.3",
        "--js-temperature", "1.0",
        "--run-evalplus", "1",
        "--evalplus-timeout", "1800",
        "--save-predictions", "1",
        "--output-dir", str(output_dir),
    ]

    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])

    print("===== HumanEval+ 7B fixed-seed completion run =====", flush=True)
    print("Fixed setting:", flush=True)
    print("  model           = Qwen/Qwen2.5-Coder-7B-Instruct", flush=True)
    print(f"  seeds           = {args.seeds}", flush=True)
    print(f"  k values        = {args.topk_values}", flush=True)
    print(f"  methods         = {methods}", flush=True)
    print("  lr              = 2e-4", flush=True)
    print("  batch_size      = 2", flush=True)
    print("  warmup_ratio    = 0.06", flush=True)
    print("  val_ratio       = 0.05", flush=True)
    print("  max_new_tokens  = 192", flush=True)
    print("  max_test        = 164", flush=True)
    print("  EvalPlus pass@1 = ON", flush=True)
    print("  output_dir      =", output_dir, flush=True)
    print("Command:", " ".join(cmd), flush=True)

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, check=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
