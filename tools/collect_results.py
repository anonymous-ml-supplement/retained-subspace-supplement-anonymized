#!/usr/bin/env python3
"""Collect per-run CSV summaries into combined CSV files.

The experiment entrypoints write CSV summaries under task-specific output
folders. This utility recursively collects those summaries and writes combined
files that can be inspected directly or imported into a table-generation
notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SUMMARY_FILES = {
    "summary_topk.csv": "all_summary_topk.csv",
    "summary_low_budget.csv": "all_summary_low_budget.csv",
    "summary_topk_pass_at_1.csv": "all_humanevalplus_summary_topk_pass_at_1.csv",
    "summary_low_budget_pass_at_1.csv": "all_humanevalplus_summary_low_budget_pass_at_1.csv",
    "evalplus_summary_all.csv": "all_humanevalplus_evalplus_summary.csv",
    "summary_methods.csv": "all_summary_methods.csv",
    "summary_full_model.csv": "all_summary_full_model.csv",
    "win_rate_vs_mag.csv": "all_win_rate_vs_mag.csv",
    "oracle_comparison_all.csv": "all_oracle_comparison.csv",
    "seed_resource_usage.csv": "all_seed_resource_usage.csv",
    "summary_results.csv": "all_gsm8k_larger_pool_summary.csv",
    "subset_results.csv": "all_gsm8k_larger_pool_subsets.csv",
}


def infer_run_metadata(path: Path, input_dir: Path) -> dict[str, str]:
    rel_parent = path.parent.relative_to(input_dir)
    parts = rel_parent.parts
    return {
        "source_file": str(path.relative_to(input_dir)),
        "run_group": parts[0] if parts else "root",
        "task_or_run": parts[-1] if parts else path.parent.name,
        "run_path": str(rel_parent),
    }


def collect_one(input_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.rglob(filename)):
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        meta = infer_run_metadata(path, input_dir)
        for key, value in reversed(list(meta.items())):
            df.insert(0, key, value)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def write_if_nonempty(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    if df.empty:
        print(f"no rows for {filename}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    df.to_csv(target, index=False)
    print(f"wrote {target} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    for source_name, out_name in SUMMARY_FILES.items():
        write_if_nonempty(collect_one(input_dir, source_name), output_dir, out_name)


if __name__ == "__main__":
    main()
