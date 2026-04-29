# Retained-Subspace Selection Supplement

This supplement contains reorganized code for the experiments in **What to Keep: Budgeted Subset Selection for Trained LoRA Adapters**. The package is organized by experiment family rather than temporary development filenames, and it avoids personal absolute paths or placeholder folder names.

## Directory layout

```text
retained_subspace_selection_supplement/
  src/retained_subspace/
    glue_experiment.py                         # GLUE / text classification runner
    vision_experiment.py                       # CIFAR / vision classification runner
  llm_sanity/
    gsm8k_qwen_math7b_experiment.py            # GSM8K Qwen2.5-Math-7B retained-subspace run
    gsm8k_mistral_larger_pool_experiment.py    # GSM8K Mistral-7B larger-rank checks
    humanevalplus_qwen_coder7b_experiment.py   # HumanEval+ Qwen2.5-Coder-7B pass@1 run
    humanevalplus_qwen_coder7b_launcher.py     # Fixed-argument launcher for the HumanEval+ run
  scripts/
    quick_start.sh
    run_main_text_and_vision.sh
    run_oracle_diagnostics.sh
    run_appendix_audits.sh
    run_additional_benchmarks.sh
    run_llm_sanity.sh
    collect_tables.sh
    run_all_core.sh
    run_all.sh
  configs/
    main_text_and_vision.yaml
    oracle_diagnostics.yaml
    appendix_audits.yaml
    additional_benchmarks.yaml
    llm_sanity.yaml
  tools/
    collect_results.py
  data/ outputs/ logs/
```

## Installation

A minimal environment for the main text and vision experiments is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the optional LLM sanity scripts, install the additional dependencies:

```bash
pip install -r requirements-llm.txt
```

The scripts use HuggingFace datasets and model checkpoints. Set `HF_HOME` if the cache should live outside this folder.

## Quick smoke run

```bash
bash scripts/quick_start.sh
```

This runs a small one-seed MRPC job with capped sample counts and writes outputs under `outputs/quick_start/`.

## Main low-budget text and vision runs

```bash
bash scripts/run_main_text_and_vision.sh
bash scripts/collect_tables.sh
```

Default settings use seeds `0,1,2,3`, LoRA rank `16`, and retained budgets `k in {1,2,4}`. The task lists can be overridden without editing the scripts:

```bash
GLUE_TASKS="mnli qnli mrpc sst2 cola qqp" VISION_TASKS="cifar10 cifar100" \
  bash scripts/run_main_text_and_vision.sh
```

## Additional benchmark runs

The main script defaults to the compact paper-facing GLUE/CIFAR group. The supplement also includes an optional benchmark script for SST-2, CoLA, QQP, Food-101, and EuroSAT:

```bash
bash scripts/run_additional_benchmarks.sh
```

The default EuroSAT setting uses `openai/clip-vit-large-patch14` with LoRA targets `q_proj,v_proj`; Food-101 uses `google/vit-base-patch16-224` with LoRA targets `query,value`. These can be overridden from the shell, for example:

```bash
EUROSAT_MODEL_NAME="openai/clip-vit-base-patch32" \
EUROSAT_TARGET_MODULES="q_proj,v_proj" \
RUN_ADDITIONAL_BENCHMARKS=1 bash scripts/run_all_core.sh
```

## Exact oracle diagnostics

```bash
RUN_ORACLE=1 bash scripts/run_all_core.sh
```

or directly:

```bash
bash scripts/run_oracle_diagnostics.sh
```

The oracle script enables exhaustive small-`k` subset enumeration through the same entrypoints as the main experiments.

## Appendix audits

```bash
RUN_APPENDIX_AUDITS=1 bash scripts/run_all_core.sh
```

or directly:

```bash
bash scripts/run_appendix_audits.sh
```

These runs reuse the frozen train-once-then-select code path with broader selector reporting and multiple split seeds.

## LLM sanity checks

```bash
RUN_LLM_SANITY=1 bash scripts/run_all_core.sh
```

The GSM8K scripts are included directly. HumanEval+ is included as a complete runner and is off by default because it requires EvalPlus execution and a 7B code model. To run it:

```bash
RUN_LLM_SANITY=1 RUN_GSM8K_QWEN=0 RUN_GSM8K_LARGER_POOL=0 RUN_HUMANEVALPLUS=1 \
  bash scripts/run_llm_sanity.sh
```

For HumanEval+, the paper-facing metric is **HumanEval+ pass@1** from EvalPlus. The runner still records syntax-valid rate as a diagnostic, but the default HumanEval+ summary files overwrite `test_acc` / `test_acc_mean` with plus-suite pass@1 when `--run-evalplus 1` is enabled. Diagnostic syntax-valid files are saved separately with the suffix `_syntax_diagnostic.csv`.

## Outputs

Each task-specific run writes CSV files such as `summary_topk.csv`, `summary_low_budget.csv`, `summary_methods.csv`, and, when enabled, `oracle_comparison_all.csv`. HumanEval+ additionally writes `evalplus_summary_all.csv`, `all_topk_results_pass_at_1.csv`, `summary_topk_pass_at_1.csv`, and `summary_low_budget_pass_at_1.csv`.

Combined CSVs are written by:

```bash
bash scripts/collect_tables.sh
```

The combined tables are placed under `outputs/tables/`.

## Reproducibility notes

The package intentionally avoids user-specific absolute paths. Runtime folders are controlled through environment variables:

```bash
HF_HOME=/path/to/hf-cache OUT_ROOT=/path/to/outputs LOG_ROOT=/path/to/logs bash scripts/run_main_text_and_vision.sh
```

The experiment scripts record arguments, library versions, GPU model names, and timing information in their output folders. Identity-bearing system fields and personal paths are not recorded.


## Result CSVs

Anonymized table-aligned CSV files are provided in `results/`. These files exclude raw logs, local paths, account names, emails, and institution-specific identifiers.
