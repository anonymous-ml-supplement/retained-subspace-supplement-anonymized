# Anonymous result tables

This folder contains anonymized CSV files that support the numeric values reported in the paper tables.

## Included files

- `alignment_report_checked_cells.csv`: long-format audit table for all automatically checked numeric cells. All rows should have `status = MATCH`.
- `table_coverage_summary.csv`: table-level coverage summary for the checked tables.
- `main_low_budget_values.csv`: checked values for the main low-budget NLP and vision table.
- `additional_low_budget_values.csv`: checked values for additional NLP/vision results.
- `llm_generative_values.csv`: checked values for GSM8K and HumanEval+ generative results. HumanEval+ is reported as EvalPlus pass@1.
- `per_seed_main_nlp.csv` and `per_seed_main_vision.csv`: checked per-seed values for the main NLP and vision results.
- `hans_shift_values.csv` / `hans_shift_summary.csv`: checked HANS-shift diagnostic values.
- `gate_active_resampling_checked.csv` / `gate_active_resampling_summary.csv`: checked gate-active resampling diagnostics.
- `humanevalplus_qwen_coder7b_pass1_summary.csv`: pass@1 summary used for the HumanEval+ rows.

## Naming note

The paper-facing name of the proposed selector is **URS**. In the released code, some internal method names retain the earlier development label `v5` for compatibility with existing scripts, configs, and result-collection utilities. Throughout this supplement, `v5` denotes **URS**, and `greedy_v5` denotes **Greedy-URS**. The baseline `mag` denotes the magnitude selector.

## Anonymity notes

The released CSVs intentionally exclude raw local paths, user names, institutional names, emails, issue-tracker notes, source inventories, and raw log/manifests. Internal audit sheets such as `Anonymity_Flags.csv`, `Source_Inventory.csv`, `Source_Values.csv`, and `Issues.csv` are not included in this public supplement.

The code uses public dataset/model identifiers where needed for reproducibility. No private checkpoints, tokens, cache paths, generated logs, or machine-specific paths are included.