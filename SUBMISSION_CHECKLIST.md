# Supplement submission checklist

- No personal absolute paths are hard-coded in scripts or documentation.
- Resource logs omit identity-bearing system fields and personal paths.
- HumanEval+ reports EvalPlus plus-suite pass@1 as the paper-facing metric.
- Syntax-valid rate for HumanEval+ is saved only as a diagnostic file.
- GLUE support includes MNLI, QNLI, MRPC, SST-2, CoLA, QQP, and RTE.
- Vision support includes CIFAR-10, CIFAR-100, Food-101, and EuroSAT aliases.
- Optional LLM sanity scripts are disabled by default in the aggregate runner.
