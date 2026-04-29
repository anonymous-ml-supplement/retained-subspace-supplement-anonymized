#!/usr/bin/env python3
"""
LLM-scale R=32 LoRA retained-subspace sanity run.

Goal
----
This is a deliberately small sanity datapoint for the NeurIPS retained-subspace
selection paper.  It trains a rank-32 LoRA adapter on a small GSM8K subset,
then compares:

  1. Magnitude: top-k LoRA directions by ||B_j|| * ||A_j||.
  2. URS-loss: top-k LoRA directions by singleton retained utility measured
     on a held-out selection split, i.e. empty-adapter loss minus singleton loss.

This is not meant to replace the main GLUE/vision evidence.  It is meant to
answer whether the retained-subspace selection signal still exists at a larger
LoRA rank in an LLM-style setting.

Expected output files
---------------------
direction_scores.csv
subset_results.csv
summary_results.csv
run_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_answer_number(text: str) -> str:
    """Extract a GSM8K-style final numeric answer."""
    text = text.replace(",", "")
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    # Prefer answers after a final-answer cue if present.
    cues = ["final answer is", "answer is", "therefore", "so"]
    lower = text.lower()
    tail = text
    for cue in cues:
        pos = lower.rfind(cue)
        if pos >= 0:
            tail = text[pos:]
            break
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
    if not nums:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return ""
    ans = nums[-1]
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


def make_train_prompt(question: str, answer: str) -> str:
    return f"Question: {question}\nLet's think step by step.\nAnswer: {answer}"


def make_eval_prompt(question: str) -> str:
    return f"Question: {question}\nLet's think step by step.\nAnswer:"


def prepare_gsm8k(args) -> Tuple[list, list, list]:
    ds = load_dataset(args.dataset_name, args.dataset_config)
    train = list(ds["train"])
    test = list(ds["test"])

    rng = random.Random(args.data_seed)
    rng.shuffle(train)
    rng.shuffle(test)

    n_train = min(args.max_train_samples, len(train)) if args.max_train_samples > 0 else len(train)
    n_select = min(args.max_selection_samples, max(1, len(train) - n_train))
    train_examples = train[:n_train]
    selection_examples = train[n_train:n_train + n_select]
    if len(selection_examples) < n_select:
        selection_examples = train[-n_select:]

    n_test = min(args.max_test_samples, len(test)) if args.max_test_samples > 0 else len(test)
    test_examples = test[:n_test]
    return train_examples, selection_examples, test_examples


def tokenize_sft_example(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    prompt = make_eval_prompt(question) + " "
    full = prompt + answer
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
    if tokenizer.eos_token_id is not None and len(full_ids) < max_length:
        full_ids = full_ids + [tokenizer.eos_token_id]
    labels = full_ids.copy()
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[Dict], tokenizer, max_length: int):
        self.rows = [
            tokenize_sft_example(tokenizer, ex["question"], ex["answer"], max_length)
            for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.rows[idx]


@dataclass
class CausalCollator:
    tokenizer: object

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collect_lora_modules(model) -> List[Tuple[str, object]]:
    refs = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if "default" in module.lora_A and "default" in module.lora_B:
                refs.append((name, module))
    if not refs:
        raise RuntimeError("No PEFT LoRA modules found. Check target modules and PEFT version.")
    return refs


def infer_lora_rank(lora_refs: Sequence[Tuple[str, object]]) -> int:
    _, module = lora_refs[0]
    return int(module.lora_A["default"].weight.shape[0])


@contextmanager
def temporary_lora_rank_mask(lora_refs: Sequence[Tuple[str, object]], keep_indices: Sequence[int], rank: int):
    keep = set(int(i) for i in keep_indices)
    mask_a = torch.zeros(rank, dtype=torch.float32)
    mask_b = torch.zeros(rank, dtype=torch.float32)
    for i in keep:
        if 0 <= i < rank:
            mask_a[i] = 1.0
            mask_b[i] = 1.0

    backups = []
    with torch.no_grad():
        for _, module in lora_refs:
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            backups.append((A, B, A.detach().clone(), B.detach().clone()))
            ma = mask_a.to(device=A.device, dtype=A.dtype).view(-1, 1)
            mb = mask_b.to(device=B.device, dtype=B.dtype).view(1, -1)
            A.mul_(ma)
            B.mul_(mb)
    try:
        yield
    finally:
        with torch.no_grad():
            for A, B, A0, B0 in backups:
                A.copy_(A0)
                B.copy_(B0)


def lora_magnitude_scores(lora_refs: Sequence[Tuple[str, object]], rank: int) -> np.ndarray:
    scores = np.zeros(rank, dtype=np.float64)
    with torch.no_grad():
        for _, module in lora_refs:
            A = module.lora_A["default"].weight.detach().float().cpu().numpy()  # [r, in]
            B = module.lora_B["default"].weight.detach().float().cpu().numpy()  # [out, r]
            for j in range(rank):
                scores[j] += float(np.linalg.norm(A[j, :]) * np.linalg.norm(B[:, j]))
    return scores


def evaluate_loss(model, dataset: SFTDataset, collator: CausalCollator, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    losses, counts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n = int((batch["labels"] != -100).sum().item())
            losses.append(float(out.loss.detach().cpu()) * max(n, 1))
            counts.append(max(n, 1))
    return float(sum(losses) / max(sum(counts), 1))


def topk(scores: np.ndarray, k: int) -> List[int]:
    return [int(i) for i in np.argsort(-np.asarray(scores))[:k]]


def selected_to_str(indices: Sequence[int]) -> str:
    return ",".join(str(int(i)) for i in indices)


def generate_answer(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


def evaluate_gsm8k_exact(
    model,
    tokenizer,
    examples: Sequence[Dict],
    device: torch.device,
    max_new_tokens: int,
    max_examples: int,
) -> Dict[str, float]:
    model.eval()
    rows = []
    correct = 0
    total = min(len(examples), max_examples) if max_examples > 0 else len(examples)
    for ex in examples[:total]:
        prompt = make_eval_prompt(ex["question"])
        pred_text = generate_answer(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        pred = normalize_answer_number(pred_text)
        gold = normalize_answer_number(ex["answer"])
        ok = int(pred == gold and pred != "")
        correct += ok
        rows.append({"question": ex["question"], "gold": gold, "pred": pred, "correct": ok, "generation": pred_text})
    return {"exact_acc": float(correct / max(total, 1)), "n": int(total), "rows": rows}


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--dataset-name", type=str, default="gsm8k")
    ap.add_argument("--dataset-config", type=str, default="main")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=3407)
    ap.add_argument("--max-train-samples", type=int, default=512)
    ap.add_argument("--max-selection-samples", type=int, default=64)
    ap.add_argument("--max-test-samples", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    ap.add_argument("--load-4bit", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--selection-batch-size", type=int, default=1)
    ap.add_argument("--topk-values", type=str, default="1,2,4,8,16")
    ap.add_argument("--eval-generations", type=int, default=128)
    args = ap.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples, selection_examples, test_examples = prepare_gsm8k(args)
    model, tokenizer = load_model_and_tokenizer(args)
    collator = CausalCollator(tokenizer)

    train_dataset = SFTDataset(train_examples, tokenizer, args.max_length)
    selection_dataset = SFTDataset(selection_examples, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=bool(torch.cuda.is_available()),
        fp16=False,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if args.load_4bit else "adamw_torch",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    lora_refs = collect_lora_modules(model)
    rank = infer_lora_rank(lora_refs)
    if rank != args.lora_r:
        print(f"[warn] inferred LoRA rank {rank} differs from args.lora_r {args.lora_r}")

    # Scores
    mag = lora_magnitude_scores(lora_refs, rank)
    all_ranks = list(range(rank))
    empty_indices: List[int] = []
    with temporary_lora_rank_mask(lora_refs, empty_indices, rank):
        empty_loss = evaluate_loss(model, selection_dataset, collator, args.selection_batch_size, device)

    singleton_losses = []
    singleton_utility = []
    for j in range(rank):
        with temporary_lora_rank_mask(lora_refs, [j], rank):
            loss_j = evaluate_loss(model, selection_dataset, collator, args.selection_batch_size, device)
        singleton_losses.append(loss_j)
        singleton_utility.append(empty_loss - loss_j)

    direction_df = pd.DataFrame({
        "direction": list(range(rank)),
        "magnitude_score": mag,
        "singleton_selection_loss": singleton_losses,
        "urs_loss_score": singleton_utility,
    })
    direction_df.to_csv(output_dir / "direction_scores.csv", index=False)

    # Subset evaluation
    topk_values = [int(x.strip()) for x in args.topk_values.split(",") if x.strip()]
    result_rows = []

    with temporary_lora_rank_mask(lora_refs, all_ranks, rank):
        full_loss = evaluate_loss(model, selection_dataset, collator, args.selection_batch_size, device)
        full_eval = evaluate_gsm8k_exact(model, tokenizer, test_examples, device, args.max_new_tokens, args.eval_generations)
    result_rows.append({
        "method": "full_lora",
        "topk": rank,
        "selected_indices": selected_to_str(all_ranks),
        "selection_loss": full_loss,
        "test_exact_acc": full_eval["exact_acc"],
        "eval_n": full_eval["n"],
    })
    pd.DataFrame(full_eval["rows"]).to_csv(output_dir / "generations_full_lora.csv", index=False)

    for method, scores in [("mag", mag), ("urs_loss", np.asarray(singleton_utility))]:
        for k in topk_values:
            if k > rank:
                continue
            selected = topk(scores, k)
            with temporary_lora_rank_mask(lora_refs, selected, rank):
                sel_loss = evaluate_loss(model, selection_dataset, collator, args.selection_batch_size, device)
                gen_eval = evaluate_gsm8k_exact(model, tokenizer, test_examples, device, args.max_new_tokens, args.eval_generations)
            result_rows.append({
                "method": method,
                "topk": k,
                "selected_indices": selected_to_str(selected),
                "selection_loss": sel_loss,
                "test_exact_acc": gen_eval["exact_acc"],
                "eval_n": gen_eval["n"],
            })
            pd.DataFrame(gen_eval["rows"]).to_csv(output_dir / f"generations_{method}_k{k}.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "subset_results.csv", index=False)

    # Compact summary against magnitude.
    summary_rows = []
    for k in topk_values:
        rows_k = results[results["topk"] == k]
        if rows_k.empty:
            continue
        mag_row = rows_k[rows_k["method"] == "mag"]
        urs_row = rows_k[rows_k["method"] == "urs_loss"]
        if not mag_row.empty and not urs_row.empty:
            mag_acc = float(mag_row.iloc[0]["test_exact_acc"])
            urs_acc = float(urs_row.iloc[0]["test_exact_acc"])
            summary_rows.append({
                "topk": k,
                "mag_test_exact_acc": mag_acc,
                "urs_loss_test_exact_acc": urs_acc,
                "urs_minus_mag_acc_pp": (urs_acc - mag_acc) * 100.0,
                "mag_selection_loss": float(mag_row.iloc[0]["selection_loss"]),
                "urs_loss_selection_loss": float(urs_row.iloc[0]["selection_loss"]),
                "urs_minus_mag_selection_loss": float(urs_row.iloc[0]["selection_loss"] - mag_row.iloc[0]["selection_loss"]),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_results.csv", index=False)

    meta = {
        "train_seconds": train_seconds,
        "rank": rank,
        "empty_selection_loss": empty_loss,
        "num_lora_modules": len(lora_refs),
        "device": str(device),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done. Key files:")
    print(output_dir / "direction_scores.csv")
    print(output_dir / "subset_results.csv")
    print(output_dir / "summary_results.csv")


if __name__ == "__main__":
    main()
