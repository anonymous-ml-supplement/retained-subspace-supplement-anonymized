#!/usr/bin/env python3
"""
HumanEval+ retained-subspace validation for Qwen2.5-Coder-7B.

This runner is intentionally PACF-free. It tests the v5-vs-magnitude retained-
subspace story on a causal-LLM code-generation benchmark rather than on a GLUE
proxy. The paper-facing HumanEval+ metric is EvalPlus pass@1. Syntax-valid rate
is retained only as a lightweight diagnostic and is not the main reported score.

Pipeline:
- train a LoRA adapter on CodeFeedback instruction/code pairs
- score LoRA directions on a held-out CodeFeedback validation split
- build retained-direction subsets for mag / v5 / greedy_v5 / etc.
- generate one completion per HumanEval task for each retained subset
- run EvalPlus and report HumanEval+ pass@1 from the plus test suite
- dump per-problem generations and EvalPlus-compatible samples for auditing
"""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(output_dir.name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_status(output_dir: Path, payload: Dict) -> None:
    with open(output_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rho = spearmanr(x, y).statistic
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


def js_from_example_scores(full_scores: np.ndarray, sub_scores: np.ndarray, temperature: float = 1.0, eps: float = 1e-8) -> float:
    f = np.asarray(full_scores, dtype=np.float64) / max(temperature, eps)
    s = np.asarray(sub_scores, dtype=np.float64) / max(temperature, eps)
    f = f - f.max()
    s = s - s.max()
    p = np.exp(np.clip(f, -30.0, 30.0))
    q = np.exp(np.clip(s, -30.0, 30.0))
    p = p / np.clip(p.sum(), eps, None)
    q = q / np.clip(q.sum(), eps, None)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def cosine_similarity_flat(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / denom)


def pairwise_similarity_from_probs(probs: np.ndarray, metric: str = "cosine") -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    rank = probs.shape[0]
    sim = np.eye(rank, dtype=np.float64)
    for i in range(rank):
        for j in range(i + 1, rank):
            if metric == "cosine":
                s = cosine_similarity_flat(probs[i], probs[j])
            elif metric == "js":
                p = np.clip(probs[i], 1e-6, 1.0)
                q = np.clip(probs[j], 1e-6, 1.0)
                m = 0.5 * (p + q)
                js = 0.5 * np.mean(np.sum(p * np.log(p / m), axis=1) + np.sum(q * np.log(q / m), axis=1))
                s = float(1.0 / (1.0 + js))
            elif metric == "corr":
                x = probs[i].reshape(-1)
                y = probs[j].reshape(-1)
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    s = 0.0
                else:
                    s = float(np.corrcoef(x, y)[0, 1])
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
            sim[i, j] = s
            sim[j, i] = s
    return sim


def greedy_select_with_redundancy(base_scores: np.ndarray, similarity: np.ndarray, k: int, lam: float) -> List[int]:
    n = len(base_scores)
    remaining = set(range(n))
    selected: List[int] = []
    while len(selected) < min(k, n) and remaining:
        best_i = None
        best_gain = None
        for i in remaining:
            red = 0.0 if not selected else max(float(similarity[i, j]) for j in selected)
            gain = float(base_scores[i]) - lam * red
            if (best_gain is None) or (gain > best_gain):
                best_gain = gain
                best_i = i
        selected.append(int(best_i))
        remaining.remove(best_i)
    return selected


def topk_indices_from_scores(scores: np.ndarray, k: int) -> List[int]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return [int(x) for x in order[: max(1, min(k, len(order)))]]


def mask_from_indices(rank: int, indices: Sequence[int]) -> np.ndarray:
    mask = np.zeros(rank, dtype=np.float32)
    mask[list(indices)] = 1.0
    return mask


def selected_indices_to_str(indices: Sequence[int]) -> str:
    return ",".join(str(int(i)) for i in indices)


def get_system_resource_info() -> Dict:
    # Keep resource logs double-blind safe: record software/GPU configuration
    # but omit identity-bearing system fields.
    info = {
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_names": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False, capture_output=True, text=True)
        if out.stdout.strip():
            info["nvidia_smi"] = [line.strip() for line in out.stdout.strip().splitlines()]
    except Exception:
        pass
    return info


def gpu_memory_snapshot() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"max_memory_allocated_mb": 0.0, "max_memory_reserved_mb": 0.0, "memory_allocated_mb": 0.0, "memory_reserved_mb": 0.0}
    return {
        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
        "memory_allocated_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)),
        "memory_reserved_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)),
    }


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()
        self.rank_mask: Optional[torch.Tensor] = None
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_rank_mask(self, rank_mask: Optional[torch.Tensor]) -> None:
        self.rank_mask = rank_mask

    def _masked_lora_B(self) -> torch.Tensor:
        if self.rank_mask is None:
            return self.lora_B
        mask = self.rank_mask.to(self.lora_B.device, dtype=self.lora_B.dtype).view(1, -1)
        return self.lora_B * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        b_eff = self._masked_lora_B().to(device=x.device, dtype=x.dtype)
        a_eff = self.lora_A.to(device=x.device, dtype=x.dtype)
        lora_out = self.dropout(x) @ a_eff.t()
        lora_out = lora_out @ b_eff.t()
        return base_out + self.scaling * lora_out


@dataclass
class LoRAModuleRef:
    module_path: str
    module: LoRALinear


def get_parent_module(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_target_modules_with_lora(model: nn.Module, target_module_names: Sequence[str], rank: int, alpha: float, dropout: float, logger: logging.Logger) -> List[LoRAModuleRef]:
    refs: List[LoRAModuleRef] = []
    for module_path, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short_name = module_path.split(".")[-1]
        if short_name not in target_module_names:
            continue
        parent, child_name = get_parent_module(model, module_path)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, child_name, wrapped)
        refs.append(LoRAModuleRef(module_path=module_path, module=wrapped))
    logger.info("Replaced %d target Linear layers with LoRA wrappers.", len(refs))
    for ref in refs:
        logger.info("  LoRA target: %s", ref.module_path)
    if not refs:
        raise ValueError("No target modules were replaced. Check --target-modules for the chosen model.")
    return refs


def set_global_rank_mask(lora_refs: Sequence[LoRAModuleRef], rank_mask: Optional[torch.Tensor]) -> None:
    for ref in lora_refs:
        ref.module.set_rank_mask(rank_mask)


@contextmanager
def temporary_rank_mask(lora_refs: Sequence[LoRAModuleRef], rank_mask: Optional[torch.Tensor]):
    old_masks = [ref.module.rank_mask for ref in lora_refs]
    set_global_rank_mask(lora_refs, rank_mask)
    try:
        yield
    finally:
        for ref, old in zip(lora_refs, old_masks):
            ref.module.set_rank_mask(old)


def gather_rank_statistics(lora_refs: Sequence[LoRAModuleRef], rank: int) -> Dict[str, np.ndarray]:
    mag = np.zeros(rank, dtype=np.float64)
    grad = np.zeros(rank, dtype=np.float64)
    fisher = np.zeros(rank, dtype=np.float64)
    short_gain = np.zeros(rank, dtype=np.float64)
    for ref in lora_refs:
        a = ref.module.lora_A.detach().float().cpu().numpy()
        b = ref.module.lora_B.detach().float().cpu().numpy()
        a_grad = np.zeros_like(a)
        b_grad = np.zeros_like(b)
        if ref.module.lora_A.grad is not None:
            a_grad = ref.module.lora_A.grad.detach().float().cpu().numpy()
        if ref.module.lora_B.grad is not None:
            b_grad = ref.module.lora_B.grad.detach().float().cpu().numpy()
        for k in range(rank):
            mag[k] += np.linalg.norm(b[:, k]) * np.linalg.norm(a[k, :])
            grad_k = np.linalg.norm(b_grad[:, k]) * np.linalg.norm(a[k, :]) + np.linalg.norm(b[:, k]) * np.linalg.norm(a_grad[k, :])
            grad[k] += grad_k
            fisher[k] += grad_k ** 2
            short_gain[k] += -(np.sum(b[:, k] * b_grad[:, k]) + np.sum(a[k, :] * a_grad[k, :]))
    curv = grad ** 2
    return {"mag": mag, "grad": grad, "curv": curv, "Fisher": fisher, "short_gain": short_gain}


def freeze_non_lora_backbone(model: nn.Module, logger: logging.Logger) -> None:
    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        allow = name.endswith("lora_A") or name.endswith("lora_B")
        param.requires_grad = allow
        if allow:
            trainable_names.append(name)
    logger.info("Trainable parameter tensors: %d", len(trainable_names))
    for n in trainable_names[:30]:
        logger.info("  trainable: %s", n)


TEMPLATE_WO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

ALPACA_PREFIX_TEMPLATE_MD = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Complete the following Python code.\n"
    "Notes:\n"
    "- Respond with the entire complete function definition.\n"
    "- Do not add any comments.\n"
    "- Be as concise in your code as possible.\n"
    "- Use only built-in libraries, assume no additional imports other than those provided (if any).\n"
    "- Use four spaces for each level of indentation.\n\n"
    "Code:\n{PROMPT}\n\n### Response:\n"
)


def post_process_humaneval_completion(text: str) -> str:
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("\t", "    ")
    lines = [l.rstrip() for l in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    start = 0
    for i, line in enumerate(lines):
        if "def " in line:
            start = i
            break
    lines = lines[start:]
    min_spaces = None
    for line in lines:
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        if min_spaces is None or leading < min_spaces:
            min_spaces = leading
    if min_spaces is None:
        min_spaces = 0
    trimmed = [line[min_spaces:] if len(line) >= min_spaces else line for line in lines]
    return "\n".join(trimmed).strip() + ("\n" if trimmed else "")


def syntax_valid_python(code: str) -> bool:
    code = code.strip()
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def load_codefeedback_with_retry(cache_dir: Optional[str], logger: logging.Logger, retries: int = 3):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Loading CodeFeedback (attempt %d/%d)", attempt, retries)
            return load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", cache_dir=cache_dir)
        except Exception as exc:
            last_exc = exc
            logger.warning("CodeFeedback load failed on attempt %d: %r", attempt, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError("Failed to load CodeFeedback") from last_exc


def load_humaneval_with_retry(cache_dir: Optional[str], logger: logging.Logger, retries: int = 3):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Loading HumanEval (attempt %d/%d)", attempt, retries)
            return load_dataset("openai_humaneval", split="test", cache_dir=cache_dir)
        except Exception as exc:
            last_exc = exc
            logger.warning("HumanEval load failed on attempt %d: %r", attempt, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError("Failed to load HumanEval") from last_exc


def build_codefeedback_splits(cache_dir: Optional[str], logger: logging.Logger, val_ratio: float, split_seed: int):
    raw = load_codefeedback_with_retry(cache_dir, logger)
    split = raw.train_test_split(test_size=val_ratio, seed=split_seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def maybe_subsample(ds, n: Optional[int], seed: int):
    if n is None or n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


class CodeTrainDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int, min_answer_tokens: int = 64):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_answer_tokens = min_answer_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.dataset[idx]
        prompt = TEMPLATE_WO_INPUT.format(instruction=ex["query"].strip())
        answer = ex["answer"].strip()
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        reserve = max(8, self.min_answer_tokens)
        max_prompt_len = max(1, self.max_length - reserve - (1 if eos_id is not None else 0))
        prompt_ids = prompt_ids[:max_prompt_len]
        max_answer_len = max(1, self.max_length - len(prompt_ids) - (1 if eos_id is not None else 0))
        answer_ids = answer_ids[:max_answer_len]
        input_ids = prompt_ids + answer_ids + ([eos_id] if eos_id is not None else [])
        labels = [-100] * len(prompt_ids) + answer_ids + ([eos_id] if eos_id is not None else [])
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        if not any(x != -100 for x in labels):
            labels[-1] = input_ids[-1]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class HumanEvalPromptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_prompt_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str | torch.Tensor]:
        ex = self.dataset[idx]
        prompt = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=ex["prompt"])
        toks = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_prompt_length)
        return {
            "input_ids": torch.tensor(toks["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(toks["attention_mask"], dtype=torch.long),
            "prompt_text": prompt,
            "task_id": ex["task_id"],
            "canonical_solution": ex["canonical_solution"],
        }


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            ids = ex["input_ids"]
            attn = ex["attention_mask"]
            labs = ex["labels"]
            need = max_len - len(ids)
            input_ids.append(torch.cat([torch.full((need,), pad_id, dtype=torch.long), ids], dim=0))
            attention_mask.append(torch.cat([torch.zeros(need, dtype=torch.long), attn], dim=0))
            labels.append(torch.cat([torch.full((need,), -100, dtype=torch.long), labs], dim=0))
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)}


class PromptCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask = [], []
        meta = []
        for ex in batch:
            ids = ex["input_ids"]
            attn = ex["attention_mask"]
            need = max_len - len(ids)
            input_ids.append(torch.cat([torch.full((need,), pad_id, dtype=torch.long), ids], dim=0))
            attention_mask.append(torch.cat([torch.zeros(need, dtype=torch.long), attn], dim=0))
            meta.append({"prompt_text": ex["prompt_text"], "task_id": ex["task_id"], "canonical_solution": ex["canonical_solution"]})
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "meta": meta}


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


@torch.no_grad()
def evaluate_loss_only(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray | float]:
    model.eval()
    losses = []
    per_example_target_scores = []
    for batch in dataloader:
        tbatch = batch_to_device(batch, device)
        outputs = model(**tbatch)
        losses.append(outputs.loss.detach().float().cpu().item())
        logits = outputs.logits.detach().float()
        labels = tbatch["labels"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_log_probs = torch.log_softmax(shift_logits, dim=-1)
        gathered = torch.gather(token_log_probs, dim=-1, index=torch.clamp(shift_labels, min=0).unsqueeze(-1)).squeeze(-1)
        mask = shift_labels.ne(-100)
        seq_scores = (gathered * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        per_example_target_scores.extend(seq_scores.detach().cpu().numpy().tolist())
    finite_losses = [x for x in losses if np.isfinite(x)]
    loss_value = float(np.mean(finite_losses)) if finite_losses else float("nan")
    return {"loss": loss_value, "example_scores": np.asarray(per_example_target_scores, dtype=np.float64)}


@torch.no_grad()
def evaluate_generation_syntax_rate(model: nn.Module, dataloader: DataLoader, tokenizer, device: torch.device, max_new_tokens: int, predictions_path: Optional[Path] = None) -> Dict[str, float]:
    model.eval()
    correct = 0
    contains_def = 0
    total = 0
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        fout = open(predictions_path, "w", encoding="utf-8")
        evalplus_path = predictions_path.with_name(predictions_path.stem + "_evalplus.jsonl")
        fout_evalplus = open(evalplus_path, "w", encoding="utf-8")
    else:
        fout = None
        fout_evalplus = None
    try:
        for batch in dataloader:
            meta = batch["meta"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = input_ids.shape[1]
            for j in range(gen.shape[0]):
                new_tokens = gen[j, prompt_len:]
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completion = post_process_humaneval_completion(raw_text)
                valid = syntax_valid_python(completion)
                has_def = int("def " in completion)
                correct += int(valid)
                contains_def += has_def
                total += 1
                if fout is not None:
                    fout.write(json.dumps({
                        "task_id": meta[j]["task_id"],
                        "prompt_text": meta[j]["prompt_text"],
                        "raw_generation": raw_text,
                        "completion": completion,
                        "solution": completion,
                        "canonical_solution": meta[j]["canonical_solution"],
                        "syntax_valid": bool(valid),
                        "contains_def": bool(has_def),
                    }, ensure_ascii=False) + "\n")
                if fout_evalplus is not None:
                    fout_evalplus.write(json.dumps({
                        "task_id": meta[j]["task_id"],
                        "solution": completion,
                    }, ensure_ascii=False) + "\n")
    finally:
        if fout is not None:
            fout.close()
        if fout_evalplus is not None:
            fout_evalplus.close()
    return {"acc": float(correct / max(total, 1)), "contains_def_rate": float(contains_def / max(total, 1)), "n": int(total)}


def train_one_seed(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], train_loader: DataLoader, val_loader: DataLoader, device: torch.device, learning_rate: float, weight_decay: float, epochs: int, warmup_ratio: float, early_stop_patience: int, grad_accum_steps: int, logger: logging.Logger) -> pd.DataFrame:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) * epochs / max(1, grad_accum_steps)))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: List[Dict] = []
    set_global_rank_mask(lora_refs, None)
    logger.info("Start training: epochs=%d, total_steps=%d, warmup_steps=%d", epochs, total_steps, warmup_steps)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            tbatch = batch_to_device(batch, device)
            outputs = model(**tbatch)
            loss = outputs.loss / max(1, grad_accum_steps)
            loss.backward()
            if step % grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        train_eval = evaluate_loss_only(model, train_loader, device)
        val_eval = evaluate_loss_only(model, val_loader, device)
        history_rows.append({"epoch": epoch, "train_loss": train_eval["loss"], "val_loss": val_eval["loss"], "lr": float(scheduler.get_last_lr()[0])})
        logger.info("Epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, train_eval["loss"], val_eval["loss"])
        if val_eval["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_eval["loss"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break
    model.load_state_dict(best_state)
    return pd.DataFrame(history_rows)


def compute_direction_table(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], train_loader: DataLoader, val_loader: DataLoader, device: torch.device, rank: int, logger: logging.Logger, lambda_v1: float, lambda_v2: float, alpha_v4: float, beta_v4: float, gamma_v4: float, alpha_v5: float, beta_v5: float, gamma_v5: float, js_temperature: float) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    logger.info("Computing direction-level scores and single-direction caches...")
    set_global_rank_mask(lora_refs, None)
    model.zero_grad(set_to_none=True)
    batch = next(iter(train_loader))
    tbatch = batch_to_device(batch, device)
    model(**tbatch).loss.backward()
    full_val = evaluate_loss_only(model, val_loader, device)
    full_train = evaluate_loss_only(model, train_loader, device)
    rank_stats = gather_rank_statistics(lora_refs, rank)
    rows, val_behavior_cache = [], []
    for k in range(rank):
        only_mask = torch.zeros(rank, dtype=torch.float32, device=device); only_mask[k] = 1.0
        drop_mask = torch.ones(rank, dtype=torch.float32, device=device); drop_mask[k] = 0.0
        with temporary_rank_mask(lora_refs, only_mask):
            only_val = evaluate_loss_only(model, val_loader, device)
        with temporary_rank_mask(lora_refs, drop_mask):
            drop_val = evaluate_loss_only(model, val_loader, device)
        standalone_effect = full_val["loss"] - only_val["loss"]
        loo_utility = drop_val["loss"] - full_val["loss"]
        js_val = js_from_example_scores(full_val["example_scores"], only_val["example_scores"], temperature=js_temperature)
        rows.append({
            "direction": k,
            "mag": float(rank_stats["mag"][k]),
            "grad": float(rank_stats["grad"][k]),
            "curv": float(rank_stats["curv"][k]),
            "standalone_effect": float(standalone_effect),
            "loo_utility": float(loo_utility),
            "JS": float(js_val),
            "Fisher": float(rank_stats["Fisher"][k]),
            "short_gain": float(rank_stats["short_gain"][k]),
        })
        val_behavior_cache.append(np.asarray(only_val["example_scores"], dtype=np.float32))
    df = pd.DataFrame(rows)
    df["v1"] = df["standalone_effect"] - lambda_v1 * df["JS"]
    df["v2"] = df["standalone_effect"] - lambda_v2 * df["JS"]
    df["v3"] = df["short_gain"] - df["Fisher"]
    df["v4"] = alpha_v4 * df["short_gain"] + beta_v4 * df["standalone_effect"] - gamma_v4 * df["Fisher"]
    a = zscore(df["standalone_effect"].values)
    b = zscore(df["JS"].values)
    c = zscore(df["Fisher"].values)
    gate = sigmoid_np(alpha_v5 * a - beta_v5 * b - gamma_v5 * c)
    df["gate_v5"] = gate
    df["v5"] = gate * a
    meta = {"full_train_loss": full_train["loss"], "full_val_loss": full_val["loss"]}
    cache = {"val_behavior": np.stack(val_behavior_cache, axis=0)}
    return df, meta, cache


def build_method_subsets(direction_df: pd.DataFrame, methods: Sequence[str], topk_values: Sequence[int], similarity: Optional[np.ndarray], greedy_base_method: str, greedy_lambda: float, greedy_name: str, seed: int) -> Tuple[Dict[str, Dict[int, List[int]]], pd.DataFrame]:
    rng = np.random.default_rng(seed + 2027)
    subset_map: Dict[str, Dict[int, List[int]]] = {}
    method_rows: List[Dict] = []
    valid_columns = set(direction_df.columns)
    for method in methods:
        subset_map[method] = {}
        if method == "random":
            scores = rng.standard_normal(len(direction_df))
            rho = safe_spearman(scores, direction_df["loo_utility"].values)
            method_rows.append({"method": method, "spearman_loo_utility": rho, "selection_type": "independent"})
            for k in topk_values:
                subset_map[method][k] = topk_indices_from_scores(scores, k)
        elif method == greedy_name:
            if similarity is None:
                raise ValueError("Greedy selector requested but similarity matrix is None.")
            if greedy_base_method not in valid_columns:
                raise ValueError(f"Greedy base method {greedy_base_method} not found.")
            base_scores = direction_df[greedy_base_method].values.astype(np.float64)
            rho = safe_spearman(base_scores, direction_df["loo_utility"].values)
            method_rows.append({"method": method, "spearman_loo_utility": rho, "selection_type": f"greedy_from_{greedy_base_method}"})
            max_k = max(topk_values)
            full_selected = greedy_select_with_redundancy(base_scores, similarity, max_k, greedy_lambda)
            for k in topk_values:
                subset_map[method][k] = full_selected[:k]
        else:
            if method not in valid_columns:
                raise ValueError(f"Requested method {method} not found in direction table")
            scores = direction_df[method].values.astype(np.float64)
            rho = safe_spearman(scores, direction_df["loo_utility"].values)
            method_rows.append({"method": method, "spearman_loo_utility": rho, "selection_type": "independent"})
            for k in topk_values:
                subset_map[method][k] = topk_indices_from_scores(scores, k)
    return subset_map, pd.DataFrame(method_rows)


def evaluate_method_subsets(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], subset_map: Dict[str, Dict[int, List[int]]], val_loader: DataLoader, test_loader: DataLoader, tokenizer, device: torch.device, rank: int, max_new_tokens: int, predictions_dir: Optional[Path] = None) -> pd.DataFrame:
    rows: List[Dict] = []
    for method, kmap in subset_map.items():
        for k, indices in kmap.items():
            mask = torch.tensor(mask_from_indices(rank, indices), dtype=torch.float32, device=device)
            pred_path = None
            if predictions_dir is not None:
                pred_path = predictions_dir / f"predictions_{method}_top{k}.jsonl"
            with temporary_rank_mask(lora_refs, mask):
                val_out = evaluate_loss_only(model, val_loader, device)
                test_out = evaluate_generation_syntax_rate(model, test_loader, tokenizer, device, max_new_tokens=max_new_tokens, predictions_path=pred_path)
            rows.append({
                "method": method,
                "topk": int(k),
                "keep_ratio": float(k / rank),
                "selected_indices": selected_indices_to_str(indices),
                "val_loss": float(val_out["loss"]),
                # ``test_acc`` is initially a syntax-valid diagnostic so that the
                # script can still produce lightweight smoke outputs when EvalPlus
                # is disabled. When EvalPlus is enabled, the final paper-facing
                # CSV files overwrite ``test_acc`` with HumanEval+ pass@1 and keep
                # this value under ``test_syntax_valid_rate``.
                "test_acc": float(test_out["acc"]),
                "test_syntax_valid_rate": float(test_out["acc"]),
                "test_contains_def_rate": float(test_out["contains_def_rate"]),
            })
    return pd.DataFrame(rows)


def run_exact_subset_oracle(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], val_loader: DataLoader, device: torch.device, rank: int, max_k: int, logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict] = []
    for k in range(1, min(max_k, rank) + 1):
        best_loss, best_subset = None, None
        subset_counter = 0
        for subset in itertools.combinations(range(rank), k):
            subset_counter += 1
            mask = torch.tensor(mask_from_indices(rank, subset), dtype=torch.float32, device=device)
            with temporary_rank_mask(lora_refs, mask):
                out = evaluate_loss_only(model, val_loader, device)
            if (best_loss is None) or (out["loss"] < best_loss - 1e-12):
                best_loss = float(out["loss"])
                best_subset = subset
        rows.append({"topk": int(k), "oracle_val_loss": float(best_loss), "oracle_subset": selected_indices_to_str(best_subset or []), "num_subsets": int(subset_counter)})
        logger.info("Exact oracle finished for k=%d | subsets=%d | best_val_loss=%.6f", k, subset_counter, best_loss)
    return pd.DataFrame(rows)


def compare_to_oracle(oracle_df: pd.DataFrame, subset_map: Dict[str, Dict[int, List[int]]]) -> pd.DataFrame:
    rows = []
    if oracle_df is None or oracle_df.empty:
        return pd.DataFrame()
    oracle_by_k = {int(r.topk): r for r in oracle_df.itertuples(index=False)}
    for method, kmap in subset_map.items():
        for k, indices in kmap.items():
            if k not in oracle_by_k:
                continue
            oracle_subset = [int(x) for x in str(oracle_by_k[k].oracle_subset).split(",") if str(x) != ""]
            overlap = len(set(indices) & set(oracle_subset))
            union = len(set(indices) | set(oracle_subset))
            rows.append({"method": method, "topk": int(k), "selected_indices": selected_indices_to_str(indices), "oracle_subset": selected_indices_to_str(oracle_subset), "oracle_overlap": int(overlap), "oracle_jaccard": float(overlap / union) if union else 0.0})
    return pd.DataFrame(rows)


def compute_win_rate_table(all_topk_df: pd.DataFrame, reference_method: str = "mag") -> pd.DataFrame:
    rows = []
    refs = all_topk_df[all_topk_df["method"] == reference_method][["seed", "topk", "test_acc"]].rename(columns={"test_acc": "ref_test_acc"})
    merged = all_topk_df.merge(refs, on=["seed", "topk"], how="left")
    for method in sorted(all_topk_df["method"].unique()):
        if method == reference_method:
            continue
        sub = merged[merged["method"] == method].copy()
        sub["delta_vs_ref"] = sub["test_acc"] - sub["ref_test_acc"]
        grouped = sub.groupby("topk")
        out = grouped.agg(win_rate=("delta_vs_ref", lambda x: float(np.mean(x > 0))), tie_rate=("delta_vs_ref", lambda x: float(np.mean(np.isclose(x, 0.0, atol=1e-8)))), mean_delta=("delta_vs_ref", "mean"), std_delta=("delta_vs_ref", "std")).reset_index()
        out["method"] = method
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["method", "topk", "win_rate", "tie_rate", "mean_delta", "std_delta"])
    return pd.concat(rows, ignore_index=True)[["method", "topk", "win_rate", "tie_rate", "mean_delta", "std_delta"]]


def save_ranking_plot(summary_methods: pd.DataFrame, output_dir: Path) -> None:
    if summary_methods.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df = summary_methods.sort_values("spearman_mean", ascending=False)
    ax.bar(plot_df["method"], plot_df["spearman_mean"], yerr=plot_df["spearman_std"].fillna(0.0), capsize=4)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("Spearman with LOO utility")
    ax.set_title("HumanEval smoke: ranking fidelity")
    plt.tight_layout()
    plt.savefig(output_dir / "ranking_fidelity.png", dpi=200)
    plt.close(fig)


def save_topk_curve_plot(summary_topk: pd.DataFrame, output_dir: Path, methods: Sequence[str]) -> None:
    if summary_topk.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in methods:
        sub = summary_topk[summary_topk["method"] == method].sort_values("topk")
        if sub.empty:
            continue
        ax.plot(sub["topk"], sub["test_acc_mean"], marker="o", label=method)
        std = sub["test_acc_std"].fillna(0.0)
        ax.fill_between(sub["topk"], sub["test_acc_mean"] - std, sub["test_acc_mean"] + std, alpha=0.15)
    ax.set_xlabel("Retained directions (top-k)")
    ax.set_ylabel("HumanEval+ pass@1 if EvalPlus is enabled; otherwise syntax-valid rate")
    ax.set_title("HumanEval smoke top-k retention curves")
    ax.legend(ncol=min(4, max(1, len(methods))))
    plt.tight_layout()
    plt.savefig(output_dir / "topk_retention_curves.png", dpi=200)
    plt.close(fig)


def save_delta_plot(win_rate_df: pd.DataFrame, output_dir: Path, methods: Sequence[str]) -> None:
    if win_rate_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False
    for method in methods:
        if method == "mag":
            continue
        sub = win_rate_df[win_rate_df["method"] == method].sort_values("topk")
        if sub.empty:
            continue
        plotted = True
        std = sub["std_delta"].fillna(0.0)
        ax.plot(sub["topk"], sub["mean_delta"], marker="o", label=method)
        ax.fill_between(sub["topk"], sub["mean_delta"] - std, sub["mean_delta"] + std, alpha=0.15)
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Retained directions (top-k)")
    ax.set_ylabel("Syntax-valid rate delta vs mag")
    ax.set_title("HumanEval smoke low-budget gain over magnitude")
    ax.legend(ncol=min(4, max(1, len(methods))))
    plt.tight_layout()
    plt.savefig(output_dir / "delta_vs_mag_curves.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--early-stop-patience", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-train-samples", type=int, default=1024)
    ap.add_argument("--max-val-samples", type=int, default=128)
    ap.add_argument("--max-test-samples", type=int, default=64)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--dataset-split-seed", type=int, default=3407)
    ap.add_argument("--topk-values", type=str, default="1,2")
    ap.add_argument("--report-methods", type=str, default="mag,v5,greedy_v5")
    ap.add_argument("--plot-methods", type=str, default="mag,v5,greedy_v5")
    ap.add_argument("--lambda-v1", type=float, default=0.8)
    ap.add_argument("--lambda-v2", type=float, default=0.4)
    ap.add_argument("--alpha-v4", type=float, default=0.6)
    ap.add_argument("--beta-v4", type=float, default=0.8)
    ap.add_argument("--gamma-v4", type=float, default=0.2)
    ap.add_argument("--alpha-v5", type=float, default=1.2)
    ap.add_argument("--beta-v5", type=float, default=0.9)
    ap.add_argument("--gamma-v5", type=float, default=0.3)
    ap.add_argument("--js-temperature", type=float, default=1.0)
    ap.add_argument("--enable-greedy-selector", type=int, default=1)
    ap.add_argument("--greedy-base-method", type=str, default="v5")
    ap.add_argument("--greedy-similarity", type=str, default="cosine", choices=["cosine", "js", "corr"])
    ap.add_argument("--greedy-lambda", type=float, default=0.30)
    ap.add_argument("--greedy-method-name", type=str, default="greedy_v5")
    ap.add_argument("--enable-exact-subset-oracle", type=int, default=0)
    ap.add_argument("--oracle-max-k", type=int, default=4)
    ap.add_argument("--oracle-max-seeds", type=int, default=1)
    ap.add_argument("--save-predictions", type=int, default=1)
    ap.add_argument("--run-evalplus", type=int, default=1)
    ap.add_argument("--evalplus-timeout", type=int, default=1800)
    return ap.parse_args()



def summarize_evalplus_results(eval_results_path: Path) -> Dict[str, float | int | str]:
    payload = json.loads(eval_results_path.read_text(encoding="utf-8"))
    eval_block = payload.get("eval", {}) if isinstance(payload, dict) else {}
    total = len(eval_block)
    base_pass = 0
    plus_pass = 0
    for _, records in eval_block.items():
        rec = records[0] if isinstance(records, list) and records else {}
        if rec.get("base_status") == "pass":
            base_pass += 1
        if rec.get("plus_status") == "pass":
            plus_pass += 1
    summary = {
        "eval_results_path": str(eval_results_path),
        "n_tasks": int(total),
        "base_pass_count": int(base_pass),
        "plus_pass_count": int(plus_pass),
        "base_pass_rate": float(base_pass / total) if total else 0.0,
        "plus_pass_rate": float(plus_pass / total) if total else 0.0,
    }
    return summary


def parse_evalplus_tag(tag: str) -> tuple[str, int | None]:
    """Parse tags such as ``mag_top1`` or ``greedy_v5_top4``."""
    match = re.match(r"^(?P<method>.+)_top(?P<topk>\d+)$", str(tag))
    if not match:
        return str(tag), None
    return match.group("method"), int(match.group("topk"))


def build_humanevalplus_pass_at_1_tables(all_topk_df: pd.DataFrame, all_evalplus_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge EvalPlus results into the normal top-k tables.

    The returned ``pass1_topk_df`` uses HumanEval+ plus-suite pass@1 as
    ``test_acc`` so downstream plotting / win-rate code treats HumanEval+ the
    same way as the other benchmark tables. The original syntax-valid diagnostic
    remains in ``test_syntax_valid_rate``.
    """
    eval_df = all_evalplus_df.copy()
    parsed = eval_df["tag"].apply(parse_evalplus_tag)
    eval_df["method"] = [x[0] for x in parsed]
    eval_df["topk"] = [x[1] for x in parsed]
    eval_df = eval_df.dropna(subset=["topk"]).copy()
    eval_df["topk"] = eval_df["topk"].astype(int)

    metric_cols = [
        "seed", "method", "topk", "n_tasks",
        "base_pass_count", "plus_pass_count", "base_pass_rate", "plus_pass_rate",
    ]
    available_metric_cols = [c for c in metric_cols if c in eval_df.columns]
    merged = all_topk_df.merge(eval_df[available_metric_cols], on=["seed", "method", "topk"], how="left")
    if "test_syntax_valid_rate" not in merged.columns:
        merged["test_syntax_valid_rate"] = merged["test_acc"]
    merged["humaneval_base_pass_at_1"] = merged.get("base_pass_rate", np.nan)
    merged["humaneval_plus_pass_at_1"] = merged.get("plus_pass_rate", np.nan)
    # Paper-facing HumanEval+ score: EvalPlus plus-suite pass@1.
    merged["test_acc"] = merged["humaneval_plus_pass_at_1"]

    pass1_topk_df = merged.dropna(subset=["test_acc"]).copy()
    summary_topk = pass1_topk_df.groupby(["method", "topk", "keep_ratio"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        humaneval_plus_pass_at_1_mean=("humaneval_plus_pass_at_1", "mean"),
        humaneval_plus_pass_at_1_std=("humaneval_plus_pass_at_1", "std"),
        humaneval_base_pass_at_1_mean=("humaneval_base_pass_at_1", "mean"),
        humaneval_base_pass_at_1_std=("humaneval_base_pass_at_1", "std"),
        test_syntax_valid_mean=("test_syntax_valid_rate", "mean"),
        test_syntax_valid_std=("test_syntax_valid_rate", "std"),
        test_contains_def_mean=("test_contains_def_rate", "mean"),
        test_contains_def_std=("test_contains_def_rate", "std"),
        val_loss_mean=("val_loss", "mean"),
        val_loss_std=("val_loss", "std"),
    ).reset_index()
    summary_low_budget = summary_topk[summary_topk["topk"].isin([k for k in [1, 2, 4, 8] if k in set(summary_topk["topk"])])].copy()
    win_rate_df = compute_win_rate_table(pass1_topk_df, reference_method="mag")
    return pass1_topk_df, summary_topk, summary_low_budget, win_rate_df


def maybe_run_evalplus(predictions_dir: Path, timeout_seconds: int, logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict] = []
    if not predictions_dir.exists():
        logger.warning("Predictions directory does not exist: %s", predictions_dir)
        return pd.DataFrame()
    evalplus_inputs = sorted(predictions_dir.glob("*_evalplus.jsonl"))
    if not evalplus_inputs:
        logger.warning("No EvalPlus prediction files found under %s", predictions_dir)
        return pd.DataFrame()
    for samples_path in evalplus_inputs:
        tag = samples_path.stem.replace("predictions_", "").replace("_evalplus", "")
        cmd = [sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", str(samples_path)]
        logger.info("Running EvalPlus for %s", samples_path.name)
        status = "ok"
        stderr_text = ""
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds, check=True)
            stdout_text = proc.stdout
            stderr_text = proc.stderr
        except subprocess.CalledProcessError as exc:
            status = f"error_returncode_{exc.returncode}"
            stdout_text = exc.stdout or ""
            stderr_text = exc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            stdout_text = exc.stdout or ""
            stderr_text = exc.stderr or ""
        (samples_path.parent / f"{tag}_evalplus_stdout.txt").write_text(stdout_text, encoding="utf-8")
        (samples_path.parent / f"{tag}_evalplus_stderr.txt").write_text(stderr_text, encoding="utf-8")
        eval_results_path = samples_path.with_name(samples_path.stem + "_eval_results.json")
        method, topk = parse_evalplus_tag(tag)
        row = {"tag": tag, "method": method, "topk": topk, "samples_path": str(samples_path), "status": status}
        if eval_results_path.exists():
            row.update(summarize_evalplus_results(eval_results_path))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(predictions_dir / "evalplus_summary.csv", index=False)
        (predictions_dir / "evalplus_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return df

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    write_status(output_dir, {"stage": "start", "ok": True})
    logger.info("Arguments: %s", vars(args))
    run_start = time.perf_counter()
    manifest = {"args": vars(args), "system": get_system_resource_info(), "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [s.strip() for s in args.report_methods.split(",") if s.strip()]
    plot_methods = [s.strip() for s in args.plot_methods.split(",") if s.strip()]
    if args.enable_greedy_selector:
        if args.greedy_method_name not in methods:
            methods.append(args.greedy_method_name)
        if args.greedy_method_name not in plot_methods:
            plot_methods.append(args.greedy_method_name)
    else:
        methods = [m for m in methods if m != args.greedy_method_name]
        plot_methods = [m for m in plot_methods if m != args.greedy_method_name]
    topk_values = sorted(set(int(s.strip()) for s in args.topk_values.split(",") if s.strip()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Methods to evaluate: %s", methods)

    ds = build_codefeedback_splits(args.cache_dir, logger, args.val_ratio, args.dataset_split_seed)
    humaneval_raw = load_humaneval_with_retry(args.cache_dir, logger)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer pad_token_id=%s eos_token_id=%s", tokenizer.pad_token_id, tokenizer.eos_token_id)

    all_method_rows, all_topk_rows, all_direction_rows, full_rows, seed_resource_rows = [], [], [], [], []
    oracle_compare_rows = []
    for seed_idx, seed in enumerate(seeds):
        seed_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info("================ Seed %d ================", seed)
        write_status(output_dir, {"stage": f"seed_{seed}", "ok": True})
        set_seed(seed)
        train_split = maybe_subsample(ds["train"], args.max_train_samples if args.max_train_samples > 0 else None, seed)
        val_split = maybe_subsample(ds["validation"], args.max_val_samples if args.max_val_samples > 0 else None, seed + 100)
        test_split = maybe_subsample(humaneval_raw, args.max_test_samples if args.max_test_samples > 0 else None, seed + 200)
        logger.info("Dataset sizes | train=%d val=%d test=%d", len(train_split), len(val_split), len(test_split))
        train_ds = CodeTrainDataset(train_split, tokenizer, args.max_length)
        val_ds = CodeTrainDataset(val_split, tokenizer, args.max_length)
        test_prompt_ds = HumanEvalPromptDataset(test_split, tokenizer, args.max_prompt_length)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=CausalLMCollator(tokenizer))
        val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=CausalLMCollator(tokenizer))
        test_loader = DataLoader(test_prompt_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=PromptCollator(tokenizer))

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, torch_dtype=torch_dtype)
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        lora_refs = replace_target_modules_with_lora(model, [s.strip() for s in args.target_modules.split(",") if s.strip()], rank=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, logger=logger)
        freeze_non_lora_backbone(model, logger)
        model.to(device)

        t_train0 = time.perf_counter()
        history_df = train_one_seed(model, lora_refs, train_loader, val_loader, device, args.learning_rate, args.weight_decay, args.epochs, args.warmup_ratio, args.early_stop_patience, args.grad_accum_steps, logger)
        train_seconds = time.perf_counter() - t_train0
        history_df.to_csv(output_dir / f"history_seed{seed}.csv", index=False)

        t_score0 = time.perf_counter()
        direction_df, full_meta, cache = compute_direction_table(model, lora_refs, train_loader, val_loader, device, args.lora_r, logger, args.lambda_v1, args.lambda_v2, args.alpha_v4, args.beta_v4, args.gamma_v4, args.alpha_v5, args.beta_v5, args.gamma_v5, args.js_temperature)
        scoring_seconds = time.perf_counter() - t_score0
        direction_df["seed"] = seed
        direction_df.to_csv(output_dir / f"direction_scores_seed{seed}.csv", index=False)
        all_direction_rows.append(direction_df)

        similarity = pairwise_similarity_from_probs(cache["val_behavior"][:, :, None], metric=args.greedy_similarity)
        subset_map, method_df = build_method_subsets(direction_df, methods, topk_values, similarity if args.enable_greedy_selector else None, args.greedy_base_method, args.greedy_lambda, args.greedy_method_name, seed)
        predictions_dir = output_dir / f"predictions_seed{seed}" if args.save_predictions else None
        topk_df = evaluate_method_subsets(model, lora_refs, subset_map, val_loader, test_loader, tokenizer, device, args.lora_r, args.max_new_tokens, predictions_dir=predictions_dir)
        method_df["seed"] = seed
        topk_df["seed"] = seed
        method_df.to_csv(output_dir / f"method_scores_seed{seed}.csv", index=False)
        topk_df.to_csv(output_dir / f"topk_results_seed{seed}.csv", index=False)
        all_method_rows.append(method_df)
        all_topk_rows.append(topk_df)

        oracle_seconds = 0.0
        if args.enable_exact_subset_oracle and seed_idx < args.oracle_max_seeds and args.lora_r <= 16:
            t_or0 = time.perf_counter()
            oracle_df = run_exact_subset_oracle(model, lora_refs, val_loader, device, args.lora_r, args.oracle_max_k, logger)
            oracle_seconds = time.perf_counter() - t_or0
            oracle_df["seed"] = seed
            oracle_df.to_csv(output_dir / f"oracle_seed{seed}.csv", index=False)
            cmp_df = compare_to_oracle(oracle_df, subset_map)
            if not cmp_df.empty:
                cmp_df["seed"] = seed
                cmp_df.to_csv(output_dir / f"oracle_comparison_seed{seed}.csv", index=False)
                oracle_compare_rows.append(cmp_df)

        full_rows.append({"seed": seed, **full_meta})
        seed_total_seconds = time.perf_counter() - seed_t0
        seed_resource_rows.append({"seed": seed, "train_seconds": train_seconds, "scoring_seconds": scoring_seconds, "oracle_seconds": oracle_seconds, "seed_total_seconds": seed_total_seconds, **gpu_memory_snapshot()})
        write_status(output_dir, {"stage": f"seed_{seed}_done", "ok": True})

    all_direction_df = pd.concat(all_direction_rows, ignore_index=True)
    all_method_df = pd.concat(all_method_rows, ignore_index=True)
    all_topk_df = pd.concat(all_topk_rows, ignore_index=True)
    full_df = pd.DataFrame(full_rows)
    seed_resource_df = pd.DataFrame(seed_resource_rows)
    summary_methods = all_method_df.groupby(["method", "selection_type"]).agg(spearman_mean=("spearman_loo_utility", "mean"), spearman_std=("spearman_loo_utility", "std")).reset_index()
    summary_topk = all_topk_df.groupby(["method", "topk", "keep_ratio"]).agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"), test_syntax_valid_mean=("test_syntax_valid_rate", "mean"), test_syntax_valid_std=("test_syntax_valid_rate", "std"), test_contains_def_mean=("test_contains_def_rate", "mean"), test_contains_def_std=("test_contains_def_rate", "std"), val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std")).reset_index()
    summary_low_budget = summary_topk[summary_topk["topk"].isin([k for k in [1, 2, 4, 8] if k in set(summary_topk["topk"])])].copy()
    summary_full = full_df.agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    win_rate_df = compute_win_rate_table(all_topk_df, reference_method="mag")
    resource_summary = {"system": get_system_resource_info(), "total_wall_clock_seconds": time.perf_counter() - run_start, "per_seed_summary": seed_resource_df.to_dict(orient="records")}
    if oracle_compare_rows:
        oracle_compare_df = pd.concat(oracle_compare_rows, ignore_index=True)
        oracle_compare_df.to_csv(output_dir / "oracle_comparison_all.csv", index=False)
    all_direction_df.to_csv(output_dir / "all_direction_scores.csv", index=False)
    all_method_df.to_csv(output_dir / "all_method_scores.csv", index=False)
    all_topk_df.to_csv(output_dir / "all_topk_results.csv", index=False)
    summary_methods.to_csv(output_dir / "summary_methods.csv", index=False)
    summary_topk.to_csv(output_dir / "summary_topk.csv", index=False)
    summary_low_budget.to_csv(output_dir / "summary_low_budget.csv", index=False)
    summary_full.to_csv(output_dir / "summary_full_model.csv", index=False)
    win_rate_df.to_csv(output_dir / "win_rate_vs_mag.csv", index=False)
    seed_resource_df.to_csv(output_dir / "seed_resource_usage.csv", index=False)
    (output_dir / "resource_summary.json").write_text(json.dumps(resource_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_ranking_plot(summary_methods, output_dir)
    save_topk_curve_plot(summary_topk, output_dir, plot_methods)
    save_delta_plot(win_rate_df, output_dir, plot_methods)
    if args.run_evalplus and args.save_predictions:
        evalplus_frames = []
        for seed in seeds:
            pred_dir = output_dir / f"predictions_seed{seed}"
            df_evalplus = maybe_run_evalplus(pred_dir, timeout_seconds=args.evalplus_timeout, logger=logger)
            if not df_evalplus.empty:
                df_evalplus.insert(0, "seed", seed)
                evalplus_frames.append(df_evalplus)
        if evalplus_frames:
            all_evalplus_df = pd.concat(evalplus_frames, ignore_index=True)
            all_evalplus_df.to_csv(output_dir / "evalplus_summary_all.csv", index=False)
            pass1_topk_df, pass1_summary_topk, pass1_summary_low_budget, pass1_win_rate_df = build_humanevalplus_pass_at_1_tables(all_topk_df, all_evalplus_df)
            # Keep the initial syntax-valid diagnostic files explicitly, then make
            # the standard HumanEval+ files paper-facing pass@1 tables.
            all_topk_df.to_csv(output_dir / "all_topk_results_syntax_diagnostic.csv", index=False)
            summary_topk.to_csv(output_dir / "summary_topk_syntax_diagnostic.csv", index=False)
            summary_low_budget.to_csv(output_dir / "summary_low_budget_syntax_diagnostic.csv", index=False)
            pass1_topk_df.to_csv(output_dir / "all_topk_results_pass_at_1.csv", index=False)
            pass1_summary_topk.to_csv(output_dir / "summary_topk_pass_at_1.csv", index=False)
            pass1_summary_low_budget.to_csv(output_dir / "summary_low_budget_pass_at_1.csv", index=False)
            pass1_win_rate_df.to_csv(output_dir / "win_rate_vs_mag_pass_at_1.csv", index=False)
            # Default filenames used by table-collection scripts now contain
            # HumanEval+ plus-suite pass@1 in ``test_acc`` / ``test_acc_mean``.
            pass1_topk_df.to_csv(output_dir / "all_topk_results.csv", index=False)
            pass1_summary_topk.to_csv(output_dir / "summary_topk.csv", index=False)
            pass1_summary_low_budget.to_csv(output_dir / "summary_low_budget.csv", index=False)
            pass1_win_rate_df.to_csv(output_dir / "win_rate_vs_mag.csv", index=False)
            save_topk_curve_plot(pass1_summary_topk, output_dir, plot_methods)
            save_delta_plot(pass1_win_rate_df, output_dir, plot_methods)
    logger.info("Saved summary outputs, oracle outputs (if enabled), prediction dumps (if enabled), and resource logs.")
    logger.info("Done.")
    write_status(output_dir, {"stage": "done", "ok": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output-dir" in sys.argv:
            try:
                out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                write_status(out_dir, {"stage": "error", "ok": False, "error": repr(exc)})
                with open(out_dir / "fatal_error.txt", "w", encoding="utf-8") as f:
                    f.write(repr(exc) + "\n")
            except Exception:
                pass
        raise
