
#!/usr/bin/env python3
"""
Vision runner for retained-subspace selection in frozen LoRA adapters.

The script supports CIFAR-style and image-classification benchmarks, trains a
LoRA adapter once, scores rank directions on a held-out selection split, and
evaluates retained top-k subspaces with optional oracle diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


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


def probs_from_logits(logits: np.ndarray, temperature: float = 2.0, eps: float = 1e-6) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64) / temperature
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape for probability conversion: {logits.shape}")
    if logits.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -20.0, 20.0)))
        p = np.clip(p, eps, 1.0 - eps)
        probs = np.stack([1.0 - p, p], axis=1)
    else:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits, -30.0, 30.0))
        probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), eps, None)
        probs = np.clip(probs, eps, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def js_from_logits(logits_p: np.ndarray, logits_q: np.ndarray, temperature: float = 2.0, eps: float = 1e-6) -> float:
    p = probs_from_logits(logits_p, temperature=temperature, eps=eps)
    q = probs_from_logits(logits_q, temperature=temperature, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    return float(np.nanmean(0.5 * (kl_pm + kl_qm)))


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if logits.ndim == 1:
        preds = (logits >= 0.0).astype(np.int64)
    elif logits.ndim == 2:
        if logits.shape[1] == 1:
            preds = (logits[:, 0] >= 0.0).astype(np.int64)
        else:
            preds = np.argmax(logits, axis=1).astype(np.int64)
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    if preds.shape[0] != labels.shape[0]:
        raise ValueError(f"Prediction/label length mismatch: preds={preds.shape}, labels={labels.shape}")
    return float((preds == labels).mean())


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
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if out.stdout.strip():
            info["nvidia_smi"] = [line.strip() for line in out.stdout.strip().splitlines()]
    except Exception:
        pass
    return info


def gpu_memory_snapshot() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "max_memory_allocated_mb": 0.0,
            "max_memory_reserved_mb": 0.0,
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
        }
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
        b_eff = self._masked_lora_B()
        lora_out = self.dropout(x) @ self.lora_A.t()
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
        a = ref.module.lora_A.detach().cpu().numpy()
        b = ref.module.lora_B.detach().cpu().numpy()
        a_grad = np.zeros_like(a)
        b_grad = np.zeros_like(b)
        if ref.module.lora_A.grad is not None:
            a_grad = ref.module.lora_A.grad.detach().cpu().numpy()
        if ref.module.lora_B.grad is not None:
            b_grad = ref.module.lora_B.grad.detach().cpu().numpy()
        for k in range(rank):
            mag[k] += np.linalg.norm(b[:, k]) * np.linalg.norm(a[k, :])
            grad_k = np.linalg.norm(b_grad[:, k]) * np.linalg.norm(a[k, :]) + np.linalg.norm(b[:, k]) * np.linalg.norm(a_grad[k, :])
            grad[k] += grad_k
            fisher[k] += grad_k ** 2
            short_gain[k] += -(np.sum(b[:, k] * b_grad[:, k]) + np.sum(a[k, :] * a_grad[k, :]))
    curv = grad ** 2
    return {"mag": mag, "grad": grad, "curv": curv, "Fisher": fisher, "short_gain": short_gain}


def save_single_direction_cache(output_dir: Path, seed: int, labels: np.ndarray, val_logits: np.ndarray, val_probs: np.ndarray, similarity: np.ndarray) -> None:
    np.savez_compressed(
        output_dir / f"single_direction_val_cache_seed{seed}.npz",
        labels=labels.astype(np.int64),
        val_logits=val_logits.astype(np.float32),
        val_probs=val_probs.astype(np.float32),
        similarity=similarity.astype(np.float32),
    )


def run_exact_subset_oracle(
    model: nn.Module,
    lora_refs: Sequence[LoRAModuleRef],
    val_loader: DataLoader,
    device: torch.device,
    rank: int,
    max_k: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for k in range(1, min(max_k, rank) + 1):
        best_loss = None
        best_acc = None
        best_subset = None
        subset_counter = 0
        for subset in itertools.combinations(range(rank), k):
            subset_counter += 1
            mask = torch.tensor(mask_from_indices(rank, subset), dtype=torch.float32, device=device)
            with temporary_rank_mask(lora_refs, mask):
                out = evaluate_model(model, val_loader, device)
            val_loss = float(out["loss"])
            val_acc = float(out["acc"])
            if (best_loss is None) or (val_loss < best_loss - 1e-12):
                best_loss = val_loss
                best_acc = val_acc
                best_subset = subset
        rows.append({
            "topk": int(k),
            "oracle_val_loss": float(best_loss),
            "oracle_val_acc": float(best_acc),
            "oracle_subset": selected_indices_to_str(best_subset or []),
            "num_subsets": int(subset_counter),
        })
        logger.info("Exact oracle finished for k=%d | subsets=%d | best_val_loss=%.6f", k, subset_counter, best_loss)
    return pd.DataFrame(rows)


def default_topk_values_for_rank(rank: int) -> List[int]:
    if rank <= 4:
        vals = [1, 2, 3, 4]
    elif rank <= 8:
        vals = [1, 2, 4, 8]
    elif rank <= 16:
        vals = [1, 2, 4, 8, 12, 16]
    else:
        vals = [1, 2, 4, 8, rank // 2, rank]
    return sorted(set(v for v in vals if v <= rank))

from datasets import DatasetDict, load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup


def freeze_non_lora_backbone(model: nn.Module, logger: logging.Logger) -> None:
    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        allow = (
            name.endswith("lora_A")
            or name.endswith("lora_B")
            or name.startswith("classifier")
            or ".classifier." in name
            or name.startswith("vit.classifier")
        )
        param.requires_grad = allow
        if allow:
            trainable_names.append(name)
    logger.info("Trainable parameter tensors: %d", len(trainable_names))
    for n in trainable_names[:30]:
        logger.info("  trainable: %s", n)



def resolve_vision_dataset_name(dataset_name: str) -> str:
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "food101": "food101",
        "food-101": "food101",
        "eurosat": "timm/eurosat-rgb",
        "eurosat-rgb": "timm/eurosat-rgb",
    }
    key = dataset_name.lower()
    return aliases.get(key, dataset_name)

def load_dataset_with_retry(dataset_name: str, cache_dir: Optional[str], logger: logging.Logger, retries: int = 3):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Loading dataset=%s (attempt %d/%d)", dataset_name, attempt, retries)
            return load_dataset(dataset_name, cache_dir=cache_dir)
        except Exception as exc:
            last_exc = exc
            logger.warning("Dataset load failed on attempt %d: %r", attempt, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to load dataset={dataset_name} after {retries} attempts") from last_exc


def prepare_vision_dataset(dataset_name: str, data_split_seed: int, val_ratio: float, test_ratio: float, cache_dir: Optional[str], logger: logging.Logger) -> Tuple[DatasetDict, str, str, List[str]]:
    hf_dataset_name = resolve_vision_dataset_name(dataset_name)
    raw = load_dataset_with_retry(hf_dataset_name, cache_dir=cache_dir, logger=logger)
    train_split_name = "train" if "train" in raw else list(raw.keys())[0]
    train_features = raw[train_split_name].features
    train_columns = list(raw[train_split_name].column_names)
    image_candidates = ["img", "image", "images", "pixel_values"]
    label_candidates = ["label", "labels", "fine_label", "coarse_label"]
    image_col = next((c for c in image_candidates if c in train_columns), None)
    label_col = next((c for c in label_candidates if c in train_columns), None)
    if image_col is None or label_col is None:
        raise KeyError(f"Could not infer image/label columns for dataset={dataset_name}. columns={train_columns}")
    label_feature = train_features[label_col]
    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = list(label_feature.names)
    else:
        unique_labels = sorted(set(int(x) for x in raw[train_split_name][label_col]))
        label_names = [str(x) for x in unique_labels]
    if "validation" in raw and "test" in raw:
        prepared = DatasetDict({"train": raw[train_split_name], "validation": raw["validation"], "test": raw["test"]})
    elif "validation" in raw:
        # Datasets such as Food-101 provide train/validation but no official test
        # split. Use a clean train holdout for model selection and keep the
        # official validation split as the paper-facing test split.
        first = raw[train_split_name].train_test_split(test_size=val_ratio, seed=data_split_seed, stratify_by_column=label_col)
        prepared = DatasetDict({"train": first["train"], "validation": first["test"], "test": raw["validation"]})
    elif "test" in raw:
        first = raw[train_split_name].train_test_split(test_size=val_ratio, seed=data_split_seed, stratify_by_column=label_col)
        prepared = DatasetDict({"train": first["train"], "validation": first["test"], "test": raw["test"]})
    else:
        first = raw[train_split_name].train_test_split(test_size=test_ratio, seed=data_split_seed, stratify_by_column=label_col)
        second = first["train"].train_test_split(test_size=val_ratio / max(1e-8, 1.0 - test_ratio), seed=data_split_seed + 1, stratify_by_column=label_col)
        prepared = DatasetDict({"train": second["train"], "validation": second["test"], "test": first["test"]})
    logger.info("Prepared dataset %s (HF: %s) | train=%d val=%d test=%d num_labels=%d image_col=%s label_col=%s", dataset_name, hf_dataset_name, len(prepared["train"]), len(prepared["validation"]), len(prepared["test"]), len(label_names), image_col, label_col)
    return prepared, image_col, label_col, label_names


class ProcessedVisionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor, image_col: str, label_col: str):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.dataset[idx]
        image = ex[self.image_col]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(int(ex[self.label_col]), dtype=torch.long)}


def build_dataloaders(prepared: DatasetDict, processor, image_col: str, label_col: str, batch_size: int, max_train: Optional[int], max_val: Optional[int], max_test: Optional[int], seed: int):
    def maybe_subsample(ds, n: Optional[int], local_seed: int):
        if n is None or n <= 0 or n >= len(ds):
            return ds
        return ds.shuffle(seed=local_seed).select(range(n))
    train_ds = maybe_subsample(prepared["train"], max_train, seed)
    val_ds = maybe_subsample(prepared["validation"], max_val, seed + 100)
    test_ds = maybe_subsample(prepared["test"], max_test, seed + 200)
    train_wrapped = ProcessedVisionDataset(train_ds, processor, image_col, label_col)
    val_wrapped = ProcessedVisionDataset(val_ds, processor, image_col, label_col)
    test_wrapped = ProcessedVisionDataset(test_ds, processor, image_col, label_col)
    def collate_fn(batch):
        return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "labels": torch.stack([b["labels"] for b in batch])}
    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_wrapped, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_wrapped, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray | float]:
    model.eval()
    losses, logits_list, labels_list = [], [], []
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.detach().cpu().item())
        logits = outputs.logits.detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy().reshape(-1)
        logits_list.append(logits)
        labels_list.append(labels)
    logits_all = np.concatenate(logits_list, axis=0)
    labels_all = np.concatenate(labels_list)
    return {"loss": float(np.mean(losses)), "acc": accuracy_from_logits(logits_all, labels_all), "logits": logits_all, "labels": labels_all}


def train_one_seed(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], train_loader: DataLoader, val_loader: DataLoader, device: torch.device, learning_rate: float, weight_decay: float, epochs: int, warmup_ratio: float, early_stop_patience: int, logger: logging.Logger) -> pd.DataFrame:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
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
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
        train_eval = evaluate_model(model, train_loader, device)
        val_eval = evaluate_model(model, val_loader, device)
        history_rows.append({"epoch": epoch, "train_loss": train_eval["loss"], "train_acc": train_eval["acc"], "val_loss": val_eval["loss"], "val_acc": val_eval["acc"], "lr": float(scheduler.get_last_lr()[0])})
        logger.info("Epoch %d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f", epoch, train_eval["loss"], train_eval["acc"], val_eval["loss"], val_eval["acc"])
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


def compute_direction_table(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, rank: int, logger: logging.Logger, lambda_v1: float, lambda_v2: float, alpha_v4: float, beta_v4: float, gamma_v4: float, alpha_v5: float, beta_v5: float, gamma_v5: float, js_temperature: float) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    logger.info("Computing direction-level scores and single-direction caches...")
    set_global_rank_mask(lora_refs, None)
    model.zero_grad(set_to_none=True)
    batch = next(iter(train_loader))
    batch = batch_to_device(batch, device)
    model(**batch).loss.backward()
    full_val = evaluate_model(model, val_loader, device)
    full_test = evaluate_model(model, test_loader, device)
    full_train = evaluate_model(model, train_loader, device)
    rank_stats = gather_rank_statistics(lora_refs, rank)
    rows, val_logits_cache, val_probs_cache = [], [], []
    labels_cache = None
    for k in range(rank):
        only_mask = torch.zeros(rank, dtype=torch.float32, device=device); only_mask[k] = 1.0
        drop_mask = torch.ones(rank, dtype=torch.float32, device=device); drop_mask[k] = 0.0
        with temporary_rank_mask(lora_refs, only_mask):
            only_val = evaluate_model(model, val_loader, device)
            only_test = evaluate_model(model, test_loader, device)
        with temporary_rank_mask(lora_refs, drop_mask):
            drop_val = evaluate_model(model, val_loader, device)
        standalone_effect = full_val["loss"] - only_val["loss"]
        loo_utility = drop_val["loss"] - full_val["loss"]
        js_val = js_from_logits(full_val["logits"], only_val["logits"], temperature=js_temperature)
        rows.append({"direction": k, "mag": float(rank_stats["mag"][k]), "grad": float(rank_stats["grad"][k]), "curv": float(rank_stats["curv"][k]), "standalone_effect": float(standalone_effect), "loo_utility": float(loo_utility), "JS": float(js_val), "Fisher": float(rank_stats["Fisher"][k]), "short_gain": float(rank_stats["short_gain"][k]), "only_val_acc": float(only_val["acc"]), "only_test_acc": float(only_test["acc"])})
        val_logits_cache.append(np.asarray(only_val["logits"], dtype=np.float32))
        val_probs_cache.append(probs_from_logits(np.asarray(only_val["logits"], dtype=np.float32), temperature=js_temperature).astype(np.float32))
        if labels_cache is None:
            labels_cache = np.asarray(only_val["labels"], dtype=np.int64)
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
    meta = {"full_train_acc": full_train["acc"], "full_val_acc": full_val["acc"], "full_test_acc": full_test["acc"], "full_train_loss": full_train["loss"], "full_val_loss": full_val["loss"], "full_test_loss": full_test["loss"]}
    cache = {"labels": labels_cache, "val_logits": np.stack(val_logits_cache, axis=0), "val_probs": np.stack(val_probs_cache, axis=0)}
    return df, meta, cache


def build_method_subsets(direction_df: pd.DataFrame, methods: Sequence[str], topk_values: Sequence[int], similarity: Optional[np.ndarray], greedy_base_method: str, greedy_lambda: float, greedy_name: str, seed: int) -> Tuple[Dict[str, Dict[int, List[int]]], pd.DataFrame]:
    rng = np.random.default_rng(seed + 2027)
    subset_map, method_rows = {}, []
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
            base_scores = direction_df[greedy_base_method].values.astype(np.float64)
            rho = safe_spearman(base_scores, direction_df["loo_utility"].values)
            method_rows.append({"method": method, "spearman_loo_utility": rho, "selection_type": f"greedy_from_{greedy_base_method}"})
            full_selected = greedy_select_with_redundancy(base_scores, similarity, max(topk_values), greedy_lambda)
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


def evaluate_method_subsets(model: nn.Module, lora_refs: Sequence[LoRAModuleRef], subset_map: Dict[str, Dict[int, List[int]]], val_loader: DataLoader, test_loader: DataLoader, device: torch.device, rank: int) -> pd.DataFrame:
    rows = []
    for method, kmap in subset_map.items():
        for k, indices in kmap.items():
            mask = torch.tensor(mask_from_indices(rank, indices), dtype=torch.float32, device=device)
            with temporary_rank_mask(lora_refs, mask):
                val_out = evaluate_model(model, val_loader, device)
                test_out = evaluate_model(model, test_loader, device)
            rows.append({"method": method, "topk": int(k), "keep_ratio": float(k / rank), "selected_indices": selected_indices_to_str(indices), "val_acc": float(val_out["acc"]), "test_acc": float(test_out["acc"]), "val_loss": float(val_out["loss"]), "test_loss": float(test_out["loss"])})
    return pd.DataFrame(rows)


def compare_to_oracle(oracle_df: pd.DataFrame, subset_map: Dict[str, Dict[int, List[int]]]) -> pd.DataFrame:
    if oracle_df is None or oracle_df.empty:
        return pd.DataFrame()
    rows = []
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


def save_ranking_plot(summary_methods: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    if summary_methods.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df = summary_methods.sort_values("spearman_mean", ascending=False)
    ax.bar(plot_df["method"], plot_df["spearman_mean"], yerr=plot_df["spearman_std"].fillna(0.0), capsize=4)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("Spearman with LOO utility")
    ax.set_title(f"{dataset_name}: ranking fidelity")
    plt.tight_layout()
    plt.savefig(output_dir / "ranking_fidelity.png", dpi=200)
    plt.close(fig)


def save_topk_curve_plot(summary_topk: pd.DataFrame, output_dir: Path, dataset_name: str, methods: Sequence[str]) -> None:
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
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{dataset_name}: top-k retention curves")
    ax.legend(ncol=min(4, max(1, len(methods))))
    plt.tight_layout()
    plt.savefig(output_dir / "topk_retention_curves.png", dpi=200)
    plt.close(fig)


def save_delta_plot(win_rate_df: pd.DataFrame, output_dir: Path, dataset_name: str, methods: Sequence[str]) -> None:
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
    ax.set_ylabel("Test accuracy delta vs mag")
    ax.set_title(f"{dataset_name}: low-budget gain over magnitude")
    ax.legend(ncol=min(4, max(1, len(methods))))
    plt.tight_layout()
    plt.savefig(output_dir / "delta_vs_mag_curves.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    ap.add_argument("--dataset-name", type=str, default="cifar10")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0,1,2,3")
    ap.add_argument("--data-split-seed", type=int, default=123)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--target-modules", type=str, default="query,value")
    ap.add_argument("--report-methods", type=str, default="random,mag,grad,curv,v1,v2,v5,greedy_v5")
    ap.add_argument("--plot-methods", type=str, default="mag,v1,v2,v5,greedy_v5")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--early-stop-patience", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--max-test-samples", type=int, default=0)
    ap.add_argument("--topk-values", type=str, default="1,2,4,8,12,16")
    ap.add_argument("--lambda-v1", type=float, default=0.8)
    ap.add_argument("--lambda-v2", type=float, default=0.4)
    ap.add_argument("--alpha-v4", type=float, default=0.6)
    ap.add_argument("--beta-v4", type=float, default=0.8)
    ap.add_argument("--gamma-v4", type=float, default=0.2)
    ap.add_argument("--alpha-v5", type=float, default=1.0)
    ap.add_argument("--beta-v5", type=float, default=1.0)
    ap.add_argument("--gamma-v5", type=float, default=1.0)
    ap.add_argument("--js-temperature", type=float, default=2.0)
    ap.add_argument("--save-single-direction-cache", type=int, default=1)
    ap.add_argument("--single-direction-cache-format", type=str, default="logits_and_probs", choices=["probs", "logits_and_probs"])
    ap.add_argument("--enable-greedy-selector", type=int, default=1)
    ap.add_argument("--greedy-base-method", type=str, default="v5")
    ap.add_argument("--greedy-similarity", type=str, default="cosine", choices=["cosine", "js", "corr"])
    ap.add_argument("--greedy-lambda", type=float, default=0.10)
    ap.add_argument("--greedy-method-name", type=str, default="greedy_v5")
    ap.add_argument("--enable-exact-subset-oracle", type=int, default=0)
    ap.add_argument("--oracle-max-k", type=int, default=4)
    ap.add_argument("--oracle-max-seeds", type=int, default=1)
    return ap.parse_args()


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
    if args.topk_values.strip().lower() == "auto":
        topk_values = default_topk_values_for_rank(args.lora_r)
    else:
        topk_values = sorted(set(int(s.strip()) for s in args.topk_values.split(",") if s.strip()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Methods to evaluate: %s", methods)

    prepared, image_col, label_col, label_names = prepare_vision_dataset(args.dataset_name, args.data_split_seed, args.val_ratio, args.test_ratio, args.cache_dir, logger)
    processor = AutoImageProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    all_method_rows, all_topk_rows, all_direction_rows, full_rows, seed_resource_rows = [], [], [], [], []
    oracle_compare_rows = []
    for seed_idx, seed in enumerate(seeds):
        seed_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info("================ Seed %d ================", seed)
        write_status(output_dir, {"stage": f"seed_{seed}", "ok": True})
        set_seed(seed)
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = build_dataloaders(prepared, processor, image_col, label_col, args.batch_size, args.max_train_samples if args.max_train_samples > 0 else None, args.max_val_samples if args.max_val_samples > 0 else None, args.max_test_samples if args.max_test_samples > 0 else None, seed)
        logger.info("Dataset sizes | train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
        model = AutoModelForImageClassification.from_pretrained(args.model_name, num_labels=len(label_names), ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id, cache_dir=args.cache_dir)
        lora_refs = replace_target_modules_with_lora(model, [s.strip() for s in args.target_modules.split(",") if s.strip()], rank=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, logger=logger)
        freeze_non_lora_backbone(model, logger)
        model.to(device)

        t_train0 = time.perf_counter()
        history_df = train_one_seed(model, lora_refs, train_loader, val_loader, device, args.learning_rate, args.weight_decay, args.epochs, args.warmup_ratio, args.early_stop_patience, logger)
        train_seconds = time.perf_counter() - t_train0
        history_df.to_csv(output_dir / f"history_seed{seed}.csv", index=False)

        t_score0 = time.perf_counter()
        direction_df, full_meta, cache = compute_direction_table(model, lora_refs, train_loader, val_loader, test_loader, device, args.lora_r, logger, args.lambda_v1, args.lambda_v2, args.alpha_v4, args.beta_v4, args.gamma_v4, args.alpha_v5, args.beta_v5, args.gamma_v5, args.js_temperature)
        scoring_seconds = time.perf_counter() - t_score0
        direction_df["seed"] = seed
        direction_df.to_csv(output_dir / f"direction_scores_seed{seed}.csv", index=False)
        all_direction_rows.append(direction_df)

        similarity = pairwise_similarity_from_probs(cache["val_probs"], metric=args.greedy_similarity)
        if args.save_single_direction_cache:
            save_single_direction_cache(output_dir, seed, cache["labels"], cache["val_logits"], cache["val_probs"], similarity)
        subset_map, method_df = build_method_subsets(direction_df, methods, topk_values, similarity if args.enable_greedy_selector else None, args.greedy_base_method, args.greedy_lambda, args.greedy_method_name, seed)
        topk_df = evaluate_method_subsets(model, lora_refs, subset_map, val_loader, test_loader, device, args.lora_r)
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
    summary_topk = all_topk_df.groupby(["method", "topk", "keep_ratio"]).agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"), test_loss_mean=("test_loss", "mean"), test_loss_std=("test_loss", "std"), val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"), val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std")).reset_index()
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
    save_ranking_plot(summary_methods, output_dir, args.dataset_name)
    save_topk_curve_plot(summary_topk, output_dir, args.dataset_name, plot_methods)
    save_delta_plot(win_rate_df, output_dir, args.dataset_name, plot_methods)
    logger.info("Saved summary outputs, oracle outputs (if enabled), prediction caches (if enabled), and resource logs.")
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
