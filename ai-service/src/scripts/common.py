"""Shared training/evaluation utilities for VoiceGuard scripts.

Everything here is CPU-safe and memory-conscious:

    * :class:`WaveformDataset` — wraps a ``BaseAudioDataset`` iterator in a
      torch ``Dataset`` that loads and preprocesses **one sample at a
      time**; no waveforms are held in RAM beyond the current batch.
    * :func:`collate_variable_length` — pads a batch to its longest clip
      and returns real per-item lengths for masked pooling.
    * :func:`compute_metrics` — accuracy/precision/recall/F1/confusion.
    * :func:`evaluate_dataset` / :func:`evaluate_model` — shared loops.
    * :func:`save_checkpoint` / :func:`load_checkpoint` — head + config.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from dataset.base_audio_dataset import BaseAudioDataset
from dataset.labels import Label
from model.classifier import VoiceGuardClassifier
from model.config import ID_TO_LABEL, LABEL_TO_ID

__all__ = [
    "TrainingConfig",
    "WaveformDataset",
    "collate_variable_length",
    "compute_metrics",
    "evaluate_dataset",
    "evaluate_model",
    "save_checkpoint",
    "load_checkpoint",
    "set_seeds",
]

NUM_CLASSES = len(Label)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Everything needed to reproduce a head-only training run."""

    dataset_path: str
    train_split: str = "train"
    validation_split: str = "validation"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0          # keep 0 on Windows/CPU: simplest + safe
    seed: int = 26104
    encoder_name: str = "facebook/wav2vec2-base"
    best_metric: str = "f1"       # checkpoint selection metric
    label_to_id: dict = field(default_factory=lambda: {
        label.value: idx for label, idx in LABEL_TO_ID.items()
    })

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "TrainingConfig":
        data = json.loads(raw)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Dataset / batching
# ---------------------------------------------------------------------------


class WaveformDataset(Dataset):
    """Torch dataset over a ``BaseAudioDataset`` — loads one item at a time.

    ``__getitem__`` preprocesses exactly one sample (file decode, mono,
    resample, duration check) and returns it; nothing is cached, so RAM
    usage stays at one clip regardless of dataset size.
    """

    def __init__(self, dataset: BaseAudioDataset) -> None:
        self._samples = list(dataset)  # metadata only (paths/labels); no audio
        if not self._samples:
            raise ValueError(f"Dataset split {dataset.split_name!r} is empty.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        clip = sample.waveform()  # mono float32 16 kHz, duration-validated
        return clip.samples, LABEL_TO_ID[sample.label], sample.key


def collate_variable_length(batch):
    """Pad waveforms to the batch max; return (values, lengths, labels, keys)."""
    waveforms, labels, keys = zip(*batch)
    lengths = torch.as_tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())
    values = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    for row, wave in enumerate(waveforms):
        values[row, : wave.shape[0]] = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    label_ids = torch.as_tensor(labels, dtype=torch.long)
    return values, lengths, label_ids, list(keys)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(label_ids: Sequence[int], predicted_ids: Sequence[int]) -> dict:
    """Accuracy, precision/recall/F1 (macro + per-class) and confusion matrix.

    Confusion matrix convention: ``cm[true][pred]``.
    Pure Python/NumPy — no sklearn dependency needed here.
    """
    y_true = np.asarray(label_ids, dtype=np.int64)
    y_pred = np.asarray(predicted_ids, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(
            f"Metric inputs must be non-empty and equal length, got "
            f"{y_true.shape} vs {y_pred.shape}."
        )

    accuracy = float((y_true == y_pred).mean())
    per_class = {}
    for class_id, name in ((0, Label.BONAFIDE.value), (1, Label.SPOOF.value)):
        tp = int(((y_pred == class_id) & (y_true == class_id)).sum())
        fp = int(((y_pred == class_id) & (y_true != class_id)).sum())
        fn = int(((y_pred != class_id) & (y_true == class_id)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int((y_true == class_id).sum())}
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))

    cm = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        cm[truth, pred] += 1

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,  # cm[true][pred]
        "class_counts": {
            Label.BONAFIDE.value: int((y_true == 0).sum()),
            Label.SPOOF.value: int((y_true == 1).sum()),
        },
    }


# ---------------------------------------------------------------------------
# Evaluation loops (shared by training-time validation and evaluate.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: VoiceGuardClassifier,
    loader: Iterable,
    criterion: Optional[nn.Module] = None,
) -> dict:
    """Run one full pass over ``loader``; return loss (optional) + metrics."""
    model.eval()
    all_labels: list = []
    all_preds: list = []
    total_loss = 0.0
    total_items = 0
    for values, lengths, label_ids, _keys in loader:
        logits = model(values, waveform_lengths=lengths)
        if criterion is not None:
            loss = criterion(logits, label_ids)
            total_loss += float(loss.item()) * label_ids.shape[0]
            total_items += label_ids.shape[0]
        all_labels.extend(int(v) for v in label_ids.tolist())
        all_preds.extend(int(v) for v in logits.argmax(dim=-1).tolist())
    metrics = compute_metrics(all_labels, all_preds)
    if criterion is not None and total_items:
        metrics["loss"] = total_loss / total_items
    return metrics


def evaluate_dataset(
    model: VoiceGuardClassifier,
    dataset: BaseAudioDataset,
    batch_size: int = 4,
    criterion: Optional[nn.Module] = None,
    num_workers: int = 0,
) -> dict:
    """Evaluate ``model`` over an entire ``BaseAudioDataset`` split."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        WaveformDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=num_workers,
    )
    return evaluate_model(model, loader, criterion=criterion)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: VoiceGuardClassifier,
    config: TrainingConfig,
    epoch: int,
    metrics: dict,
) -> Path:
    """Save head weights + encoder name + training config + metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "encoder_name": model.encoder_name,
        "head_state_dict": {k: v for k, v in model.head.state_dict().items()},
        "config": json.loads(config.to_json()),
        "epoch": epoch,
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "label_to_id": {label.value: idx for label, idx in LABEL_TO_ID.items()},
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    path: Path,
    build_model,
) -> tuple:
    """Rebuild the model via ``build_model(encoder_name)`` and load the head.

    Returns ``(model, config: TrainingConfig, metadata: dict)``. Only the
    head weights are restored — the encoder comes from the pretrained
    checkpoint named inside the file, matching the frozen-encoder design.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(state["encoder_name"])
    model.head.load_state_dict(state["head_state_dict"])
    model.eval()
    config = TrainingConfig.from_json(json.dumps(state["config"]))
    metadata = {"epoch": state.get("epoch"), "metrics": state.get("metrics", {})}
    return model, config, metadata


def set_seeds(seed: int) -> None:
    """Seed python/numpy/torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
