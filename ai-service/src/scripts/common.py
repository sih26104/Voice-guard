"""Shared training/evaluation utilities for VoiceGuard scripts.

Everything here is CPU-safe and memory-conscious:

    * :class:`WaveformDataset` — wraps a ``BaseAudioDataset`` iterator in a
      torch ``Dataset`` that loads and preprocesses **one sample at a
      time**; no waveforms are held in RAM beyond the current batch. For
      training, clips longer than ``max_duration_s`` (default 5 s) are
      deterministically cropped so CPU epochs stay practical.
    * :func:`collate_variable_length` — pads a batch to its longest clip
      and returns real per-item lengths for masked pooling.
    * :func:`compute_metrics` — accuracy/precision/recall/F1/confusion
      plus balanced accuracy, ROC-AUC (from spoof probabilities), FPR/FNR
      and per-spoof-class metrics.
    * :func:`evaluate_dataset` / :func:`evaluate_model` — shared loops.
    * :func:`save_checkpoint` / :func:`load_checkpoint` — head + config.
"""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import Dataset

from dataset.base_audio_dataset import BaseAudioDataset
from dataset.labels import Label
from model.classifier import VoiceGuardClassifier
from model.config import ID_TO_LABEL, LABEL_TO_ID

__all__ = [
    "TrainingConfig",
    "confusion_matrix_rows",
    "WaveformDataset",
    "collate_variable_length",
    "compute_metrics",
    "evaluate_dataset",
    "evaluate_model",
    "save_checkpoint",
    "load_checkpoint",
    "set_seeds",
    "spoof_probabilities_only",
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
    max_train_duration_s: float = 5.0  # training-only crop cap; None disables
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

    Args:
        dataset: The ``BaseAudioDataset`` split to iterate.
        max_duration_s: If set, clips longer than this are deterministically
            cropped to this many seconds (see :meth:`crop_window`). Used for
            CPU-feasible training; short clips are preserved untouched.
    """

    #: Crop length cap (seconds) for training batches.
    MAX_TRAIN_DURATION_S = 5.0

    def __init__(
        self,
        dataset: BaseAudioDataset,
        max_duration_s: Optional[float] = MAX_TRAIN_DURATION_S,
    ) -> None:
        self._samples = list(dataset)  # metadata only (paths/labels); no audio
        self._max_duration_s = max_duration_s
        if not self._samples:
            raise ValueError(f"Dataset split {dataset.split_name!r} is empty.")

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def crop_window(total_samples: int, max_samples: int, key: str, seed: int = 0) -> int:
        """Deterministic crop start for a clip longer than the cap.

        The window is derived from a stable hash of the sample key (plus a
        global seed), so every access to the same clip yields the same
        5-second segment across epochs and runs — reproducible training —
        while different clips spread across their full lengths instead of
        always cropping the file start (which is often silence).
        """
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        max_start = total_samples - max_samples
        return int.from_bytes(digest[:4], "little") % (max_start + 1)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        clip = sample.waveform()  # mono float32 16 kHz, duration-validated
        if (
            self._max_duration_s is not None
            and clip.duration > self._max_duration_s
        ):
            max_samples = int(round(self._max_duration_s * clip.sample_rate))
            start = self.crop_window(
                clip.num_samples, max_samples, sample.key
            )
            samples = clip.samples[start : start + max_samples]
        else:
            samples = clip.samples
        return samples, LABEL_TO_ID[sample.label], sample.key


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


def confusion_matrix_rows(confusion_matrix) -> list:
    """Confusion matrix (ndarray or nested lists) -> list of plain rows.

    Tolerant helper for display/report paths: ``compute_metrics`` yields a
    NumPy array, while JSON round-trips (checkpoints, summary files) turn
    it into nested Python lists — both must print without crashing.
    """
    return [[int(value) for value in row] for row in confusion_matrix]


def spoof_probabilities_only(spoof_probabilities) -> np.ndarray:
    """Coerce a spoof-probability sequence into a finite ``float64`` array.

    Accepts lists/arrays/tensors of raw sigmoid probabilities **or** the
    per-batch ``model(values).exp()[:, 1]`` style values (which are also
    just sigmoid probabilities). Raises ``ValueError`` on shape mismatch
    or NaN/Inf — a corrupted probability vector must never silently
    degrade ROC-AUC.
    """
    probs = np.asarray(spoof_probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        raise ValueError("Spoof probability vector is empty.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Spoof probabilities contain NaN or infinite values.")
    return probs


def _roc_auc_score_safe(y_true: np.ndarray, spoof_probs: Optional[np.ndarray]) -> Optional[float]:
    """ROC-AUC from the *continuous* spoof probability — or ``None``.

    Defined only when both classes are present in ``y_true``: with a
    single class sklearn returns ``nan`` (and warns), which we refuse to
    surface as a number. ``None`` is therefore also returned whenever
    probabilities were not collected. Undefined-metric sklearn warnings
    are suppressed for the single-class path we deliberately guard.
    """
    if spoof_probs is None or len(spoof_probs) != len(y_true):
        return None
    if int((y_true == 0).sum()) == 0 or int((y_true == 1).sum()) == 0:
        return None  # undefined for a single class — do not fabricate a value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return float(roc_auc_score(y_true, spoof_probs))


def compute_metrics(
    label_ids: Sequence[int],
    predicted_ids: Sequence[int],
    spoof_probabilities: Optional[Sequence[float]] = None,
) -> dict:
    """Accuracy, precision/recall/F1 (macro + per-class) and confusion matrix.

    Extended metrics (backward-compatible additions — every key that
    existed before is unchanged):

    * ``balanced_accuracy`` — mean of BONAFIDE recall and SPOOF recall;
      ``None`` (not a fabricated value) when only one class is present.
    * ``roc_auc`` — from the **continuous spoof probability** (never from
      argmax predictions); ``None`` when probabilities were not supplied
      or only one class is present.
    * ``spoof_precision`` / ``spoof_recall`` / ``spoof_f1`` — SPOOF-as-
      positive-class metrics (mirror ``per_class['spoof']``).
    * ``false_positive_rate`` — FP / (FP + TN) (bonafide misclassified).
    * ``false_negative_rate`` — FN / (FN + TP) (spoof missed).

    Confusion matrix convention: ``cm[true][pred]``.
    Pure Python/NumPy for the classic metrics; sklearn only for
    balanced accuracy and ROC-AUC.
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

    # Confusion-cell counts, cm[true][pred]: [[TN, FP], [FN, TP]].
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    # SPOOF-as-positive-class FPR/FNR. Both denominators are guaranteed
    # non-zero because per-class support has already been validated > 0.
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None

    # Balanced accuracy = mean(BONAFIDE recall, SPOOF recall). Defined
    # only when both classes are present: sklearn returns a degenerate
    # value (with a warning) on single-class input, so we refuse it.
    both_classes = tn + fp > 0 and fn + tp > 0
    if both_classes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    else:
        balanced_accuracy = None

    spoof_probs: Optional[np.ndarray] = None
    if spoof_probabilities is not None:
        spoof_probs = spoof_probabilities_only(spoof_probabilities)
        if len(spoof_probs) != len(y_true):
            raise ValueError(
                f"spoof_probabilities length {len(spoof_probs)} does not "
                f"match {len(y_true)} labels."
            )
    roc_auc = _roc_auc_score_safe(y_true, spoof_probs)

    cm = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        cm[truth, pred] += 1

    return {
        # --- pre-existing keys: unchanged -------------------------------
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
        # --- new keys ----------------------------------------------------
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,          # None unless probabilities provided
        "spoof_precision": per_class[Label.SPOOF.value]["precision"],
        "spoof_recall": per_class[Label.SPOOF.value]["recall"],
        "spoof_f1": per_class[Label.SPOOF.value]["f1"],
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


# ---------------------------------------------------------------------------
# Evaluation loops (shared by training-time validation and evaluate.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: VoiceGuardClassifier,
    loader: Iterable,
    criterion: Optional[nn.Module] = None,
    collect_probabilities: bool = True,
) -> dict:
    """Run one full pass over ``loader``; return loss (optional) + metrics.

    Spoof probabilities are collected by default (``collect_probabilities=True``)
    so ROC-AUC is computed from the continuous score; set it to ``False``
    for a loss-only pass over a huge split. The probabilities themselves
    stay **out of** the returned metrics dict: they are consumed by
    :func:`compute_metrics` and only surfaced as scalar ``roc_auc``, so
    checkpoint payloads (``save_checkpoint`` stores the metrics dict)
    remain scalar-only.
    """
    model.eval()
    all_labels: list = []
    all_preds: list = []
    all_spoof_probs: Optional[list] = [] if collect_probabilities else None
    total_loss = 0.0
    total_items = 0
    for values, lengths, label_ids, _keys in loader:
        logits = model(values, waveform_lengths=lengths)
        if criterion is not None:
            loss = criterion(logits, label_ids)
            total_loss += float(loss.item()) * label_ids.shape[0]
            total_items += label_ids.shape[0]
        if all_spoof_probs is not None:
            # Softmax over both logits, SPOOF column (class id 1): the
            # continuous score ROC-AUC must see. Same value the API's
            # ``predict`` reports as ``spoof_probability``.
            all_spoof_probs.extend(
                float(v) for v in torch.softmax(logits, dim=-1)[:, 1].tolist()
            )
        all_labels.extend(int(v) for v in label_ids.tolist())
        all_preds.extend(int(v) for v in logits.argmax(dim=-1).tolist())
    metrics = compute_metrics(all_labels, all_preds, spoof_probabilities=all_spoof_probs)
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
    """Evaluate ``model`` over an entire ``BaseAudioDataset`` split.

    Evaluation is deliberately **full-length and deterministic**: no
    cropping is applied to validation/test clips.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        WaveformDataset(dataset, max_duration_s=None),
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
    """Save head weights + encoder name + training config + metrics.

    Array-valued metric entries (e.g. the confusion matrix) are converted
    to plain lists so the ``.pt`` payload stays JSON-serializable; scalar
    and dict entries (including all evaluation metrics and ``loss``) pass
    through unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "encoder_name": model.encoder_name,
        "head_state_dict": {k: v for k, v in model.head.state_dict().items()},
        "config": json.loads(config.to_json()),
        "epoch": epoch,
        "metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in metrics.items()
            if k != "confusion_matrix"
        },
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
