"""Head-only training for the VoiceGuard spoof detector.

Trains **only** the small classification head; the pretrained Wav2Vec2
encoder stays frozen (``requires_grad=False``) and in eval mode. Audio is
loaded and preprocessed one sample at a time by ``WaveformDataset`` — no
split is ever fully resident in RAM.

Run (from ``ai-service/``)::

    .venv/Scripts/python.exe -m src.scripts.train_head --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.mlaad import MlaadDataset
from model.classifier import VoiceGuardClassifier

from .common import (
    TrainingConfig,
    WaveformDataset,
    collate_variable_length,
    confusion_matrix_rows,
    evaluate_dataset,
    save_checkpoint,
    set_seeds,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the VoiceGuard classification head (encoder frozen)."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset root containing splits.csv")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (keep 0 on Windows/CPU)")
    parser.add_argument("--max-train-seconds", type=float, default=5.0,
                        help="Deterministically crop training clips to this "
                             "duration (validation/test stay full length); "
                             "0 disables cropping")
    parser.add_argument("--seed", type=int, default=26104)
    parser.add_argument("--encoder", default="facebook/wav2vec2-base")
    parser.add_argument("--best-metric", choices=("f1", "loss"), default="f1",
                        help="Checkpoint selection metric")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the best checkpoint + training summary")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        dataset_path=str(args.dataset),
        train_split=args.train_split,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_duration_s=(args.max_train_seconds
                              if args.max_train_seconds > 0 else None),
        encoder_name=args.encoder,
        best_metric=args.best_metric,
    )


def train_head(config: TrainingConfig, output_dir: Path, log=print) -> dict:
    """Run head-only training; return the summary (best metrics + paths)."""
    set_seeds(config.seed)

    train_set = WaveformDataset(
        MlaadDataset(config.dataset_path, split=config.train_split),
        max_duration_s=config.max_train_duration_s,
    )
    val_dataset = MlaadDataset(config.dataset_path, split=config.validation_split)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=config.num_workers,
    )

    model = VoiceGuardClassifier(encoder_name=config.encoder_name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in trainable)
    log(f"encoder frozen: {frozen:,} params | head trainable: {sum(p.numel() for p in trainable):,} params")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_score = float("-inf")
    best_epoch = -1
    best_metrics = {}
    best_path = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        started = time.time()
        for values, lengths, label_ids, _keys in train_loader:
            optimizer.zero_grad()
            logits = model(values, waveform_lengths=lengths)
            loss = criterion(logits, label_ids)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * label_ids.shape[0]
            seen += label_ids.shape[0]
        train_loss = running_loss / max(seen, 1)

        val_metrics = evaluate_dataset(
            model, val_dataset, batch_size=config.batch_size,
            criterion=criterion, num_workers=config.num_workers,
        )
        elapsed = time.time() - started
        log(
            f"epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} ({elapsed:.1f}s)"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        })

        score = (val_metrics["f1"] if config.best_metric == "f1"
                 else -val_metrics.get("loss", float("inf")))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_path = save_checkpoint(
                output_dir / "best_head.pt",
                model, config, epoch, val_metrics,
            )
            log(f"  -> new best ({config.best_metric}={score:.4f}); checkpoint saved")

    summary = {
        "best_epoch": best_epoch,
        "best_metric": config.best_metric,
        "best_score": best_score if best_score != float("-inf") else None,
        "best_val_metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in best_metrics.items()
        },
        "history": history,
        "checkpoint": str(best_path) if best_path else None,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def print_training_summary(summary: dict, best_metric: str = "f1") -> None:
    """Print the end-of-training report.

    The confusion matrix may be a NumPy array (fresh metrics) or nested
    lists (JSON round-trip through checkpoint/summary files) —
    :func:`confusion_matrix_rows` handles both.
    """
    print()
    print("=== Training finished ===")
    print(f"best epoch: {summary['best_epoch']} "
          f"({best_metric}={summary['best_score']})")
    best = summary["best_val_metrics"]
    print(f"val accuracy={best['accuracy']:.4f} f1={best['f1']:.4f} "
          f"precision={best['precision']:.4f} recall={best['recall']:.4f}")
    cm = best.get("confusion_matrix")
    if cm is not None:
        print("confusion matrix (rows=true [bonafide, spoof], cols=pred):")
        for row in confusion_matrix_rows(cm):
            print("  ", row)
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"summary:    {summary.get('summary_path')}")


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    output_dir = Path(args.output_dir)

    print("Training configuration:")
    print(config.to_json())
    print()
    summary = train_head(config, output_dir)
    print_training_summary(summary, config.best_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
