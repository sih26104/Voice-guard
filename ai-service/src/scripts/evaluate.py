"""Evaluation for a trained VoiceGuard head on the MLAAD test split.

Loads a checkpoint saved by ``train_head.py``, rebuilds the frozen-encoder
model, runs inference over the requested split and reports accuracy,
precision, recall, F1, the confusion matrix and class counts.

Run (from ``ai-service/``)::

    .venv/Scripts/python.exe -m src.scripts.evaluate --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset.mlaad import MlaadDataset
from model.classifier import load_voiceguard_model

from .common import evaluate_dataset, load_checkpoint


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VoiceGuard head on an MLAAD split."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_head.pt saved by train_head.py")
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset root containing splits.csv")
    parser.add_argument("--split", default="test",
                        help="Split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=None,
                        help="Optional path for a JSON metrics report")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    def build_model(encoder_name: str):
        return load_voiceguard_model(encoder_name=encoder_name)

    model, config, metadata = load_checkpoint(args.checkpoint, build_model)
    print(f"checkpoint: {args.checkpoint} (epoch {metadata.get('epoch')})")
    print(f"encoder (frozen): {model.encoder_name}")
    print(f"training run used: dataset={config.dataset_path} "
          f"batch_size={config.batch_size} lr={config.learning_rate}")
    print()

    dataset = MlaadDataset(args.dataset, split=args.split)
    print(f"evaluating split {args.split!r}: {len(dataset)} samples "
          f"(batch_size={args.batch_size})")

    metrics = evaluate_dataset(model, dataset, batch_size=args.batch_size)

    print()
    print("=== Results (prototype baseline — NOT production performance) ===")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f} (macro)")
    print(f"recall   : {metrics['recall']:.4f} (macro)")
    print(f"f1       : {metrics['f1']:.4f} (macro)")
    if "loss" in metrics:
        print(f"loss     : {metrics['loss']:.4f}")
    print()
    print("per class:")
    for name, values in metrics["per_class"].items():
        print(f"  {name:8s} precision={values['precision']:.4f} "
              f"recall={values['recall']:.4f} f1={values['f1']:.4f} "
              f"support={values['support']}")
    print()
    print(f"class counts: {metrics['class_counts']}")
    cm = metrics["confusion_matrix"]
    print("confusion matrix (rows=true [bonafide, spoof], cols=pred):")
    for row in cm:
        print("  ", row.tolist())

    print()
    print("NOTE: the current 1,000-file MLAAD subset splits were NOT made")
    print("group-aware: the same TTS systems appear in train and test, so")
    print("these numbers measure the prototype pipeline, NOT unseen-TTS")
    print("generalization. Do not treat them as production performance.")

    if args.output:
        report = {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "split": args.split,
            "encoder_name": model.encoder_name,
            "epoch": metadata.get("epoch"),
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "per_class": metrics["per_class"],
                "class_counts": metrics["class_counts"],
                "confusion_matrix": cm.tolist(),
            },
            "caveat": (
                "TTS systems overlap between train and test in this subset; "
                "results are a prototype baseline, not production performance."
            ),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"report written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
