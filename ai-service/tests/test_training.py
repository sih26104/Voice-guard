"""Lightweight tests for the training/evaluation pipeline.

Everything runs on the tiny random-weights model with synthetic waveforms
and temp-dir fixtures — **no dataset training, no downloads, no real
checkpoint loading**. A full-epoch smoke test over a real split is
deliberately excluded per task scope.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset.labels import Label
from model.classifier import VoiceGuardClassifier
from model.config import ID_TO_LABEL, LABEL_TO_ID

from scripts.common import (
    TrainingConfig,
    WaveformDataset,
    collate_variable_length,
    compute_metrics,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    set_seeds,
)

SR = 16_000


def sine(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def tiny_model() -> VoiceGuardClassifier:
    model = VoiceGuardClassifier.with_tiny_config()
    model.eval()
    return model


class TestTrainingConfig(unittest.TestCase):
    def test_roundtrip_json(self) -> None:
        config = TrainingConfig(dataset_path="data/mlaad_subset", epochs=5,
                                batch_size=8, learning_rate=3e-4)
        restored = TrainingConfig.from_json(config.to_json())
        self.assertEqual(restored, config)

    def test_defaults_are_head_only_cpu_safe(self) -> None:
        config = TrainingConfig(dataset_path="x")
        self.assertEqual(config.num_workers, 0)      # Windows-safe
        self.assertEqual(config.best_metric, "f1")
        self.assertEqual(config.label_to_id, {"bonafide": 0, "spoof": 1})

    def test_unknown_json_keys_ignored(self) -> None:
        raw = json.dumps({"dataset_path": "x", "not_a_field": 1})
        config = TrainingConfig.from_json(raw)
        self.assertEqual(config.dataset_path, "x")


class TestLabelMappingConsistency(unittest.TestCase):
    def test_checkpoint_mapping_matches_model(self) -> None:
        self.assertEqual(
            {label.value: idx for label, idx in LABEL_TO_ID.items()},
            TrainingConfig(dataset_path="x").label_to_id,
        )
        self.assertEqual(set(ID_TO_LABEL), {0, 1})


class TestCollateVariableLength(unittest.TestCase):
    def test_pads_and_reports_real_lengths(self) -> None:
        short, long = sine(0.2), sine(0.5)
        batch = [(short, 0, "a"), (long, 1, "b")]
        values, lengths, label_ids, keys = collate_variable_length(batch)
        self.assertEqual(tuple(values.shape), (2, long.shape[0]))
        self.assertTrue(torch.all(values[0, short.shape[0]:] == 0))  # padded
        self.assertEqual(lengths.tolist(), [short.shape[0], long.shape[0]])
        self.assertEqual(label_ids.tolist(), [0, 1])
        self.assertEqual(keys, ["a", "b"])

    def test_labels_are_canonical_ids(self) -> None:
        values, lengths, label_ids, _ = collate_variable_length(
            [(sine(0.1), LABEL_TO_ID[Label.SPOOF], "s")]
        )
        self.assertEqual(label_ids.tolist(), [1])


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y_true = [0, 0, 1, 1]
        metrics = compute_metrics(y_true, [0, 0, 1, 1])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"].tolist(),
                         [[2, 0], [0, 2]])
        self.assertEqual(metrics["class_counts"], {"bonafide": 2, "spoof": 2})

    def test_known_confusion(self) -> None:
        # y_true: 0 0 0 1 1 1 ; y_pred: 0 1 1 1 1 0
        metrics = compute_metrics([0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0])
        self.assertAlmostEqual(metrics["accuracy"], 3 / 6)
        bonafide = metrics["per_class"]["bonafide"]
        spoof = metrics["per_class"]["spoof"]
        self.assertAlmostEqual(bonafide["precision"], 1 / 2)   # 1 TP / (1+1 FP)
        self.assertAlmostEqual(bonafide["recall"], 1 / 3)
        self.assertAlmostEqual(spoof["recall"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"].tolist(),
                         [[1, 2], [1, 2]])

    def test_rejects_mismatched_and_empty(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([0, 1], [0])
        with self.assertRaises(ValueError):
            compute_metrics([], [])


class TestLossAndTinyTrainingStep(unittest.TestCase):
    def test_cross_entropy_decreases_over_steps(self) -> None:
        set_seeds(7)
        model = tiny_model()
        # Feed a *fixed* batch with a learnable label pattern (short clips
        # = bonafide, long = spoof) using distinct frequencies per class so
        # the head can separate them.
        wave_a, wave_b = sine(0.3, 300.0), sine(0.9, 900.0)
        values = torch.zeros(4, wave_b.shape[0])
        values[0, : wave_a.shape[0]] = torch.from_numpy(wave_a)
        values[1, : wave_a.shape[0]] = torch.from_numpy(wave_a)
        values[2] = torch.from_numpy(wave_b)
        values[3] = torch.from_numpy(wave_b)
        lengths = torch.tensor([wave_a.shape[0], wave_a.shape[0],
                                wave_b.shape[0], wave_b.shape[0]])
        labels = torch.tensor([0, 0, 1, 1])

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=5e-3
        )
        criterion = nn.CrossEntropyLoss()
        model.train()

        first_loss = None
        last_loss = None
        for _ in range(30):
            optimizer.zero_grad()
            logits = model(values, waveform_lengths=lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            first_loss = first_loss if first_loss is not None else float(loss.detach())
            last_loss = float(loss.detach())
        self.assertIsNotNone(first_loss)
        self.assertLess(last_loss, first_loss)

    def test_only_head_receives_gradients(self) -> None:
        model = tiny_model()
        criterion = nn.CrossEntropyLoss()
        values = torch.stack([torch.from_numpy(sine(0.3)), torch.from_numpy(sine(0.3))])
        labels = torch.tensor([0, 1])
        logits = model(values)
        criterion(logits, labels).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))


class TestCheckpointSaveLoad(unittest.TestCase):
    def test_roundtrip_head_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_head.pt"
            model_a = tiny_model()
            config = TrainingConfig(dataset_path="data/mlaad_subset", epochs=2)
            metrics = {"accuracy": 0.9, "f1": 0.8}
            save_checkpoint(path, model_a, config, epoch=2, metrics=metrics)

            model_b, restored_config, metadata = load_checkpoint(
                path, lambda encoder_name: tiny_model()
            )
            self.assertEqual(restored_config, config)
            self.assertEqual(metadata["epoch"], 2)
            for (k_a, v_a), (k_b, v_b) in zip(
                model_a.head.state_dict().items(),
                model_b.head.state_dict().items(),
            ):
                self.assertEqual(k_a, k_b)
                self.assertTrue(torch.equal(v_a, v_b))

    def test_checkpoint_payload_contains_config_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_head.pt"
            save_checkpoint(path, tiny_model(),
                            TrainingConfig(dataset_path="x"), epoch=1,
                            metrics={"accuracy": 0.5})
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("encoder_name", state)
            self.assertIn("head_state_dict", state)
            self.assertIn("config", state)
            self.assertEqual(state["label_to_id"], {"bonafide": 0, "spoof": 1})


class TestEvaluateModelLoop(unittest.TestCase):
    def test_metrics_from_mock_loader(self) -> None:
        model = tiny_model()

        def fake_loader():
            values = torch.stack([torch.from_numpy(sine(0.3)), torch.from_numpy(sine(0.3))])
            lengths = torch.tensor([values.shape[1]] * 2)
            label_ids = torch.tensor([0, 1])
            yield values, lengths, label_ids, ["k1", "k2"]

        metrics = evaluate_model(model, fake_loader())
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        total = metrics["class_counts"]["bonafide"] + metrics["class_counts"]["spoof"]
        self.assertEqual(total, 2)

    def test_loss_included_when_criterion_given(self) -> None:
        model = tiny_model()

        def fake_loader():
            values = torch.stack([torch.from_numpy(sine(0.3))])
            lengths = torch.tensor([values.shape[1]])
            yield values, lengths, torch.tensor([0]), ["k"]

        metrics = evaluate_model(model, fake_loader(), criterion=nn.CrossEntropyLoss())
        self.assertIn("loss", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)


class TestWaveformDatasetContract(unittest.TestCase):
    def test_rejects_empty_dataset(self) -> None:
        class EmptyDataset:
            split_name = "train"

            def __len__(self):
                return 0

            def __iter__(self):
                return iter(())

        with self.assertRaises(ValueError):
            WaveformDataset(EmptyDataset())


if __name__ == "__main__":
    unittest.main()
