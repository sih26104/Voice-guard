"""Unit tests for :mod:`model` (VoiceGuard classifier).

Lightweight by design: most tests run against a tiny random-weights
model (~30k params, CPU, milliseconds, no downloads). The real
pretrained checkpoint is exercised by one self-skipping test that runs
only when the model is already in the local HF cache.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from dataset.labels import Label
from model import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    MODEL_SAMPLE_RATE,
    PRETRAINED_MODEL_NAME,
    SpoofPrediction,
    VoiceGuardClassifier,
    predict,
    prepare_waveform,
)
from model.classifier import load_voiceguard_model

SR = MODEL_SAMPLE_RATE


def sine(seconds: float, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Synthesize a mono 16 kHz sine waveform (no dataset needed)."""
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def tiny_model() -> VoiceGuardClassifier:
    model = VoiceGuardClassifier.with_tiny_config()
    model.eval()
    return model


class TestLabelMapping(unittest.TestCase):
    def test_mapping_covers_both_classes(self) -> None:
        self.assertEqual(set(LABEL_TO_ID), {Label.BONAFIDE, Label.SPOOF})
        self.assertEqual(sorted(ID_TO_LABEL), [0, 1])

    def test_mapping_is_deterministic(self) -> None:
        self.assertEqual(LABEL_TO_ID[Label.BONAFIDE], 0)
        self.assertEqual(LABEL_TO_ID[Label.SPOOF], 1)

    def test_no_new_label_enum_created(self) -> None:
        # The model must reuse the dataset taxonomy, not redefine labels.
        self.assertTrue(all(isinstance(key, Label) for key in LABEL_TO_ID))
        self.assertTrue(all(isinstance(value, Label) for value in ID_TO_LABEL.values()))


class TestModelConstruction(unittest.TestCase):
    def test_tiny_model_builds_and_freezes_encoder(self) -> None:
        model = tiny_model()
        counts = model.parameter_counts()
        self.assertGreater(counts["total"], 0)
        self.assertLess(counts["trainable"], counts["total"])
        encoder_frozen = all(not p.requires_grad for p in model.encoder.parameters())
        self.assertTrue(encoder_frozen)
        head_trainable = all(p.requires_grad for p in model.head.parameters())
        self.assertTrue(head_trainable)

    def test_default_model_name(self) -> None:
        self.assertEqual(PRETRAINED_MODEL_NAME, "facebook/wav2vec2-base")

    def test_tiny_config_head_uses_canonical_labels(self) -> None:
        model = tiny_model()
        self.assertEqual(model.head[-1].out_features, len(Label))


class TestPrepareWaveform(unittest.TestCase):
    def test_1d_becomes_batch_of_one(self) -> None:
        out = prepare_waveform(sine(0.5))
        self.assertEqual(tuple(out.shape), (1, SR // 2))
        self.assertEqual(out.dtype, torch.float32)

    def test_2d_passthrough(self) -> None:
        batch = np.stack([sine(0.1), sine(0.1)])
        out = prepare_waveform(batch)
        self.assertEqual(tuple(out.shape), (2, int(0.1 * SR)))

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            prepare_waveform(np.array([], dtype=np.float32))

    def test_rejects_non_finite(self) -> None:
        bad = sine(0.1)
        bad[3] = np.nan
        with self.assertRaises(ValueError):
            prepare_waveform(bad)

    def test_rejects_3d(self) -> None:
        with self.assertRaises(ValueError):
            prepare_waveform(np.zeros((2, 2, 10), dtype=np.float32))


class TestForwardAndProbabilities(unittest.TestCase):
    def test_output_shape_single(self) -> None:
        model = tiny_model()
        with torch.no_grad():
            logits = model(sine(1.0))
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_output_shape_batch(self) -> None:
        model = tiny_model()
        batch = np.stack([sine(0.5), sine(0.5)])
        with torch.no_grad():
            logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_probabilities_valid(self) -> None:
        model = tiny_model()
        probs = model.probabilities(sine(0.4))
        self.assertEqual(tuple(probs.shape), (1, 2))
        self.assertTrue(torch.isfinite(probs).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_variable_length_inputs_all_work(self) -> None:
        model = tiny_model()
        for seconds in (0.5, 1.0, 2.0):
            with self.subTest(seconds=seconds):
                probs = model.probabilities(sine(seconds))
                self.assertEqual(tuple(probs.shape), (1, 2))
                self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_encoder_frozen_no_grad_leak(self) -> None:
        model = tiny_model()
        out = model(sine(0.3))
        out.sum().backward()
        head_grads = [p.grad for p in model.head.parameters()]
        self.assertTrue(any(g is not None for g in head_grads))
        encoder_grads = [p.grad for p in model.encoder.parameters()]
        self.assertTrue(all(g is None for g in encoder_grads))


class TestPredictInterface(unittest.TestCase):
    def test_predict_returns_spoof_prediction(self) -> None:
        model = tiny_model()
        result = predict(model, sine(0.5))
        self.assertIsInstance(result, SpoofPrediction)
        self.assertIn(result.label, {"bonafide", "spoof"})
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=5)
        self.assertEqual(
            result.spoof_probability, result.probabilities["spoof"]
        )
        self.assertGreaterEqual(result.confidence, 0.5)

    def test_predict_matches_probabilities_argmax(self) -> None:
        model = tiny_model()
        probs = model.probabilities(sine(0.5))[0]
        result = predict(model, sine(0.5))
        expected = "bonafide" if probs[0] >= probs[1] else "spoof"
        self.assertEqual(result.label, expected)

    def test_predict_rejects_bad_input(self) -> None:
        model = tiny_model()
        with self.assertRaises(ValueError):
            predict(model, np.array([], dtype=np.float32))


class TestRealModel(unittest.TestCase):
    """Exercises the real pretrained checkpoint only if already cached."""

    def setUp(self) -> None:
        try:
            from transformers import Wav2Vec2Config

            Wav2Vec2Config.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)
        except Exception:
            self.skipTest("facebook/wav2vec2-base not in local HF cache")

    def test_pretrained_forward_pass(self) -> None:
        model = load_voiceguard_model()
        counts = model.parameter_counts()
        self.assertGreater(counts["total"], 90_000_000)
        self.assertLess(counts["trainable"], 1_000_000)
        probs = model.probabilities(sine(1.0))
        self.assertEqual(tuple(probs.shape), (1, 2))
        self.assertTrue(torch.isfinite(probs).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
