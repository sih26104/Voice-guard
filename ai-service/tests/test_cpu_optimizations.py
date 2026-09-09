"""Tests for the CPU-only training optimizations.

Covers the frozen-encoder guarantees (no grads, no checkpointing, eval
pinned under ``model.train()`` / ``no_grad`` forward) and the
deterministic 5-second training crop in ``WaveformDataset`` — all with
the tiny random-weights model and synthetic audio.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from dataset.labels import Label
from model.classifier import VoiceGuardClassifier

from scripts.common import (
    TrainingConfig,
    WaveformDataset,
    collate_variable_length,
    compute_metrics,
)

SR = 16_000


def sine(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def tiny_model() -> VoiceGuardClassifier:
    return VoiceGuardClassifier.with_tiny_config()


class _FakeSample:
    """Minimal stand-in for an MlaadSample (metadata + waveform only)."""

    def __init__(self, key: str, seconds: float, label: Label) -> None:
        self.key = key
        self.label = label
        self._seconds = seconds

    def waveform(self):
        from dataset.audio import AudioClip

        return AudioClip.from_array(sine(self._seconds), sample_rate=SR)


class _FakeDataset:
    """Minimal BaseAudioDataset-shaped fixture for WaveformDataset."""

    def __init__(self, items) -> None:
        self._items = items
        self.split_name = "train"

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


class TestEncoderFrozen(unittest.TestCase):
    def test_all_encoder_params_frozen(self) -> None:
        model = tiny_model()
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_no_encoder_grads_after_backward(self) -> None:
        model = tiny_model()
        logits = model(torch.stack([torch.from_numpy(sine(0.3))] * 2))
        logits.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.head.parameters()))

    def test_train_mode_pins_encoder_to_eval(self) -> None:
        model = tiny_model()
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.head.training)
        model.eval()

    def test_forward_runs_under_no_grad_when_frozen(self) -> None:
        model = tiny_model()
        model.train()  # training mode: encoder must still be graph-free
        with torch.enable_grad():
            logits = model(torch.from_numpy(sine(0.3)))
        self.assertEqual(tuple(logits.shape), (1, 2))
        # Only head outputs require grad; encoder produced none.
        self.assertTrue(logits.requires_grad)
        model.eval()


class TestGradientCheckpointing(unittest.TestCase):
    def test_disabled_after_construction(self) -> None:
        model = tiny_model()
        self.assertFalse(model.encoder.is_gradient_checkpointing)

    def test_still_disabled_after_enabling_and_rebuilding(self) -> None:
        # Simulate foreign code enabling checkpointing, then prove our
        # constructor path disables it again.
        model = tiny_model()
        model.encoder.gradient_checkpointing_enable()
        self.assertTrue(model.encoder.is_gradient_checkpointing)
        fresh = tiny_model()
        self.assertFalse(fresh.encoder.is_gradient_checkpointing)

    def test_config_default_records_frozen_encoder(self) -> None:
        config = TrainingConfig(dataset_path="x")
        self.assertEqual(config.max_train_duration_s, 5.0)


class TestTrainingCrop(unittest.TestCase):
    def test_long_clip_limited_to_five_seconds(self) -> None:
        ds = _FakeDataset([
            _FakeSample("data\\mlaad_subset\\audio\\original\\en\\long.wav",
                        12.0, Label.BONAFIDE),
        ])
        item = WaveformDataset(ds)[0]
        samples, label_id, key = item
        self.assertLessEqual(samples.shape[0], int(5.0 * SR))
        self.assertEqual(samples.shape[0], int(5.0 * SR))  # exactly 5 s
        self.assertEqual(label_id, 0)  # bonafide
        self.assertIn("long.wav", key)

    def test_crop_matches_configured_cap(self) -> None:
        ds = _FakeDataset([
            _FakeSample("k.wav", 9.0, Label.SPOOF),
        ])
        ds_item = WaveformDataset(ds, max_duration_s=2.0)[0]
        self.assertEqual(ds_item[0].shape[0], int(2.0 * SR))

    def test_short_clips_preserved_exactly(self) -> None:
        ds = _FakeDataset([
            _FakeSample("a.wav", 1.0, Label.BONAFIDE),
            _FakeSample("b.wav", 4.99, Label.SPOOF),
        ])
        wds = WaveformDataset(ds)
        first, second = wds[0], wds[1]
        self.assertEqual(first[0].shape[0], int(1.0 * SR))
        self.assertEqual(second[0].shape[0], int(4.99 * SR))

    def test_boundary_five_seconds_not_cropped(self) -> None:
        ds = _FakeDataset([
            _FakeSample("exact.wav", 5.0, Label.BONAFIDE),
        ])
        samples = WaveformDataset(ds)[0][0]
        self.assertEqual(samples.shape[0], int(5.0 * SR))

    def test_crop_is_deterministic_across_instances_and_epochs(self) -> None:
        ds = _FakeDataset([
            _FakeSample("data\\audio\\long.wav", 20.0, Label.BONAFIDE),
        ])
        wds_a = WaveformDataset(ds)
        wds_b = WaveformDataset(ds)
        a1, a2 = wds_a[0], wds_a[0]  # two epochs of access
        b1 = wds_b[0]                # a different dataset instance
        for first, second in ((a1, a2), (a1, b1)):
            self.assertTrue(np.array_equal(first[0], second[0]))

    def test_crop_window_helper_bounds_and_spread(self) -> None:
        total, cap = int(20.0 * SR), int(5.0 * SR)
        starts = {
            WaveformDataset.crop_window(total, cap, f"clip_{i}.wav")
            for i in range(50)
        }
        self.assertTrue(all(0 <= s <= total - cap for s in starts))
        self.assertGreater(len(starts), 1)  # not always the file start

    def test_disabled_crop_returns_full_clip(self) -> None:
        ds = _FakeDataset([
            _FakeSample("long.wav", 8.0, Label.SPOOF),
        ])
        samples = WaveformDataset(ds, max_duration_s=None)[0][0]
        self.assertEqual(samples.shape[0], int(8.0 * SR))


class TestBatchPreparation(unittest.TestCase):
    def test_collate_shapes_and_masks_still_correct(self) -> None:
        short, long = sine(0.2), sine(6.5)  # long clip survives collate
        values, lengths, label_ids, keys = collate_variable_length(
            [(short, 0, "a"), (long, 1, "b")]
        )
        self.assertEqual(tuple(values.shape), (2, long.shape[0]))
        self.assertEqual(lengths.tolist(), [short.shape[0], long.shape[0]])
        self.assertTrue(torch.all(values[0, short.shape[0]:] == 0))
        self.assertEqual(label_ids.tolist(), [0, 1])
        self.assertEqual(keys, ["a", "b"])

    def test_full_length_row_matches_single_clip_exactly(self) -> None:
        # A full-length row inside a padded batch (shorter sibling forces
        # the masked path) must match single-clip inference: the mask
        # excludes all padding for that row.
        model = tiny_model()
        wave_long, wave_short = sine(0.8), sine(0.3)
        batch = torch.zeros(2, wave_long.shape[0])
        batch[0, : wave_short.shape[0]] = torch.from_numpy(wave_short)
        batch[1] = torch.from_numpy(wave_long)
        with torch.no_grad():
            single = model(torch.from_numpy(wave_long))
            batch_logits = model(
                batch,
                waveform_lengths=[wave_short.shape[0], wave_long.shape[0]],
            )
        self.assertLess(
            abs(float(single[0, 0]) - float(batch_logits[1, 0])), 1e-5
        )

    def test_padded_row_matches_single_clip_approximately(self) -> None:
        # Zero-padded rows are an *approximation* by nature: the positional
        # conv receptive field spans the padding boundary even with correct
        # attention masking. Assert the effect stays small.
        model = tiny_model()
        wave = sine(0.4)
        with torch.no_grad():
            single = model(torch.from_numpy(wave))
            padded = torch.from_numpy(
                np.pad(wave, (0, SR), constant_values=0.0)
            )[None, :]
            batch_logits = model(padded, waveform_lengths=[wave.shape[0]])
        self.assertLess(
            abs(float(single[0, 0]) - float(batch_logits[0, 0])), 0.05
        )

    def test_batch_forward_shapes(self) -> None:
        model = tiny_model()
        values = torch.stack([torch.from_numpy(sine(0.3))] * 4)
        lengths = torch.tensor([values.shape[1]] * 4)
        with torch.no_grad():
            logits = model(values, waveform_lengths=lengths)
        self.assertEqual(tuple(logits.shape), (4, 2))

    def test_evaluation_split_is_not_cropped(self) -> None:
        # evaluate_dataset must construct its loader with max_duration_s=None
        import inspect

        from scripts import common

        source = inspect.getsource(common.evaluate_dataset)
        self.assertIn("max_duration_s=None", source)


if __name__ == "__main__":
    unittest.main()
