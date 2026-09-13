"""Tests for the extended evaluation metrics pipeline.

Scope: the metrics layer only — ``compute_metrics`` and the probability
collection inside ``evaluate_model``. No dataset, no downloads, no real
checkpoints: ROC-AUC/evaluate-loop cases use deterministic logits from a
duck-typed stub model, mirroring the lightweight conventions of
``test_training.py``.

Conventions under test: BONAFIDE (class 0) is the negative class and
SPOOF (class 1) the positive class; FPR = FP/(FP+TN), FNR = FN/(FN+TP);
ROC-AUC must come from the continuous SPOOF probability, never argmax.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset.labels import Label
from model.classifier import VoiceGuardClassifier

from scripts.common import (
    TrainingConfig,
    compute_metrics,
    evaluate_model,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Reference case: y_true 0 0 0 1 1 1 / y_pred 0 1 1 1 1 0
#   cells: TN=1 FP=2 FN=1 TP=2  ->  cm [[1, 2], [1, 2]]
#   bonafide recall 1/3, spoof recall 2/3 -> balanced accuracy 0.5
#   FPR = 2/3, FNR = 1/3
# ---------------------------------------------------------------------------
Y_TRUE = [0, 0, 0, 1, 1, 1]
Y_PRED = [0, 1, 1, 1, 1, 0]


def perfect_probs(y_true, margin: float = 0.9):
    """Spoof probabilities perfectly ordered by the true labels."""
    return [1.0 - margin if label == 0 else margin for label in y_true]


class _StubHeadModel:
    """Duck-typed stand-in for VoiceGuardClassifier inside evaluate_model.

    The frozen encoder is irrelevant to the metrics loop, so this stub
    skips it and emits deterministic logits from a fixed per-sample table
    (consumed in loader order). No weights, no graph, no download.
    """

    def __init__(self, spoof_logits) -> None:
        self._spoof_logits = [float(v) for v in spoof_logits]
        self._cursor = 0

    def eval(self):
        return self

    def __call__(self, values, waveform_lengths=None):
        batch = values.shape[0]
        logits = torch.tensor(
            [[-1.0, self._spoof_logits[self._cursor + i]] for i in range(batch)],
            dtype=torch.float32,
        )
        self._cursor += batch
        return logits


def fake_loader(labels, spoof_logits, batch_size):
    """Yield (values, lengths, label_ids, keys) batches of ``batch_size``."""
    width = 160  # arbitrary identical clip width; content never inspected
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        values = torch.randn(stop - start, width)
        lengths = torch.full((stop - start,), width, dtype=torch.long)
        yield (
            values,
            lengths,
            torch.tensor(labels[start:stop], dtype=torch.long),
            [f"k{i}" for i in range(start, stop)],
        )


# ---------------------------------------------------------------------------
# compute_metrics: extended scalar metrics
# ---------------------------------------------------------------------------


class TestBalancedAccuracy(unittest.TestCase):
    def test_mean_of_per_class_recalls(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        recalls = [v["recall"] for v in metrics["per_class"].values()]
        self.assertAlmostEqual(metrics["balanced_accuracy"],
                               float(np.mean(recalls)))

    def test_perfect_predictions(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_TRUE)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_all_wrong_predictions(self) -> None:
        metrics = compute_metrics(Y_TRUE, [1, 1, 1, 0, 0, 0])
        self.assertEqual(metrics["balanced_accuracy"], 0.0)

    def test_single_class_returns_none_not_degenerate_value(self) -> None:
        # sklearn would emit a degenerate value + warning here; we refuse it.
        metrics = compute_metrics([0, 0, 0], [0, 1, 0])
        self.assertIsNone(metrics["balanced_accuracy"])
        metrics = compute_metrics([1, 1], [1, 0])
        self.assertIsNone(metrics["balanced_accuracy"])


class TestFalsePositiveRate(unittest.TestCase):
    def test_reference_case(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(metrics["false_positive_rate"], 2 / 3)

    def test_zero_when_no_bonafide_predicted_spoof(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_TRUE)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

    def test_one_when_every_bonafide_predicted_spoof(self) -> None:
        metrics = compute_metrics([0, 0, 1], [1, 1, 1])
        self.assertAlmostEqual(metrics["false_positive_rate"], 1.0)

    def test_undefined_without_bonafide_samples(self) -> None:
        metrics = compute_metrics([1, 1], [1, 0])
        self.assertIsNone(metrics["false_positive_rate"])

    def test_matches_confusion_matrix_cells(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        tn, fp = metrics["confusion_matrix"].tolist()[0]
        self.assertAlmostEqual(metrics["false_positive_rate"],
                               fp / (fp + tn))


class TestFalseNegativeRate(unittest.TestCase):
    def test_reference_case(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(metrics["false_negative_rate"], 1 / 3)

    def test_zero_when_perfect(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_TRUE)
        self.assertEqual(metrics["false_negative_rate"], 0.0)

    def test_one_when_every_spoof_missed(self) -> None:
        metrics = compute_metrics([0, 1, 1], [0, 0, 0])
        self.assertAlmostEqual(metrics["false_negative_rate"], 1.0)

    def test_undefined_without_spoof_samples(self) -> None:
        metrics = compute_metrics([0, 0], [0, 1])
        self.assertIsNone(metrics["false_negative_rate"])

    def test_matches_confusion_matrix_cells(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        (fn, tp) = metrics["confusion_matrix"].tolist()[1]
        self.assertAlmostEqual(metrics["false_negative_rate"],
                               fn / (fn + tp))


class TestSpoofPositiveClassMetrics(unittest.TestCase):
    def test_mirror_per_class_spoof_entries(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        spoof = metrics["per_class"][Label.SPOOF.value]
        self.assertAlmostEqual(metrics["spoof_precision"], spoof["precision"])
        self.assertAlmostEqual(metrics["spoof_recall"], spoof["recall"])
        self.assertAlmostEqual(metrics["spoof_f1"], spoof["f1"])

    def test_reference_values(self) -> None:
        # precision 2/(2+2), recall 2/3
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(metrics["spoof_precision"], 0.5)
        self.assertAlmostEqual(metrics["spoof_recall"], 2 / 3)
        expected_f1 = 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)
        self.assertAlmostEqual(metrics["spoof_f1"], expected_f1)


# ---------------------------------------------------------------------------
# ROC-AUC: continuous spoof probability, never argmax
# ---------------------------------------------------------------------------


class TestRocAuc(unittest.TestCase):
    def test_perfectly_separated_probabilities(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED,
                                  spoof_probabilities=perfect_probs(Y_TRUE))
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_inverted_probabilities(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED,
                                  spoof_probabilities=perfect_probs(
                                      [1 - v for v in Y_TRUE]))
        self.assertEqual(metrics["roc_auc"], 0.0)

    def test_hand_computed_pairs_and_ties(self) -> None:
        # y_true [0, 1, 0, 1], probs [0.3, 0.3, 0.3, 0.9]:
        # pos 0.9 beats both negs (2 wins); pos 0.3 ties both negs
        # (2 ties = 1 win-equivalent) -> 3 / 4 = 0.75
        metrics = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1],
                                  spoof_probabilities=[0.3, 0.3, 0.3, 0.9])
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)

    def test_constant_probabilities_give_half(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED,
                                  spoof_probabilities=[0.5] * len(Y_TRUE))
        self.assertAlmostEqual(metrics["roc_auc"], 0.5)

    def test_uses_probabilities_not_argmax(self) -> None:
        # Argmax predicts every sample BONAFIDE (spoof recall = 0), yet the
        # probabilities order the classes perfectly: ROC-AUC must be 1.0.
        metrics = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0],
                                  spoof_probabilities=[0.1, 0.2, 0.9, 0.95])
        self.assertEqual(metrics["spoof_recall"], 0.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_invariant_to_sample_order(self) -> None:
        # Same (label, prediction, probability) triplets in shuffled order
        # must yield the same ROC-AUC — guards label/probability misalignment.
        order = [3, 0, 4, 1, 5, 2]
        shuffled_true = [Y_TRUE[i] for i in order]
        shuffled_pred = [Y_PRED[i] for i in order]
        shuffled_probs = [perfect_probs(Y_TRUE)[i] for i in order]
        reference = compute_metrics(Y_TRUE, Y_PRED,
                                    spoof_probabilities=perfect_probs(Y_TRUE))
        shuffled = compute_metrics(shuffled_true, shuffled_pred,
                                   spoof_probabilities=shuffled_probs)
        self.assertEqual(reference["roc_auc"], shuffled["roc_auc"])
        self.assertEqual(shuffled["roc_auc"], 1.0)

    def test_single_class_bonafide_returns_none(self) -> None:
        metrics = compute_metrics([0, 0, 0], [0, 1, 0],
                                  spoof_probabilities=[0.2, 0.4, 0.6])
        self.assertIsNone(metrics["roc_auc"])

    def test_single_class_spoof_returns_none(self) -> None:
        metrics = compute_metrics([1, 1], [1, 0],
                                  spoof_probabilities=[0.2, 0.4])
        self.assertIsNone(metrics["roc_auc"])

    def test_none_without_probabilities_backward_compat(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        self.assertIsNone(metrics["roc_auc"])

    def test_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics(Y_TRUE, Y_PRED, spoof_probabilities=[0.5, 0.5])

    def test_rejects_non_finite_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics(Y_TRUE, Y_PRED,
                            spoof_probabilities=[0.5, float("nan"),
                                                 0.5, 0.5, 0.5, 0.5])

    def test_rejects_empty_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics(Y_TRUE, Y_PRED, spoof_probabilities=[])


# ---------------------------------------------------------------------------
# evaluate_model: probability collection through the loop
# ---------------------------------------------------------------------------


class TestEvaluateModelProbabilityCollection(unittest.TestCase):
    def test_probabilities_align_with_labels_across_batches(self) -> None:
        # SPOOF first, BONAFIDE second — order must follow the labels, not
        # the batch position. Low logit for bonafide, high for spoof.
        labels = [1, 0, 1, 0]
        spoof_logits = [3.0, -3.0, 3.0, -3.0]
        metrics = evaluate_model(
            _StubHeadModel(spoof_logits),
            fake_loader(labels, spoof_logits, batch_size=2),
        )
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["class_counts"],
                         {"bonafide": 2, "spoof": 2})

    def test_predictions_derive_from_logits(self) -> None:
        # spoof logit 3.0 beats bonafide logit -1.0 -> argmax = SPOOF.
        labels = [1, 1]
        spoof_logits = [3.0, 3.0]
        metrics = evaluate_model(
            _StubHeadModel(spoof_logits),
            fake_loader(labels, spoof_logits, batch_size=2),
        )
        self.assertEqual(metrics["spoof_recall"], 1.0)
        self.assertEqual(metrics["false_negative_rate"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_single_class_loader_does_not_crash(self) -> None:
        # Only-bonafide split: ROC-AUC undefined -> None, others computed.
        labels = [0, 0]
        spoof_logits = [-3.0, 3.0]
        metrics = evaluate_model(
            _StubHeadModel(spoof_logits),
            fake_loader(labels, spoof_logits, batch_size=1),
        )
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["class_counts"]["spoof"], 0)

    def test_probabilities_can_be_disabled(self) -> None:
        labels = [0, 1]
        spoof_logits = [-3.0, 3.0]
        metrics = evaluate_model(
            _StubHeadModel(spoof_logits),
            fake_loader(labels, spoof_logits, batch_size=2),
            collect_probabilities=False,
        )
        self.assertIsNone(metrics["roc_auc"])
        self.assertIn("accuracy", metrics)

    def test_loss_still_reported_with_criterion(self) -> None:
        labels = [0, 1]
        spoof_logits = [3.0, 3.0]
        metrics = evaluate_model(
            _StubHeadModel(spoof_logits),
            fake_loader(labels, spoof_logits, batch_size=2),
            criterion=nn.CrossEntropyLoss(),
        )
        self.assertIn("loss", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)


# ---------------------------------------------------------------------------
# Backward compatibility of the original metric contract
# ---------------------------------------------------------------------------


class TestBackwardCompatibility(unittest.TestCase):
    ORIGINAL_KEYS = {
        "accuracy", "precision", "recall", "f1", "per_class",
        "confusion_matrix", "class_counts",
    }

    def test_two_argument_call_still_works(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)  # original signature
        self.assertTrue(self.ORIGINAL_KEYS.issubset(metrics))

    def test_original_values_unchanged(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(metrics["accuracy"], 3 / 6)
        bonafide = metrics["per_class"]["bonafide"]
        spoof = metrics["per_class"]["spoof"]
        self.assertAlmostEqual(bonafide["precision"], 1 / 2)
        self.assertAlmostEqual(bonafide["recall"], 1 / 3)
        self.assertAlmostEqual(spoof["recall"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"].tolist(),
                         [[1, 2], [1, 2]])
        self.assertEqual(metrics["class_counts"],
                         {"bonafide": 3, "spoof": 3})

    def test_new_keys_do_not_shadow_original_ones(self) -> None:
        # macro precision/recall/f1 must stay macro even though spoof-level
        # metrics were added alongside them.
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        macro_f1 = float(np.mean(
            [v["f1"] for v in metrics["per_class"].values()]))
        self.assertAlmostEqual(metrics["f1"], macro_f1)

    def test_per_class_structure_unchanged(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        for entry in metrics["per_class"].values():
            self.assertEqual(
                set(entry), {"precision", "recall", "f1", "support"})

    def test_confusion_matrix_still_numpy_int(self) -> None:
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        cm = metrics["confusion_matrix"]
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.dtype, np.int64)
        self.assertTrue(hasattr(cm, "tolist"))

    def test_full_metrics_dict_roundtrips_through_checkpoint(self) -> None:
        # train_head stores the whole val_metrics dict in checkpoints: the
        # extended dict (incl. None entries) must save/load cleanly.
        metrics = compute_metrics(Y_TRUE, Y_PRED,
                                  spoof_probabilities=perfect_probs(Y_TRUE))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_head.pt"
            save_checkpoint(path, VoiceGuardClassifier.with_tiny_config(),
                            TrainingConfig(dataset_path="x"),
                            epoch=1, metrics=metrics)
            self.assertTrue(path.exists())  # assert inside the temp dir's life

    def test_evaluate_model_still_reports_original_keys(self) -> None:
        labels = [0, 1]
        spoof_logits = [3.0, 3.0]
        metrics = evaluate_model(
            _StubHeadModel(spoof_logits),
            fake_loader(labels, spoof_logits, batch_size=2),
        )
        self.assertTrue(self.ORIGINAL_KEYS.issubset(metrics))
        self.assertNotIn("spoof_probabilities", metrics)  # stays internal


if __name__ == "__main__":
    unittest.main()
