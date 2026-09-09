"""Waveform preprocessing for the VoiceGuard model.

Bridges the canonical ``dataset.audio`` output (1-D float32 mono 16 kHz)
and the pretrained checkpoint's expected input. Normalization behavior is
read from the checkpoint's own preprocessor config on the Hub — never
guessed or hardcoded here.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from transformers import Wav2Vec2FeatureExtractor

from .config import FEAT_EXTRACTOR_NAME

__all__ = [
    "load_feature_extractor",
    "input_should_normalize",
    "normalize_waveform",
]

_extractor: Optional[Wav2Vec2FeatureExtractor] = None


def load_feature_extractor() -> Wav2Vec2FeatureExtractor:
    """Load (once) the checkpoint's feature extractor config.

    Only the small preprocessor config is fetched — no model weights.
    Cached after the first call.
    """
    global _extractor
    if _extractor is None:
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(FEAT_EXTRACTOR_NAME)
    return _extractor


def input_should_normalize() -> bool:
    """Whether the checkpoint expects zero-mean/unit-variance input."""
    return bool(getattr(load_feature_extractor(), "do_normalize", False))


def normalize_waveform(values: Tensor) -> Tensor:
    """Apply the checkpoint's input normalization to a (batch, time) tensor.

    Mirrors ``Wav2Vec2FeatureExtractor``: per-sample zero mean and unit
    variance over the time axis (``(x - mean) / sqrt(var + 1e-7)``).
    A no-op when the checkpoint's config disables normalization.
    """
    if not input_should_normalize():
        return values
    mean = values.mean(dim=-1, keepdim=True)
    variance = values.var(dim=-1, keepdim=True, unbiased=False)
    return (values - mean) / torch.sqrt(variance + 1e-7)
