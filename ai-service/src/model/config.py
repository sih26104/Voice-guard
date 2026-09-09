"""Model constants and configuration for VoiceGuard.

Central place for the pretrained encoder choice and default classifier
hyper-parameters. The label taxonomy itself is NOT redefined here — it is
imported from :mod:`dataset.labels` so there is a single source of truth
for BONAFIDE vs SPOOF.

Input normalization is deliberately NOT a constant here: whether the
checkpoint expects zero-mean/unit-variance input is read from the
checkpoint's own preprocessor config (see :mod:`model.processor`).
"""

from __future__ import annotations

from dataset.labels import Label

__all__ = [
    "PRETRAINED_MODEL_NAME",
    "MODEL_SAMPLE_RATE",
    "FEAT_EXTRACTOR_NAME",
    "DEFAULT_DROPOUT",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
]

#: Smallest practical pretrained speech encoder for CPU-only inference.
#: Base (95M params) — NOT large/300M — chosen for the 1,000-sample
#: CPU-only VoiceGuard prototype. Weights downloaded once, cached locally.
PRETRAINED_MODEL_NAME: str = "facebook/wav2vec2-base"

#: All model input must be 16 kHz mono float32 (matches MlaadDataset).
MODEL_SAMPLE_RATE: int = 16_000

#: Feature extractor config used for checkpoint-driven input normalization.
FEAT_EXTRACTOR_NAME: str = "facebook/wav2vec2-base"

#: Classification head defaults (small by design; the encoder is frozen).
DEFAULT_DROPOUT: float = 0.1
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_BATCH_SIZE: int = 4

#: Canonical index <-> label mapping shared by training, metrics and the
#: future FastAPI layer. Derived from dataset.labels.Label — no new enum.
LABEL_TO_ID = {label: idx for idx, label in enumerate(Label)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
