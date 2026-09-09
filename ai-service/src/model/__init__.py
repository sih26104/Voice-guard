"""VoiceGuard model package.

Smallest practical pretrained baseline for CPU-only spoof detection:
a frozen Wav2Vec2-base encoder with a small trainable 2-class head
(BONAFIDE vs SPOOF). Imports mirror the dataset package convention:
import from here, not from submodules.
"""

from .classifier import (
    SpoofPrediction,
    VoiceGuardClassifier,
    load_voiceguard_model,
    predict,
    prepare_batch,
    prepare_waveform,
)
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    FEAT_EXTRACTOR_NAME,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MODEL_SAMPLE_RATE,
    PRETRAINED_MODEL_NAME,
)
from .processor import (
    input_should_normalize,
    load_feature_extractor,
    normalize_waveform,
)

__all__ = [
    "VoiceGuardClassifier",
    "SpoofPrediction",
    "load_voiceguard_model",
    "predict",
    "prepare_waveform",
    "prepare_batch",
    "PRETRAINED_MODEL_NAME",
    "MODEL_SAMPLE_RATE",
    "FEAT_EXTRACTOR_NAME",
    "DEFAULT_DROPOUT",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "input_should_normalize",
    "load_feature_extractor",
    "normalize_waveform",
]
