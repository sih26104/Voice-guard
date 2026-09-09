"""VoiceGuard dataset module.

Future home of the BONAFIDE vs SPOOF audio classification data
pipeline. Nothing here assumes a concrete dataset yet: names, folder
structures and label sources are supplied entirely via configuration.

Public surface (import from here, not from submodules):
    Label            - canonical two-class label enum
    normalize_label  - parse raw label strings
    DatasetConfig    - validated path/label configuration
    load_config      - parse a dataset config file
    BaseAudioDataset - dataset interface every concrete loader extends
    AudioSample      - single labelled audio item
    AudioClip        - canonical mono float32 audio + sample rate
    preprocess       - file/array -> mono, resampled, validated clip
    MlaadSample      - one MLAAD row (audio loaded lazily)
    MlaadDataset     - concrete loader for the local MLAAD subset
"""

from .labels import Label, normalize_label
from .config import DatasetConfig, SplitConfig, load_config, parse_config
from .base_audio_dataset import AudioSample, BaseAudioDataset, LoaderNotImplementedError
from .audio import (
    AudioClip,
    AudioValidationError,
    DurationValidationError,
    TARGET_SAMPLE_RATE,
    load_audio,
    preprocess,
)
from .mlaad import MLAAD_SPLITS, MlaadDataset, MlaadDatasetError, MlaadSample

__all__ = [
    "Label",
    "normalize_label",
    "DatasetConfig",
    "SplitConfig",
    "load_config",
    "parse_config",
    "AudioSample",
    "BaseAudioDataset",
    "LoaderNotImplementedError",
    "AudioClip",
    "AudioValidationError",
    "DurationValidationError",
    "TARGET_SAMPLE_RATE",
    "load_audio",
    "preprocess",
    "MLAAD_SPLITS",
    "MlaadSample",
    "MlaadDataset",
    "MlaadDatasetError",
]
