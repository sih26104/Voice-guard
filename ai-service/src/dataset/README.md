# VoiceGuard Dataset Module

Initial **structure only** — no dataset is downloaded, no model weights are
fetched, nothing is trained, and no filenames or folder layouts are assumed
by the code. A concrete dataset (e.g. ASVspoof 2019 LA) is described entirely
via configuration, added later.

## Purpose

Binary audio classification: **BONAFIDE** vs **SPOOF**. This module owns
everything between "raw dataset somewhere on disk" and "iterable of labelled
audio samples" for the rest of the VoiceGuard pipeline.

## Layout

```
src/dataset/
├── __init__.py            public exports (import from `dataset`, not submodules)
├── labels.py              canonical Label enum + normalize_label()
├── config.py              DatasetConfig / SplitConfig + load_config()
├── base_audio_dataset.py  AudioSample dataclass + BaseAudioDataset ABC
├── audio.py               dataset-agnostic audio loading/preprocessing
└── README.md              this file

../tests/test_audio.py      unit tests for audio.py (synthesized audio only)
dataset_config.example.json template for per-dataset path configs (repo root
                            of ai-service; copy to dataset_config.json)
```

## Audio preprocessing vs dataset loading

These are deliberately separate concerns:

- **Dataset loading** (`base_audio_dataset.py` + future concrete loaders):
  *which* files exist and *what label* each has. Loaders never preprocess.
- **Audio preprocessing** (`audio.py`): *how* any audio becomes model-ready.
  It knows nothing about datasets — no filenames, folders, or labels.

`audio.py` pipeline (all pure functions over waveforms/paths):

```
load_audio(path) -> AudioClip        # native SR, mono float32
AudioClip.resampled(target_sr)       # SOXR HQ resampling
AudioClip.validated(min_s, max_s)    # duration range check
preprocess(path_or_array, ...)       # full path: mono -> 16 kHz -> validated
```

`AudioClip` is the single canonical representation downstream (features,
model input) should consume: a 1-D float32 mono waveform in [-1, 1] plus
its integer sample rate. Failures raise `AudioValidationError` /
`DurationValidationError` so callers catch one exception family.

Default duration bounds: 0.5 s – 30 s (both overridable or disable-able
with `None`); default target rate 16 kHz (`TARGET_SAMPLE_RATE`).

## Configuration

Copy `dataset_config.example.json` → `dataset_config.json`, fill in paths.
Shape:

| Key                | Type   | Meaning                                              |
|--------------------|--------|------------------------------------------------------|
| `name`             | str    | Human-readable dataset identifier                    |
| `root`             | str    | Filesystem root the dataset lives under              |
| `audio_extensions` | [str]  | Lowercase extensions to recognize (e.g. `[".flac"]`) |
| `splits`           | obj    | Per-split `manifest` / `labels` path references      |
| `label_overrides`  | obj    | Optional per-key label fixes (`{"file": "spoof"}`)   |

`manifest` / `labels` are opaque strings at this layer — a future concrete
loader decides how to parse them. Unknown labels and unknown splits raise
immediately (fail loudly, never guess).

## Loader contract

Concrete loaders subclass `BaseAudioDataset` and implement:

- `__len__() -> int`
- `__iter__() -> Iterator[AudioSample]`

The base class never scans the filesystem. It resolves labels *only* from
configured `label_overrides`; anything else stays `None` until a concrete
loader supplies a real source. `LoaderNotImplementedError` marks methods
awaiting concrete behavior.

## Extending later (non-goals for now)

- A concrete `AsvspoofDataset(BaseAudioDataset)` implementing discovery of
  its own protocol files
- Feature extraction (MFCC / mel-spectrogram) — belongs in a separate
  `features/` module, not here
- Torch `Dataset`/`DataLoader` wrappers — separate adapters, keep this
  module dependency-light (stdlib + config only)
