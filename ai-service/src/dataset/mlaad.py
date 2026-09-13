"""MLAAD subset loader for VoiceGuard.

Concrete, dataset-specific loader for the local MLAAD-tiny subset
(``data/mlaad_subset/``). It implements the dataset-agnostic
:class:`~dataset.base_audio_dataset.BaseAudioDataset` contract and reuses
the shared audio preprocessing in :mod:`dataset.audio` — no preprocessing
logic is duplicated here.

Source of truth: ``splits.csv`` with the schema::

    local_path,hf_path,label,language,tts_system,split

Facts this loader encodes (and nothing more):
    * ``local_path`` is relative to the *repository root* (it uses
      Windows-style separators as produced by the split script; parsing
      normalizes them so the code stays platform-neutral).
    * ``label`` is BONAFIDE/SPOOF (any casing accepted, resolved via the
      canonical :func:`dataset.labels.normalize_label` — no new enum).
    * ``split`` is ``train`` / ``validation`` / ``test`` (every MLAAD
      split file), plus ``test-unseen-tts`` when the CSV provides it
      (the full-dataset splits do; see ``create_mlaad_full_splits.py``).
    * ``tts_system`` is empty for BONAFIDE rows.

Design:
    * CSV rows are parsed and validated eagerly (cheap; fails fast).
    * Audio is loaded lazily and one sample at a time: iterating the
      dataset never touches the WAV files, and :meth:`MlaadSample.waveform`
      preprocesses on demand. The full dataset never sits in memory.
    * Unknown labels, unknown splits, missing required columns, wrong
      column counts, missing audio files and duplicate local paths all
      raise immediately with row-level context.

Another dataset loader can later be added the same way (subclass
:class:`BaseAudioDataset`, own its manifest format, reuse ``audio.py``)
without touching this file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from .audio import AudioClip, preprocess
from .base_audio_dataset import AudioSample, BaseAudioDataset
from .config import DatasetConfig, SplitConfig
from .labels import Label, normalize_label

__all__ = [
    "MLAAD_SPLITS",
    "MlaadSample",
    "MlaadDataset",
    "MlaadDatasetError",
]

#: Splits defined by the MLAAD split script (also validated eagerly).
#: ``test-unseen-tts`` exists only in split files produced by
#: ``scripts/create_mlaad_full_splits.py``; the prototype subset keeps
#: the original three.
MLAAD_SPLITS: Tuple[str, ...] = ("train", "validation", "test", "test-unseen-tts")

#: Splits every valid MLAAD splits.csv must be able to serve. Historic
#: behaviour for the 1,000-file prototype subset — unchanged.
CORE_MLAAD_SPLITS: Tuple[str, ...] = ("train", "validation", "test")

_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "local_path",
    "label",
    "language",
    "tts_system",
    "split",
)


class MlaadDatasetError(ValueError):
    """Raised for malformed MLAAD metadata (CSV structure/content issues)."""


@dataclass(frozen=True)
class MlaadSample(AudioSample):
    """One MLAAD row, resolved against the splits CSV.

    Adds the MLAAD-specific metadata (``language``, ``tts_system``) on
    top of the dataset-agnostic :class:`AudioSample`. Audio is **not**
    loaded here; call :meth:`waveform` to preprocess the file on demand.
    """

    language: str = ""
    tts_system: Optional[str] = None

    def waveform(
        self,
        sample_rate: int = 16_000,
        min_duration_s: Optional[float] = 0.5,
        max_duration_s: Optional[float] = None,
    ) -> AudioClip:
        """Load and preprocess this sample's audio via ``dataset.audio``.

        Uses the shared preprocessing pipeline (mono -> resample ->
        validate). The max-duration cap is disabled by default because
        real MLAAD clips run up to ~38.5 s, past the 30 s module default.

        Raises:
            AudioValidationError: missing/unreadable/undecodable file.
            DurationValidationError: duration below ``min_duration_s``.
        """
        if self.audio_path is None:  # pragma: no cover - defensive
            raise MlaadDatasetError(
                f"Sample {self.key!r} has no audio path; cannot load waveform."
            )
        return preprocess(
            self.audio_path,
            sample_rate=sample_rate,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )


class MlaadDataset(BaseAudioDataset):
    """Loader for the local MLAAD subset driven by ``splits.csv``.

    Args:
        dataset_root: The dataset directory containing ``splits.csv``.
            Audio ``local_path`` values are resolved relative to the
            repository root (two levels above this package), matching
            how the split script wrote them.
        split: One of ``train`` / ``validation`` / ``test`` — or
            ``test-unseen-tts`` for split files that contain it (the
            full-dataset splits; the CSV row values are validated
            independently of this argument).
        splits_filename: CSV file name inside ``dataset_root``. Kept as
            a parameter only for tests; production code uses the default.
        repo_root: Base directory ``local_path`` values resolve against.
            Defaults to the discovered repository root; tests override it
            to point at temporary fixtures.

    Raises:
        MlaadDatasetError: missing/empty CSV, missing required columns,
            malformed or duplicate rows, unknown labels, unknown splits.
        KeyError: unknown split name (via the base-class config).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        splits_filename: str = "splits.csv",
        repo_root: Optional[str | Path] = None,
    ) -> None:
        self._dataset_root = Path(dataset_root)
        self._splits_path = self._dataset_root / splits_filename
        self._repo_root = (
            Path(repo_root) if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )

        if split not in MLAAD_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of: {list(MLAAD_SPLITS)}"
            )
        rows = self._load_rows()
        if split == "test-unseen-tts" and not any(
            row["split"] == "test-unseen-tts" for row in rows
        ):
            raise ValueError(
                f"Split {split!r} requested, but {self._splits_path} has no "
                f"'test-unseen-tts' rows (regenerate splits with "
                f"scripts/create_mlaad_full_splits.py)."
            )
        self._rows: Dict[str, dict] = {row["local_path"]: row for row in rows}

        config = self._build_config(len(rows))
        super().__init__(config, split=split)

    # ------------------------------------------------------------------
    # Parsing / validation
    # ------------------------------------------------------------------

    def _load_rows(self) -> list:
        """Parse and validate every CSV row; raise on the first problem."""
        if not self._splits_path.is_file():
            raise MlaadDatasetError(f"Splits CSV not found: {self._splits_path}")

        try:
            with self._splits_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise MlaadDatasetError(
                        f"Splits CSV {self._splits_path} is empty (no header row)."
                    )
                missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
                if missing:
                    raise MlaadDatasetError(
                        f"Splits CSV {self._splits_path} is missing required "
                        f"columns: {missing}; found {reader.fieldnames}"
                    )
                parsed: list = []
                seen: set = set()
                for line_number, row in enumerate(reader, start=2):
                    record = self._parse_row(row, line_number)
                    if record["local_path"] in seen:
                        raise MlaadDatasetError(
                            f"Splits CSV {self._splits_path} line {line_number}: "
                            f"duplicate local_path {record['local_path']!r}"
                        )
                    seen.add(record["local_path"])
                    parsed.append(record)
        except MlaadDatasetError:
            raise
        except (OSError, csv.Error) as exc:
            raise MlaadDatasetError(
                f"Failed to read splits CSV {self._splits_path}: {exc}"
            ) from exc

        if not parsed:
            raise MlaadDatasetError(f"Splits CSV {self._splits_path} has no data rows.")
        return parsed

    def _parse_row(self, row: dict, line_number: int) -> dict:
        """Validate one CSV row into a normalized record.

        ``csv.DictReader`` never raises on its own for ragged rows: extra
        fields land in ``None`` under the ``None`` key and missing fields
        become ``None`` values. Both are rejected here with the CSV line
        number for context.
        """
        context = f"Splits CSV line {line_number}"

        if row.get(None) is not None:
            raise MlaadDatasetError(
                f"{context}: row has more values than header columns "
                f"(extra: {row[None]!r})"
            )

        def column(name: str) -> str:
            value = row.get(name)
            if value is None:
                raise MlaadDatasetError(f"{context}: missing value for column {name!r}")
            return value.strip()

        local_path = column("local_path")
        if not local_path:
            raise MlaadDatasetError(f"{context}: 'local_path' must be non-empty")

        raw_label = column("label")
        try:
            label = normalize_label(raw_label)
        except ValueError as exc:
            raise MlaadDatasetError(f"{context}: {exc}") from exc

        split = column("split")
        if split not in MLAAD_SPLITS:
            raise MlaadDatasetError(
                f"{context}: unknown split {split!r}; expected one of: "
                f"{list(MLAAD_SPLITS)}"
            )

        language = column("language")
        if not language:
            raise MlaadDatasetError(f"{context}: 'language' must be non-empty")

        tts_system = column("tts_system") or None

        return {
            "local_path": local_path,
            "label": label,
            "split": split,
            "language": language,
            "tts_system": tts_system,
        }

    def _build_config(self, row_count: int) -> DatasetConfig:
        """Wrap the CSV into the dataset-agnostic config the ABC expects."""
        splits = {
            name: SplitConfig(
                name=name,
                manifest=str(self._splits_path),
                labels=str(self._splits_path),
            )
            for name in MLAAD_SPLITS
        }
        return DatasetConfig(
            name="mlaad-subset",
            root=self._dataset_root,
            audio_extensions=(".wav",),
            splits=splits,
            label_overrides={},
        )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_audio_path(self, local_path: str) -> Path:
        """Resolve a CSV ``local_path`` to an absolute filesystem path.

        Values are relative to the repository root (e.g.
        ``data\\mlaad_subset\\audio\\...``); separators are normalized so
        the loader behaves identically on Windows and POSIX.
        """
        return (self._repo_root / Path(local_path.replace("\\", "/"))).resolve()

    # ------------------------------------------------------------------
    # BaseAudioDataset contract
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(1 for row in self._rows.values() if row["split"] == self.split_name)

    def __iter__(self) -> Iterator[MlaadSample]:
        """Yield validated samples for the selected split (lazy audio).

        The existence of each referenced audio file is verified here, but
        files are **not** opened or decoded — audio loads one sample at a
        time via :meth:`MlaadSample.waveform`.
        """
        for local_path, row in self._rows.items():
            if row["split"] != self.split_name:
                continue
            audio_path = self._resolve_audio_path(local_path)
            if not audio_path.is_file():
                raise MlaadDatasetError(
                    f"Audio file referenced by splits.csv does not exist: "
                    f"{audio_path} (local_path={local_path!r})"
                )
            yield MlaadSample(
                key=local_path,
                audio_path=audio_path,
                label=row["label"],
                split=row["split"],
                metadata={},
                language=row["language"],
                tts_system=row["tts_system"],
            )

    def by_label(self, label: Label) -> Tuple[MlaadSample, ...]:
        """Convenience accessor: samples of this split with exactly ``label``."""
        wanted = label if isinstance(label, Label) else normalize_label(label)
        return tuple(sample for sample in self if sample.label is wanted)
