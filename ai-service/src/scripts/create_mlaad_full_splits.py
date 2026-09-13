"""Create and validate the four evaluation splits for the FULL MLAAD-tiny dataset.

Run as a module (from ``ai-service/``)::

    .venv/Scripts/python.exe -m src.scripts.create_mlaad_full_splits
    .venv/Scripts/python.exe -m src.scripts.create_mlaad_full_splits --validate-only

Inputs and outputs
------------------

* Inputs:  ``data/mlaad_full/manifest.csv`` (from ``scripts/download_mlaad_full.py``)
           **and** the on-disk ``audio/`` tree. The disk scan is the source of
           truth for the file inventory; the manifest is cross-checked against
           it (same paths, same derived metadata) before anything is split.
* Outputs: ``data/mlaad_full/splits.csv`` with the loader schema::

      local_path,hf_path,label,language,tts_system,split

           and ``data/mlaad_full/split_report.json`` with the full audit
           (class counts, TTS-system counts, unseen-TTS disjointness and the
           pass/fail result of every validation check).

Splits
------

``train`` / ``validation`` / ``test``
    Seen spoof TTS systems are split per (language, system) at 70/15/15;
    bonafide files are split per language at 70/15/15. Both classes are
    present in all three.

``test-unseen-tts``
    Per language, whole spoof TTS systems (never split atomically) are held
    out deterministically until ~15% of the language's spoof files are
    covered. Those systems occur in **no other split** — the split measures
    generalization to TTS systems the model has never seen. A dedicated,
    disjoint per-language slice of bonafide files is held out as this
    split's bonafide support.

Why this module exists (difference to ``scripts/create_mlaad_full_splits.py``)
-----------------------------------------------------------------------------

The older standalone script aliased the bonafide ``test`` rows into
``test-unseen-tts``, so those ``local_path`` values appeared **twice** in
``splits.csv``. That violates the "every file represented exactly once" /
"no duplicate local_path" contract and is rejected outright by
:class:`dataset.mlaad.MlaadDataset` (``duplicate local_path`` error). This
module gives the unseen split its own disjoint bonafide support instead:
every WAV lands in exactly one row. It also adds the disk-vs-manifest
cross-check and writes ``split_report.json``.

Conventions reused from the existing code
-----------------------------------------

* Split names come from :data:`dataset.mlaad.MLAAD_SPLITS` (loader stays in
  sync with the generator by construction).
* Labels are validated with :func:`dataset.labels.normalize_label` (any
  casing accepted; the CSV uses the manifest's uppercase ``SPOOF`` /
  ``BONAFIDE`` spelling).
* ``local_path`` is repo-root-relative with forward slashes (platform
  neutral; :mod:`dataset.mlaad` normalizes separators when resolving).
* ``tts_system`` is empty for BONAFIDE rows.

Determinism: all randomness flows through one ``random.Random(SEED)``
instance, so the output is a pure function of the dataset + seed.

Read-only guarantee: the script only *reads* the audio tree (existence
checks); no audio file is created, moved, modified or deleted. ``data/`` is
git-ignored, so no raw WAVs (and no derived CSV/report) can be committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# The dataset package lives in ai-service/src. Importing this module as
# ``src.scripts.create_mlaad_full_splits`` (the documented invocation) puts
# ai-service — not ai-service/src — on sys.path, so the bootstrap below makes
# ``from dataset...`` work in both invocation styles (module or PYTHONPATH=src).
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dataset.labels import normalize_label  # noqa: E402
from dataset.mlaad import MLAAD_SPLITS  # noqa: E402

__all__ = [
    "SEED",
    "SplitError",
    "derive_metadata",
    "scan_audio_files",
    "load_manifest_rows",
    "cross_check_manifest",
    "build_splits",
    "hold_out_unseen_systems",
    "run_validations",
    "validate_or_raise",
    "build_report",
    "main",
]

SEED = 26104

PROJECT_ROOT = Path(__file__).resolve().parents[3]      # Voice-guard/
DATA_DIR = PROJECT_ROOT / "data" / "mlaad_full"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST = DATA_DIR / "manifest.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"
REPORT_JSON = DATA_DIR / "split_report.json"

#: Only ``*.wav`` files are inventoried (suffix match is case-insensitive).
AUDIO_SUFFIX = ".wav"

LABEL_SPOOF = "SPOOF"
LABEL_BONAFIDE = "BONAFIDE"

#: Fraction of each language's spoof files to hold out — in whole TTS
#: systems — for ``test-unseen-tts``.
UNSEEN_FRACTION = 0.15

#: Never hold out so many systems that fewer than this many remain seen per
#: language (train/validation/test must stay meaningful).
MIN_SEEN_SYSTEMS = 2

#: A language contributes a bonafide slice to ``test-unseen-tts`` only if it
#: has at least this many bonafide files; smaller languages keep every
#: bonafide file in the core splits.
MIN_BONAFIDE_FOR_SUPPORT = 10

#: Standard split fractions applied inside every (language, system) or
#: (language) bucket of the rows destined for the core splits.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

#: Core splits that must contain both classes.
CORE_SPLITS: Tuple[str, ...] = ("train", "validation", "test")

FIELDNAMES = [
    "local_path",
    "hf_path",
    "label",
    "language",
    "tts_system",
    "split",
]


class SplitError(RuntimeError):
    """Raised when the dataset, manifest or generated splits violate an invariant."""


# ---------------------------------------------------------------------------
# Inventory: disk scan + manifest cross-check
# ---------------------------------------------------------------------------


def derive_metadata(hf_path: str) -> Tuple[str, str, str]:
    """Derive ``(label, language, tts_system)`` from an audio-relative path.

    Mirrors ``download_mlaad_full.derive_metadata`` conventions but is
    **strict**: every inventoried file must be classifiable, because the
    split contract requires every file to be represented exactly once —
    silently skipping a file here would break that guarantee downstream.

    * ``fake/<language>/<tts_system>/<file>.wav`` → ``("SPOOF", lang, system)``
    * ``original/<language>/<file>.wav``          → ``("BONAFIDE", lang, "")``

    Raises:
        SplitError: any other layout (wrong depth, unknown top-level dir).
    """
    parts = hf_path.split("/")
    if len(parts) == 4 and parts[0] == "fake":
        return LABEL_SPOOF, parts[1], parts[2]
    if len(parts) == 3 and parts[0] == "original":
        return LABEL_BONAFIDE, parts[1], ""
    raise SplitError(
        f"Unexpected audio layout {hf_path!r}: expected "
        f"'fake/<language>/<tts_system>/<file>.wav' or "
        f"'original/<language>/<file>.wav'"
    )


def build_local_path(hf_path: str, data_dir: Path, repo_root: Path) -> str:
    """Repo-root-relative ``local_path`` for an audio-relative ``hf_path``.

    Forward slashes throughout (platform neutral; the loader normalizes).
    Raises ``SplitError`` if ``data_dir`` is not inside ``repo_root``.
    """
    absolute = data_dir / "audio" / hf_path
    try:
        return absolute.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise SplitError(
            f"Dataset dir {data_dir} is not inside repo root {repo_root}; "
            f"cannot build a repo-relative local_path for {hf_path!r}"
        ) from exc


def scan_audio_files(audio_dir: Path, data_dir: Path, repo_root: Path) -> List[dict]:
    """Inventory every WAV under ``audio_dir`` (read-only).

    Returns one record per file with the splits.csv schema (minus ``split``),
    sorted by ``hf_path`` for deterministic output regardless of the
    filesystem's enumeration order. Empty system directories naturally
    contribute nothing: metadata is derived from files, never from directory
    names.
    """
    if not audio_dir.is_dir():
        raise SplitError(
            f"Audio directory not found: {audio_dir}. "
            f"Run scripts/download_mlaad_full.py first."
        )
    records: List[dict] = []
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != AUDIO_SUFFIX:
            continue
        hf_path = path.relative_to(audio_dir).as_posix()
        label, language, tts_system = derive_metadata(hf_path)
        records.append(
            {
                "local_path": build_local_path(hf_path, data_dir, repo_root),
                "hf_path": hf_path,
                "label": label,
                "language": language,
                "tts_system": tts_system,
            }
        )
    if not records:
        raise SplitError(f"No WAV files found under {audio_dir}.")
    records.sort(key=lambda record: record["hf_path"])
    return records


def load_manifest_rows(path: Path) -> List[dict]:
    """Read ``manifest.csv``, validating its shape and labels (fail fast)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run scripts/download_mlaad_full.py first."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SplitError(f"Manifest is empty (no header row): {path}")
        required = {"local_path", "hf_path", "label", "language", "tts_system"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise SplitError(
                f"Manifest {path} is missing columns: {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise SplitError(f"Manifest has no data rows: {path}")
    for index, row in enumerate(rows, start=2):
        try:
            normalize_label(row["label"])
        except ValueError as exc:
            raise SplitError(f"Manifest line {index}: {exc}") from exc
    return rows


def cross_check_manifest(
    disk_records: Sequence[dict], manifest_rows: Sequence[dict]
) -> dict:
    """Disk inventory vs manifest: same paths, same derived metadata.

    Both sides are compared as a path multiset (normalized to forward
    slashes) and per-path metadata. Any disagreement is a hard error: the
    split must cover exactly the files that actually exist, with ground-truth
    labels the dataset itself declares.

    Returns a summary dict for ``split_report.json``.
    """
    disk_by_path = {record["local_path"]: record for record in disk_records}
    manifest_by_path = {}
    for row in manifest_rows:
        normalized = row["local_path"].replace("\\", "/")
        manifest_by_path[normalized] = row

    missing_in_manifest = sorted(set(disk_by_path) - set(manifest_by_path))
    missing_on_disk = sorted(set(manifest_by_path) - set(disk_by_path))

    mismatches: List[dict] = []
    for local_path in sorted(set(disk_by_path) & set(manifest_by_path)):
        record = disk_by_path[local_path]
        row = manifest_by_path[local_path]
        try:
            label = normalize_label(row["label"]).value.upper()
        except ValueError as exc:
            raise SplitError(
                f"Manifest row for {local_path!r} has an invalid label: {exc}"
            ) from exc
        expected = (
            label,
            row["language"].strip(),
            (row["tts_system"] or "").strip(),
        )
        actual = (record["label"], record["language"], record["tts_system"])
        if expected != actual:
            mismatches.append(
                {
                    "local_path": local_path,
                    "manifest": expected,
                    "derived": actual,
                }
            )

    if missing_in_manifest or missing_on_disk or mismatches:
        raise SplitError(
            "Manifest/disk cross-check failed: "
            f"missing_in_manifest={len(missing_in_manifest)} "
            f"(e.g. {missing_in_manifest[:3]}), "
            f"missing_on_disk={len(missing_on_disk)} "
            f"(e.g. {missing_on_disk[:3]}), "
            f"metadata_mismatches={len(mismatches)} "
            f"(e.g. {mismatches[:3]})"
        )

    return {
        "manifest_rows": len(manifest_rows),
        "disk_files": len(disk_records),
        "matched_paths": len(disk_by_path),
        "missing_in_manifest": [],
        "missing_on_disk": [],
        "metadata_mismatches": [],
    }


# ---------------------------------------------------------------------------
# Deterministic splitting primitives
# ---------------------------------------------------------------------------


def _cut(values: List[dict], rng: random.Random) -> Tuple[List[dict], List[dict], List[dict]]:
    """Deterministically cut ``values`` into train/validation/test parts.

    Sizes use ``round()`` on the fractions and are clamped so no part goes
    negative; order is randomized before cutting, then each part is shuffled,
    so no positional signal leaks into the splits.
    """
    values = list(values)
    rng.shuffle(values)
    total = len(values)
    train_size = min(int(round(TRAIN_FRACTION * total)), total)
    validation_size = min(
        int(round(VALIDATION_FRACTION * total)), total - train_size
    )
    train = values[:train_size]
    validation = values[train_size:train_size + validation_size]
    test = values[train_size + validation_size:]
    for part in (train, validation, test):
        rng.shuffle(part)
    return train, validation, test


def _split_grouped(rows: List[dict], group_key, rng: random.Random) -> Dict[str, List[dict]]:
    """70/15/15-split ``rows`` independently inside every group."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    assigned: Dict[str, List[dict]] = {
        "train": [], "validation": [], "test": []
    }
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        train, validation, test = _cut(groups[key], rng)
        assigned["train"].extend(train)
        assigned["validation"].extend(validation)
        assigned["test"].extend(test)
    return assigned


def hold_out_unseen_systems(
    spoof_rows: Sequence[dict], rng: random.Random
) -> Tuple[List[dict], List[dict]]:
    """Hold out complete TTS systems globally for test-unseen-tts.

    A TTS system is treated as one global group regardless of language.
    Therefore, if a system is selected for test-unseen-tts, none of its
    files may appear in train, validation, or test.

    Systems are selected atomically until approximately UNSEEN_FRACTION
    of all spoof files are held out, while keeping at least
    MIN_SEEN_SYSTEMS systems available for the core splits.

    Returns:
        (unseen_rows, seen_rows)
    """
    by_system: Dict[str, List[dict]] = defaultdict(list)

    for row in spoof_rows:
        tts_system = row.get("tts_system")
        if not tts_system:
            raise SplitError(
                f"SPOOF row without tts_system cannot be group-split: "
                f"{row.get('local_path')!r}"
            )

        by_system[tts_system].append(row)

    names = sorted(by_system)
    total_files = sum(len(rows) for rows in by_system.values())

    if len(names) <= MIN_SEEN_SYSTEMS:
        raise SplitError(
            f"Need more than MIN_SEEN_SYSTEMS={MIN_SEEN_SYSTEMS} "
            f"TTS systems for an unseen-system split; found {len(names)}."
        )

    # Shuffle system identities, not individual files.
    rng.shuffle(names)

    unseen_names: List[str] = []
    held_files = 0

    for name in names:
        if held_files / total_files >= UNSEEN_FRACTION:
            break

        remaining_after = len(names) - len(unseen_names) - 1

        # Always preserve enough complete TTS systems for train/validation/test.
        if remaining_after < MIN_SEEN_SYSTEMS:
            break

        unseen_names.append(name)
        held_files += len(by_system[name])

    unseen_set = set(unseen_names)

    unseen_rows: List[dict] = []
    seen_rows: List[dict] = []

    for name, rows in by_system.items():
        if name in unseen_set:
            unseen_rows.extend(rows)
        else:
            seen_rows.extend(rows)

    return unseen_rows, seen_rows


def split_bonafide_support(
    bonafide_rows: Sequence[dict], rng: random.Random
) -> Tuple[List[dict], List[dict]]:
    """Hold out a disjoint per-language bonafide slice for ``test-unseen-tts``.

    BONAFIDE files carry no TTS identity, so they cannot be held out "by
    system". Instead each language contributes a dedicated ~15% random slice
    (minimum one file) as the unseen split's bonafide support — **moved**,
    never copied, so every bonafide file lands in exactly one split and
    ``splits.csv`` contains no duplicate ``local_path``. Languages with fewer
    than ``MIN_BONAFIDE_FOR_SUPPORT`` files keep everything in the core
    splits.

    Returns ``(support_rows, core_rows)``.
    """
    by_language: Dict[str, List[dict]] = defaultdict(list)
    for row in bonafide_rows:
        by_language[row["language"]].append(row)

    support_rows: List[dict] = []
    core_rows: List[dict] = []
    for language in sorted(by_language):
        rows = list(by_language[language])
        total = len(rows)
        if total < MIN_BONAFIDE_FOR_SUPPORT:
            core_rows.extend(rows)
            continue
        rng.shuffle(rows)
        support_size = max(1, int(round(UNSEEN_FRACTION * total)))
        support_size = min(support_size, total)
        support_rows.extend(rows[:support_size])
        core_rows.extend(rows[support_size:])
    return support_rows, core_rows


def build_splits(disk_records: Sequence[dict], rng: random.Random) -> List[dict]:
    """Assign every inventoried file to exactly one split.

    Returns flat rows tagged with ``split`` in the splits.csv schema. Spoof
    unseen-system rows and the disjoint bonafide support together form
    ``test-unseen-tts``; nothing is duplicated anywhere.
    """
    spoof = [row for row in disk_records if row["label"] == LABEL_SPOOF]
    bonafide = [row for row in disk_records if row["label"] == LABEL_BONAFIDE]

    unseen_spoof, seen_spoof = hold_out_unseen_systems(spoof, rng)
    support_bonafide, core_bonafide = split_bonafide_support(bonafide, rng)

    seen_assigned = _split_grouped(
        seen_spoof, lambda row: (row["language"], row["tts_system"]), rng
    )
    bonafide_assigned = _split_grouped(core_bonafide, lambda row: row["language"], rng)

    final_rows: List[dict] = []
    for split in CORE_SPLITS:
        final_rows.extend(dict(row, split=split) for row in seen_assigned[split])
        final_rows.extend(dict(row, split=split) for row in bonafide_assigned[split])
    final_rows.extend(dict(row, split="test-unseen-tts") for row in unseen_spoof)
    final_rows.extend(dict(row, split="test-unseen-tts") for row in support_bonafide)
    return final_rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_validations(
    rows: Sequence[dict],
    disk_records: Sequence[dict],
    repo_root: Path = PROJECT_ROOT,
) -> List[dict]:
    """Run every split invariant; return one ``{name, passed, detail}`` per check.

    Checks (all must pass before anything is written):

    1. ``every_wav_represented_exactly_once`` — the split rows and the disk
       inventory are the same path multiset (this also catches files that
       were dropped or invented).
    2. ``no_duplicate_local_path`` — no path appears more than once.
    3. ``all_manifest_paths_exist`` — every ``local_path`` resolves to a real
       file under ``repo_root`` (read-only ``isfile`` checks).
    4. ``all_expected_splits_present`` — every name in
       :data:`dataset.mlaad.MLAAD_SPLITS` has at least one row.
    5. ``core_splits_contain_both_classes`` — train/validation/test each
       contain BONAFIDE **and** SPOOF rows.
    6. ``test_unseen_tts_contains_spoof`` — the unseen split has spoof rows.
    7. ``zero_tts_overlap_<split>_vs_unseen`` — no spoof TTS system of
       ``test-unseen-tts`` occurs in that core split. The ``train`` variant
       is the headline guarantee; validation/test are checked too (the holdout
       removes those systems from all core splits).
    """
    checks: List[dict] = []

    def add(name: str, passed: bool, **detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    row_paths = Counter(row["local_path"] for row in rows)
    disk_paths = Counter(record["local_path"] for record in disk_records)

    missing = sorted(disk_paths - row_paths)
    extra = sorted(row_paths - disk_paths)
    duplicated = sorted(path for path, count in row_paths.items() if count > 1)
    add(
        "every_wav_represented_exactly_once",
        not missing and not extra and not duplicated,
        expected_files=len(disk_records),
        assigned_rows=sum(row_paths.values()),
        missing_on_disk_side=missing[:10],
        extra_in_splits=extra[:10],
        duplicated=duplicated[:10],
    )
    add("no_duplicate_local_path", not duplicated, duplicates=duplicated[:10])

    missing_files = sorted(
        path for path in row_paths
        if not (Path(repo_root) / path).is_file()
    )
    add(
        "all_manifest_paths_exist",
        not missing_files,
        checked=len(row_paths),
        missing_files=missing_files[:10],
    )

    by_split: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    counts = {name: len(by_split.get(name, [])) for name in MLAAD_SPLITS}
    add(
        "all_expected_splits_present",
        all(counts[name] > 0 for name in MLAAD_SPLITS),
        row_counts=counts,
    )

    core_class_counts = {
        split: {
            LABEL_BONAFIDE: sum(
                1 for row in by_split.get(split, [])
                if row["label"] == LABEL_BONAFIDE
            ),
            LABEL_SPOOF: sum(
                1 for row in by_split.get(split, [])
                if row["label"] == LABEL_SPOOF
            ),
        }
        for split in CORE_SPLITS
    }
    core_ok = all(
        counts_by_label[LABEL_BONAFIDE] > 0 and counts_by_label[LABEL_SPOOF] > 0
        for counts_by_label in core_class_counts.values()
    )
    add(
        "core_splits_contain_both_classes",
        core_ok,
        class_counts=core_class_counts,
    )

    unseen_spoof_rows = [
        row for row in by_split.get("test-unseen-tts", [])
        if row["label"] == LABEL_SPOOF
    ]
    add(
        "test_unseen_tts_contains_spoof",
        bool(unseen_spoof_rows),
        spoof_rows=len(unseen_spoof_rows),
    )

    unseen_systems = {
        row["tts_system"] for row in unseen_spoof_rows if row["tts_system"]
    }
    for split in CORE_SPLITS:
        split_systems = {
            row["tts_system"] for row in by_split.get(split, [])
            if row["label"] == LABEL_SPOOF and row["tts_system"]
        }
        leaked = sorted(unseen_systems & split_systems)
        add(
            f"zero_tts_overlap_{split}_vs_unseen",
            not leaked,
            leaked_systems=leaked,
            unseen_system_count=len(unseen_systems),
            split_system_count=len(split_systems),
        )
    return checks


def validate_or_raise(
    rows: Sequence[dict],
    disk_records: Sequence[dict],
    repo_root: Path = PROJECT_ROOT,
) -> List[dict]:
    """Run all validations; raise :class:`SplitError` on the first failure.

    Returns the checks list (for the report) when everything passes.
    """
    checks = run_validations(rows, disk_records, repo_root)
    failed = [check for check in checks if not check["passed"]]
    if failed:
        lines = [
            f"  FAIL {check['name']}: {json.dumps(check['detail'], sort_keys=True)}"
            for check in failed
        ]
        raise SplitError(
            f"{len(failed)} split validation check(s) failed:\n" + "\n".join(lines)
        )
    return checks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_report(
    disk_records: Sequence[dict],
    rows: Sequence[dict],
    checks: Sequence[dict],
    cross_check: dict,
    seed: int,
    data_dir: Path = DATA_DIR,
) -> dict:
    """Assemble the full audit for ``split_report.json``."""

    def _class_counts(split: str) -> dict:
        counter = Counter(
            row["label"] for row in rows if row["split"] == split
        )
        return {
            LABEL_BONAFIDE: counter.get(LABEL_BONAFIDE, 0),
            LABEL_SPOOF: counter.get(LABEL_SPOOF, 0),
        }

    def _language_counts(split: str) -> dict:
        counter = Counter(
            row["language"] for row in rows if row["split"] == split
        )
        return dict(sorted(counter.items()))

    def _label_counts_per_language(split: str) -> dict:
        per_language: Dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            if row["split"] == split:
                per_language[row["language"]][row["label"]] += 1
        return {
            language: {
                LABEL_BONAFIDE: counter.get(LABEL_BONAFIDE, 0),
                LABEL_SPOOF: counter.get(LABEL_SPOOF, 0),
            }
            for language, counter in sorted(per_language.items())
        }

    def _systems_in(split: str) -> List[str]:
        return sorted({
            row["tts_system"] for row in rows
            if row["split"] == split and row["label"] == LABEL_SPOOF
            and row["tts_system"]
        })

    splits_section = {}
    for name in MLAAD_SPLITS:
        systems = _systems_in(name)
        splits_section[name] = {
            "total": sum(1 for row in rows if row["split"] == name),
            "class_counts": _class_counts(name),
            "language_counts": _language_counts(name),
            "label_counts_per_language": _label_counts_per_language(name),
            "tts_systems": systems,
            "tts_system_count": len(systems),
        }

    totals = Counter(record["label"] for record in disk_records)
    language_totals = Counter(record["language"] for record in disk_records)
    all_systems = sorted({
        record["tts_system"] for record in disk_records
        if record["label"] == LABEL_SPOOF and record["tts_system"]
    })
    system_file_counts = Counter(
        record["tts_system"] for record in disk_records
        if record["label"] == LABEL_SPOOF and record["tts_system"]
    )

    unseen_systems = splits_section["test-unseen-tts"]["tts_systems"]
    overlap = {
        f"train_vs_{name}": sorted(
            set(splits_section["train"]["tts_systems"])
            & set(splits_section[name]["tts_systems"])
        )
        for name in ("validation", "test", "test-unseen-tts")
    }

    return {
        "seed": seed,
        "generated_by": "src.scripts.create_mlaad_full_splits",
        "dataset": {
            "data_dir": data_dir.as_posix(),
            "total_files": len(disk_records),
            LABEL_BONAFIDE.lower(): totals.get(LABEL_BONAFIDE, 0),
            LABEL_SPOOF.lower(): totals.get(LABEL_SPOOF, 0),
            "languages": dict(sorted(language_totals.items())),
            "spoof_tts_system_count": len(all_systems),
            "spoof_tts_systems": all_systems,
            "spoof_files_per_tts_system": dict(sorted(system_file_counts.items())),
        },
        "manifest_cross_check": cross_check,
        "splits": splits_section,
        "tts_system_overlap": overlap,
        "unseen_tts_check": {
            "unseen_systems": unseen_systems,
            "overlap_with_train": sorted(
                set(unseen_systems) & set(splits_section["train"]["tts_systems"])
            ),
            "overlap_with_validation": sorted(
                set(unseen_systems)
                & set(splits_section["validation"]["tts_systems"])
            ),
            "overlap_with_test": sorted(
                set(unseen_systems) & set(splits_section["test"]["tts_systems"])
            ),
            "zero_overlap_with_train": not set(unseen_systems) & set(
                splits_section["train"]["tts_systems"]
            ),
            "confirmed_disjoint": (
                bool(unseen_systems)
                and not set(unseen_systems)
                & set(splits_section["train"]["tts_systems"])
                and not set(unseen_systems)
                & set(splits_section["validation"]["tts_systems"])
                and not set(unseen_systems)
                & set(splits_section["test"]["tts_systems"])
            ),
        },
        "validation": {
            "passed": all(check["passed"] for check in checks),
            "check_count": len(checks),
            "checks": list(checks),
        },
        "notes": [
            "BONAFIDE files carry no TTS identity (dataset structure has none), "
            "so the unseen split's bonafide side is a dedicated random "
            "per-language slice; only the SPOOF side of test-unseen-tts is "
            "guaranteed system-disjoint.",
            "Every file is assigned to exactly one split; no local_path appears "
            "twice (the unseen split's bonafide support is moved, not aliased).",
            "Validation intentionally shares TTS systems with train (matched "
            "condition) because it drives checkpoint selection; unseen-system "
            "generalization is measured exclusively on test-unseen-tts.",
            "Splits are file-level; the dataset exposes no speaker IDs to group by.",
            "TTS system directories without any downloaded file contribute no "
            "rows and no systems.",
        ],
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def write_splits_csv(rows: Sequence[dict], path: Path = SPLITS_CSV) -> Path:
    """Write ``splits.csv`` (schema = FIELDNAMES); parent dirs are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDNAMES})
    return path


def load_existing_splits_csv(path: Path) -> List[dict]:
    """Load a written ``splits.csv`` for re-validation (schema-checked)."""
    if not path.is_file():
        raise FileNotFoundError(f"Splits CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SplitError(f"Splits CSV is empty (no header row): {path}")
        missing = set(FIELDNAMES) - set(reader.fieldnames)
        if missing:
            raise SplitError(
                f"Splits CSV {path} is missing columns: {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            label = normalize_label(row["label"]).value.upper()
            split = row["split"].strip()
            if split not in MLAAD_SPLITS:
                raise SplitError(
                    f"Splits CSV {path} line {line_number}: unknown split "
                    f"{split!r}; expected one of {list(MLAAD_SPLITS)}"
                )
            rows.append(
                {
                    "local_path": row["local_path"].strip(),
                    "hf_path": row["hf_path"].strip(),
                    "label": label,
                    "language": row["language"].strip(),
                    "tts_system": (row["tts_system"] or "").strip(),
                    "split": split,
                }
            )
    if not rows:
        raise SplitError(f"Splits CSV has no data rows: {path}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(report: dict, validate_only: bool) -> None:
    dataset = report["dataset"]
    title = "MLAAD full splits" + (" (validate-only)" if validate_only else "")
    print()
    print("================================")
    print(title)
    print("================================")
    print(
        f"Files: {dataset['total_files']}  "
        f"(BONAFIDE {dataset['bonafide']} / SPOOF {dataset['spoof']}, "
        f"{dataset['spoof_tts_system_count']} TTS systems)"
    )
    print()
    print("Split counts (BONAFIDE / SPOOF):")
    for name in MLAAD_SPLITS:
        section = report["splits"][name]
        counts = section["class_counts"]
        print(
            f"  {name:<16} {section['total']:>6}  "
            f"({counts[LABEL_BONAFIDE]} / {counts[LABEL_SPOOF]})  "
            f"systems: {section['tts_system_count']}"
        )
    unseen = report["unseen_tts_check"]
    print()
    print(f"Unseen-TTS systems ({len(unseen['unseen_systems'])}):")
    for system in unseen["unseen_systems"]:
        print(f"  - {system}")
    print()
    print(
        "Zero TTS overlap train vs test-unseen-tts: "
        f"{'CONFIRMED' if unseen['zero_overlap_with_train'] else 'VIOLATION'}"
    )
    print(f"Validation: {'PASSED' if report['validation']['passed'] else 'FAILED'} "
          f"({report['validation']['check_count']} checks)")
    if not validate_only:
        print(f"\nSplits CSV: {SPLITS_CSV}")
        print(f"Report:     {REPORT_JSON}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate (or re-validate) the four MLAAD-full evaluation splits "
            "with full validation and a split_report.json audit."
        )
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-validate an existing splits.csv against the disk+manifest; "
             "write nothing.",
    )
    args = parser.parse_args(argv)

    try:
        disk_records = scan_audio_files(AUDIO_DIR, DATA_DIR, PROJECT_ROOT)
        manifest_rows = load_manifest_rows(MANIFEST)
        cross_check = cross_check_manifest(disk_records, manifest_rows)
        print(
            f"Disk inventory: {len(disk_records)} WAV files; manifest rows: "
            f"{len(manifest_rows)} — cross-check OK"
        )

        if args.validate_only:
            rows = load_existing_splits_csv(SPLITS_CSV)
        else:
            rows = build_splits(disk_records, random.Random(SEED))

        checks = validate_or_raise(rows, disk_records, PROJECT_ROOT)

        if args.validate_only:
            report = build_report(
                disk_records, rows, checks, cross_check, SEED, DATA_DIR
            )
            report["validate_only"] = True
            REPORT_JSON.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            _print_summary(report, validate_only=True)
            return 0

        # Deterministic row order in the CSV (rows were built with the same
        # seeded rng; shuffling keeps the file order uncorrelated with the
        # source tree layout).
        rows = list(rows)
        random.Random(SEED).shuffle(rows)
        write_splits_csv(rows, SPLITS_CSV)
        report = build_report(disk_records, rows, checks, cross_check, SEED, DATA_DIR)
        REPORT_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        _print_summary(report, validate_only=False)
        return 0
    except (SplitError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
