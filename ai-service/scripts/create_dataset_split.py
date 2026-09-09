from pathlib import Path
import csv
import random
from collections import Counter, defaultdict


SEED = 26104

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = PROJECT_ROOT / "data" / "mlaad_subset" / "manifest.csv"
OUTPUT = PROJECT_ROOT / "data" / "mlaad_subset" / "splits.csv"

random.seed(SEED)


def stratified_split(rows, train_count, validation_count, test_count):
    rows = rows.copy()
    random.shuffle(rows)

    train = rows[:train_count]

    validation = rows[
        train_count:train_count + validation_count
    ]

    test = rows[
        train_count + validation_count:
        train_count + validation_count + test_count
    ]

    return train, validation, test


def main():
    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------

    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    print(f"Total samples: {len(rows)}")

    spoof = [
        r for r in rows
        if r["label"] == "SPOOF"
    ]

    bonafide = [
        r for r in rows
        if r["label"] == "BONAFIDE"
    ]

    print(f"SPOOF: {len(spoof)}")
    print(f"BONAFIDE: {len(bonafide)}")

    # ---------------------------------------------------------
    # SPOOF
    #
    # Distribute samples across TTS systems before splitting.
    # This avoids the split being dominated by one generator.
    # ---------------------------------------------------------

    spoof_groups = defaultdict(list)

    for row in spoof:
        spoof_groups[row["tts_system"]].append(row)

    for group in spoof_groups.values():
        random.shuffle(group)

    systems = list(spoof_groups.keys())
    random.shuffle(systems)

    # Round-robin the TTS systems to create a mixed ordering.
    spoof_ordered = []

    while True:
        added = False

        for system in systems:
            if spoof_groups[system]:
                spoof_ordered.append(
                    spoof_groups[system].pop()
                )
                added = True

        if not added:
            break

    # Exact 350 / 75 / 75 split.
    train_spoof = spoof_ordered[:350]

    validation_spoof = spoof_ordered[
        350:425
    ]

    test_spoof = spoof_ordered[
        425:500
    ]

    # ---------------------------------------------------------
    # BONAFIDE
    # ---------------------------------------------------------

    train_bonafide, validation_bonafide, test_bonafide = (
        stratified_split(
            bonafide,
            350,
            75,
            75
        )
    )

    # ---------------------------------------------------------
    # Add split labels
    # ---------------------------------------------------------

    final_rows = []

    for row in train_spoof:
        row["split"] = "train"
        final_rows.append(row)

    for row in train_bonafide:
        row["split"] = "train"
        final_rows.append(row)

    for row in validation_spoof:
        row["split"] = "validation"
        final_rows.append(row)

    for row in validation_bonafide:
        row["split"] = "validation"
        final_rows.append(row)

    for row in test_spoof:
        row["split"] = "test"
        final_rows.append(row)

    for row in test_bonafide:
        row["split"] = "test"
        final_rows.append(row)

    random.shuffle(final_rows)

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------

    split_counts = Counter(
        row["split"]
        for row in final_rows
    )

    label_counts = Counter(
        row["label"]
        for row in final_rows
    )

    if len(final_rows) != 1000:
        raise RuntimeError(
            f"Expected 1000 samples, got {len(final_rows)}"
        )

    if split_counts != Counter({
        "train": 700,
        "validation": 150,
        "test": 150,
    }):
        raise RuntimeError(
            f"Unexpected split counts: {split_counts}"
        )

    if label_counts != Counter({
        "SPOOF": 500,
        "BONAFIDE": 500,
    }):
        raise RuntimeError(
            f"Unexpected label counts: {label_counts}"
        )

    # ---------------------------------------------------------
    # Report TTS distribution
    # ---------------------------------------------------------

    print()
    print("TTS systems represented in each split:")

    for split in ("train", "validation", "test"):
        systems_in_split = {
            row["tts_system"]
            for row in final_rows
            if row["split"] == split
            and row["label"] == "SPOOF"
        }

        print(
            f"{split}: {len(systems_in_split)} TTS systems"
        )

    # ---------------------------------------------------------
    # Write splits.csv
    # ---------------------------------------------------------

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        fieldnames = [
            "local_path",
            "hf_path",
            "label",
            "language",
            "tts_system",
            "split",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(final_rows)

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print("================================")
    print("Dataset split created")
    print("================================")

    print(f"Total: {len(final_rows)}")
    print(f"Splits: {split_counts}")
    print(f"Labels: {label_counts}")

    print()
    print("Samples by split and label:")

    for split in ("train", "validation", "test"):
        counts = Counter(
            row["label"]
            for row in final_rows
            if row["split"] == split
        )

        print(
            f"{split}: "
            f"SPOOF={counts['SPOOF']}, "
            f"BONAFIDE={counts['BONAFIDE']}"
        )

    print()
    print("Note:")
    print(
        "TTS systems are distributed across splits because "
        "the 500-file subset contains samples from many "
        "systems. A completely unseen-TTS test requires a "
        "larger group-aware dataset subset."
    )

    print()
    print("Output:")
    print(OUTPUT)


if __name__ == "__main__":
    main()