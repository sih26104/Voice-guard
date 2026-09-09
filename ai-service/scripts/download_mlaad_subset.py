from pathlib import Path
from collections import defaultdict
import csv
import random

from huggingface_hub import list_repo_files, hf_hub_download


REPO_ID = "mueller91/MLAAD-tiny"
SEED = 26104

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mlaad_subset"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST = DATA_DIR / "manifest.csv"

random.seed(SEED)


def round_robin_by_group(files, group_index):
    groups = defaultdict(list)

    for path in files:
        parts = path.split("/")
        groups[parts[group_index]].append(path)

    for values in groups.values():
        random.shuffle(values)

    group_names = list(groups.keys())
    random.shuffle(group_names)

    selected = []

    while True:
        added = False

        for group in group_names:
            if groups[group]:
                selected.append(groups[group].pop())
                added = True

        if not added:
            break

    return selected


def select_fake(files, language, count):
    candidates = [
        x for x in files
        if x.startswith(f"fake/{language}/")
    ]

    ordered = round_robin_by_group(candidates, 2)

    return ordered[:count]


def select_original(files, language, count):
    candidates = [
        x for x in files
        if x.startswith(f"original/{language}/")
    ]

    random.shuffle(candidates)

    return candidates[:count]


def main():
    print("Listing MLAAD-tiny files...")

    files = list_repo_files(
        REPO_ID,
        repo_type="dataset"
    )

    wav_files = [
        x for x in files
        if x.lower().endswith(".wav")
    ]

    print(f"Total WAV files available: {len(wav_files)}")

    fake_en = select_fake(wav_files, "en", 400)
    fake_de = select_fake(wav_files, "de", 100)

    original_en = select_original(wav_files, "en", 400)
    original_de = select_original(wav_files, "de", 100)

    selected = []

    for path in fake_en:
        selected.append(("SPOOF", "en", path))

    for path in fake_de:
        selected.append(("SPOOF", "de", path))

    for path in original_en:
        selected.append(("BONAFIDE", "en", path))

    for path in original_de:
        selected.append(("BONAFIDE", "de", path))

    random.shuffle(selected)

    print(f"Selected files: {len(selected)}")

    if len(selected) != 1000:
        raise RuntimeError(
            f"Expected 1000 files, selected {len(selected)}"
        )

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_rows = []

    for index, (label, language, hf_path) in enumerate(
        selected,
        start=1
    ):
        print(f"[{index}/1000] {hf_path}")

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            cache_dir="C:/hf_cache",
        )

        local_path = Path(local_path)

        destination = AUDIO_DIR / Path(hf_path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        import shutil
        shutil.copy2(local_path, destination)

        local_path = destination

        local_path = Path(local_path)

        parts = hf_path.split("/")

        tts_system = (
            parts[2]
            if label == "SPOOF"
            else ""
        )

        manifest_rows.append({
            "local_path": str(
                local_path.relative_to(PROJECT_ROOT)
            ),
            "hf_path": hf_path,
            "label": label,
            "language": language,
            "tts_system": tts_system,
        })

    with MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "local_path",
                "hf_path",
                "label",
                "language",
                "tts_system",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("================================")
    print("MLAAD subset download complete")
    print("================================")
    print(f"Files:    {len(manifest_rows)}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()