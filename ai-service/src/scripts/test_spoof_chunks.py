import os
import tempfile
import httpx
import librosa
import soundfile as sf
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_RATE = 16000
TARGET_FILE = "northandsouth_16_f000117.wav"


def find_file():
    root = Path("data/mlaad_full/audio/fake")

    matches = list(root.rglob(TARGET_FILE))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {TARGET_FILE} under {root}"
        )

    return matches[0]


def test_chunks(audio, sr, chunk_seconds):
    total_seconds = len(audio) / sr
    total_chunks = int(total_seconds // chunk_seconds)

    print()
    print("=" * 60)
    print(f"SPOOF {chunk_seconds}-SECOND CHUNK TEST")
    print("=" * 60)
    print(f"Duration: {total_seconds:.2f} sec")
    print(f"Complete chunks: {total_chunks}")
    print()

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        with httpx.Client(timeout=120.0) as client:

            for i in range(total_chunks):

                start_sample = int(i * chunk_seconds * sr)
                end_sample = int((i + 1) * chunk_seconds * sr)

                chunk = audio[start_sample:end_sample]

                chunk_path = os.path.join(
                    temp_dir,
                    f"chunk_{i + 1:02d}.wav"
                )

                sf.write(
                    chunk_path,
                    chunk,
                    sr,
                    subtype="PCM_16"
                )

                with open(chunk_path, "rb") as f:
                    response = client.post(
                        API_URL,
                        files={
                            "file": (
                                f"chunk_{i + 1:02d}.wav",
                                f,
                                "audio/wav"
                            )
                        }
                    )

                response.raise_for_status()

                result = response.json()

                spoof = result["spoofProbability"] * 100
                label = result["label"]

                results.append(spoof)

                start_time = i * chunk_seconds
                end_time = start_time + chunk_seconds

                print(
                    f"Chunk {i + 1:02d} "
                    f"({start_time:02d}-{end_time:02d} sec) -> "
                    f"{spoof:6.2f}% spoof -> {label}"
                )

    if results:
        print()
        print("-" * 60)
        print(f"Average: {sum(results) / len(results):.2f}%")
        print(f"Lowest:  {min(results):.2f}%")
        print(f"Highest: {max(results):.2f}%")
        print(
            f"Chunks >= 50%: "
            f"{sum(x >= 50 for x in results)}/{len(results)}"
        )


def main():

    input_file = find_file()

    print("=" * 60)
    print("SPOOF CHUNK TEST")
    print("=" * 60)
    print(f"File: {input_file}")

    audio, sr = librosa.load(
        str(input_file),
        sr=SAMPLE_RATE,
        mono=True
    )

    print(f"Duration: {len(audio) / sr:.2f} sec")
    print(f"Sample rate: {sr} Hz")

    test_chunks(audio, sr, 3)
    test_chunks(audio, sr, 5)

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
