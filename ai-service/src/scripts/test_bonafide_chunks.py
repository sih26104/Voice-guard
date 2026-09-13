import os
import tempfile
import httpx
import librosa
import soundfile as sf

INPUT_FILE = r"data\mlaad_full\audio\original\de\altehaus_001_f000016.wav"
API_URL = "http://127.0.0.1:8000/predict"

CHUNK_SECONDS = 3
SAMPLE_RATE = 16000


def main():
    print("=" * 60)
    print("BONAFIDE 3-SECOND CHUNK TEST")
    print("=" * 60)
    print(f"File: {os.path.basename(INPUT_FILE)}")

    audio, sr = librosa.load(
        INPUT_FILE,
        sr=SAMPLE_RATE,
        mono=True
    )

    total_seconds = len(audio) / sr
    total_chunks = int(total_seconds // CHUNK_SECONDS)

    print(f"Duration: {total_seconds:.2f} sec")
    print(f"Sample rate: {sr} Hz")
    print(f"Testing: {total_chunks} complete 3-second chunks")
    print()

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        with httpx.Client(timeout=120.0) as client:

            for i in range(total_chunks):
                start = i * CHUNK_SECONDS
                end = start + CHUNK_SECONDS

                chunk = audio[start * sr:end * sr]

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

                print(
                    f"Chunk {i + 1:02d} "
                    f"({start:02d}-{end:02d} sec) -> "
                    f"{spoof:6.2f}% spoof -> {label}"
                )

    print()
    print("=" * 60)

    if results:
        average = sum(results) / len(results)

        print(f"Average spoof probability: {average:.2f}%")
        print(f"Lowest: {min(results):.2f}%")
        print(f"Highest: {max(results):.2f}%")

        bonafide_count = sum(x < 50 for x in results)
        spoof_count = len(results) - bonafide_count

        print()
        print(f"Chunks below 50%: {bonafide_count}/{len(results)}")
        print(f"Chunks at/above 50%: {spoof_count}/{len(results)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
