from scipy.io import wavfile
import numpy as np

from kws_config import (
    ALL_DATASET_DIRS,
    NOISE_LABEL,
    CLIPPING_ABS_THRESHOLD,
    CLIPPING_RATIO_THRESHOLD,
    DATASET_DIR,
    LOW_PEAK_THRESHOLD,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    SAMPLE_RATE,
)


def main():
    bad_duration = []
    bad_samplerate = []
    clipped = []
    empty_or_low = []
    stereo_files = []

    for label in ALL_DATASET_DIRS:
        label_dir = DATASET_DIR / label
        if not label_dir.exists():
            continue

        for wav_path in sorted(label_dir.glob("*.wav")):
            try:
                fs, audio = wavfile.read(wav_path)
            except Exception as exc:
                bad_samplerate.append((str(wav_path), f"read_error:{exc}"))
                continue

            if audio.ndim > 1:
                stereo_files.append(str(wav_path))
                audio = audio[:, 0]

            duration = len(audio) / float(fs)
            if fs != SAMPLE_RATE:
                bad_samplerate.append((str(wav_path), fs))
            if label != NOISE_LABEL and (duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS):
                bad_duration.append((str(wav_path), round(duration, 3)))

            clip_points = int(np.sum(np.abs(audio) >= CLIPPING_ABS_THRESHOLD))
            clip_ratio = clip_points / max(len(audio), 1)
            if clip_ratio > CLIPPING_RATIO_THRESHOLD:
                clipped.append(str(wav_path))

            peak = int(np.max(np.abs(audio))) if len(audio) else 0
            if peak < LOW_PEAK_THRESHOLD:
                empty_or_low.append(str(wav_path))

    print("=== KET QUA KIEM TRA ===")
    print("Sai sample rate:", len(bad_samplerate))
    print("Sai thoi luong:", len(bad_duration))
    print("Bi clipping/vo tieng:", len(clipped))
    print("Qua nho/im lang bat thuong:", len(empty_or_low))
    print("Stereo files:", len(stereo_files))

    print("\n--- File sai sample rate ---")
    for item in bad_samplerate[:20]:
        print(item)

    print("\n--- File sai thoi luong ---")
    for item in bad_duration[:20]:
        print(item)

    print("\n--- File bi clipping ---")
    for item in clipped[:20]:
        print(item)

    print("\n--- File qua nho/im lang bat thuong ---")
    for item in empty_or_low[:20]:
        print(item)

    print("\n--- File stereo ---")
    for item in stereo_files[:20]:
        print(item)


if __name__ == "__main__":
    main()
