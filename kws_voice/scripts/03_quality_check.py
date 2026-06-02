from scipy.io import wavfile
import numpy as np

from kws_config import (
    ALL_DATASET_DIRS,
    CLASS_NAMES,
    NOISE_LABEL,
    OTHER_LABEL,
    CLIPPING_ABS_THRESHOLD,
    CLIPPING_RATIO_THRESHOLD,
    DATASET_DIR,
    LOW_PEAK_THRESHOLD,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    SAMPLE_RATE,
)

# INMP441 on ESP32-S3 firmware (I2S 16-bit, no AGC): normal speech ~30-50cm → RMS 800-5000.
# Targets: RMS 800-6000, clipping < 1% of files.
_RMS_LOW = 300
_RMS_HIGH = 12000
_CLIP_FILE_RATIO = 0.01   # warn if >1% samples clipped in a file


def _gain_summary():
    print("=== GAIN SUMMARY (INMP441 target: RMS 800-6000, clip<1%) ===")
    all_rms = {}
    known = set(CLASS_NAMES)
    all_labels = sorted(d.name for d in DATASET_DIR.iterdir() if d.is_dir())
    for label in all_labels:
        label_dir = DATASET_DIR / label
        if not label_dir.exists():
            continue
        rms_list = []
        clipped_files = 0
        total = 0
        for wav_path in sorted(label_dir.glob("*.wav")):
            try:
                fs, audio = wavfile.read(wav_path)
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio[:, 0]
            total += 1
            rms_list.append(int(np.sqrt(np.mean(audio.astype(np.int32) ** 2))))
            clip_ratio = np.sum(np.abs(audio) >= CLIPPING_ABS_THRESHOLD) / max(len(audio), 1)
            if clip_ratio > _CLIP_FILE_RATIO:
                clipped_files += 1

        if not rms_list:
            continue
        rms_arr = np.array(rms_list)
        med = int(np.median(rms_arr))
        p10 = int(np.percentile(rms_arr, 10))
        p90 = int(np.percentile(rms_arr, 90))
        clip_pct = 100 * clipped_files / total if total else 0
        all_rms[label] = med

        if med > _RMS_HIGH or clip_pct > 5:
            status = "⚠ QUA TO / CLIP - ha gain"
        elif med < _RMS_LOW:
            status = "⚠ QUA NHO - tang gain"
        else:
            status = "OK"

        tag = "" if label in known else "  [chua trong CLASS_NAMES]"
        print(f"  {label:<12} files={total:4d}  RMS p10={p10:5d} med={med:5d} p90={p90:5d}"
              f"  clipped_files={clipped_files:3d} ({clip_pct:.0f}%)  {status}{tag}")

    if len(all_rms) >= 2:
        vals = list(all_rms.values())
        ratio = max(vals) / max(min(vals), 1)
        if ratio > 5:
            print(f"\n  [!] Chenh lech gain giua cac class qua lon (max/min={ratio:.0f}x) -"
                  " model se hoc bien do thay vi am hoc. Can thu lai cho dong deu.")
        else:
            print(f"\n  [OK] Cac class tuong doi dong deu (max/min={ratio:.1f}x)")
    print()


def main():
    _gain_summary()

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
