import argparse
import shutil

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from kws_config import (
    ALL_DATASET_DIRS,
    NOISE_LABEL,
    CLASS_NAMES,
    CLIPPING_ABS_THRESHOLD,
    CLIPPING_RATIO_THRESHOLD,
    DATASET_DIR,
    FIXED_DATASET_DIR,
    LOW_PEAK_THRESHOLD,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    N_SAMPLES,
    SAMPLE_RATE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="xoa file wav cu trong kws_dataset_fixed truoc khi tao lai")
    return parser.parse_args()


def fix_length(audio):
    if len(audio) == N_SAMPLES:
        return audio
    if len(audio) > N_SAMPLES:
        return audio[:N_SAMPLES]
    return np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")


def load_and_prepare(wav_path, label):
    fs, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    if len(audio) == 0:
        return None, "empty_audio"

    if label not in {NOISE_LABEL, "silence"}:
        duration = len(audio) / float(fs)
        if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
            return None, f"duration={duration:.3f}s outside [{MIN_DURATION_SECONDS:.2f}, {MAX_DURATION_SECONDS:.2f}]"

        clip_points = int(np.sum(np.abs(audio) >= CLIPPING_ABS_THRESHOLD))
        clip_ratio = clip_points / max(len(audio), 1)
        if clip_ratio > CLIPPING_RATIO_THRESHOLD:
            return None, f"clipping_ratio={clip_ratio:.3f} > {CLIPPING_RATIO_THRESHOLD:.3f}"

        peak = int(np.max(np.abs(audio))) if len(audio) else 0
        if peak < LOW_PEAK_THRESHOLD:
            return None, f"peak={peak} < {LOW_PEAK_THRESHOLD}"

    audio = audio.astype(np.float32)
    if fs != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, fs)

    audio = fix_length(audio)
    fixed = np.clip(audio, -32768, 32767).astype(np.int16)
    return fixed, None


def build_noise_chunks(wav_path):
    fs, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    audio = audio.astype(np.float32)
    if fs != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, fs)

    chunks = []
    for start in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES):
        chunk = audio[start : start + N_SAMPLES]
        chunk = np.clip(chunk, -32768, 32767).astype(np.int16)
        chunks.append((start, chunk))
    return chunks


def count_wavs(directory):
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.wav"))


def process_label(label, source_dir, target_dir):
    kept = 0
    skipped = 0

    for wav_path in sorted(source_dir.glob("*.wav")):
        try:
            fixed_audio, skip_reason = load_and_prepare(wav_path, label)
            if fixed_audio is None:
                print(f"  SKIP: {wav_path.name} ({skip_reason})")
                skipped += 1
                continue
            wavfile.write(target_dir / wav_path.name, SAMPLE_RATE, fixed_audio)
            kept += 1
        except Exception as exc:
            print(f"  SKIP: {wav_path.name} ({exc})")
            skipped += 1

    return kept, skipped


def process_background_label(source_dir, target_dir):
    kept = 0
    skipped = 0

    for wav_path in sorted(source_dir.glob("*.wav")):
        try:
            chunks = build_noise_chunks(wav_path)
            if not chunks:
                print(f"  SKIP: {wav_path.name} (too_short_for_chunk)")
                skipped += 1
                continue
            for start, chunk in chunks:
                out_name = f"{wav_path.stem}_{start}.wav"
                wavfile.write(target_dir / out_name, SAMPLE_RATE, chunk)
                kept += 1
        except Exception as exc:
            print(f"  SKIP: {wav_path.name} ({exc})")
            skipped += 1

    return kept, skipped


def process_dataset(args):
    total_kept = 0
    total_skipped = 0

    if args.clean:
        for label in ALL_DATASET_DIRS:
            target_dir = FIXED_DATASET_DIR / label
            if target_dir.exists():
                shutil.rmtree(target_dir)

    for label in ALL_DATASET_DIRS:
        source_dir = DATASET_DIR / label
        target_dir = FIXED_DATASET_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            continue

        if label == NOISE_LABEL:
            kept, skipped = process_background_label(source_dir, target_dir)
        else:
            kept, skipped = process_label(label, source_dir, target_dir)

        total_kept += kept
        total_skipped += skipped
        print(f"{label}: kept={kept}, skipped={skipped}")

    return total_kept, total_skipped


def print_summary(total_kept, total_skipped):
    train_total = 0
    for label in CLASS_NAMES:
        train_total += count_wavs(FIXED_DATASET_DIR / label)

    noise_total = count_wavs(FIXED_DATASET_DIR / NOISE_LABEL)
    all_total = train_total + noise_total

    print(f"\n=== TONG KET ===")
    print(f"Train classes total (script 06 se dung): {train_total}")
    print(f"Background noise total:                 {noise_total}")
    print(f"All fixed wav total:                    {all_total}")
    print(f"Giu lai (dem theo process):             {total_kept}")
    print(f"Bo qua:                                 {total_skipped}")
    print(f"Output:                                 {FIXED_DATASET_DIR}")


def run_pipeline(args):
    total_kept, total_skipped = process_dataset(args)
    print_summary(total_kept, total_skipped)

def main():
    args = parse_args()
    FIXED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    run_pipeline(args)


if __name__ == "__main__":
    main()
