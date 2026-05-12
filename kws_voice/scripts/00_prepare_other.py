import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from kws_config import (
    DATASET_DIR,
    N_SAMPLES,
    SAMPLE_RATE,
    NOISE_LABEL,
    OTHER_LABEL,
    OTHER_TARGET_COUNT,
    OTHER_NOISE_SOURCE_FRACTION,
    OTHER_HOP_SAMPLES,
    OTHER_NOISE_MANIFEST_PATH,
    KEYWORD_CLASS_NAMES,
)

ARCHIVE_DIR = DATASET_DIR / "archive"
KEYWORDS = set(KEYWORD_CLASS_NAMES)


def parse_noise_fraction(value):
    fraction = float(value)
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return 1.0
    return fraction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="chi thong ke, khong ghi file")
    parser.add_argument("--clean", action="store_true", help="xoa file cu trong other truoc khi tao lai")
    parser.add_argument("--target-other", type=int, default=OTHER_TARGET_COUNT, help="so file other muc tieu")
    parser.add_argument(
        "--noise-source-fraction",
        type=parse_noise_fraction,
        default=OTHER_NOISE_SOURCE_FRACTION,
        help="ti le noise source duoc dung de tao other",
    )
    return parser.parse_args()


def ensure_mono(audio):
    if audio.ndim > 1:
        return audio[:, 0]
    return audio


def resample_audio(audio, fs):
    audio = audio.astype(np.float32)
    if fs != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, fs)
    return audio


def fix_length(audio):
    if len(audio) > N_SAMPLES:
        return audio[:N_SAMPLES]
    if len(audio) < N_SAMPLES:
        return np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    return audio


def to_int16(audio):
    return np.clip(audio, -32768, 32767).astype(np.int16)


def load_fixed_audio(wav_path):
    fs, audio = wavfile.read(wav_path)
    audio = ensure_mono(audio)
    audio = resample_audio(audio, fs)
    audio = fix_length(audio)
    return to_int16(audio)


def iter_noise_wavs():
    noise_dir = DATASET_DIR / NOISE_LABEL
    return sorted(p for p in noise_dir.glob("*.wav") if p.is_file())


def select_noise_wavs(noise_source_fraction):
    all_noise_wavs = iter_noise_wavs()
    if not all_noise_wavs or noise_source_fraction <= 0.0:
        print(f"other noise sources: selected=0, total={len(all_noise_wavs)}")
        return []

    keep_count = max(1, int(round(len(all_noise_wavs) * noise_source_fraction)))
    keep_count = min(keep_count, len(all_noise_wavs))
    selected = all_noise_wavs[:keep_count]
    print(f"other noise sources: selected={len(selected)}, total={len(all_noise_wavs)}")
    return selected


def build_noise_other_candidates(noise_wavs):
    candidates = []
    for wav_path in noise_wavs:
        fs, audio = wavfile.read(wav_path)
        audio = ensure_mono(audio)
        audio = resample_audio(audio, fs)
        if len(audio) < N_SAMPLES:
            continue

        step = max(1, OTHER_HOP_SAMPLES)
        local_count = 0
        for start in range(0, len(audio) - N_SAMPLES + 1, step):
            chunk = to_int16(audio[start : start + N_SAMPLES])
            candidates.append(
                {
                    "audio": chunk,
                    "out_name": f"noise__{wav_path.stem}__{start}.wav",
                    "group_key": f"noise::{wav_path.stem}",
                    "source_rel": str(wav_path.relative_to(DATASET_DIR)).replace("\\", "/"),
                }
            )
            local_count += 1

        if local_count == 0:
            candidates.append(
                {
                    "audio": to_int16(fix_length(audio)),
                    "out_name": f"noise__{wav_path.stem}__0.wav",
                    "group_key": f"noise::{wav_path.stem}",
                    "source_rel": str(wav_path.relative_to(DATASET_DIR)).replace("\\", "/"),
                }
            )
    return candidates


def iter_archive_other_class_dirs():
    if not ARCHIVE_DIR.exists():
        return []
    return [
        path
        for path in sorted(ARCHIVE_DIR.iterdir())
        if path.is_dir() and path.name not in KEYWORDS and path.name != "_background_noise_"
    ]


def build_archive_other_candidates():
    candidates = []
    for class_dir in iter_archive_other_class_dirs():
        for wav_path in sorted(class_dir.glob("*.wav")):
            rel = wav_path.relative_to(ARCHIVE_DIR)
            candidates.append(
                {
                    "wav_path": wav_path,
                    "out_name": f"archive__{class_dir.name}__{wav_path.stem}.wav",
                    "group_key": str(rel).replace("\\", "/"),
                    "source_rel": str(rel).replace("\\", "/"),
                }
            )
    return candidates


def interleave_candidates(noise_candidates, archive_candidates, target_total):
    selected = []
    noise_index = 0
    archive_index = 0
    prefer_noise = True

    while len(selected) < target_total and (noise_index < len(noise_candidates) or archive_index < len(archive_candidates)):
        if prefer_noise and noise_index < len(noise_candidates):
            selected.append(noise_candidates[noise_index])
            noise_index += 1
        elif archive_index < len(archive_candidates):
            selected.append(archive_candidates[archive_index])
            archive_index += 1
        elif noise_index < len(noise_candidates):
            selected.append(noise_candidates[noise_index])
            noise_index += 1
        prefer_noise = not prefer_noise

    return selected


def clean_outputs(*output_dirs):
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for wav_path in output_dir.glob("*.wav"):
            wav_path.unlink()


def load_candidate_audio(item):
    if "audio" in item:
        return item["audio"]
    return load_fixed_audio(item["wav_path"])


def write_candidates(output_dir, candidates):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in candidates:
        wavfile.write(output_dir / item["out_name"], SAMPLE_RATE, load_candidate_audio(item))
        written += 1
    return written


def write_other_manifest(candidates, dry_run):
    manifest = {
        "noise_label": NOISE_LABEL,
        "other_label": OTHER_LABEL,
        "sources": sorted(
            {
                item["source_rel"]
                for item in candidates
                if item["source_rel"].startswith(f"{NOISE_LABEL}/")
            }
        ),
    }
    if dry_run:
        print(f"other_manifest_sources={len(manifest['sources'])} -> {OTHER_NOISE_MANIFEST_PATH}")
        return
    with open(OTHER_NOISE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def print_other_summary(noise_candidates, archive_candidates, selected, target_total):
    selected_groups = defaultdict(int)
    noise_selected = 0
    archive_selected = 0
    for item in selected:
        selected_groups[item["group_key"]] += 1
        if item["source_rel"].startswith(f"{NOISE_LABEL}/"):
            noise_selected += 1
        else:
            archive_selected += 1

    print(f"other: target={target_total}, selected={len(selected)}")
    print(f"other: noise_candidates={len(noise_candidates)}, archive_candidates={len(archive_candidates)}")
    print(f"other: selected_noise={noise_selected}, selected_archive={archive_selected}")
    print(f"other: unique_groups={len(selected_groups)}")


def main():
    args = parse_args()

    other_dir = DATASET_DIR / OTHER_LABEL
    if args.clean and not args.dry_run:
        clean_outputs(other_dir)

    selected_noise_wavs = select_noise_wavs(args.noise_source_fraction)
    noise_candidates = build_noise_other_candidates(selected_noise_wavs)
    archive_candidates = build_archive_other_candidates()
    selected = interleave_candidates(noise_candidates, archive_candidates, args.target_other)

    print_other_summary(noise_candidates, archive_candidates, selected, args.target_other)

    if args.dry_run:
        write_other_manifest(selected, dry_run=True)
        print("Dry run complete. Khong ghi file nao.")
        return

    written = write_candidates(other_dir, selected)
    write_other_manifest(selected, dry_run=False)
    print(f"Saved other: {written} file(s) -> {other_dir}")


if __name__ == "__main__":
    main()
