import hashlib
import json
from collections import defaultdict

from kws_config import (
    CLASS_NAMES,
    FFT_LENGTH,
    FIXED_DATASET_DIR,
    FRAME_LENGTH,
    FRAME_STEP,
    MEL_BINS,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    TFDATA_DIR,
    WAKE_WORD,
    NOISE_LABEL,
    OTHER_LABEL,
    OTHER_NOISE_MANIFEST_PATH,
)

ARCHIVE_DIR = FIXED_DATASET_DIR.parent / "kws_dataset" / "archive"
VALIDATION_LIST_PATH = ARCHIVE_DIR / "validation_list.txt"
TESTING_LIST_PATH = ARCHIVE_DIR / "testing_list.txt"


def load_split_list(list_path):
    items = set()
    if not list_path.exists():
        return items

    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.add(line.replace("\\", "/"))
    return items


def load_other_noise_sources():
    if not OTHER_NOISE_MANIFEST_PATH.exists():
        return set()
    with open(OTHER_NOISE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("sources", []))


def stable_split_from_group(group_key):
    digest = hashlib.sha1(group_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 15:
        return "val"
    if bucket < 30:
        return "test"
    return "train"


def infer_other_source_rel(wav_path):
    stem = wav_path.stem
    if stem.startswith("noise__"):
        _, source_stem, _ = stem.split("__", 2)
        return f"{NOISE_LABEL}/{source_stem}.wav"
    if stem.startswith("archive__"):
        _, class_name, rest = stem.split("__", 2)
        return f"{class_name}/{rest}.wav"
    return f"{OTHER_LABEL}/{wav_path.name}"


def infer_source_rel(class_name, wav_path):
    if class_name == OTHER_LABEL:
        return infer_other_source_rel(wav_path).replace("\\", "/")
    return f"{class_name}/{wav_path.name}"


def infer_group_key(class_name, wav_path):
    source_rel = infer_source_rel(class_name, wav_path)
    if class_name == OTHER_LABEL and wav_path.stem.startswith("noise__"):
        _, source_stem, _ = wav_path.stem.split("__", 2)
        return f"noise::{source_stem}"
    if "_nohash_" in source_rel:
        prefix, _ = source_rel.split("_nohash_", 1)
        return f"{prefix}_nohash_"
    return source_rel


def assign_split(class_name, wav_path, validation_set, testing_set):
    source_rel = infer_source_rel(class_name, wav_path).replace("\\", "/")

    if source_rel in validation_set:
        return "val"
    if source_rel in testing_set:
        return "test"

    group_key = infer_group_key(class_name, wav_path)
    return stable_split_from_group(group_key)


def list_files(validation_set, testing_set):
    splits = {"train": [], "val": [], "test": []}
    counts_by_class = defaultdict(lambda: defaultdict(int))

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = FIXED_DATASET_DIR / class_name
        if not class_dir.exists():
            continue

        for wav_path in sorted(class_dir.glob("*.wav")):
            split_name = assign_split(class_name, wav_path, validation_set, testing_set)
            splits[split_name].append((str(wav_path), class_index))
            counts_by_class[class_name][split_name] += 1

    return splits, counts_by_class


def save_meta():
    meta = {
        "classes": CLASS_NAMES,
        "class_to_id": {name: index for index, name in enumerate(CLASS_NAMES)},
        "fs": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
        "mel_bins": MEL_BINS,
        "frame_length": FRAME_LENGTH,
        "frame_step": FRAME_STEP,
        "fft_length": FFT_LENGTH,
        "n_frames": N_FRAMES,
        "wake_word": WAKE_WORD,
    }

    with open(TFDATA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_splits(splits):
    for name, split in [("train.txt", splits["train"]), ("val.txt", splits["val"]), ("test.txt", splits["test"])]:
        with open(TFDATA_DIR / name, "w", encoding="utf-8") as f:
            for path, class_id in split:
                f.write(f"{path}\t{class_id}\n")


def print_summary(splits, counts_by_class, other_noise_sources):
    total = sum(len(items) for items in splits.values())
    print(f"Total: {total}")
    print(f"Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Classes: {CLASS_NAMES}")
    if other_noise_sources:
        print(f"other noise sources used: {len(other_noise_sources)}")
    for class_name in CLASS_NAMES:
        counts = counts_by_class[class_name]
        class_total = counts["train"] + counts["val"] + counts["test"]
        print(
            f"  {class_name}: total={class_total}, train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )
    print(f"Saved: {TFDATA_DIR}")


def validate_layout():
    other_dir = FIXED_DATASET_DIR / OTHER_LABEL
    if not other_dir.exists():
        raise ValueError(f"Khong tim thay thu muc {other_dir}. Hay chay script tao class other truoc.")

    stale_dirs = [path for path in (FIXED_DATASET_DIR / "silence", FIXED_DATASET_DIR / "unknown") if path.exists()]
    if stale_dirs:
        print(f"Warning: stale dirs will be ignored: {[str(path) for path in stale_dirs]}")


def main():
    TFDATA_DIR.mkdir(parents=True, exist_ok=True)
    validate_layout()

    validation_set = load_split_list(VALIDATION_LIST_PATH)
    testing_set = load_split_list(TESTING_LIST_PATH)
    other_noise_sources = load_other_noise_sources()
    splits, counts_by_class = list_files(validation_set, testing_set)

    total = sum(len(items) for items in splits.values())
    if total == 0:
        raise ValueError("Khong tim thay file wav nao trong kws_dataset_fixed.")

    save_meta()
    save_splits(splits)
    print_summary(splits, counts_by_class, other_noise_sources)


if __name__ == "__main__":
    main()
