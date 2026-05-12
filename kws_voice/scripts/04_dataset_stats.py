import argparse

from kws_config import ALL_DATASET_DIRS, DATASET_DIR, FIXED_DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true", help="thong ke kws_dataset_fixed thay vi kws_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = FIXED_DATASET_DIR if args.fixed else DATASET_DIR

    print(f"=== THONG KE DATASET: {base_dir.name} ===")
    total = 0
    counts = []

    for label in ALL_DATASET_DIRS:
        label_dir = base_dir / label
        count = len(list(label_dir.glob("*.wav"))) if label_dir.exists() else 0
        counts.append(count)
        total += count
        print(f"{label}: {count}")

    print(f"Tong so file: {total}")

    non_zero = [count for count in counts if count > 0]
    if len(non_zero) >= 2:
        max_count = max(non_zero)
        min_count = min(non_zero)
        if min_count * 2 < max_count:
            print("Canh bao: dataset dang lech class kha nhieu.")


if __name__ == "__main__":
    main()
