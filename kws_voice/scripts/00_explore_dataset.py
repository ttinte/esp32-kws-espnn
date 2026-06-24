import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from kws_config import (
    ALL_DATASET_DIRS,
    DATASET_DIR,
    FIXED_DATASET_DIR,
    OUTPUT_DIR,
    SAMPLE_RATE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kham pha phan bo du lieu WAV cua bo du lieu KWS."
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="doc kws_dataset_fixed thay vi kws_dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="duong dan file PNG dau ra",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="duong dan file CSV thong ke tung WAV",
    )
    return parser.parse_args()


def audio_to_float(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        return audio.astype(np.float32) / scale
    return audio.astype(np.float32)


def dbfs(value):
    return 20.0 * np.log10(max(float(value), 1e-12))


def inspect_wav(path, label):
    sample_rate, audio = wavfile.read(path)
    channels = 1 if audio.ndim == 1 else audio.shape[1]

    if audio.ndim > 1:
        audio = np.mean(audio_to_float(audio), axis=1)
    else:
        audio = audio_to_float(audio)

    duration = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
    rms = np.sqrt(np.mean(np.square(audio))) if len(audio) else 0.0
    peak = np.max(np.abs(audio)) if len(audio) else 0.0

    return {
        "label": label,
        "file": str(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_s": duration,
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
    }


def collect_rows(base_dir):
    rows = []
    errors = []

    for label in ALL_DATASET_DIRS:
        label_dir = base_dir / label
        if not label_dir.exists():
            continue

        for wav_path in sorted(label_dir.glob("*.wav")):
            try:
                rows.append(inspect_wav(wav_path, label))
            except Exception as exc:
                errors.append((wav_path, str(exc)))

    return rows, errors


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "file",
        "sample_rate",
        "channels",
        "duration_s",
        "rms_dbfs",
        "peak_dbfs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def values_by_label(rows, key, labels):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row[key])
    return [grouped[label] for label in labels]


def draw_dashboard(rows, base_dir, output_path):
    counts = Counter(row["label"] for row in rows)
    labels = [label for label in ALL_DATASET_DIRS if counts[label] > 0]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Khám phá dữ liệu ban đầu — {base_dir.name} ({len(rows):,} file)",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    count_values = [counts[label] for label in labels]
    bars = ax.bar(labels, count_values, color=colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Số lượng mẫu theo lớp")
    ax.set_ylabel("Số file WAV")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    duration_groups = values_by_label(rows, "duration_s", labels)
    ax.boxplot(duration_groups, labels=labels, showfliers=False)
    ax.axhline(
        1.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="Mục tiêu 1 giây",
    )
    ax.set_title("Phân bố thời lượng theo lớp")
    ax.set_ylabel("Thời lượng (giây)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    rms_groups = values_by_label(rows, "rms_dbfs", labels)
    ax.boxplot(rms_groups, labels=labels, showfliers=False)
    ax.set_title("Phân bố âm lượng RMS")
    ax.set_ylabel("RMS (dBFS)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    peak_groups = values_by_label(rows, "peak_dbfs", labels)
    ax.boxplot(peak_groups, labels=labels, showfliers=False)
    ax.axhline(
        -0.1,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="Gần ngưỡng clipping",
    )
    ax.set_title("Phân bố mức peak")
    ax.set_ylabel("Peak (dBFS)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows, errors, base_dir):
    counts = Counter(row["label"] for row in rows)
    sample_rates = Counter(row["sample_rate"] for row in rows)
    stereo_count = sum(row["channels"] > 1 for row in rows)

    print(f"=== EDA DATASET: {base_dir.name} ===")
    for label in ALL_DATASET_DIRS:
        if counts[label]:
            print(f"{label}: {counts[label]} file")
    print(f"Tong: {len(rows)} file")
    print(f"Sample rate: {dict(sorted(sample_rates.items()))}")
    wrong_sample_rate_count = sum(
        rate != SAMPLE_RATE for rate in sample_rates.elements()
    )
    print(
        f"Khac sample rate muc tieu {SAMPLE_RATE} Hz: "
        f"{wrong_sample_rate_count}"
    )
    print(f"Stereo: {stereo_count}")
    print(f"Loi doc WAV: {len(errors)}")

    if errors:
        print("\nMot so file loi:")
        for path, message in errors[:10]:
            print(f"- {path}: {message}")


def main():
    args = parse_args()
    base_dir = FIXED_DATASET_DIR if args.fixed else DATASET_DIR
    suffix = "fixed" if args.fixed else "raw"
    plot_path = args.output or OUTPUT_DIR / f"eda_distribution_{suffix}.png"
    csv_path = args.csv or OUTPUT_DIR / f"eda_audio_stats_{suffix}.csv"

    rows, errors = collect_rows(base_dir)
    if not rows:
        raise SystemExit(f"Khong tim thay file WAV trong {base_dir}")

    draw_dashboard(rows, base_dir, plot_path)
    write_csv(rows, csv_path)
    print_summary(rows, errors, base_dir)
    print(f"\nDa luu bieu do: {plot_path}")
    print(f"Da luu thong ke: {csv_path}")


if __name__ == "__main__":
    main()
