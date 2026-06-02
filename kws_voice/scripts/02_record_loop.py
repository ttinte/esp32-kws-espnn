import argparse
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from kws_config import CLASS_NAMES, CLIP_DURATION_SECONDS, DATASET_DIR, N_SAMPLES, SAMPLE_RATE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="ten class can thu am")
    parser.add_argument("--device", type=int, default=None, help="microphone device index")
    parser.add_argument("--seconds", type=float, default=CLIP_DURATION_SECONDS, help="clip duration in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    label = args.label.strip().lower()
    if label not in CLASS_NAMES:
        print(f"[Warning:] '{label}' is not in CLASS_NAMES.")

    output_dir = DATASET_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print(f"Dang thu label: {label}")
    print(f"Moi mau dai: {args.seconds:.2f} giay")
    print(f"Tan so lay mau: {SAMPLE_RATE} Hz")
    print("Nhan Enter de thu. Go q roi Enter de thoat.")
    print("====================================")

    count = 0
    total_samples = int(round(args.seconds * SAMPLE_RATE))

    while True:
        cmd = input(">> ").strip().lower()
        if cmd == "q":
            break

        print("Dang thu...")
        audio = sd.rec(
            total_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            device=args.device,
        )
        sd.wait()

        audio = np.squeeze(audio, axis=-1)
        if len(audio) > N_SAMPLES:
            audio = audio[:N_SAMPLES]
        elif len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"{label}_{timestamp}.wav"
        write(output_path, SAMPLE_RATE, audio)
        count += 1
        print(f"Saved: {output_path} | total: {count}")

    print(f"Hoan tat. Tong so mau vua thu: {count}")


if __name__ == "__main__":
    main()
