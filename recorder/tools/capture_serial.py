"""Doc stream base64 tu firmware recorder, cat thanh clip 1s, luu vao other/.

Vi du:
    python capture_serial.py --port /dev/ttyACM0 --seconds 60 \
        --out-dir ../../kws_voice/kws_dataset/other --prefix inmp_sil

Thu nhieu canh: im lang, bat quat, noi chuyen nen... moi canh mot lan chay
voi --prefix khac nhau. Sau do chay lai pipeline train (05 -> 06 -> 07 -> 09).
"""
import argparse
import base64
import time
from pathlib import Path

import numpy as np
import serial
from scipy.io import wavfile

SAMPLE_RATE = 16000
MARKER = b"<<<STREAM"
# Neo theo vi tri script (khong phu thuoc thu muc dang chay):
# recorder/tools/ -> len esp32-kws-espnn/ -> kws_voice/kws_dataset/other
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "kws_voice" / "kws_dataset" / "other"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="vd /dev/ttyACM0 hoac /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--seconds", type=float, default=60.0, help="thoi luong thu")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--prefix", default="inmp")
    return p.parse_args()


def main():
    args = parse_args()
    ser = serial.Serial(args.port, args.baud, timeout=1)

    print("Cho marker <<<STREAM (reset board neu lau)...")
    while True:
        line = ser.readline()
        if not line:
            continue
        if MARKER in line:
            break

    print(f"Bat dau thu {args.seconds:.0f}s ...")
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        line = ser.readline().strip()
        if not line:
            continue
        try:
            data = base64.b64decode(line, validate=True)
        except Exception:
            continue  # bo qua dong rac (marker, log boot, dong bi cat)
        if len(data) % 2:
            continue  # chunk le byte (dong bi cat) -> bo de giu canh int16
        buf += data
    ser.close()

    raw = bytes(buf)
    raw = raw[: len(raw) - (len(raw) % 2)]  # an toan: tong phai chan
    pcm = np.frombuffer(raw, dtype="<i2")
    total_sec = len(pcm) / SAMPLE_RATE
    print(f"Thu duoc {total_sec:.1f}s ({len(pcm)} mau)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for start in range(0, len(pcm) - SAMPLE_RATE + 1, SAMPLE_RATE):
        clip = pcm[start:start + SAMPLE_RATE]
        wavfile.write(out_dir / f"{args.prefix}_{saved:04d}.wav", SAMPLE_RATE, clip)
        saved += 1

    print(f"Da luu {saved} clip 1s vao {out_dir}")


if __name__ == "__main__":
    main()
