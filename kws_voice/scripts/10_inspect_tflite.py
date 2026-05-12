import argparse
import json

import numpy as np
import tensorflow as tf

from kws_config import (
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_EDGE_HZ,
    MEL_BINS,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    TFLITE_MODEL_PATH,
    TFDATA_DIR,
    UPPER_EDGE_HZ,
)


K_PI = 3.14159265358979323846


def load_wav_np(path):
    audio_bin = tf.io.read_file(path)
    wave, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    wave = tf.squeeze(wave, axis=-1).numpy()
    if len(wave) > N_SAMPLES:
        wave = wave[:N_SAMPLES]
    elif len(wave) < N_SAMPLES:
        wave = np.pad(wave, (0, N_SAMPLES - len(wave)))
    return wave.astype(np.float32)


def compute_logmel_tf(wave):
    wave_tf = tf.constant(wave, dtype=tf.float32)
    stft = tf.signal.stft(
        wave_tf,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
    )
    spec = tf.abs(stft) ** 2
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BINS,
        num_spectrogram_bins=FFT_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_EDGE_HZ,
        upper_edge_hertz=UPPER_EDGE_HZ,
    )
    logmel = tf.math.log(tf.matmul(spec, mel_w) + 1e-6).numpy()
    if logmel.shape[0] > N_FRAMES:
        logmel = logmel[:N_FRAMES, :]
    elif logmel.shape[0] < N_FRAMES:
        logmel = np.pad(logmel, ((0, N_FRAMES - logmel.shape[0]), (0, 0)))
    return logmel.astype(np.float32)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_hann_window_fw():
    i = np.arange(FRAME_LENGTH, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(2.0 * K_PI * i / FRAME_LENGTH)


def build_mel_bins_fw():
    lower_mel = hz_to_mel(LOWER_EDGE_HZ)
    upper_mel = hz_to_mel(UPPER_EDGE_HZ)
    bin_points = []
    fft_bins = FFT_LENGTH // 2 + 1
    for i in range(MEL_BINS + 2):
        mel = lower_mel + (upper_mel - lower_mel) * i / (MEL_BINS + 1)
        hz = mel_to_hz(mel)
        bin_idx = int(np.floor((FFT_LENGTH + 1) * hz / SAMPLE_RATE))
        bin_idx = max(0, min(bin_idx, fft_bins - 1))
        bin_points.append(bin_idx)
    left = np.array(bin_points[:-2], dtype=np.int32)
    center = np.array(bin_points[1:-1], dtype=np.int32)
    right = np.array(bin_points[2:], dtype=np.int32)
    return left, center, right


def compute_logmel_fw(wave):
    hann = build_hann_window_fw()
    left_bins, center_bins, right_bins = build_mel_bins_fw()
    logmel = np.zeros((N_FRAMES, MEL_BINS), dtype=np.float32)
    for frame in range(N_FRAMES):
        offset = frame * FRAME_STEP
        frame_wave = wave[offset : offset + FRAME_LENGTH]
        fft_in = np.zeros(FFT_LENGTH, dtype=np.float32)
        fft_in[:FRAME_LENGTH] = frame_wave * hann
        spectrum = np.fft.rfft(fft_in, n=FFT_LENGTH)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        for mel in range(MEL_BINS):
            left = left_bins[mel]
            center = center_bins[mel]
            right = right_bins[mel]
            mel_energy = 0.0
            if center > left:
                denom = float(center - left)
                for bin_idx in range(left, center):
                    mel_energy += power[bin_idx] * float(bin_idx - left) / denom
            if right > center:
                denom = float(right - center)
                for bin_idx in range(center, right):
                    mel_energy += power[bin_idx] * float(right - bin_idx) / denom
            logmel[frame, mel] = np.log(mel_energy + 1e-6)
    return logmel


def quantize_feature(logmel, input_detail):
    scale = input_detail["quantization_parameters"]["scales"][0]
    zp = input_detail["quantization_parameters"]["zero_points"][0]
    q = np.clip(np.round(logmel / scale) + zp, -128, 127).astype(np.int8)
    return q



def quantize_feature_4d(logmel, input_detail):
    return np.expand_dims(quantize_feature(logmel, input_detail), axis=(0, -1))


def run_tflite(interpreter, input_detail, output_detail, feat):
    tensor = np.expand_dims(feat, axis=(0, -1))
    interpreter.set_tensor(input_detail["index"], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]
    if output_detail["dtype"] == np.int8:
        scale = output_detail["quantization_parameters"]["scales"][0]
        zp = output_detail["quantization_parameters"]["zero_points"][0]
        output = (output.astype(np.float32) - zp) * scale
    return output


def load_class_names():
    meta_path = TFDATA_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classes = meta.get("classes", [])
        if classes:
            return classes
    return None


def topk_str(probs, class_names, k=3):
    indices = np.argsort(probs)[::-1][:k]
    return ", ".join(f"{class_names[i]}:{probs[i]:.3f}" for i in indices)


def print_stats(name, feat, qfeat, probs, class_names):
    print(f"\n=== {name} ===")
    print(f"feat range: [{feat.min():.3f}, {feat.max():.3f}]")
    print(f"q range: [{int(qfeat.min())}, {int(qfeat.max())}]")
    print(f"top3: {topk_str(probs, class_names)}")


def compare_frontends(args, interpreter, input_detail, output_detail, class_names):
    wave = load_wav_np(args.wav)
    feat_tf = compute_logmel_tf(wave)
    feat_fw = compute_logmel_fw(wave)
    q_tf = quantize_feature(feat_tf, input_detail)
    q_fw = quantize_feature(feat_fw, input_detail)
    probs_tf = run_tflite(interpreter, input_detail, output_detail, q_tf)
    probs_fw = run_tflite(interpreter, input_detail, output_detail, q_fw)

    print(f"=== Compare frontends for: {args.wav} ===")
    print(f"audio range: [{wave.min():.3f}, {wave.max():.3f}]")
    print_stats("TensorFlow frontend", feat_tf, q_tf, probs_tf, class_names)
    print_stats("Firmware-like frontend", feat_fw, q_fw, probs_fw, class_names)
    diff = feat_fw - feat_tf
    qdiff = q_fw.astype(np.int16) - q_tf.astype(np.int16)
    print("\n=== Deltas (firmware-like - TensorFlow) ===")
    print(f"feat abs max: {np.max(np.abs(diff)):.6f}")
    print(f"feat mean abs: {np.mean(np.abs(diff)):.6f}")
    print(f"q abs max: {int(np.max(np.abs(qdiff)))}")
    print(f"q mean abs: {np.mean(np.abs(qdiff)):.6f}")
    print(f"prob abs max: {np.max(np.abs(probs_fw - probs_tf)):.6f}")
    print(f"prob mean abs: {np.mean(np.abs(probs_fw - probs_tf)):.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(TFLITE_MODEL_PATH), help="path to .tflite file")
    parser.add_argument("--wav", type=str, help="compare TensorFlow and firmware-like frontend on this wav")
    parser.add_argument("--json", action="store_true", help="print model IO details as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_detail = input_details[0]
    output_detail = output_details[0]

    class_names = load_class_names()
    if class_names is None:
        class_names = [f"class_{i}" for i in range(int(output_detail["shape"][-1]))]

    print(f"=== TFLite Model: {model_path} ===")
    print(f"Model size: {len(model_bytes)} bytes")
    print(f"Classes: {class_names}")

    if args.json:
        payload = {
            "input": {
                "name": input_detail["name"],
                "shape": input_detail["shape"].tolist(),
                "dtype": input_detail["dtype"].__name__,
                "scale": input_detail["quantization_parameters"]["scales"].tolist(),
                "zero_points": input_detail["quantization_parameters"]["zero_points"].tolist(),
            },
            "output": {
                "name": output_detail["name"],
                "shape": output_detail["shape"].tolist(),
                "dtype": output_detail["dtype"].__name__,
                "scale": output_detail["quantization_parameters"]["scales"].tolist(),
                "zero_points": output_detail["quantization_parameters"]["zero_points"].tolist(),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        for i, detail in enumerate(input_details):
            print(f"\n--- Input [{i}] ---")
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype'].__name__}")
            qp = detail.get("quantization_parameters", {})
            if qp.get("scales") is not None and len(qp["scales"]) > 0:
                print(f"  Scale: {qp['scales']}")
                print(f"  Zero point: {qp['zero_points']}")

        for i, detail in enumerate(output_details):
            print(f"\n--- Output [{i}] ---")
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype'].__name__}")
            qp = detail.get("quantization_parameters", {})
            if qp.get("scales") is not None and len(qp["scales"]) > 0:
                print(f"  Scale: {qp['scales']}")
                print(f"  Zero point: {qp['zero_points']}")

    if args.wav:
        compare_frontends(args, interpreter, input_detail, output_detail, class_names)


if __name__ == "__main__":
    main()
