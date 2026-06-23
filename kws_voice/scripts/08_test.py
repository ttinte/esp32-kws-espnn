import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from kws_config import (
    BEST_MODEL_PATH,
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_EDGE_HZ,
    MEL_BINS,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    TFDATA_DIR,
    TFLITE_MODEL_PATH,
    UPPER_EDGE_HZ,
    BACKGROUND_CLASS_NAMES,
)


MIC_WINDOW_SECONDS = 1.0
MIC_HOP_SECONDS = 0.25
MIC_MIN_RECORD_SECONDS = 1.5
BACKGROUND_CLASSES = set(BACKGROUND_CLASS_NAMES)


def load_meta():
    with open(TFDATA_DIR / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)



def load_wav_np(path):
    audio_bin = tf.io.read_file(str(path))
    wave, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    wave = tf.squeeze(wave, axis=-1).numpy()
    if len(wave) > N_SAMPLES:
        wave = wave[:N_SAMPLES]
    elif len(wave) < N_SAMPLES:
        wave = np.pad(wave, (0, N_SAMPLES - len(wave)))
    return wave



def load_wav_np_full(path):
    audio_bin = tf.io.read_file(str(path))
    wave, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    return tf.squeeze(wave, axis=-1).numpy()



def compute_logmel_np(wave):
    wave_tf = tf.constant(wave, dtype=tf.float32)
    stft = tf.signal.stft(
        wave_tf,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
    )
    spec = tf.abs(stft) ** 2
    num_spectrogram_bins = FFT_LENGTH // 2 + 1

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_EDGE_HZ,
        upper_edge_hertz=UPPER_EDGE_HZ,
    )

    mel = tf.matmul(spec, mel_w)
    logmel = tf.math.log(mel + 1e-6).numpy()

    if logmel.shape[0] > N_FRAMES:
        logmel = logmel[:N_FRAMES, :]
    elif logmel.shape[0] < N_FRAMES:
        logmel = np.pad(logmel, ((0, N_FRAMES - logmel.shape[0]), (0, 0)))

    return logmel



def predict_keras(model, logmel):
    feat = np.expand_dims(logmel, axis=(0, -1)).astype(np.float32)
    return model.predict(feat, verbose=0)[0]



def quantize_input_feature(logmel, input_detail):
    feat = np.expand_dims(logmel, axis=(0, -1)).astype(np.float32)
    if input_detail["dtype"] != np.int8:
        return feat

    scale = input_detail["quantization_parameters"]["scales"][0]
    zp = input_detail["quantization_parameters"]["zero_points"][0]
    return np.clip(np.round(feat / scale) + zp, -128, 127).astype(np.int8)



def predict_tflite(interpreter, logmel, input_detail, output_detail):
    feat = quantize_input_feature(logmel, input_detail)

    interpreter.set_tensor(input_detail["index"], feat)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]

    if output_detail["dtype"] == np.int8:
        scale = output_detail["quantization_parameters"]["scales"][0]
        zp = output_detail["quantization_parameters"]["zero_points"][0]
        output = (output.astype(np.float32) - zp) * scale

    return output



def format_probs(probs, class_names, top_k=3):
    sorted_idx = np.argsort(probs)[::-1]
    top_label = class_names[sorted_idx[0]]
    top_conf = float(probs[sorted_idx[0]])
    top_k_str = " | ".join(f"{class_names[i]}={probs[i]:.3f}" for i in sorted_idx[:top_k])
    return top_label, top_conf, top_k_str



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, help="path to single wav file")
    parser.add_argument("--dir", type=str, help="path to directory of wav files")
    parser.add_argument("--mic", action="store_true", help="record from microphone interactively")
    parser.add_argument("--seconds", type=float, default=1.0, help="microphone recording duration in seconds")
    parser.add_argument("--mode", choices=["keras", "tflite", "both"], default="both", help="model type to use")
    return parser.parse_args()



def record_wav_with_arecord(seconds):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    record_seconds = max(1, math.ceil(seconds))
    cmd = [
        "arecord",
        "-q",
        "-d",
        str(record_seconds),
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)
    return wav_path



def build_mic_windows(wave):
    window_samples = N_SAMPLES
    hop_samples = int(MIC_HOP_SECONDS * SAMPLE_RATE)

    if len(wave) <= window_samples:
        padded = np.pad(wave, (0, max(0, window_samples - len(wave))))
        return [(0, padded)]

    windows = []
    for start in range(0, len(wave) - window_samples + 1, hop_samples):
        windows.append((start, wave[start : start + window_samples]))

    last_start = len(wave) - window_samples
    if windows[-1][0] != last_start:
        windows.append((last_start, wave[last_start : last_start + window_samples]))

    return windows



def select_best_window(wave, predictor, class_names):
    best_start = 0
    best_probs = None
    best_keyword_score = -1.0
    best_top_score = -1.0

    background_indices = {i for i, name in enumerate(class_names) if name in BACKGROUND_CLASSES}

    for start, chunk in build_mic_windows(wave):
        probs = predictor(compute_logmel_np(chunk))
        top_score = float(np.max(probs))

        keyword_score = -1.0
        for i, prob in enumerate(probs):
            if i not in background_indices:
                keyword_score = max(keyword_score, float(prob))

        if keyword_score > best_keyword_score or (
            keyword_score == best_keyword_score and top_score > best_top_score
        ):
            best_start = start
            best_probs = probs
            best_keyword_score = keyword_score
            best_top_score = top_score

    return best_start, best_probs



def test_single(wav_path, class_names, model=None, interpreter=None, input_detail=None, output_detail=None, top_k=3):
    wave = load_wav_np(str(wav_path))
    logmel = compute_logmel_np(wave)

    if model is not None:
        probs = predict_keras(model, logmel)
    else:
        probs = predict_tflite(interpreter, logmel, input_detail, output_detail)

    top_label, top_conf, top_k_str = format_probs(probs, class_names, top_k)
    print(f"{Path(wav_path).name}: {top_label} ({top_conf:.3f}) [{top_k_str}]")
    return top_label, top_conf



def test_live_microphone(class_names, mode, seconds):
    requested_seconds = max(seconds, MIC_MIN_RECORD_SECONDS)
    shown_seconds = float(max(1, math.ceil(requested_seconds)))

    print("=== Test mic interactive ===")
    print(f"Moi lan bam Enter se ghi am {shown_seconds:.1f}s | Mode: {mode}")
    print(f"Classes: {class_names}")
    print("Dang quet nhieu cua so 1.0s chong lan tu audio mic de giam loi cat lech tu.")
    print("Bam Enter de ghi am, go 'q' roi Enter de thoat.\n")

    keras_model = None
    interpreter = None
    input_detail = None
    output_detail = None

    if mode in ("keras", "both"):
        keras_model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        print(f"Loaded Keras model: {BEST_MODEL_PATH}")

    if mode in ("tflite", "both"):
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        print(f"Loaded TFLite model: {TFLITE_MODEL_PATH}")

    print(f"Mic mode dang dung window={MIC_WINDOW_SECONDS:.1f}s, hop={MIC_HOP_SECONDS:.2f}s. start=... la vi tri cua so tot nhat.")
    print()

    while True:
        user_input = input("Nhan Enter de ghi am > ").strip().lower()
        if user_input in {"q", "quit", "exit"}:
            break

        try:
            wav_path = record_wav_with_arecord(requested_seconds)
        except subprocess.CalledProcessError as exc:
            print(f"Khong ghi am duoc bang arecord: {exc}")
            break

        wave = load_wav_np_full(str(wav_path))
        wav_path.unlink(missing_ok=True)

        if keras_model is not None:
            start, probs = select_best_window(
                wave,
                lambda logmel: predict_keras(keras_model, logmel),
                class_names,
            )
            top_label, top_conf, top_k_str = format_probs(probs, class_names)
            print(f"  [KERAS] {top_label} ({top_conf:.3f}) [start={start / SAMPLE_RATE:.2f}s] [{top_k_str}]")

        if interpreter is not None:
            start, probs = select_best_window(
                wave,
                lambda logmel: predict_tflite(interpreter, logmel, input_detail, output_detail),
                class_names,
            )
            top_label, top_conf, top_k_str = format_probs(probs, class_names)
            print(f"  [TFLITE] {top_label} ({top_conf:.3f}) [start={start / SAMPLE_RATE:.2f}s] [{top_k_str}]")

        print()

    print("Done.")



def describe_inference_target():
    print(f"Firmware dang dung model: {TFLITE_MODEL_PATH}")
    print("Ghi chu: script test chay tren PC, con ESP dung model C++ da export vao firmware.")
    print("Neu can doi chieu, hay chay 09_convert_to_cpp.py de dong bo artifact.")



def run_single_prediction(wav_path, class_names, mode):
    if mode == "keras":
        model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        print(f"Loaded Keras model: {BEST_MODEL_PATH}")
        pred_label, pred_conf = test_single(wav_path, class_names, model=model)
        print(f"[KERAS] => {pred_label} ({pred_conf:.3f})")
        return

    if mode == "tflite":
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        print(f"Loaded TFLite model: {TFLITE_MODEL_PATH}")
        pred_label, pred_conf = test_single(
            wav_path,
            class_names,
            interpreter=interpreter,
            input_detail=input_detail,
            output_detail=output_detail,
        )
        print(f"[TFLITE] => {pred_label} ({pred_conf:.3f})")
        return

    run_single_prediction(wav_path, class_names, "keras")
    print()
    run_single_prediction(wav_path, class_names, "tflite")



def load_batch_paths(args):
    if args.wav:
        return [Path(args.wav)]
    if args.dir:
        return sorted(Path(args.dir).glob("*.wav"))
    return []



def eval_paths(wav_paths, class_names, mode):
    if mode == "keras":
        model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        print(f"Loaded Keras model: {BEST_MODEL_PATH}")
        correct = 0
        total = 0
        for p in wav_paths:
            true_label = p.parent.name
            pred_label, _ = test_single(p, class_names, model=model)
            if true_label in class_names:
                total += 1
                if pred_label == true_label:
                    correct += 1
        if total > 0:
            print(f"[KERAS] Accuracy: {correct}/{total} = {correct/total:.4f}")
        return

    if mode == "tflite":
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        print(f"Loaded TFLite model: {TFLITE_MODEL_PATH}")
        correct = 0
        total = 0
        for p in wav_paths:
            true_label = p.parent.name
            pred_label, _ = test_single(
                p,
                class_names,
                interpreter=interpreter,
                input_detail=input_detail,
                output_detail=output_detail,
            )
            if true_label in class_names:
                total += 1
                if pred_label == true_label:
                    correct += 1
        if total > 0:
            print(f"[TFLITE] Accuracy: {correct}/{total} = {correct/total:.4f}")



def execute_mode(wav_paths, class_names, mode):
    if mode == "both":
        eval_paths(wav_paths, class_names, "keras")
        print()
        eval_paths(wav_paths, class_names, "tflite")
    else:
        eval_paths(wav_paths, class_names, mode)



def print_batch_header(wav_paths):
    print(f"\nTesting {len(wav_paths)} file(s)...\n")



def print_done():
    print("\nDone.")



def handle_single_or_mic_mode(args, class_names):
    if args.mic or (args.wav is None and args.dir is None):
        describe_inference_target()
        test_live_microphone(class_names, args.mode, args.seconds)
        return True

    if args.wav:
        describe_inference_target()
        run_single_prediction(Path(args.wav), class_names, args.mode)
        print_done()
        return True

    return False



def handle_batch_mode(args, class_names):
    wav_paths = load_batch_paths(args)
    print_batch_header(wav_paths)
    execute_mode(wav_paths, class_names, args.mode)
    print_done()



def main():
    args = parse_args()
    meta = load_meta()
    class_names = meta["classes"]

    if handle_single_or_mic_mode(args, class_names):
        return

    handle_batch_mode(args, class_names)


if __name__ == "__main__":
    main()
