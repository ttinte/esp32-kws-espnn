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
    DATASET_DIR,
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
    normalize_rms_vad_np,
)


MIC_WINDOW_SECONDS = 1.0
MIC_HOP_SECONDS = 0.20
MIC_MIN_RECORD_SECONDS = 2.5
MIC_AGGREGATE_WINDOWS = 5
# Gate im lang: wave float [-1,1] -> int16 RMS = rms_float * 32768. Nguong
# 0.0030 ~ int16 ~98 (chat hon firmware 60 mot chut de script khong nhay
# voi tieng thu nho, van loai bo im lang that).
MIC_RMS_FLOOR = 0.0030
# Neu toan bo clip mic chi quanh muc nhieu nen, ep ve other thay vi de model
# tuong tuong thanh wake_up. Normal speech trong project thuong cao hon muc nay.
MIC_NO_SPEECH_RMS = 420.0
MIC_NO_SPEECH_PEAK = 2400.0
MIC_NO_SPEECH_RMS_RATIO = 1.35
MIC_NO_SPEECH_PEAK_RATIO = 2.20
LIVE_BAT_PROMOTE_MIN = 0.045
LIVE_TAT_PROMOTE_MIN = 0.07
LIVE_WAKE_PROMOTE_MAX = 0.92
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
    wave = normalize_rms_vad_np(wave)  # Khop train/export/firmware: AGC truoc log-mel.
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



def format_probs(probs, class_names, top_k=3, override_idx=None):
    sorted_idx = np.argsort(probs)[::-1]
    top_idx = sorted_idx[0] if override_idx is None else override_idx
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx])
    display_idx = list(sorted_idx[:top_k])
    if override_idx is not None and override_idx not in display_idx:
        display_idx[-1] = override_idx
    top_k_str = " | ".join(f"{class_names[i]}={probs[i]:.3f}" for i in display_idx)
    return top_label, top_conf, top_k_str


def apply_live_command_prior(probs, class_names):
    """Promote chi bat/tat khi raw top la wake_up nhung command co score ro.

    Day la lop hien thi/decision cho mic live, khong sua xac suat model va
    khong dung cho quay/dung de tranh lam hong cac tu da on dinh.
    """
    wake_idx = next((i for i, name in enumerate(class_names) if name == "wake_up"), None)
    if wake_idx is None or int(np.argmax(probs)) != wake_idx:
        return None

    wake_score = float(probs[wake_idx])
    if wake_score >= LIVE_WAKE_PROMOTE_MAX:
        return None

    bat_idx = next((i for i, name in enumerate(class_names) if name == "bat"), None)
    tat_idx = next((i for i, name in enumerate(class_names) if name == "tat"), None)
    candidates = []
    if bat_idx is not None and float(probs[bat_idx]) >= LIVE_BAT_PROMOTE_MIN:
        candidates.append((float(probs[bat_idx]), bat_idx))
    if tat_idx is not None and float(probs[tat_idx]) >= LIVE_TAT_PROMOTE_MIN:
        candidates.append((float(probs[tat_idx]), tat_idx))
    if not candidates:
        return None

    _, idx = max(candidates)
    return idx



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, help="path to single wav file")
    parser.add_argument("--dir", type=str, help="path to directory of wav files")
    parser.add_argument("--mic", action="store_true", help="record from microphone interactively")
    parser.add_argument("--seconds", type=float, default=1.0, help="microphone recording duration in seconds")
    parser.add_argument("--mode", choices=["keras", "tflite", "both"], default="both", help="model type to use")
    parser.add_argument("--debug-windows", action="store_true", help="print top scoring mic windows for diagnosis")
    parser.add_argument("--save-label", choices=["bat", "tat", "quay", "dung", "wake_up", "other"], help="save each mic recording to this raw dataset label")
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



def select_best_window(wave, predictor, class_names, debug=False):
    """Chon nhieu cua so co nang luong cao, roi trung binh xac suat.

    Khong dung xac suat model de chon 1 cua so duy nhat, vi cach do hay nhat
    nham doan wake_up/other rat tu tin trong clip 2.5s va lam ket qua dao dong.
    Trung binh top-N cua so nang luong cao giup on dinh ma khong uu ai rieng
    bat/tat/quay/dung.
    """
    background_indices = {i for i, name in enumerate(class_names) if name in BACKGROUND_CLASSES}
    candidates = []
    rms_values = []
    peak_values = []
    max_rms = 0.0
    max_peak = 0.0
    window_debug = []

    for start, chunk in build_mic_windows(wave):
        rms = float(np.sqrt(np.mean(np.square(chunk))) * 32768.0)
        peak = float(np.max(np.abs(chunk)) * 32768.0)
        rms_values.append(rms)
        peak_values.append(peak)
        max_rms = max(max_rms, rms)
        max_peak = max(max_peak, peak)
        energy_score = rms + 0.08 * peak

        if debug:
            window_debug.append((
                energy_score,
                start,
                rms,
                peak,
            ))

        if rms >= MIC_RMS_FLOOR * 32768.0:
            candidates.append((energy_score, start, chunk))

    median_rms = float(np.median(rms_values)) if rms_values else 0.0
    median_peak = float(np.median(peak_values)) if peak_values else 0.0
    low_absolute_energy = max_rms < MIC_NO_SPEECH_RMS and max_peak < MIC_NO_SPEECH_PEAK
    flat_background = (
        median_rms > 0.0
        and max_rms < median_rms * MIC_NO_SPEECH_RMS_RATIO
        and max_peak < max(MIC_NO_SPEECH_PEAK, median_peak * MIC_NO_SPEECH_PEAK_RATIO)
    )
    if low_absolute_energy or flat_background or not candidates:
        probs = np.zeros(len(class_names), dtype=np.float32)
        if background_indices:
            probs[next(iter(background_indices))] = 1.0
        if debug:
            print_energy_debug(window_debug, prefix="    ")
            print(
                f"    [silence] max_rms={max_rms:.0f} med_rms={median_rms:.0f} "
                f"max_peak={max_peak:.0f} med_peak={median_peak:.0f} -> other"
            )
        return 0, probs, None

    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:MIC_AGGREGATE_WINDOWS]
    weight_sum = 0.0
    probs_sum = None
    best_start = selected[0][1]
    best_energy_score = selected[0][0]
    for energy_score, _, chunk in selected:
        weight = max(energy_score, 1.0)
        probs = predictor(compute_logmel_np(chunk)).astype(np.float32)
        if probs_sum is None:
            probs_sum = probs * weight
        else:
            probs_sum += probs * weight
        weight_sum += weight

    probs = probs_sum / max(weight_sum, 1e-9)
    if debug:
        print_energy_debug(window_debug, prefix="    ")
        starts = ", ".join(f"{start / SAMPLE_RATE:.2f}s" for _, start, _ in selected)
        print(f"    [selected] starts=[{starts}] best={best_start / SAMPLE_RATE:.2f}s score={best_energy_score:.0f}")
    return best_start, probs, None


def print_energy_debug(window_debug, prefix=""):
    if not window_debug:
        print(f"{prefix}[windows] no non-silent windows")
        return

    print(f"{prefix}[windows] energy start rms peak")
    for score, start, rms, peak in sorted(window_debug, reverse=True)[:5]:
        print(
            f"{prefix}  {score:.0f} {start / SAMPLE_RATE:.2f}s rms={rms:.0f} peak={peak:.0f}"
        )



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



def save_recording_for_retrain(wav_path, label):
    target_dir = DATASET_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob(f"mic_{label}_*.wav"))
    next_id = len(existing) + 1
    while True:
        out_path = target_dir / f"mic_{label}_{next_id:04d}.wav"
        if not out_path.exists():
            break
        next_id += 1
    wav_path.replace(out_path)
    print(f"  [SAVE] {out_path}")


def test_live_microphone(class_names, mode, seconds, debug_windows=False, save_label=None):
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
        if save_label:
            save_recording_for_retrain(wav_path, save_label)
        else:
            wav_path.unlink(missing_ok=True)

        if keras_model is not None:
            start, probs, override_idx = select_best_window(
                wave,
                lambda logmel: predict_keras(keras_model, logmel),
                class_names,
                debug=debug_windows,
            )
            if override_idx is None:
                override_idx = apply_live_command_prior(probs, class_names)
            top_label, top_conf, top_k_str = format_probs(probs, class_names, override_idx=override_idx)
            print(f"  [KERAS] {top_label} ({top_conf:.3f}) [start={start / SAMPLE_RATE:.2f}s] [{top_k_str}]")

        if interpreter is not None:
            start, probs, override_idx = select_best_window(
                wave,
                lambda logmel: predict_tflite(interpreter, logmel, input_detail, output_detail),
                class_names,
                debug=debug_windows,
            )
            if override_idx is None:
                override_idx = apply_live_command_prior(probs, class_names)
            top_label, top_conf, top_k_str = format_probs(probs, class_names, override_idx=override_idx)
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
        test_live_microphone(class_names, args.mode, args.seconds, args.debug_windows, args.save_label)
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
