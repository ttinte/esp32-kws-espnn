import json
import textwrap

import numpy as np
import tensorflow as tf

from kws_config import (
    BEST_MODEL_PATH,
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_EDGE_HZ,
    MEL_BINS,
    MODEL_DIR,
    MODEL_HEADER_PATH,
    MODEL_SOURCE_PATH,
    MODEL_SUMMARY_PATH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    TFDATA_DIR,
    TFLITE_MODEL_PATH,
    UPPER_EDGE_HZ,
)


def load_pairs(list_path):
    pairs = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            pairs.append(parts[0])
    return pairs


def load_wav_np(path):
    audio_bin = tf.io.read_file(path)
    wave, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    wave = tf.squeeze(wave, axis=-1).numpy()
    if len(wave) > N_SAMPLES:
        wave = wave[:N_SAMPLES]
    elif len(wave) < N_SAMPLES:
        wave = np.pad(wave, (0, N_SAMPLES - len(wave)))
    return wave


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
        pad_rows = N_FRAMES - logmel.shape[0]
        logmel = np.pad(logmel, ((0, pad_rows), (0, 0)))

    return logmel


def representative_dataset_gen(train_paths, max_samples=200):
    indices = np.random.default_rng(42).choice(len(train_paths), min(max_samples, len(train_paths)), replace=False)
    for idx in indices:
        wave = load_wav_np(train_paths[idx])
        logmel = compute_logmel_np(wave)
        logmel = np.expand_dims(logmel, axis=(0, -1)).astype(np.float32)
        yield [logmel]


def convert_to_tflite(train_paths):
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(train_paths)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model: {TFLITE_MODEL_PATH} ({len(tflite_model)} bytes)")
    return tflite_model


def verify_tflite(tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    print(f"Input shape: {input_detail['shape']}, dtype: {input_detail['dtype'].__name__}")
    print(f"Input quant: scale={input_detail['quantization_parameters']['scales']}, "
          f"zp={input_detail['quantization_parameters']['zero_points']}")
    print(f"Output shape: {output_detail['shape']}, dtype: {output_detail['dtype'].__name__}")
    print(f"Output quant: scale={output_detail['quantization_parameters']['scales']}, "
          f"zp={output_detail['quantization_parameters']['zero_points']}")

    expected_input_bytes = N_FRAMES * MEL_BINS
    actual_input_size = 1
    for dim in input_detail["shape"]:
        actual_input_size *= dim
    if actual_input_size != expected_input_bytes:
        print(f"WARNING: input size {actual_input_size} != expected {expected_input_bytes}")

    summary = {
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": input_detail["dtype"].__name__,
        "input_scale": input_detail["quantization_parameters"]["scales"].tolist(),
        "input_zero_point": input_detail["quantization_parameters"]["zero_points"].tolist(),
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": output_detail["dtype"].__name__,
        "output_scale": output_detail["quantization_parameters"]["scales"].tolist(),
        "output_zero_point": output_detail["quantization_parameters"]["zero_points"].tolist(),
        "model_size_bytes": len(tflite_model),
    }
    with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved model summary: {MODEL_SUMMARY_PATH}")
    return summary


def generate_cpp(tflite_model):
    MODEL_HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = textwrap.dedent("""\
        #ifndef G_MODEL_H
        #define G_MODEL_H

        #ifdef __cplusplus
        extern "C" {
        #endif

        extern const unsigned char g_model[];
        extern const int g_model_len;

        #ifdef __cplusplus
        }
        #endif

        #endif // G_MODEL_H
    """)

    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i : i + 12]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"  {hex_str},")

    source = '#include "model_tiny.h"\n\n'
    source += "const unsigned char g_model[] = {\n"
    source += "\n".join(hex_lines)
    source += "\n};\n\n"
    source += f"const int g_model_len = {len(tflite_model)};\n"

    with open(MODEL_HEADER_PATH, "w", encoding="utf-8") as f:
        f.write(header)
    with open(MODEL_SOURCE_PATH, "w", encoding="utf-8") as f:
        f.write(source)

    print(f"Saved: {MODEL_HEADER_PATH}")
    print(f"Saved: {MODEL_SOURCE_PATH} ({len(tflite_model)} bytes)")


def main():
    train_paths = load_pairs(TFDATA_DIR / "train.txt")

    print("=== Converting to TFLite int8 ===")
    tflite_model = convert_to_tflite(train_paths)

    print("\n=== Verifying TFLite model ===")
    verify_tflite(tflite_model)

    print("\n=== Generating C++ artifacts ===")
    generate_cpp(tflite_model)

    print(f"\nDone. Rebuild firmware with artifacts in: {MODEL_HEADER_PATH.parent}")


if __name__ == "__main__":
    main()
