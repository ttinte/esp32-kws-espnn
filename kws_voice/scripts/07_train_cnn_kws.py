import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

np.set_printoptions(suppress=True)

from kws_config import (
    ACCURACY_PLOT_PATH,
    BEST_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    EVAL_METRICS_PATH,
    NOISE_LABEL,
    OTHER_LABEL,
    OTHER_NOISE_MANIFEST_PATH,
    FFT_LENGTH,
    FINAL_MODEL_PATH,
    FIXED_DATASET_DIR,
    FRAME_LENGTH,
    FRAME_STEP,
    GAIN_MAX,
    GAIN_MIN,
    HISTORY_PATH,
    LOSS_PLOT_PATH,
    MEL_BINS,
    MODEL_DIR,
    N_FRAMES,
    N_SAMPLES,
    NOISE_MIX_PROBABILITY,
    SAMPLE_RATE,
    SEED,
    SHIFT_MAX_SAMPLES,
    TFDATA_DIR,
    LOWER_EDGE_HZ,
    UPPER_EDGE_HZ,
    CLASS_WEIGHT_MULTIPLIERS,
)

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
_OTHER_NOISE_SOURCES = None
_NOISE_CACHE = None


def load_meta():
    with open(TFDATA_DIR / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_pairs(list_path):
    pairs = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            pairs.append((parts[0], int(parts[1])))
    return pairs


def load_other_noise_sources():
    global _OTHER_NOISE_SOURCES
    if _OTHER_NOISE_SOURCES is not None:
        return _OTHER_NOISE_SOURCES

    if not OTHER_NOISE_MANIFEST_PATH.exists():
        _OTHER_NOISE_SOURCES = set()
        return _OTHER_NOISE_SOURCES

    with open(OTHER_NOISE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _OTHER_NOISE_SOURCES = set(data.get("sources", []))
    return _OTHER_NOISE_SOURCES


def infer_noise_source_rel(fixed_noise_path):
    stem = fixed_noise_path.stem
    base_stem, sep, chunk_suffix = stem.rpartition("_")
    if sep and chunk_suffix.isdigit():
        source_name = f"{base_stem}.wav"
    else:
        source_name = fixed_noise_path.name
    return f"{NOISE_LABEL}/{source_name}"


def load_noise_files():
    noise_dir = FIXED_DATASET_DIR / NOISE_LABEL
    if not noise_dir.exists():
        return []

    excluded_sources = load_other_noise_sources()
    noise_data = []
    used_files = 0
    skipped_files = 0

    for p in sorted(noise_dir.glob("*.wav")):
        source_rel = infer_noise_source_rel(p)
        if source_rel in excluded_sources:
            skipped_files += 1
            continue
        audio_bin = tf.io.read_file(str(p))
        wave, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
        wave = tf.squeeze(wave, axis=-1)
        noise_data.append(wave.numpy())
        used_files += 1

    print(f"Augmentation noise files: used={used_files}, skipped_for_other={skipped_files}")
    return noise_data


def get_noise_cache():
    global _NOISE_CACHE
    if _NOISE_CACHE is None:
        _NOISE_CACHE = load_noise_files()
    return _NOISE_CACHE


def load_wav_tf(path):
    audio_bin = tf.io.read_file(path)
    wave, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    wave = tf.squeeze(wave, axis=-1)
    wave = wave[:N_SAMPLES]
    pad = tf.maximum(0, N_SAMPLES - tf.shape(wave)[0])
    wave = tf.pad(wave, [[0, pad]])
    return wave


def augment_wave_np(wave_np):
    shift = np.random.randint(-SHIFT_MAX_SAMPLES, SHIFT_MAX_SAMPLES + 1)
    wave_np = np.roll(wave_np, shift)

    gain = np.random.uniform(GAIN_MIN, GAIN_MAX)
    wave_np = wave_np * gain

    noise_cache = get_noise_cache()
    if len(noise_cache) > 0 and np.random.rand() < NOISE_MIX_PROBABILITY:
        noise = noise_cache[np.random.randint(len(noise_cache))]
        start = np.random.randint(0, max(len(noise) - N_SAMPLES, 1))
        noise_chunk = noise[start : start + N_SAMPLES]
        if len(noise_chunk) < N_SAMPLES:
            noise_chunk = np.pad(noise_chunk, (0, N_SAMPLES - len(noise_chunk)))
        snr = np.random.uniform(0.05, 0.20)
        wave_np = wave_np + snr * noise_chunk

    return wave_np.astype(np.float32)


def compute_logmel(wave):
    stft = tf.signal.stft(
        wave,
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
    logmel = tf.math.log(mel + 1e-6)

    logmel = logmel[:N_FRAMES, :]
    pad = tf.maximum(0, N_FRAMES - tf.shape(logmel)[0])
    logmel = tf.pad(logmel, [[0, pad], [0, 0]])

    logmel = tf.expand_dims(logmel, axis=-1)
    logmel.set_shape([N_FRAMES, MEL_BINS, 1])
    return logmel


def make_ds(pairs, shuffle=True, augment=False):
    paths = [p for p, _ in pairs]
    ys = [y for _, y in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map(path, y):
        wave = load_wav_tf(path)
        if augment:
            wave = tf.py_function(augment_wave_np, [wave], tf.float32)
            wave.set_shape([N_SAMPLES])
        feat = compute_logmel(wave)
        return feat, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _make_ds_block(x, filters, name_prefix):
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, padding="same", use_bias=False, name=f"{name_prefix}_dw"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name_prefix}_dw_act")(x)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size=1, use_bias=False, name=f"{name_prefix}_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name_prefix}_pw_act")(x)
    return x


def build_ds_cnn(num_classes):
    inputs = tf.keras.Input(shape=(N_FRAMES, MEL_BINS, 1))

    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, strides=2, padding="same", use_bias=False, name="init_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="init_conv_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="init_conv_act")(x)

    x = _make_ds_block(x, 16, name_prefix="block1")
    x = _make_ds_block(x, 16, name_prefix="block2")
    x = _make_ds_block(x, 24, name_prefix="block3")

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="ds_cnn_kws")


def plot_history(history_dict):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(str(ACCURACY_PLOT_PATH))
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(str(LOSS_PLOT_PATH))
    plt.close()


def collect_predictions(model, ds):
    y_true = []
    y_pred = []

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        pred = np.argmax(probs, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    return y_true, y_pred


def compute_class_weights(train_pairs, class_names):
    counts = Counter(class_id for _, class_id in train_pairs)
    total = len(train_pairs)
    num_classes = len(class_names)
    class_weights = {}

    for class_id, class_name in enumerate(class_names):
        count = counts.get(class_id, 1)
        weight = total / (num_classes * count)
        weight *= CLASS_WEIGHT_MULTIPLIERS.get(class_name, 1.0)
        class_weights[class_id] = float(weight)

    print("\n=== TRAIN CLASS COUNTS ===")
    for class_id, class_name in enumerate(class_names):
        print(f"{class_name}: count={counts.get(class_id, 0)}, weight={class_weights[class_id]:.4f}")

    return class_weights


def save_confusion_matrix_plot(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    class_names = meta["classes"]
    train_pairs = load_pairs(TFDATA_DIR / "train.txt")
    val_pairs = load_pairs(TFDATA_DIR / "val.txt")
    test_pairs = load_pairs(TFDATA_DIR / "test.txt")

    train_ds = make_ds(train_pairs, shuffle=True, augment=True)
    val_ds = make_ds(val_pairs, shuffle=False, augment=False)
    test_ds = make_ds(test_pairs, shuffle=False, augment=False)

    model = build_ds_cnn(len(class_names))
    print("\n=== MODEL SUMMARY ===")
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(train_pairs, class_names)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    model.save(str(FINAL_MODEL_PATH))

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    plot_history(history.history)

    best_model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    y_true, y_pred = collect_predictions(best_model, test_ds)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix_plot(cm, class_names, CONFUSION_MATRIX_PLOT_PATH)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "class_names": class_names,
    }
    with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved best model: {BEST_MODEL_PATH}")
    print(f"Saved final model: {FINAL_MODEL_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PLOT_PATH}")
    print(f"Saved classification report: {CLASSIFICATION_REPORT_PATH}")
    print(f"Saved eval metrics: {EVAL_METRICS_PATH}")
    print(f"Test accuracy: {test_acc:.4f} | Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
