from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "kws_dataset"
FIXED_DATASET_DIR = ROOT / "kws_dataset_fixed"
TFDATA_DIR = ROOT / "kws_tfdata"
MODEL_DIR = ROOT / "kws_model"
OUTPUT_DIR = ROOT / "kws_output"

NOISE_LABEL = "_noise_"
OTHER_LABEL = "other"
CLASS_NAMES = ["bat", "tat", "quay", "dung", "wake_up", OTHER_LABEL]
KEYWORD_CLASS_NAMES = [name for name in CLASS_NAMES if name != OTHER_LABEL]
BACKGROUND_CLASS_NAMES = [OTHER_LABEL]
ALL_DATASET_DIRS = CLASS_NAMES + [NOISE_LABEL]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_SET = set(CLASS_NAMES)
DATASET_LABELS = tuple(ALL_DATASET_DIRS)

OTHER_TARGET_COUNT = 5000
OTHER_NOISE_SOURCE_FRACTION = 0.75
OTHER_NOISE_MANIFEST_PATH = DATASET_DIR / "other_noise_manifest.json"
# Round 6: giu wake_up du mau de khong mat wake, nhung van thap hon class other.
WAKE_TARGET_COUNT = 1500
# Lap lai mot so class trong train set de moi epoch thay nhieu bien the augment
# hon, khong nhan doi file tren dia.
CLASS_TRAIN_REPEATS = {
    "bat": 3,
    "tat": 3,
    "quay": 2,
}

SAMPLE_RATE = 16000
CLIP_DURATION_SECONDS = 1.0
N_SAMPLES = 16000
OTHER_HOP_SAMPLES = N_SAMPLES // 2
MEL_BINS = 24
FRAME_LENGTH = 512
FRAME_STEP = 320
FFT_LENGTH = 512
N_FRAMES = 49
LOWER_EDGE_HZ = 20.0
UPPER_EDGE_HZ = 7600.0
WAKE_WORD = "wake_up"

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BEST_MODEL_PATH = MODEL_DIR / "best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final.keras"
TFLITE_MODEL_PATH = MODEL_DIR / "model.tflite"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.json"
HISTORY_PATH = OUTPUT_DIR / "history.json"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "accuracy.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss.png"
CONFUSION_MATRIX_PLOT_PATH = OUTPUT_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.json"
EVAL_METRICS_PATH = OUTPUT_DIR / "eval_metrics.json"
KWS_TINY_DIR = ROOT / "kws_tiny"
MODEL_HEADER_PATH = KWS_TINY_DIR / "model_tiny.h"
MODEL_SOURCE_PATH = KWS_TINY_DIR / "model_tiny.cpp"

MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 1.5
LOW_PEAK_THRESHOLD = 80
CLIPPING_ABS_THRESHOLD = 32760
CLIPPING_RATIO_THRESHOLD = 0.35

# Far-field augmentation: tron nhieu + reverb (gia lap giong xa/phong).
# Gain-aug da bo: chuan hoa RMS (ben duoi) lam muc tin hieu vo nghia roi.
NOISE_MIX_PROBABILITY = 0.55
SHIFT_MAX_SAMPLES = 3200  # +-0.2s: day vi tri tu trong cua so de chong overfit nhip noi
REVERB_PROBABILITY = 0.5  # xac suat them echo/reverb mo phong noi xa

# SpecAugment (chi ap dung luc train): che ngau nhien frame thoi gian + bin tan so.
SPEC_AUG_TIME_MASK = 8   # so frame toi da bi che (tren 49)
SPEC_AUG_FREQ_MASK = 4   # so mel bin toi da bi che (tren 24)

# Chuan hoa RMS co VAD-gate — PHAI giong het firmware (app_state.h AGC_*).
# wave la float [-1,1] nen RMS so sanh = rms_float * 32768 (don vi int16).
AGC_TARGET_RMS = 4000.0
AGC_VAD_FLOOR_RMS = 120.0
AGC_MAX_GAIN = 12.0


def normalize_rms_vad_np(wave):
    """Chuan hoa wave float [-1,1] ve muc RMS dich (don vi int16), co VAD-gate.
    Giong het applyAgc trong firmware. Dung cho 09 (numpy)."""
    import numpy as _np
    rms = float(_np.sqrt(_np.mean(_np.square(wave))) * 32768.0)
    if rms < AGC_VAD_FLOOR_RMS:
        return wave
    gain = min(AGC_TARGET_RMS / (rms + 1e-9), AGC_MAX_GAIN)
    return _np.clip(wave * gain, -1.0, 1.0)

MODEL_CLASSES = CLASS_NAMES
# Round 6:
#  - bat/tat: tang recall live mic, vi dang bi wake_up hut ve.
#  - wake_up: tang lai so voi round 5 de khong mat wake_up that.
#  - quay/dung: gan default vi da on dinh.
CLASS_WEIGHT_MULTIPLIERS = {
    "bat":     2.0,
    "tat":     1.8,
    "quay":    1.1,
    "dung":    1.0,
    WAKE_WORD: 0.5,
    OTHER_LABEL: 1.0,
}
