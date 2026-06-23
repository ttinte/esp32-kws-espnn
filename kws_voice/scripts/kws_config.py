from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "kws_dataset"
FIXED_DATASET_DIR = ROOT / "kws_dataset_fixed"
TFDATA_DIR = ROOT / "kws_tfdata"
MODEL_DIR = ROOT / "kws_model"
OUTPUT_DIR = ROOT / "kws_output"

NOISE_LABEL = "_noise_"
OTHER_LABEL = "other"
CLASS_NAMES = ["bat", "dung", "quay", "tat", "zero", OTHER_LABEL]
KEYWORD_CLASS_NAMES = [name for name in CLASS_NAMES if name != OTHER_LABEL]
BACKGROUND_CLASS_NAMES = [OTHER_LABEL]
ALL_DATASET_DIRS = CLASS_NAMES + [NOISE_LABEL]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_SET = set(CLASS_NAMES)
DATASET_LABELS = tuple(ALL_DATASET_DIRS)

OTHER_TARGET_COUNT = 5000
OTHER_NOISE_SOURCE_FRACTION = 0.75
OTHER_NOISE_MANIFEST_PATH = DATASET_DIR / "other_noise_manifest.json"

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
WAKE_WORD = "zero"

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

# Far-field augmentation: ha gain manh de gia lap giong xa/nho (firmware KHONG co AGC),
# tang xac suat tron nhieu de model quen tin hieu yeu + nhieu.
NOISE_MIX_PROBABILITY = 0.55
GAIN_MIN = 0.2
GAIN_MAX = 1.15
SHIFT_MAX_SAMPLES = 1600

MODEL_CLASSES = CLASS_NAMES
CLASS_WEIGHT_MULTIPLIERS = {
    "bat": 1.5,   # tang weight de tot hon
    "dung": 1.0,
    "quay": 1.0,
    "tat": 1.0,
    "zero": 1.5,  # uu tien wake word
    OTHER_LABEL: 0.85,
}
