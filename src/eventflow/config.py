"""Shared paths and constants. Import from here instead of hard-coding paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"          # git-ignored: caches, checkpoints, predictions
FIGURES_DIR = ROOT / "reports" / "figures"
EXPERIMENT_LOG = ROOT / "experiments" / "results.csv"

TRAIN_CSV = DATA_DIR / "eventflow_train.csv"
TEST_CSV = DATA_DIR / "eventflow_test.csv"
TEST_LABELS_CSV = DATA_DIR / "eventflow_test_labels.csv"

RANDOM_STATE = 2210

ID_COL = "sequence_id"
ORDER_COL = "event_index"
TIME_COL = "timestamp"
EVENT_COL = "event_type"

# The 14 next-event classes in a FIXED order. Every model outputs probabilities in this column
# order, and targets are encoded as the index into this list (0..13).
EVENT_CLASSES = [
    "COMPONENT_REPLACED", "CONTROL_ALERT", "COOLING_ACTIVATED", "COOLING_FAULT",
    "EMERGENCY_STOP", "LOAD_DECREASE", "LOAD_INCREASE", "MAINTENANCE_CHECK",
    "PRESSURE_WARNING", "RECOVERY", "ROUTINE_CHECK", "SENSOR_RESET",
    "TEMP_WARNING", "VIBRATION_WARNING",
]
N_CLASSES = len(EVENT_CLASSES)

# Example construction (see notebooks/01_eda.ipynb, Section 3)
MIN_PREFIX = 90           # shortest test prefix
MAX_PREFIX = 220          # longest test prefix
TEST_MIMIC_OFFSET = 30    # test prefixes = full sequence minus its last 30 events

# Validation (see notebooks/02_validation.ipynb)
N_FOLDS = 5
TUNING_FOLD = 0           # hyperparameter tuning uses this fold only; final comparison uses all folds
