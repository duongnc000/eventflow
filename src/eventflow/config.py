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
