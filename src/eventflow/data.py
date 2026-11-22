"""Loading the raw EventFlow data.

The first call reads the CSV and writes a parquet copy to outputs/cache/; later calls read the
parquet file, which is much faster. Delete outputs/cache/ to force a re-read of the CSVs.
"""

from pathlib import Path

import pandas as pd

from eventflow.config import (
    ID_COL, ORDER_COL, OUTPUTS_DIR, TEST_CSV, TEST_LABELS_CSV, TRAIN_CSV,
)

CACHE_DIR = OUTPUTS_DIR / "cache"

CATEGORICAL_COLS = [
    "event_type", "subsystem", "shift_code",
    "operator_group", "device_family", "firmware_channel",
]


def _load_cached(csv_path: Path) -> pd.DataFrame:
    cache_path = CACHE_DIR / (csv_path.stem + ".parquet")
    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_parquet(cache_path)

    df = pd.read_csv(csv_path)
    if ORDER_COL in df.columns:
        df = df.sort_values([ID_COL, ORDER_COL], kind="stable").reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def load_train() -> pd.DataFrame:
    """Complete training histories, sorted by (sequence_id, event_index)."""
    return _load_cached(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    """Test prefixes, sorted by (sequence_id, event_index)."""
    return _load_cached(TEST_CSV)


def load_test_labels() -> pd.DataFrame:
    """True next event per test sequence. FINAL EVALUATION ONLY."""
    return pd.read_csv(TEST_LABELS_CSV)


def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into identifier / categorical / numeric groups."""
    ids = [ID_COL, ORDER_COL, "timestamp"]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    numeric = [
        c for c in df.columns
        if c not in ids and c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]
    return {"ids": ids, "categorical": categorical, "numeric": numeric}
