"""Loading the raw EventFlow data and turning it into prediction examples.

Loading: the first call reads the CSV and writes a parquet copy to outputs/cache/; later calls read
the parquet file, which is much faster. Delete outputs/cache/ to force a re-read of the CSVs.

Examples: one example = (sequence, cut point). An example does NOT copy its prefix; it stores
`last_row`, the row index (in the loaded data frame) of the last observed event. Features and
sequence models may only use rows of the same sequence up to and including `last_row`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from eventflow.config import (
    EVENT_CLASSES, EVENT_COL, ID_COL, MAX_PREFIX, MIN_PREFIX, ORDER_COL, OUTPUTS_DIR,
    TEST_CSV, TEST_LABELS_CSV, TEST_MIMIC_OFFSET, TRAIN_CSV,
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


def encode_events(events) -> np.ndarray:
    """Event names -> class codes (indices into config.EVENT_CLASSES)."""
    codes = pd.Categorical(np.asarray(events), categories=EVENT_CLASSES).codes
    if (codes < 0).any():
        raise ValueError("unknown event type(s) found")
    return codes.astype(np.int16)


def _check_layout(df: pd.DataFrame) -> None:
    """Examples address rows by position, so the frame must be as returned by load_*()."""
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0:
        raise ValueError("expected a RangeIndex starting at 0 (use the frame from load_train/load_test)")
    pos = df.groupby(ID_COL, sort=False).cumcount().to_numpy()
    if not (df[ORDER_COL].to_numpy() == pos).all():
        raise ValueError("rows must be sorted by (sequence_id, event_index) with gap-free indices")


def make_train_examples(train: pd.DataFrame, min_prefix: int = MIN_PREFIX,
                        max_prefix: int = MAX_PREFIX) -> pd.DataFrame:
    """One example per (training sequence, prefix length in [min_prefix, max_prefix]).

    Columns:
      sequence_id  system the example comes from
      last_row     row index of the last observed event (the prefix is rows ..last_row of the sequence)
      prefix_len   number of observed events
      target       class code of the next event (the event at last_row + 1)
      weight       1 / (number of examples from this sequence). Analysis only (it would make every
                   sequence count equally); the official metric is unweighted, see validation.evaluate
      is_mimic     True for the one cut made like the test set (prefix_len = length - 30)
    """
    _check_layout(train)
    g = train.groupby(ID_COL, sort=False)
    prefix_len = g.cumcount().to_numpy() + 1          # events observed if we cut after this row
    seq_len = g[ORDER_COL].transform("size").to_numpy()
    keep = (prefix_len >= min_prefix) & (prefix_len <= max_prefix) & (prefix_len < seq_len)

    codes = encode_events(train[EVENT_COL])
    last_row = np.flatnonzero(keep)
    ex = pd.DataFrame({
        ID_COL: train[ID_COL].to_numpy()[last_row],
        "last_row": last_row,
        "prefix_len": prefix_len[last_row],
        "target": codes[last_row + 1],                 # same sequence, guaranteed by prefix_len < seq_len
    })
    n_cuts = ex.groupby(ID_COL, sort=False)["last_row"].transform("size")
    ex["weight"] = 1.0 / n_cuts
    ex["is_mimic"] = ex["prefix_len"] == seq_len[last_row] - TEST_MIMIC_OFFSET
    return ex


def make_test_examples(test: pd.DataFrame) -> pd.DataFrame:
    """One example per test sequence, cut at the end of its observed prefix.

    No target column: the labels are attached only for the final evaluation (load_test_targets).
    """
    _check_layout(test)
    last_row = np.flatnonzero(test[ID_COL].ne(test[ID_COL].shift(-1)).to_numpy())
    return pd.DataFrame({
        ID_COL: test[ID_COL].to_numpy()[last_row],
        "last_row": last_row,
        "prefix_len": test[ORDER_COL].to_numpy()[last_row] + 1,
    })


def load_test_targets(test_examples: pd.DataFrame) -> np.ndarray:
    """Class codes of the true next events, aligned with `test_examples`. FINAL EVALUATION ONLY."""
    labels = load_test_labels().set_index(ID_COL)["next_event"]
    return encode_events(labels.loc[test_examples[ID_COL]].to_numpy())


def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into identifier / categorical / numeric groups."""
    ids = [ID_COL, ORDER_COL, "timestamp"]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    numeric = [
        c for c in df.columns
        if c not in ids and c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]
    return {"ids": ids, "categorical": categorical, "numeric": numeric}
