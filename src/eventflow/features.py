"""History features for the classical model (LightGBM).

make_features(df) returns one row per row of `df`, built ONLY from that row and the rows before
it in the same sequence, never from the future. For a prediction example, take the row at its cut:
    X = make_features(train)
    X_tr = X.loc[tr["last_row"]]

Feature groups (each group is one unit for the ablations):
  recent   the last 3 events and their severities, and how many times in a row the last event repeated
  counts   how often each event type appeared in the last 5 / 10 / 30 events
  since    for each event type, how many events ago it last happened (NaN = not yet in this sequence)
"""

import numpy as np
import pandas as pd

from eventflow.config import EVENT_CLASSES, EVENT_COL, ID_COL, N_CLASSES
from eventflow.data import _check_layout, encode_events

N_LAGS = 3
WINDOWS = (5, 10, 30)
# Event-code columns: tell LightGBM these are categories, not numbers (["ev_lag0", "ev_lag1", "ev_lag2"])
CATEGORICAL = [f"ev_lag{k}" for k in range(N_LAGS)]


def _positions(df):
    """Position of each row inside its sequence: 0, 1, 2, ...

    row:       0        1        2        3        4
    sequence:  TR00000  TR00000  TR00000  TR00001  TR00001
    returns:   [0, 1, 2, 0, 1]
    """
    return df.groupby(ID_COL, sort=False).cumcount().to_numpy()


def recent_features(df, n_lags=N_LAGS):
    """Group 1 RECENT: the last n_lags events (-1 = before the start of the sequence), their
    severities, and the run length of the last event (RECOVERY, RECOVERY, RECOVERY -> 3)."""
    # Example (5 rows, two sequences):
    #   events   TEMP_WARNING, RECOVERY, RECOVERY | LOAD_INCREASE, TEMP_WARNING
    #   codes    [12, 9, 9, 6, 12]     sev [2, 0, 0, 1, 2]     pos [0, 1, 2, 0, 1]
    codes = encode_events(df[EVENT_COL]).astype(np.int64)
    sev = df["severity"].to_numpy(dtype=float)
    pos = _positions(df)
    rows = np.arange(len(df))
    out = {}
    for k in range(n_lags):
        # Is there an event k steps back in this sequence?  k=1: [F, T, T, F, T]
        ok = pos >= k
        lag_ev = np.full(len(df), -1, dtype=np.int8)
        lag_sev = np.full(len(df), np.nan, dtype=np.float32)
        # The event k steps back is simply k rows up: k=1 -> codes[[0, 1, 3]] -> lag_ev [-1, 12, 9, -1, 6]
        lag_ev[ok] = codes[rows[ok] - k]
        lag_sev[ok] = sev[rows[ok] - k]
        out[f"ev_lag{k}"] = lag_ev
        out[f"sev_lag{k}"] = lag_sev

    # Run length: how many times in a row the current event has repeated.
    #   sequence    A   A   A   A   B   B   B
    #   codes      12   9   9   9   9   9  12
    #   new_run     T   T   F   F   T   F   T    (start of a sequence, or a different event than before)
    #   run_start   0   1   1   1   4   4   6    (row where the current run began)
    #   run_length  1   1   2   3   1   2   1    (rows - run_start + 1)
    new_run = (pos == 0) | (codes != np.roll(codes, 1))
    run_start = np.maximum.accumulate(np.where(new_run, rows, 0))
    out["run_length"] = (rows - run_start + 1).astype(np.int16)
    return pd.DataFrame(out, index=df.index)   # 7 columns


def count_features(df, windows=WINDOWS):
    """Group 2 COUNTS: how many times each event type appeared in the last w events (including the
    current one), for each window size w. Never reaches back past the start of the sequence."""
    codes = encode_events(df[EVENT_COL]).astype(np.int64)
    pos = _positions(df)
    rows = np.arange(len(df))
    one_hot = np.eye(N_CLASSES, dtype=np.int32)[codes]
    # running[i] = count of each event type in rows 0 .. i-1 (a leading row of zeros)
    running = np.vstack([np.zeros((1, N_CLASSES), np.int32), one_hot.cumsum(axis=0)])
    out = {}
    for w in windows:
        # First row of the window, but never before the start of this sequence
        start = rows - np.minimum(pos, w - 1)
        # Counts in the window = running total at the end - running total before the start.
        #   e.g. row 3, window rows 1-3:  running[4] = [2, 2, 0]  -  running[1] = [1, 0, 0]  =  [1, 2, 0]
        counts = running[rows + 1] - running[start]
        for j, name in enumerate(EVENT_CLASSES):
            out[f"cnt{w}_{name}"] = counts[:, j].astype(np.int16)   # small ints save memory
    # e.g. cnt30_COMPONENT_REPLACED = 1 at row 183: in the 30 events from row 154 to row 183 of
    # this sequence, COMPONENT_REPLACED happened exactly once.
    return pd.DataFrame(out, index=df.index)   # 3 windows x 14 events = 42 columns


def since_features(df):
    """Group 3 SINCE: for each event type, how many events ago it last happened in this sequence
    (0 = it is the current event, NaN = it has not happened yet in this sequence)."""
    codes = encode_events(df[EVENT_COL]).astype(np.int64)
    pos = _positions(df)
    rows = np.arange(len(df))
    # First row of each row's sequence, e.g. [0, 0, 0, 0, 0, 5, 5, 5] (a new sequence starts at row 5)
    seq_start = rows - pos
    out = {}
    for j, name in enumerate(EVENT_CLASSES):
        # e.g. TEMP_WARNING (code 12), one sequence in rows 0-4 and the next in rows 5-7:
        #   codes == 12   T   F   F   F   T   F   T   F
        #   last          0   0   0   0   4   4   6   6    (latest row so far with this event)
        #   since         0   1   2   3   0   NaN 0   1    (rows - last; NaN when last is in an earlier sequence)
        last = np.maximum.accumulate(np.where(codes == j, rows, -1))
        since = (rows - last).astype(np.float32)
        since[last < seq_start] = np.nan
        out[f"since_{name}"] = since
    return pd.DataFrame(out, index=df.index)   # 14 columns


FEATURE_GROUPS = {
    "recent": recent_features,
    "counts": count_features,
    "since": since_features,
}


def make_features(df, groups=None):
    """All feature groups side by side, or only the ones in `groups` (e.g. for an ablation).

    All three groups: 7 + 42 + 14 = 63 columns.
    """
    _check_layout(df)
    groups = list(FEATURE_GROUPS) if groups is None else groups
    return pd.concat([FEATURE_GROUPS[g](df) for g in groups], axis=1)
