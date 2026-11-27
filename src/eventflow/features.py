"""History features for the classical model (LightGBM).

make_features(df) returns one row per row of `df`, built ONLY from that row and the rows before
it in the same sequence, never from the future. For a prediction example, take the row at its cut:
    X = make_features(train)
    X_tr = X.loc[tr["last_row"]]

Feature groups (each group is one unit for the ablations):
  recent   the last 3 events and their severities, and how many times in a row the last event repeated
  counts   how often each event type appeared in the last 5 / 10 / 30 events
  since    for each event type, how many events ago it last happened (NaN = not yet in this sequence)
  habits   what this machine is like so far: share of each event type, average severity, events so far
  timing   how fast events are happening: last 3 gaps, time taken by the last 5 / 20 events, usual pace, burst
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

def habit_features(df): #output the share of each event so far and the mean of severyty so far for each state by the total amount of events so far
    """Group 4 HABITS: what this machine is like SO FAR (from the start of its sequence up to this
    row, never the whole diary): the share of each event type, the average severity, and how many
    events it has had."""
    codes = encode_events(df[EVENT_COL]).astype(np.int64) #[0, 1, 1, 0, 1, 2, 1]
    pos = _positions(df) # [0, 1, 2, 3, 4, 0, 1]    => machine B starts again at 0
    rows = np.arange(len(df)) #[0, 1, 2, 3, 4, 5, 6]
    seq_start = rows - pos  # first row of each row's sequence => [0, 0, 0, 0, 0, 5, 5]
    n_so_far = pos + 1   # events so far, including this one #[1, 2, 3, 4, 5, 1, 2]
    one_hot = np.eye(N_CLASSES, dtype=np.int32)[codes] # create one hot encoding for each event for every row
    running = np.vstack([np.zeros((1, N_CLASSES), np.int32), one_hot.cumsum(axis=0)]) # total down each column(eventype) for every row, eg at row:7 =>	2TEMP	4REC	1LOAD
    counts_so_far = running[rows + 1] - running[seq_start]   # total up to this row total before each machine started(if a machine start remove the count for the prev machine) at row 5 => [2,3,1] - running[5]: [2,3,0] = 0, 0, 1 (only LOAD)
    out = {}
    for j, name in enumerate(EVENT_CLASSES): # why share not count bc counts grow as the diary gets longer, so they're harder to compare using count
        out[f"share_{name}"] = (counts_so_far[:, j] / n_so_far).astype(np.float32) # share of count => share = (how many of this event so far) ÷ (how many events so far)

    sev = df["severity"].to_numpy(dtype=float)
    sev_running = np.concatenate([[0.0], np.cumsum(sev)])    # running total of severity
    out["mean_severity_so_far"] = ((sev_running[rows + 1] - sev_running[seq_start]) / n_so_far).astype(np.float32) # same thing for severity (mean = the total severity so far for each machine or seq / total event so far each machine)
    out["events_so_far"] = n_so_far.astype(np.int16) #[1, 2, 3, 4, 5, 1, 2]
    return pd.DataFrame(out, index=df.index) #14 event shares  +  1 mean severity  +  1 events so far  =  16 columns

def timing_features(df):
    """Group 5 TIMING: how fast events have been happening, from the timestamps up to this row (never
    the gap to the next event, which is unknown at prediction time). Gaps are skewed (0.05 to 7.5,
    median about 1), so everything is on a log scale. NaN = not enough earlier events in this sequence."""
    t = df["timestamp"].to_numpy(dtype=float)  # time when each event happen [10.0, 11.0, 11.5, 13.5, 1.3, 2.1]
    pos = _positions(df) #[0, 1, 2, 3, 0, 1]
    rows = np.arange(len(df)) #[0,1,2,3,4,5]

    def time_back(n):
        """Time from the event n steps back up to this event (NaN if the sequence has no such event)."""
        span = np.full(len(df), np.nan) # [nan, nan, nan, nan, nan, nan]
        ok = pos >= n # for n = 1 => [F, T, T, T, F, T] is there an event 1 step back in this machine
        span[ok] = t[rows[ok]] - t[rows[ok] - n] # time now − time n events back, time_back(1) 
        return span # time_back(1)  =[nan, 1.0, 0.5, 2.0, nan, 0.8], time_back(2)  =[nan, nan, 1.5, 2.5, nan, nan] 

    out = {}
    gap = time_back(1)                                # gap just before each event
    for k in range(N_LAGS): # select the specific gap from the gaps(from [nan, 1.0, 0.5, 2.0, nan, 0.8]) => log_gap0 = [nan, 1.0, 0.5, 2.0, nan, 0.8]=>no shift,log_gap1 = [nan, nan, 1.0, 0.5, nan(bc it from machine B), nan]=> shift 1 in all seq, log_gap2 = [nan, nan, nan, 1.0, nan(bc it from machine B), nan] => shift 2 in all seq
        # log_gap0 = gap before the current event, log_gap1 = gap before the previous event, ...
        lag = np.full(len(df), np.nan) # [nan, nan, nan, nan, nan, nan]
        ok = pos >= k # for k= 1 => [F, T, T, T, F, T]
        lag[ok] = gap[rows[ok] - k] # rows[ok] = [1, 2, 3, 5], rows[ok - k] = [0, 1, 2, 4], gap[rows[ok] - k] = [nan, 1.0, 0.5, nan], lag = [nan, nan, 1.0, 0.5, nan, nan] here is log_gap1: The gap of the event 1 step before the current one
        out[f"log_gap{k}"] = np.log(lag).astype(np.float32) #log_gap0: The gap of the current/last event, log_gap1: The gap of the event 1 step before the current one, log_gap2: The gap of the event 2 steps before the current one
    for n in (5, 20):
        # time taken by the last n events (n gaps), e.g. small = a burst of events
        out[f"log_span{n}"] = np.log(time_back(n)).astype(np.float32) 

    # This machine's usual pace so far: average gap from its first event up to now
    seq_start = rows - pos #seq_start = [0, 0, 0, 0, 0, 0, 0, 0, 0 | 9, 9]
    mean_gap = np.full(len(df), np.nan)
    ok = pos > 0 #[F, T, T, T, T, T, T, T, T | F, T]
    mean_gap[ok] = (t[ok] - t[seq_start[ok]]) / pos[ok] # minus the current event time witht the start of seq event time the divide by the total event from start to current
    out["log_mean_gap_so_far"] = np.log(mean_gap).astype(np.float32)
    # Burst: last 5 gaps compared with the usual pace (0 = usual, -0.7 = twice as fast, +0.7 = twice as slow)
    out["burst5"] = (np.log(time_back(5)) - np.log(5 * mean_gap)).astype(np.float32) # log( time the last 5 gaps took  ÷  (5 × mean gap so far) ), Negative → faster than usual (a burst),0 → the usual speed,Positive → slower than usual
    return pd.DataFrame(out, index=df.index)   # 3 gaps + 2 spans + usual pace + burst = 7 columns

FEATURE_GROUPS = {
    "recent": recent_features,
    "counts": count_features,
    "since": since_features,
    "habits": habit_features,
    "timing": timing_features,
}


def make_features(df, groups=None):
    """All feature groups side by side, or only the ones in `groups` (e.g. for an ablation).

    All five groups: 7 + 42 + 14 + 16 + 7 = 86 columns.
    """
    _check_layout(df)
    groups = list(FEATURE_GROUPS) if groups is None else groups
    return pd.concat([FEATURE_GROUPS[g](df) for g in groups], axis=1)
