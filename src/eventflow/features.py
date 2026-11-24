"""History features for the classical model (LightGBM).

make_features(df) returns one row per row of `df`, built ONLY from that row and the rows before
it in the same sequence, never from the future. For a prediction example, take the row at its cut:
    X = make_features(train)
    X_tr = X.loc[tr["last_row"]]

Feature groups (each group is one unit for the ablations):
  recent   the last 3 events and their severities, and how many times in a row the last event repeated
  counts   how often each event type appeared in the last 5 / 10 / 30 events
"""

import numpy as np
import pandas as pd

from eventflow.config import EVENT_CLASSES, EVENT_COL, ID_COL, N_CLASSES
from eventflow.data import _check_layout, encode_events

N_LAGS = 3
WINDOWS = (5, 10, 30)
CATEGORICAL = [f"ev_lag{k}" for k in range(N_LAGS)]   # event codes: tell LightGBM these are categories (["ev_lag0", "ev_lag1", "ev_lag2"])


def _positions(df):#return the position of each event in the sequence
    #row:      0        1        2        3        4
    #machine:  TR00000  TR00000  TR00000  TR00001  TR00001
    #returns:  array([0, 1, 2, 0, 1])
    """Position of each row inside its sequence: 0, 1, 2, ..."""
    return df.groupby(ID_COL, sort=False).cumcount().to_numpy() 


def recent_features(df, n_lags=N_LAGS):
    """Group 1 RECENT: the last n_lags events (-1 = before the start of the sequence), their
    severities, and the run length of the last event (RECOVERY, RECOVERY, RECOVERY -> 3)."""
    codes = encode_events(df[EVENT_COL]).astype(np.int64) #convert eventype to code/number (TEMP_WARNING, RECOVERY, RECOVERY, LOAD_INCREASE, TEMP_WARNING => [12, 9, 9, 6, 12])
    sev = df["severity"].to_numpy(dtype=float) # serverity as numpy array of decimal number (sev = [2.0, 0.0, 0.0, 1.0, 2.0])
    pos = _positions(df) #postion of event for each seq (pos = [0, 1, 2, 0, 1])
    rows = np.arange(len(df)) # [0,..,num of total event]
    out = {}
    for k in range(n_lags): # run k= 0,1,2
        ok = pos >= k                            # is there an event k steps back in this sequence? ([0, 1, 2, 0, 1] =>[F, T, T, F, T] at k=1)
        lag_ev = np.full(len(df), -1, dtype=np.int8) #[-1, -1, -1, -1, -1]
        lag_sev = np.full(len(df), np.nan, dtype=np.float32) #[nan, nan, nan, nan, nan]
        lag_ev[ok] = codes[rows[ok] - k] # what happen at k event ago, rows[ok] = [1, 2, 4], rows[ok] - k  = [0, 1, 3], codes[[0, 1, 3]] = [12, 9, 6], lag_ev = [-1, 12, 9, -1, 6]     
        lag_sev[ok] = sev[rows[ok] - k] # what happen at k  severty ago, sev[[0, 1, 3]] = [2, 0, 1], lag_sev = [nan, 2, 0, nan, 1]
        out[f"ev_lag{k}"] = lag_ev
        out[f"sev_lag{k}"] = lag_sev

    # how many time in a row the event repeat REC, REC, REC => run_length = 3    
    #machines= A	A	A	A	B	B	B, codes =12	9	9	9	9	9	12
    new_run = (pos == 0) | (codes != np.roll(codes, 1))         # a new run start either the start of a new seq or the previous is != the current, (T,T,F,F,T,F,T)
    run_start = np.maximum.accumulate(np.where(new_run, rows, 0)) # if run start here, replace with rows num [0,..,len] else 0 => 0,	1	0	0	4	0	6 => keep the biggest num seen so far 0	1	1	1	4	4	6
    out["run_length"] = (rows - run_start + 1).astype(np.int16) # (0	1	2	3	4	5	6) - (0	1	1	1	4	4	6) + 1 = 1	1	2	3	1	2	1 => 1,2,3 => 9,9,9 | 1,2 => 9,9
    return pd.DataFrame(out, index=df.index) # 7 col ev_lag0/1/2, sev_lag0/1/2, run_length


def count_features(df, windows=WINDOWS):
    """Group 2 COUNTS: how many times each event type appeared in the last w events (including the
    current one), for each window size w. Never reaches back past the start of the sequence."""
    codes = encode_events(df[EVENT_COL]).astype(np.int64) 
    pos = _positions(df)
    rows = np.arange(len(df))
    one_hot = np.eye(N_CLASSES, dtype=np.int32)[codes]  # one hot encode the code
    running = np.vstack([np.zeros((1, N_CLASSES), np.int32), one_hot.cumsum(axis=0)])  #  count eventype for every sequence
    out = {}
    # how many event of each type in the last w events of each seq
    for w in windows:
        start = rows - np.minimum(pos, w - 1)    # first row of the window, inside this sequence
        counts = running[rows + 1] - running[start] # cummulative quantiy of the row - the cummulative of the start = cummaltive quanty of the window
        #running[3 + 1] = running[4] = [2, 2, 0]   ← counts in rows 0–3
        #running[1]                  = [1, 0, 0]   ← counts in row 0 (before the window)
        #counts                      = [1, 2, 0]   → TEMP 1, REC 2, LOAD 0 (REC, REC, TEMP)
        for j, name in enumerate(EVENT_CLASSES):
            out[f"cnt{w}_{name}"] = counts[:, j].astype(np.int16)   # small ints save memory 
    return pd.DataFrame(out, index=df.index)
#            Machine           cnt30_COMPONENT_REPLACED  cnt30_CONTROL_ALERT  …  cnt30_TEMP_WARNING
#row...      ?                   ?                       ?                       ?
#row 183     (TR00000)          1                       0            …          4
#cnt30_COMPONENT_REPLACED = 1 at row 183 means that in the 30 events from row 154 to row 183 of machine TR00000, COMPONENT_REPLACED happened exactly once.


FEATURE_GROUPS = {
    "recent": recent_features,
    "counts": count_features,
}


def make_features(df, groups=None):
    """All feature groups side by side, or only the ones in `groups` (e.g. for an ablation)."""
    _check_layout(df)
    groups = list(FEATURE_GROUPS) if groups is None else groups
    return pd.concat([FEATURE_GROUPS[g](df) for g in groups], axis=1)
    #recent (7 columns) + counts (42 columns) => one table, 49 columns