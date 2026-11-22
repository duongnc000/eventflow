"""Shared validation setup: fixed sequence-grouped folds and one evaluation function.

Every model must be validated through these functions so that results are comparable:
- folds are assigned per sequence_id (a system is never split across folds), deterministically
  from config.RANDOM_STATE, so every machine gets the same folds;
- tuning uses fold config.TUNING_FOLD only; the final model comparison uses all folds;
- `evaluate` reports the primary metric (log loss over all test-like cuts of the held-out systems)
  plus the test-mimic check.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from eventflow.config import ID_COL, N_FOLDS, RANDOM_STATE
from eventflow.metrics import multiclass_log_loss


def assign_folds(sequence_ids, n_folds: int = N_FOLDS, seed: int = RANDOM_STATE) -> pd.Series:
    """Map each unique sequence_id to a fold in 0..n_folds-1 (balanced, deterministic)."""
    ids = np.sort(pd.unique(np.asarray(sequence_ids)))
    order = np.random.default_rng(seed).permutation(len(ids))
    folds = np.empty(len(ids), dtype=np.int8)
    folds[order] = np.arange(len(ids)) % n_folds
    return pd.Series(folds, index=pd.Index(ids, name=ID_COL), name="fold")


def add_folds(examples: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a copy of `examples` with a `fold` column (from assign_folds)."""
    out = examples.copy()
    out["fold"] = out[ID_COL].map(assign_folds(out[ID_COL], n_folds, seed)).to_numpy()
    return out


def split(examples: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training part, validation part) for one fold."""
    is_val = examples["fold"].to_numpy() == fold
    return examples[~is_val], examples[is_val]


def iter_folds(examples: pd.DataFrame, folds=None) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train part, validation part); all folds by default."""
    for fold in (range(int(examples["fold"].max()) + 1) if folds is None else folds):
        yield (fold, *split(examples, fold))


def evaluate(val_examples: pd.DataFrame, proba: np.ndarray) -> dict[str, float]:
    """Score predictions for validation examples (rows of `proba` aligned with `val_examples`).

    logloss         PRIMARY: all test-like cuts (prefix length 90-220), every example weighted
                    equally. Low variance; its prefix-length mix is close to the test set's.
    logloss_mimic   CHECK: only the one cut per sequence made like the test set (length - 30).
                    Closest copy of the test set, but about 5x noisier (one example per sequence).

    Weighting every sequence equally was considered and rejected: it over-weights short prefixes
    (see notebooks/02_validation.ipynb, Section 3).
    """
    y = val_examples["target"].to_numpy()
    mimic = val_examples["is_mimic"].to_numpy()
    return {
        "logloss": multiclass_log_loss(y, proba),
        "logloss_mimic": multiclass_log_loss(y[mimic], proba[mimic]),
        "n_examples": int(len(y)),
        "n_sequences": int(val_examples[ID_COL].nunique()),
    }
