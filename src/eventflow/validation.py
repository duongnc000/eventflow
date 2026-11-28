"""Shared validation setup: fixed sequence-grouped folds, one evaluation function and one
cross-validation loop that every model runs through.

- Folds are assigned per sequence_id (a system is never split across folds), deterministically
  from config.RANDOM_STATE, so every machine gets the same folds.
- Tuning uses fold config.TUNING_FOLD only; the final model comparison uses all folds.
- `evaluate` reports the primary metric (log loss over all test-like cuts of the held-out systems)
  plus the test-mimic check.

The model contract (see `Model`): a model is any object with
    fit(data, examples)            -> self
    predict_proba(data, examples)  -> array (len(examples), N_CLASSES), columns in EVENT_CLASSES order
where `data` is the whole loaded frame (load_train(), or load_test() for the final evaluation) and
`examples` are the cuts (data.make_train_examples / make_test_examples). A model may only use rows
of the same sequence up to and including each example's `last_row`.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from eventflow.config import EVENT_CLASSES, ID_COL, N_CLASSES, N_FOLDS, OUTPUTS_DIR, RANDOM_STATE
from eventflow.metrics import multiclass_log_loss
from eventflow.tracking import log_experiment

OOF_DIR = OUTPUTS_DIR / "oof"
PROBA_COLS = [f"p_{c}" for c in EVENT_CLASSES]        # probability columns of a saved prediction file
KEY_COLS = [ID_COL, "last_row", "fold", "target"]     # columns that identify a saved prediction row


# ---------------------------------------------------------------------------------------- folds

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
    for fold in _fold_list(examples, folds):
        yield (fold, *split(examples, fold))


def holdout_sequences(examples: pd.DataFrame, frac: float = 0.1,
                      seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split TRAINING examples into (fit part, held-back part) by whole sequences.

    For early stopping or any other choice made during fit: the held-back machines act as a
    validation set inside the training folds, so the validation fold is only used for the score.
    """
    ids = np.sort(examples[ID_COL].unique())
    held = np.random.default_rng(seed).choice(ids, size=int(round(len(ids) * frac)), replace=False)
    is_held = examples[ID_COL].isin(held).to_numpy()
    return examples[~is_held], examples[is_held]


def _fold_list(examples: pd.DataFrame, folds) -> list[int]:
    return list(range(int(examples["fold"].max()) + 1)) if folds is None else [int(f) for f in folds]


# ----------------------------------------------------------------------------------- evaluation

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


# ------------------------------------------------------------------------------ model contract

class Model(Protocol):
    """What cross_validate expects from a model (see the module docstring)."""

    def fit(self, data: pd.DataFrame, examples: pd.DataFrame) -> "Model": ...

    def predict_proba(self, data: pd.DataFrame, examples: pd.DataFrame) -> np.ndarray: ...


def check_proba(proba, n_rows: int) -> np.ndarray:
    """Raise if `proba` is not a valid (n_rows, N_CLASSES) probability table."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape != (n_rows, N_CLASSES):
        raise ValueError(f"predict_proba returned shape {proba.shape}, expected {(n_rows, N_CLASSES)} "
                         f"(one row per example, one column per class in EVENT_CLASSES order)")
    if not np.isfinite(proba).all():
        raise ValueError("predict_proba returned NaN or infinite values")
    if (proba < 0).any():
        raise ValueError("predict_proba returned negative probabilities")
    worst = np.abs(proba.sum(axis=1) - 1).max()
    if worst > 1e-3:
        raise ValueError(f"predict_proba rows must sum to 1 (largest deviation {worst:.4f})")
    return proba


# ------------------------------------------------------------------------------ cross-validation

@dataclass
class CVResult:
    """Output of cross_validate.

    per_fold  one row per fold: logloss, logloss_mimic, n_examples, n_sequences, seconds
    keys      sequence_id, last_row, fold, target of every predicted example (aligned with proba)
    proba     out-of-fold predictions, shape (len(keys), N_CLASSES)
    name      file name used by save() / load_oof(), or None
    """
    per_fold: pd.DataFrame
    keys: pd.DataFrame
    proba: np.ndarray
    name: str | None = None

    @property
    def mean(self) -> float:
        """Mean over folds of the primary metric."""
        return float(self.per_fold["logloss"].mean())

    @property
    def std(self) -> float:
        """Fold-to-fold standard deviation of the primary metric (NaN for a single fold)."""
        return float(self.per_fold["logloss"].std())

    @property
    def mimic(self) -> float:
        """Mean over folds of the test-mimic check."""
        return float(self.per_fold["logloss_mimic"].mean())

    @property
    def folds(self) -> str:
        """Which folds were run, e.g. '0' (tuning) or '0-4' (all)."""
        f = sorted(int(x) for x in self.per_fold.index)
        if len(f) > 1 and f == list(range(f[0], f[-1] + 1)):
            return f"{f[0]}-{f[-1]}"
        return ",".join(map(str, f))

    def __repr__(self) -> str:
        spread = "" if np.isnan(self.std) else f" ± {self.std:.4f}"
        return (f"CVResult({self.name or 'unnamed'}: folds {self.folds}, logloss {self.mean:.4f}{spread}, "
                f"mimic {self.mimic:.4f})")

    def save(self, name: str | None = None) -> None:
        """Write the out-of-fold predictions to outputs/oof/<name>.parquet."""
        self.name = name or self.name
        if not self.name:
            raise ValueError("give the result a name to save it")
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.concat([self.keys.reset_index(drop=True),
                        pd.DataFrame(self.proba.astype(np.float32), columns=PROBA_COLS)], axis=1)
        df.to_parquet(OOF_DIR / f"{self.name}.parquet", index=False)

    def log(self, run_id: str, author: str, model: str, feature_set: str,
            params: dict | None = None, notes: str = "") -> None:
        """Append this result to experiments/results.csv (mean, std, mimic and per-fold scores)."""
        log_experiment(run_id=run_id, author=author, model=model, feature_set=feature_set,
                       val_logloss=self.mean, val_std=self.std, val_mimic=self.mimic,
                       val_folds=self.per_fold["logloss"].tolist(), folds=self.folds,
                       params=params, notes=notes)


def cross_validate(make_model: Callable[[], Model], data: pd.DataFrame, examples: pd.DataFrame,
                   folds=None, name: str | None = None, verbose: bool = True) -> CVResult:
    """Fit a fresh model on each fold's training part and score it on the held-out part.

    make_model  function returning a NEW, unfitted model (a class, or e.g. lambda: MyModel(alpha=1))
    data        the full loaded frame the examples point into (load_train())
    examples    training examples WITH a fold column (add_folds(make_train_examples(train)))
    folds       folds to run; None = all. Use [TUNING_FOLD] while tuning.
    name        if given, the out-of-fold predictions are saved to outputs/oof/<name>.parquet
    """
    fold_of = examples["fold"].to_numpy()
    folds = _fold_list(examples, folds)
    proba_all = np.full((len(examples), N_CLASSES), np.nan)
    rows = []
    for fold in folds:
        is_val = fold_of == fold
        tr, va = examples[~is_val], examples[is_val]
        start = time.perf_counter()
        model = make_model()
        model.fit(data, tr)
        proba = check_proba(model.predict_proba(data, va), len(va))
        rows.append({"fold": fold, **evaluate(va, proba), "seconds": round(time.perf_counter() - start, 1)})
        proba_all[is_val] = proba
        if verbose:
            r = rows[-1]
            print(f"fold {fold}: logloss {r['logloss']:.4f}  mimic {r['logloss_mimic']:.4f}  ({r['seconds']} s)")

    covered = np.isin(fold_of, folds)
    result = CVResult(per_fold=pd.DataFrame(rows).set_index("fold"),
                      keys=examples.loc[covered, KEY_COLS].reset_index(drop=True),
                      proba=proba_all[covered], name=name)
    if name:
        result.save()
    return result


def load_oof(name: str) -> pd.DataFrame:
    """Saved out-of-fold predictions: KEY_COLS + PROBA_COLS, one row per predicted example."""
    return pd.read_parquet(OOF_DIR / f"{name}.parquet")
