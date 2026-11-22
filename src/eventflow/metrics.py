"""Evaluation helpers. Every model should be scored through `multiclass_log_loss`
so that validation and test numbers are computed identically."""

import numpy as np
from sklearn.metrics import log_loss

from eventflow.config import N_CLASSES

EPS = 1e-7


def clip_and_normalize(proba: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip probabilities away from 0 and renormalise rows to sum to 1."""
    proba = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def multiclass_log_loss(y_true, proba: np.ndarray, sample_weight=None,
                        classes=range(N_CLASSES)) -> float:
    """Multiclass log loss.

    `y_true` holds class codes (indices into config.EVENT_CLASSES) and the columns of `proba`
    are ordered the same way. `sample_weight` gives a weighted average (None = all equal).
    """
    return float(log_loss(y_true, clip_and_normalize(proba),
                          sample_weight=sample_weight, labels=list(classes)))
