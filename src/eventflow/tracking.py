"""Minimal experiment log: one row per run appended to experiments/results.csv.

Every number that ends up in the report should be traceable to a row here. Runs made with
validation.cross_validate are logged with CVResult.log(), which fills the per-fold columns:

  folds        which folds were run: "0" (tuning) or "0-4" (all)
  val_logloss  mean primary metric over those folds
  val_std      fold-to-fold std of the primary metric (empty for a single fold)
  val_mimic    mean test-mimic check
  val_folds    the primary metric per fold (JSON list), for paired fold-by-fold comparisons
"""

import csv
import json
import math
from datetime import datetime

from eventflow.config import EXPERIMENT_LOG

FIELDS = [
    "timestamp", "run_id", "author", "model", "feature_set", "folds",
    "val_logloss", "val_std", "val_mimic", "val_folds", "test_logloss", "params", "notes",
]


def _fmt(x: float | None) -> str:
    return "" if x is None or math.isnan(x) else f"{x:.5f}"


def log_experiment(
    run_id: str,
    author: str,
    model: str,
    feature_set: str,
    val_logloss: float,
    test_logloss: float | None = None,
    params: dict | None = None,
    notes: str = "",
    val_std: float | None = None,
    val_mimic: float | None = None,
    val_folds: list[float] | None = None,
    folds: str = "",
) -> None:
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not EXPERIMENT_LOG.exists() or EXPERIMENT_LOG.stat().st_size == 0
    if not new_file:
        with EXPERIMENT_LOG.open(encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            raise ValueError(f"{EXPERIMENT_LOG} has the header {header}, expected {FIELDS}")
    with EXPERIMENT_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "author": author,
            "model": model,
            "feature_set": feature_set,
            "folds": folds,
            "val_logloss": _fmt(val_logloss),
            "val_std": _fmt(val_std),
            "val_mimic": _fmt(val_mimic),
            "val_folds": "" if val_folds is None else json.dumps([round(float(v), 5) for v in val_folds]),
            "test_logloss": _fmt(test_logloss),
            "params": json.dumps(params or {}, sort_keys=True),
            "notes": notes,
        })
