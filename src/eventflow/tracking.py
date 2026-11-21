"""Minimal experiment log: one row per run appended to experiments/results.csv.

Every number that ends up in the report should be traceable to a row here.
"""

import csv
import json
from datetime import datetime

from eventflow.config import EXPERIMENT_LOG

FIELDS = [
    "timestamp", "run_id", "author", "model", "feature_set",
    "val_logloss", "test_logloss", "params", "notes",
]


def log_experiment(
    run_id: str,
    author: str,
    model: str,
    feature_set: str,
    val_logloss: float,
    test_logloss: float | None = None,
    params: dict | None = None,
    notes: str = "",
) -> None:
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not EXPERIMENT_LOG.exists() or EXPERIMENT_LOG.stat().st_size == 0
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
            "val_logloss": f"{val_logloss:.5f}",
            "test_logloss": "" if test_logloss is None else f"{test_logloss:.5f}",
            "params": json.dumps(params or {}, sort_keys=True),
            "notes": notes,
        })
