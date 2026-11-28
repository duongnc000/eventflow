"""LightGBM on the history features (features.py): the classical model.

LightGBMModel follows the shared model interface (validation.py):
    fit(data, examples)            learn from the training examples, return the model itself
    predict_proba(data, examples)  array (n_examples, 14), columns in config.EVENT_CLASSES order

Features are made for every row of `data`, and each example reads the row at its cut (`last_row`),
so they only use the past (checked in 06_features_check).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from eventflow.config import N_CLASSES, RANDOM_STATE
from eventflow.data import _check_layout
from eventflow.features import CATEGORICAL, FEATURE_GROUPS
from eventflow.validation import holdout_sequences

EVENT_CODE = pd.CategoricalDtype(categories=range(-1, N_CLASSES))   # declare the codes in the ev lag as category(-1 if there no previous event for ev_lag)
_cache = {"data": None, "groups": {}}


def group_features(data, group):# (data:training or testing df, group: 6 df featture group: recent,count,since,etc)
    # return the feature group from this data and store in cache for later use(dont need to compute again)
    """The columns of one feature group, for every row of `data`.

    Making features takes a while and gives the same result in every fold, so each group is made
    once and kept. A different frame (e.g. the test set) starts a new cache.
    """
    if _cache["data"] is not data:
        _check_layout(data) # check the row order
        _cache["data"], _cache["groups"] = data, {} #_cache = {"data": train, "groups": {"recent": ..., "counts": ..., ...}}
    if group not in _cache["groups"]: # add the faeture group
        _cache["groups"][group] = FEATURE_GROUPS[group](data)
    return _cache["groups"][group]

class LightGBMModel:
    """LightGBM on the history features of the chosen groups (all six by default).

    groups    feature groups to use; leave one out for an ablation, e.g. every group but "sensors"
    patience  early stopping: stop when the held-back machines have not improved for this many rounds
    params    LightGBM settings, e.g. num_leaves=127 (defaults below)

    Early stopping uses 10% of the TRAINING machines (holdout_sequences), never the validation fold,
    so the validation score stays honest.
    """

    def __init__(self, groups=None, patience=50, print_round=True, **params):
        self.groups = list(FEATURE_GROUPS) if groups is None else list(groups) # if None => all 6 groups, else only the selected group
        self.patience = patience #early stop,  stop when the model not improve for 50 round in a row
        self.print_round = print_round #Whether to print "stopped at round …" after training
        self.params = {"n_estimators": 5000, "learning_rate": 0.05, "num_leaves": 63,
                       "random_state": RANDOM_STATE, "verbose": -1, **params} #n_estimators:num of trees, num_leaves:max end points per tree (how detailed each tree can be)

    def _rows(self, data, examples): #get the feature rows for each last row of the questions
        """The feature row at each example's cut, with the event-code columns as categories."""
        cut = examples["last_row"].to_numpy() #Where each question stops eg [119, 205, 5120]
        X = pd.concat([group_features(data, g).iloc[cut] for g in self.groups], axis=1) # get the group rows at the cut rows so features at row 119, 205 and 5120 (so 3 row each feature group)
        return X.astype({c: EVENT_CODE for c in CATEGORICAL if c in X.columns}) # go rhough each c in CATEGORICAL(["ev_lag0", "ev_lag1", "ev_lag2"]) then build a dictionary, column -> new type: {"ev_lag0": EVENT_CODE, "ev_lag1": EVENT_CODE, "ev_lag2": EVENT_CODE}
        #before:  ev_lag0 = 2, 9, 12   (numbers)
        #after:   ev_lag0 = 2, 9, 12   (categories: labels from the -1..13 menu)

    def fit(self, data, examples):
        fit_part, stop_part = holdout_sequences(examples)        # 90% to learn, 10% for early stopping
        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(self._rows(data, fit_part), fit_part["target"].to_numpy(),
                        eval_X=self._rows(data, stop_part), eval_y=stop_part["target"].to_numpy(),
                        callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        self.best_round_ = self.model_.best_iteration_
        if self.print_round:
            print(f"  stopped at round {self.best_round_}")
        return self

    def predict_proba(self, data, examples):
        proba = np.zeros((len(examples), N_CLASSES)) #An empty num of questions × 14 table to fill in.
        proba[:, self.model_.classes_] = self.model_.predict_proba(self._rows(data, examples)) # makes the predictions using trees 1 → 412 (the best round), adding up the tree, Each tree adds a little bit (+ or −) to each event's score
        return proba #return the prediction table for each row