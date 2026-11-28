"""Simple baseline models: the reference rows for the final comparison table.

Every model has the same two methods (the shared model interface in TASKS.md):
    fit(data, train_examples)       learn from the training examples, return the model itself
    predict_proba(data, examples)   array (n_examples, 14), columns in config.EVENT_CLASSES order

`data` is the frame from load_train(). An example only stores `last_row`, the row of the last
event it may see (see data.py), so every model reads the history through `last_row`.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eventflow.config import EVENT_COL, ID_COL, N_CLASSES
from eventflow.data import column_groups, encode_events


class Uniform:
    """Every event gets the same chance, 1/14. Learns nothing, ignores the history."""

    def fit(self, data, train_examples):
        return self

    def predict_proba(self, data, examples):
        return np.full((len(examples), N_CLASSES), 1 / N_CLASSES)

class NGram:
    """Lookup table: "after these last `order` events, what came next, and how often?"

    order=0: no history, the usual event mix (class prior)
    order=1: last event only (transition table)
    order=3: last 3 events

    Smoothing: a context seen only a few times gives unreliable chances, so each table is mixed
    with the table one order shorter, as if we added `alpha` extra examples that follow it.
    Example with alpha=10: a 3-event context seen 2 times is mostly decided by the last 2 events,
    while one seen 5,000 times hardly changes. A context never seen uses the shorter table.

    source="cuts": learn from the test-like training examples only (answers are events 91-221)
    source="all":  learn from every event of the training machines, warm-up included
    """

    def __init__(self, order=1, alpha=1.0, source="cuts"):
        self.order = order
        self.alpha = alpha
        self.source = source

    @staticmethod
    def _context(codes, rows, k): #hash the location for the pattern on the table eg: COOLING → RECOVERY store at row 37 on the table then later: the predciton candiate eg(60%rountin,30%TEMP,10%COOLING after the pattern)
        # rows list of stop row of each question [3,5], k is cal on the last 2 event(k=2)
        """One number per row for its last k events: last event + 14 x the one before + 14^2 x ...

        Example (k=2): last event RECOVERY (9), the one before TEMP_WARNING (12) -> 9 + 14*12 = 177.
        Dropping the oldest event is then just "% 14^(k-1)": 177 % 14 = 9, the last event alone.
        """
        #eg row	event	code
        #   2	COOLING_ACTIVATED	2
        #   3	RECOVERY	9
        #   4	TEMP_WARNING	12
        #   5	RECOVERY	9
        key = np.zeros(len(rows), dtype=np.int64) #len(rows) = 2(num of question) = [0,0]
        for back in range(k):                       # back = 0,1
            key += codes[rows - back] * N_CLASSES ** back # BACK = 0 (last event) key = [0, 0] + [9, 9] = [9, 9] , back =1 the event before key = [9, 9] + [28, 168] = [37, 177]
        return key #[37, 177] (COOLing ACT -> REC at row 37, TEMP -> REC at row 177)

    def fit(self, data, train_examples):
        codes = encode_events(data[EVENT_COL]).astype(np.int64) # turn event into codes -> ["ROUTINE_CHECK", "TEMP_WARNING", "RECOVERY", ...] -> [10, 12, 9, ...]

        if self.source == "cuts":
            rows = train_examples["last_row"].to_numpy() # take the last row of each question (last row of 1-90,1-91,..,1-220)
            targets = train_examples["target"].to_numpy() # get the target for each quesiton
        elif self.source == "all":
            ids = data[ID_COL].to_numpy() #ids = ["TR00000", "TR00000", ..., "TR00000", "TR00001", "TR00001", ...]
                                          #         row 0      row 1          row 199    row 200    row 201
            pos = data.groupby(ID_COL, sort=False).cumcount().to_numpy() # postion inside machine [0, 1, 2, 0, 1]
            has_next = np.append(ids[1:] == ids[:-1], False)           # is the next row the same machine [T, T, F, T, F]
            in_train = data[ID_COL].isin(train_examples[ID_COL].unique()).to_numpy() # take the sequnces that not in the valdation data eg A yes, B no: [T, T, T, F, F]
            enough_history = pos >= self.order - 1                    # if the current row have enough previous event to cal the pattern, eg for 1 events order/pattern [T, T, T, T, T]
            rows = np.flatnonzero(in_train & has_next & enough_history) # Keep the rows where all are True [T, T, F, F, F] ,rows = [0, 1]( only row 0 and 1 is where the pattern can end)
            targets = codes[rows + 1] # get the ans throught codes[[1, 2]]
        else:
            raise ValueError("source must be 'cuts' or 'all'")

        table = np.full((1, N_CLASSES), 1 / N_CLASSES)   # start from equal chances [[0.071, 0.071, ..., 0.071]]
        for k in range(self.order + 1):                  # build order 0, 1, ..., self.order
            n_contexts = N_CLASSES ** k #total pattern that could happen for each k(k=1:14, k=2:14x14 =196,etc)
            context = self._context(codes, rows, k) #return the hash/row number for every training pair (TEMP ->,TEMP ->,TEMP ->,RECOVERY ->,COOLING->) = context = [12, 12, 12, 9, 2]
            counts = np.bincount(context * N_CLASSES + targets, minlength=n_contexts * N_CLASSES) # count the target for each pattern (TEMP ->COOL,TEMP->COOL,TEMP->REC,REC->ROUTIN,COOLING->REC) = [170,170,177,136,37] => 170:2,177:1,136:1,37:1
            counts = counts.reshape(n_contexts, N_CLASSES).astype(float) # turn the counts into grid 14x14 for k=1 (eg TEMP->COOLING row 12 column 2 at cell 170)
            shorter = table[np.arange(n_contexts) % max(n_contexts // N_CLASSES, 1)] #take the percentage row from the shorter pattern (eg k=2 then take k=1 eg TEMP->COOL-> get backup row from pattern COOL->)
            total = counts.sum(axis=1, keepdims=True) + self.alpha # sum of the count + alpha(alpha is for adding the backup row then cal percentage still sum to 100%)
            with np.errstate(invalid="ignore", divide="ignore"):
                table = (counts + self.alpha * shorter) / total #cal perecntage using the counts
            table[total[:, 0] == 0] = shorter[total[:, 0] == 0]   # if the pattern is never seen and alpha =0 so it haven't add the shorter version row, now add the shorter version row
        self.table_ = table
        return self

    def predict_proba(self, data, examples): #make prediction
        codes = encode_events(data[EVENT_COL]).astype(np.int64)
        context = self._context(codes, examples["last_row"].to_numpy(), self.order)
        return self.table_[context]#question ends with TEMP  →  row 12  →  COOLING 59%, RECOVERY 34%, ...

class CurrentRowLR:
    """The course template's baseline: logistic regression on the last observed row only.

    Categorical columns (event type, subsystem, shift, ...) are one-hot encoded. Numeric columns
    (severity, sensors) get missing values filled with the median and are rescaled. It uses no
    history at all, only the current row. Unlike the template, it learns from our test-like
    examples instead of every row.
    """

    def __init__(self, max_iter=400):
        self.max_iter = max_iter

    def fit(self, data, train_examples):
        groups = column_groups(data)
        self.columns_ = groups["categorical"] + groups["numeric"]
        prep = ColumnTransformer([
            ("categorical", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             groups["categorical"]),
            ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]),
             groups["numeric"]),
        ])
        self.model_ = Pipeline([("prep", prep), ("lr", LogisticRegression(max_iter=self.max_iter))])
        rows = data.iloc[train_examples["last_row"].to_numpy()][self.columns_]
        self.model_.fit(rows, train_examples["target"].to_numpy())
        return self

    def predict_proba(self, data, examples):
        rows = data.iloc[examples["last_row"].to_numpy()][self.columns_]
        proba = np.zeros((len(examples), N_CLASSES))
        proba[:, self.model_.classes_] = self.model_.predict_proba(rows)   # a class never seen stays 0
        return proba # ouput the 14 percentages for each question