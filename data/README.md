# Data

Place the three EventFlow CSV files from the LMS here (they are git-ignored):

- `eventflow_train.csv`: 6,500 complete training histories
- `eventflow_test.csv`: 900 test prefixes, each ending right before the event to predict
- `eventflow_test_labels.csv`: true next event per test sequence (**final evaluation only, never for training or tuning**)
