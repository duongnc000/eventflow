# EventFlow: Temporal Event Prediction

AI2210 Deep Learning & Neural Networks, group project.

Given the observed history of a system, predict a probability distribution over the 14 possible
**next events**. The metric is multiclass **Log Loss** (lower is better).

## Setup

Requires Python 3.13. Recommended: [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # creates .venv and installs everything from uv.lock
# or, with plain pip:
pip install -r requirements.txt
```

Put the three course CSVs in [data/](data/) (see [data/README.md](data/README.md)).

## Reproducing the results

> TODO: fill in as the pipeline is built. Target: a short list of commands that regenerate
> every number, table and figure in the report, with no manual steps.

1. `...` (preprocessing / feature cache)
2. `...` (train + evaluate each model)
3. `...` (final model: test-set Log Loss)
4. `...` (generative rollouts + diagnostics)

## Repository layout

```
data/                 course CSVs (git-ignored)
notebooks/            exploration and analysis notebooks (numbered in pipeline order)
src/eventflow/        reusable code imported by notebooks and scripts
  config.py           paths, seed, column names
  metrics.py          log loss (shared by every model)
  tracking.py         appends runs to experiments/results.csv
scripts/              command-line entry points that reproduce the final results
experiments/
  results.csv         experiment log: every reported number must trace back here
reports/figures/      figures used in the report and slides
outputs/              caches, checkpoints, predictions (git-ignored, regenerated)
```

## Ground rules

- Validation split is **by `sequence_id`**. No system may appear in both train and validation.
- Anything fitted on data (scalers, encoders, imputers, vocabularies) is fitted on the local
  training split only.
- **Test labels are for the final evaluation only.** Never use them for tuning or model selection.
- No pretrained models or external data.
- Log every run you might report with `eventflow.tracking.log_experiment(...)`.
- Fixed seed: `RANDOM_STATE = 2210` in `config.py`.

See [TASKS.md](TASKS.md) for who does what.
