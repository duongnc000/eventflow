# Task split

**A** = me, **B** = teammate. Tick items off as they are done.

## Phase 1: Foundations (A), with B on independent groundwork

These define the shared interface for everything after, so they should be **reusable modules in
`src/eventflow/`**, not only notebook cells.

### A
- [ ] **EDA** (`notebooks/01_eda.ipynb`): event classes and class balance, sequence lengths,
      time gaps and bursts, missingness, correlated or redundant sensors, per-system differences,
      transition matrix. Save the key figures to `reports/figures/`.
- [ ] **Sample construction** (`src/eventflow/data.py`): turn complete training histories into
      (prefix → next event) examples whose prefix lengths match the test set (90–250).
      Cache to parquet in `outputs/`.
- [ ] **Validation strategy** (`src/eventflow/validation.py`): group split by `sequence_id`
      (GroupKFold or a fixed holdout), with no future information. **Freeze it** so every model
      uses the same folds.
- [ ] **Baselines**: reproduce uniform (~2.64), transition table (~1.81) and template LR (~1.75)
      through `metrics.multiclass_log_loss`, logged to `experiments/results.csv`.

### B (in parallel, needs only the event column)
- [ ] **Generation diagnostics** (`src/eventflow/gen_metrics.py`): event-frequency JS divergence,
      bigram/trigram comparison, diversity (unique n-gram ratio, duplicate rate,
      nearest-neighbour similarity to training), and optionally inter-arrival/burst statistics.
      Test them by comparing real sequences against shuffled ones.
- [ ] Read the EDA as it lands. Draft the deep-model data loader (padding/truncation, embedding
      of categorical columns).

**Handover:** A shares the `data.py` / `validation.py` API, and everyone runs from the same folds.

## Phase 2: Features and models

| | A | B |
|---|---|---|
| Representation | History features: recent events and n-grams, counts in windows, time since each event type, time-gap statistics, sensor trends/deltas, missingness flags | Per-step input encoding for sequences: event and categorical embeddings, normalised time gaps, sensors |
| Models | **Classical:** LightGBM (and/or LR) on engineered features | **Deep #1:** GRU/LSTM. **Deep #2:** Transformer or TCN |
| Ablations | Remove feature groups one at a time (sensors, time gaps, history window, ...) | Context length, no sensors, no time gaps, architectural component off |

- [ ] Each model is tuned on the shared folds and every run is logged.

## Phase 3: Analysis and generation

### A
- [ ] Fair comparison table across all models and final model selection (validation only).
- [ ] Final test-set evaluation (once).
- [ ] Error analysis of the final model: per-class loss, confusion between similar events,
      performance vs prefix length, calibration.

### B
- [ ] Train a reduced event (+ time-gap) model for rollouts.
- [ ] Sample several continuations per held-out prefix (temperature/top-k sampling, not greedy).
- [ ] Run the diagnostics, build the summary table, pick example rollouts, and discuss failure
      modes and memorisation.

## Shared (both)
- [ ] `scripts/` entry points plus a README "Reproducing" section. Test them on a clean clone.
- [ ] Report (5–8 pages): each person drafts their own sections, and the other reviews them.
- [ ] Slides: include at least one rollout and the generation diagnostics.
- [ ] **Oral-exam prep (40%)**: walk each other through your half. Each must be able to explain
      the *whole* pipeline.
