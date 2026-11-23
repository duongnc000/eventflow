# Task split

**A** = me, **B** = teammate. Tick items off as they are done.

A builds the data/validation foundations, the deep sequence models and the generation model.
B builds the baselines, the feature-based models and the generation diagnostics, runs the final
comparison and analysis, and leads the slides and report editing.

## Phase 1: Foundations

These define the shared interface for everything after, so they should be **reusable modules in
`src/eventflow/`**, not only notebook cells.

### A
- [x] **EDA** (`notebooks/01_eda.ipynb`): event classes and class balance, sequence lengths,
      time gaps and bursts, missingness, correlated or redundant sensors, per-system differences,
      transition matrix. Key figures in `reports/figures/`.
- [x] **Sample construction** (`src/eventflow/data.py`): (sequence, cut point) examples at the
      test-like prefix lengths 90–220.
- [x] **Validation strategy** (`src/eventflow/validation.py`, `notebooks/02_validation.ipynb`):
      frozen 5 folds grouped by `sequence_id`, unweighted primary metric + test-mimic check.
- [ ] **Shared model interface**: every model implements `fit(data, train_examples)` and
      `predict_proba(data, examples)`; `cross_validate(make_model, data, examples)` runs all folds
      and returns per-fold scores; every model saves its out-of-fold predictions to `outputs/`.
      Add the fold std and test-mimic score to `tracking.log_experiment`. Short usage example for B.

### B
- [ ] **Baselines** (`notebooks/03_baselines.ipynb`, through the shared interface): uniform,
      class prior, transition table (smoothing tuned on fold 0), template LR (current row only),
      last-3-events n-gram. Experiment: transition table trained on all events vs. on test-like
      cuts only. All on the 5 folds, logged to `experiments/results.csv`. No test labels.
      (Brief's test scores for reference: uniform ~2.64, transition ~1.81, template LR ~1.75.)
- [ ] **Generation diagnostics** (`src/eventflow/gen_metrics.py`): event-frequency JS divergence,
      bigram/trigram comparison, diversity (unique n-gram ratio, duplicate rate,
      nearest-neighbour similarity to training), and optionally inter-arrival/burst statistics.
      Test them by comparing real sequences against shuffled ones.
- [ ] Read `01_eda` and `02_validation` (decisions in their Notes cells and EDA Section 9).

**Handover:** A shares the model interface; everyone runs from the same folds and the same
`evaluate`.

## Phase 2: Features and models

| | A | B |
|---|---|---|
| Representation | Per-step input encoding for sequences: event and categorical embeddings, normalised time gaps, sensors. Deep data loader (padding/truncation) | History features: recent events and n-grams, counts in windows, time since each event type, time-gap statistics, sensor trends/deltas, missingness flags |
| Models | **Deep #1:** GRU/LSTM. **Deep #2:** Transformer or TCN | **Classical:** LightGBM (and/or LR) on engineered features |
| Ablations | Context length, no sensors, no time gaps, architectural component off | Remove feature groups one at a time (sensors, time gaps, history window, ...) |

- [ ] Each model is tuned on fold 0, compared on all 5 folds, and every run is logged with its
      out-of-fold predictions saved.
- [ ] End of Phase 2: walk each other through your models and features (oral-exam prep).

## Phase 3: Analysis and generation

### A
- [ ] Train a reduced event (+ time-gap) model for rollouts.
- [ ] Sample several continuations per held-out prefix (temperature/top-k sampling, not greedy).
- [ ] Re-check "loss by prefix length" (02_validation, Section 3) for the long-context deep models:
      if they improve with longer prefixes, compare the primary metric with the length-matched one.

### B
- [ ] Run the generation diagnostics on the rollouts, with an n-gram generator as a reference;
      build the summary table, pick example rollouts, and discuss failure modes and memorisation.
- [ ] Fair comparison table across all models (from the saved out-of-fold predictions) and final
      model selection (validation only).
- [ ] Final test-set evaluation (once): the final model and all baselines.
- [ ] Error analysis of the final model: per-class loss, confusion between similar events,
      performance vs prefix length, calibration.

## Shared (both)
- [ ] `scripts/` entry points plus a README "Reproducing" section. Test them on a clean clone.
- [ ] Report (5–8 pages): each person drafts their own sections, the other reviews them;
      B leads the structure and final editing.
- [ ] Slides (B leads): include at least one rollout and the generation diagnostics.
- [ ] **Oral-exam prep (40%)**: walk each other through your half. Each must be able to explain
      the *whole* pipeline.
