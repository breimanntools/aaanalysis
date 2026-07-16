# ADR-0061 — ModelEvaluator: repeated cross-validation, bootstrap CIs, and paired model comparison

Status: Accepted — 2026-07-16

## Context

Going from a feature matrix to a *rigorously evaluated* model was hand-rolled boilerplate
(issue #91). Repeated/stratified cross-validation, multi-seed mean±std, bootstrap confidence
intervals, and paired ΔMCC significance were all reimplemented outside AAanalysis by downstream
projects (a `summ()` / `boot_ci()` / 5-seed loop). The package shipped only `comp_bootstrap_ci`,
and the existing `.eval` methods are per-class and tied to *feature-set* quality (`CPP.eval`) or a
single fitted deployment model (`AAPred.eval`, `TreeModel.eval`) — there was no general,
model-agnostic evaluation harness. AAanalysis's stated stance is "MCC + honest CIs," yet it offered
no turn-key way to produce exactly that, and the rigor matters: on one project dPULearn looked like
+0.16 MCC on a single hold-out but −0.07 under repeated CV.

The issue offered three shapes (standalone `aa.evaluate`/`aa.compare` helpers; a dedicated `Tool`
class + plot; a hybrid) and asked to pick one during triage.

## Decision

Add a dedicated **`ModelEvaluator(Tool)`** class (option 2, the recommended shape) paired 1:1 with
**`ModelEvaluatorPlot`**, in `aaanalysis/prediction/`. No new dependency — scikit-learn and scipy
are already core; the bootstrap reuses `comp_bootstrap_ci` / `ut.bootstrap_ci_`.

- **`run(X, labels, n_cv=5, n_rounds=1, metrics=None, ci=0.95, random_state=None) -> df_eval`** runs
  `n_rounds` repeats of stratified `n_cv`-fold cross-validation (each repeat reshuffled with a
  distinct seed `random_state + round`), scoring **every model on the same folds** within a round.
  Per (model, metric) it returns `score` (mean), `score_std` (population std), a percentile
  bootstrap CI (`ci_low`/`ci_high`) of the mean, and `n_scores` (= `n_cv * n_rounds`). The raw
  per-fold scores are kept on `df_scores_`.
- **`eval(metric="mcc", ci=0.95, random_state=None) -> df_eval`** compares the evaluated models
  **pairwise on the shared folds** using `df_scores_` (so no cross-validation is repeated): a signed
  `delta = score_a - score_b` (mean over folds), `delta_std`, a bootstrap CI on the paired
  differences, and a two-sided **Wilcoxon signed-rank** `p_value`. Requires ≥2 models.
- `ModelEvaluatorPlot.scores(df_eval)` draws the CI bars; `ModelEvaluatorPlot.compare(df_eval)`
  draws the signed-delta bars with CI whiskers and significance stars.

Metrics extend `LIST_METRICS_PRED` with **`mcc`** (Matthews correlation coefficient, the headline
model-quality metric); default `list_metrics=["accuracy", "balanced_accuracy", "mcc"]` (all
label-value agnostic, no `predict_proba` needed). Probability metrics (`roc_auc`) require
`predict_proba`, validated capability-based only when requested.

### Why `run` + `eval` (not `evaluate`/`compare` helpers)

The `Tool` template mandates `run` + `eval`. `run` produces the primary artifact (the evaluation
table); `eval` evaluates the *relationship between* models (the paired comparison). This keeps a
single stateful object holding the per-fold scores that both the table and the comparison derive
from — a comparison never re-runs CV — and matches the package's 1:1 logic/plot architecture. The
plot methods are named `scores` / `compare` (not `run` / `eval`) so the plot class is not a
`Tool`-contract (`run`+`eval`) false positive.

### Reproducibility & determinism

All randomness (fold shuffling, bootstrap resampling) is seeded from `random_state` (constructor
seed, per-call override). Same seed → **byte-identical** `df_eval` / comparison table.

## Consequences

- **New public surface (minor bump):** `ModelEvaluator`, `ModelEvaluatorPlot` in
  `aaanalysis/__init__.py` `__all__`; abbreviations `me` / `me_plot`.
- **No new dependency;** additive; nothing existing changes, so no regression/perf risk to the
  shipped byte-identical paths.
- Scope is deliberately narrow: **not** hyper-parameter search / AutoML, **not** the paper-fidelity
  nested-CV Monte-Carlo engine (#276), **not** the `find_features` selection-leakage fix (#411).
  Those remain separate. `predict_oof` (ADR-0059) is the per-sample out-of-fold counterpart; this is
  the aggregate repeated-CV + CI + comparison harness.

## Alternatives considered

- **Standalone `aa.evaluate` / `aa.compare` helpers** — lowest ceremony but two loose top-level
  functions with no natural plot pair and no shared per-fold state; rejected for architectural
  consistency.
- **Bootstrapping the sample rows instead of the fold scores** — a different (and heavier)
  uncertainty model; the per-fold-score bootstrap matches the downstream `boot_ci()` precedent and
  `comp_bootstrap_ci`'s contract, and is documented as such.
