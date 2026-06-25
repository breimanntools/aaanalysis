# ADR-0046 — `predict_samples` multi-model comparison harness; paper-fidelity training engine deferred

Status: Accepted — 2026-06-25

## Context

`predict_samples` shipped (in the unreleased `aaanalysis.pipe` layer) as a thin wrapper that
rebuilt a feature matrix from one `df_feat` and fit a single `TreeModel`, returning the uniform
`(model, None, df_eval)` triple. Two pressures reshaped it:

1. **The reference ML protocol.** The γ-secretase paper ([Breimann25]) trains and compares many
   model families — random forest, extra trees, xgboost, catboost, LDA, logistic regression, SVM,
   MLP, plus voting and stacking ensembles — over **25 Monte-Carlo rounds** of balanced 80/20
   splits with **nested cross-validation** (inner 5-fold feature-selection + `GridSearchCV`
   hyperparameter tuning, outer hold-out scoring), averaging predicted probabilities across all
   models and rounds into one mean ± std "prediction score". The maintainer wanted that capability
   reflected in the pipeline: *"a wrapper for many different scikit-learn tools … users provide a
   list of models, can also provide different feature sets and labels, all combined to get
   predictors which are returned."*
2. **The golden-pipeline charter.** Golden pipelines are defined as **thin, no own algorithm,
   defaults matching the explicit primitive path** (ADR-0040 / ADR-0041). The paper's full protocol
   is the opposite of thin — a substantial training engine — and its full model set forces heavy,
   install-fragile dependencies (`xgboost`, `catboost`).

A decisive fact removed the migration cost: the **entire `aaanalysis.pipe` layer is unreleased**
(the latest tag/PyPI release `v1.0.3` predates it), so `predict_samples` can be redesigned as a
**clean break** with no deprecation.

## Decision

**D1 — `predict_samples` becomes a thin multi-model × multi-feature-set comparison harness.**
Inputs: `list_df_feat` (a single `df_feat`, a list, or a `{name: df_feat}` dict), `df_seq` +
`labels`, and `models` (scikit-learn estimator instances or model-name strings; default set
`RandomForest`, `ExtraTrees`, `SVM`, `LogisticRegression`). For every `(feature set × model)` cell
it builds `X`, cross-validates (`balanced_accuracy`, `accuracy`, `f1`, `precision`, `recall`,
`roc_auc`), and refits the model on all samples. It returns `(predictors, None, df_eval)`:
`predictors` is a dict keyed `(feature_set, model)`; `df_eval` is the comparison table, one row per
cell, with `<metric>_mean` / `<metric>_std`, `n_features`, `is_shap_ready`, and `is_best`.

**D2 — Clean break, no deprecation.** Because the layer is unreleased, the old
`(df_feat, df_parts, …) → single TreeModel` contract is replaced outright (input switches from
`df_parts` to `df_seq`; the result slot becomes a dict of predictors). No semver impact; the example
notebook and tests are rewritten to the new contract.

**D3 — It stays a thin wrapper (no own algorithm).** No Monte-Carlo rounds, no nested CV, no
`GridSearchCV`, no ensembles inside this function — only a per-cell `cross_validate` + refit over the
estimators the user brings. Core scikit-learn only; **no new dependencies**. `random_state` is
injected into each estimator only where it exposes that parameter and the user left it unset.

**D4 — The paper's full training engine is deferred to a tracked issue.** The 25-round /
nested-CV / `GridSearchCV` / voting+stacking / averaged-probability-score protocol (with
`xgboost` / `catboost` behind a new extra) becomes a future **core** primitive that `predict_samples`
may later wrap behind an opt-in flag — it is not baked into the thin wrapper.

**D5 — SHAP continuity is surfaced, not assumed.** `df_eval` carries `is_shap_ready` (the refit
predictor exposes `feature_importances_`), so users know which predictors can feed `explain_features`
(tree-based only). Arbitrary scikit-learn estimators are not SHAP-explainable through that path.

## Rejected alternatives

- **Implement the full paper protocol inside `predict_samples` now.** Violates the thin-wrapper
  charter (ADR-0040/0041), forces heavy/fragile dependencies into a core pipeline, and bloats one
  function with a research-grade engine. Kept as the deferred core primitive (D4).
- **Keep the `TreeModel`-only contract.** Cannot compare the non-tree families (SVM, linear, MLP)
  the reference protocol relies on; the maintainer explicitly wanted "many scikit-learn tools".
- **Return only the single best predictor** (mirroring `find_features`' single winner). The
  maintainer wants every trained predictor returned for comparison, so the result is the full dict.
- **Estimator classes (`Type`) instead of instances.** Instances let users set hyperparameters and
  are the scikit-learn-idiomatic input; name strings remain as a convenience shortcut.

## Consequences

- Breaking change to `predict_samples`, absorbed cleanly because the layer is unreleased.
- `df_eval` is a new grid-shaped comparison table; `predictors` is a `(feature_set, model)`-keyed
  dict. The example notebook and the unit test suite were rewritten.
- `explain_features` remains the SHAP step; only `is_shap_ready` predictors can feed it.
- A follow-up issue tracks the paper-fidelity training engine (D4) as future work.

## Out of scope

- The full paper training engine (deferred — see D4).
- Multiclass / regression targets — the harness is binary (test vs reference).
- A comparison figure in the `figs` slot — reserved (currently always `None`).
