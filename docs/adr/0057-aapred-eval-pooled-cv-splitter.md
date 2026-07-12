# ADR-0057 — `AAPred.eval` accepts a custom CV splitter and scores it by pooled out-of-fold prediction

Status: Accepted — 2026-07-12

## Context

`AAPred.eval` hardcoded `StratifiedKFold(n_splits=n_cv)` in the backend and capped the integer
`n_cv` at the smallest class count (`check_n_cv`). On a small, imbalanced benchmark — e.g. the
γ-secretase substrate set (63 substrates vs. 14 known non-substrates) — this made `LeaveOneOut`
cross-validation impossible through the shipped API, so every such evaluation dropped back to
hand-rolled sklearn:

```python
balanced_accuracy_score(y, cross_val_predict(SVC(kernel="linear"), X, y, cv=LeaveOneOut()))
```

There is a second, subtler gap. The existing `cv` principle scores **per fold, then averages**
(`cross_val_score`). Under `LeaveOneOut` each test fold holds a single sample, so the per-fold
`balanced_accuracy` is degenerate (0 or 1) and its average does **not** equal the study's metric,
which pools every held-out prediction and scores **once**. Supporting `LeaveOneOut` correctly
therefore means adding both a custom-splitter argument *and* a pooled-prediction scoring principle
— averaging per-fold scores over singleton folds would be silently wrong.

## Decision

**D1 — Add a `cv=` splitter parameter to `AAPred.eval`; do not overload `n_cv`.** `cv` accepts any
scikit-learn cross-validation splitter (validated to expose a callable `split`). It is optional and
keyword-shaped; `n_cv` keeps its exact integer meaning and is ignored when `cv` is given. Overloading
`n_cv` to accept either an int or a splitter was rejected: the scoring principle then depends on the
argument's runtime type, which is surprising, and the type hint degrades.

**D2 — A splitter is scored by a new `cv_pooled` principle.** `eval` runs `cross_val_predict` once
over the splitter and applies each metric a single time on the pooled held-out predictions,
reproducing `metric(labels, cross_val_predict(estimator, X, labels, cv=cv))`. The rows are tagged
`cv_pooled` in the `principle` column (distinct from the per-fold `cv`), and `score_std` is `NaN`
because a pooled score is one estimate, not a fold distribution — the same convention already used
for `holdout`. The `cv` splitter path **replaces** the per-fold `cv` rows (it does not add a second
block); the two principles never coexist in one call.

**D3 — A splitter bypasses the smallest-class-count cap.** `check_n_cv` still caps the integer path
(a k > smallest-class-count `StratifiedKFold` cannot be built), but a passed splitter defines its own
folds and is trusted (validated only for a `split` method). This is exactly what unblocks
`LeaveOneOut` on an imbalanced set.

**D4 — The default (`cv=None`) output is byte-identical.** The per-metric row order (cv/pooled row,
then the optional holdout row) is preserved, and the `cv=None` branch calls the unchanged
`StratifiedKFold` + `cross_val_score` path. `baseline=` composition matrices are scored with the same
splitter for a coherent comparison.

**D5 — Metric dispatch for the pooled path.** Five metrics (`accuracy`, `balanced_accuracy`,
`precision`, `recall`, `f1`) score hard class labels from `cross_val_predict(method="predict")`;
`roc_auc` scores the positive-class probability from a second `cross_val_predict(method="predict_proba")`
(the positive class is the greater label, the last column of the class-sorted output). Estimators
already guarantee `predict_proba` at construction, so the proba path is always available.

## Rejected alternatives

- **Overload `n_cv` to accept a splitter** (D1): type-dependent scoring semantics, worse type hint.
- **Emit both `cv` and `cv_pooled` rows when `cv` is passed** (D2): doubles the table for no
  information — the user chose the pooled principle by passing a splitter.
- **Keep per-fold averaging for a passed splitter** (D2): silently wrong for `LeaveOneOut`, the
  motivating case — a singleton test fold makes per-fold `balanced_accuracy` degenerate.
- **Relax the class-count cap for the integer path too** (D3): a `StratifiedKFold` with
  k > smallest-class-count genuinely cannot be constructed; the cap is correct there.

## Consequences

`AAPred.eval` now absorbs the small-n `LeaveOneOut` boilerplate the prediction layer was meant to
replace (the γ-secretase Appendix-3 `balanced_acc` helper). `df_eval` gains a third possible
`principle` value (`cv_pooled`); `AAPredPlot.eval` already enumerates the distinct principles
dynamically, so it renders the new value without change. The change is additive and backward
compatible; the default path is regression-guarded byte-identical, and the pooled `LeaveOneOut`
score is golden-tested to equal the sklearn reference within `1e-9`.
