# ADR-0059 — AAPred.predict_oof: cross-validated out-of-fold scores for the training set

Status: Accepted — 2026-07-12

## Context

`AAPred.predict` deploys models fit on **all** the training data (`fit_models` →
`.fit(X, labels)`, then `predict_proba_models`) and scores samples with them. Scoring the
*training* proteins that way is in-sample and optimistically biased. To get honest per-protein
scores for the training set, the γ-secretase Use Case hand-rolled sklearn twice (Appendix-3
substratome ranking and the SHAP prediction context):

```python
P = np.vstack([cross_val_predict(m, X, y, cv=cv5, method="predict_proba")[:, 1] for m in ensemble])
score, std = P.mean(0), P.std(0)          # mean over the ensemble, std across models
```

The gap is only half-open: `predict` already returns `score` and `score_std` (std across the
ensemble) for *new* proteins. What was missing is the **out-of-fold** (cross-validated) score for
the *training* set — the per-sample counterpart of `eval`'s aggregate cross-validated metrics (#398,
surfaced while reducing the Appendix-3 boilerplate of epic #305). This is the minimal out-of-fold
primitive, **not** the paper-fidelity nested-CV engine (#276) or bootstrap CIs (#91).

## Decision

Add an **additive** method `AAPred.predict_oof(X, labels, label_pos=1, n_cv=5)` returning a
per-sample `df_pred` with columns `score` (mean positive-class out-of-fold probability over the
ensemble) and `score_std` (std across models; `0` for a single model) — the same aggregation shape
as `predict_proba_models`, so the output matches `predict`.

- **A dedicated method, not `predict(cv=...)` or an `oof_scores_` attribute.** OOF operates on the
  *training* matrix `X` + `labels`, exactly like `fit` and `eval` — not on the raw `df_seq` +
  bound `df_feat` that the deployment `predict` takes. Threading a `cv=` branch (and `labels`) into
  `predict` would muddy the deployment path; an `oof_scores_` attribute would either add CV cost to
  every `fit` (breaking the byte-identical-`fit` requirement) or fail to score a freshly passed `X`.
  `predict_oof` is the per-sample twin of `eval` (aggregate CV metrics → per-sample CV scores).
- **No prior `fit` required.** Like `eval`, it clones the constructor's estimators and
  cross-validates them itself; it never reads or writes the deployment models in `list_models_`, so
  `fit`'s output stays byte-identical and the two scoring paths stay cleanly separate.
- **Backend `predict_proba_oof`** runs, per estimator, `cross_val_predict(clone(est), X, labels,
  cv=StratifiedKFold(n_cv, shuffle=True, random_state), method="predict_proba")`, selects the
  positive-class column by `label_pos`'s index in the sorted class order (what `cross_val_predict`
  returns; for `{0, 1}` and `label_pos=1` this is `[:, 1]`), stacks across models, and reduces with
  `mean(0)` / `std(0)` — reproducing the hand-rolled block **within 1e-9** (regression-tested).
- **Deterministic under `random_state`** via the `StratifiedKFold(shuffle=True, random_state=...)`
  seed contract, threaded from the constructor exactly as `eval` does (no per-call `seed`, for
  consistency with `eval`).

**Rejected — `predict(..., cv=...)` overload.** Rejected because `predict` is the deployment path
(new proteins, `df_seq` + `df_feat`, prior `fit`), and folding a training-set cross-validation mode
into it conflates deployment with evaluation and forces an `X`/`df_seq` branch through the method.

**Rejected — `oof_scores_` fitted attribute.** Rejected because it couples `fit` to cross-validation
(cost on every fit, or lazy state) and cannot score a freshly supplied `X`; a method keeps `fit`
byte-identical and the API explicit.

**ADR-0032 tier:** purely additive. `fit` / `predict` / `eval` outputs and `list_models_` are
unchanged; the new method is orthogonal. No new required dependency (uses the already-present
scikit-learn `cross_val_predict` + `StratifiedKFold`).

## Consequences

- Honest, in-sample-free per-protein scores for the training set are now a single library-native
  call; the γ-secretase Appendix-3 ensemble block and the SHAP proba block reduce to one
  `predict_oof` each (the notebook swap rides the Appendix-3 rewrite, not this PR).
- `predict_oof` and `eval` now share the same stratified-CV seed contract, so a user can read the
  aggregate `eval` metric and the per-sample OOF scores as two views of the same cross-validation.
- Advances the minimal out-of-fold primitive underneath the deferred paper-fidelity training
  engine (#276) and repeated-CV / CIs (#91), without pre-committing their shapes.
