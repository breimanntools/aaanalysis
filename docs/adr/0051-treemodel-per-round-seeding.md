# ADR-0051 — TreeModel per-round seeding (fixed seed → independent rounds)

Status: Accepted — 2026-07-06

## Context

`fit_tree_based_models` runs `n_rounds` of RFE + importance fits and averages the per-round
feature importances. It passed a **constant** `random_state` to both the RFE
`RandomForestClassifier` and the importance-model kwargs, so under a fixed seed every round fit
identical estimators: `feat_importance_std` (and `predict_proba`'s `pred_std`) collapsed to exactly
`0` and rounds 2..N were wasted. This hit the encouraged reproducibility path (a fixed
`random_state`, or the global `options["random_state"]`) and contradicted the documented "average
across training rounds enhances robustness" claim. `ShapModel` already reseeds per round
(`random_state + round`).

Part of #343 (defect #2). Under the `None` default the rounds already varied (fresh global entropy
each fit), so only the fixed-seed path was degenerate.

## Decision

Reseed per round: round `i` uses `random_state + i` for the RFE `RandomForestClassifier` and for
each importance model that carries a `random_state` (added by `check_model_kwargs` when the
estimator supports it), via a local `_seed_model_kwargs` mirroring `ShapModel`'s. **No-op when
`random_state is None`** (the `None` default path is byte-identical to before).

**ADR-0032 tier:** this changes fixed-seed importance output (mean + std), so it is output-affecting
— but the prior output was a degenerate `std=0` with wasted rounds, not a meaningful result worth
preserving, and the `None` default is unchanged. Applied **directly** (no legacy/opt-in), per the
maintainer decision on #343 (unlike defect #1, there is no reproducible-but-useful behavior to keep).

## Regolden / anchors

- **No existing regression anchor pinned TreeModel importances** (the unit + integration suites
  assert structure, not exact values), so nothing needed regolding; those suites stay green.
- **New guard:** platform-independent structural invariants (`test_tm_per_round_seed.py`) — non-zero
  std under a fixed seed, same-seed reproducibility, single-round `std=0`. A frozen exact-value hash
  was **rejected**: RandomForest Gini importances are not bit-identical across the CI matrix (BLAS /
  sklearn build), so an exact anchor would be flaky; the invariants encode exactly what the fix
  guarantees and gate on every runner (not just the nightly).
- **No example/notebook change:** `examples/explainable_ai/tm.ipynb` exercises `TreeModel.eval`
  (a separate CV path, `eval_feature_selections`), not the fixed `fit` importance path, so its
  output is unaffected; the fix adds no public parameter.

## Consequences

- Fixed-seed TreeModel importances change once (degenerate → real Monte-Carlo mean + non-zero std);
  reproducibility within a seed is preserved. The `None`-default output is unchanged.
- `ShapModel` is unchanged (already correct); `TreeModel` now matches its per-round reseeding.
- #343 defect #3 (BH p-value monotonicity) remains open.
