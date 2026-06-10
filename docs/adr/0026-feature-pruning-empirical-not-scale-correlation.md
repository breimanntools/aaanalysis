# ADR-0026 — Feature pruning is empirical (sample-level), df_feat-in/out methods on `SequenceFeature`

Status: Accepted — 2026-06-11

## Context

Issue #32 (narrowed scope) asks for **model-free variance + correlation
filtering** of a CPP feature table — a reduction step that runs on a fitted
`df_feat`, returns a row-filtered `df_feat`, and composes *before* the
model-based `TreeModel.select_features` (ADR-0023). The model/importance-based
half is already shipped and must not be reimplemented here.

Two facts complicated the design:

1. **`CPP.run` already redundancy-reduces internally.** The backend
   `_backend/cpp/_filters/_redundancy_filter.py::filtering` walks candidate
   features in descending `abs_auc` and drops a feature when its **scale-vector
   correlation** (`df_scales.corr()`, the 20-dim AA scales) exceeds `max_cor`
   *and* its positions overlap an already-kept feature (`max_overlap`), within
   category. "Drop correlated, keep higher `abs_auc`" therefore already exists —
   but on *scale vectors + position overlap*, computed once from the scales,
   independent of any dataset.
2. **The reusable primitive correlates something else.** The helper the issue
   points at — `NumericalFeature.filter_correlation(X, max_cor)` — correlates the
   **empirical feature matrix** `X` (realized feature values across the user's
   actual proteins). Two features built from different scales/parts can be
   near-perfectly correlated on a *specific* dataset while their scales are not.

So the question was whether #32's correlation step is just a re-export of CPP's
in-run filter, or a genuinely different surface — and where the whole thing
should live.

## Decision

**D1 — Pruning is empirical (sample-level), not a re-export of CPP's redundancy
filter.** `prune_by_correlation` filters on the **empirical** correlation of `X`
over the user's samples (reusing `filter_correlation`), with no position-overlap
or category checks. This catches dataset-specific redundancy CPP's scale-vector
filter cannot, making the two **complementary**, not duplicative. The glossary
coins **feature pruning** and separates it from CPP **redundancy reduction**
(scale-corr + overlap), **feature selection** (model-based), CPP **feature
filtering** (in-run split/scale screening), and **feature simplification**.

**D2 — Two dedicated df_feat-in/df_feat-out methods on `SequenceFeature`.**
`prune_by_variance(df_feat, df_parts, threshold=0.0)` and
`prune_by_correlation(df_feat, df_parts, max_cor=0.7)` — not one
strategy-dispatched method. They are a *pipeline* (variance → correlation →
`select_features`), independently testable, and chain to compose. They live on
`SequenceFeature` because it is the only class that turns `df_feat` + `df_parts`
into `X` (via `feature_matrix`); the low-level X→mask work stays on
`NumericalFeature.filter_correlation` (existing) and a private
`filter_variance_` backend helper. Both are additive methods on an
already-exported class — no new public symbol, outside the CONFIRM-FIRST API
surface.

**D3 — Deterministic `abs_auc` tie-break; variance precedes correlation.**
`prune_by_correlation` sorts `df_feat` by `[abs_auc, abs_mean_dif]` descending
before pruning, so the stronger feature of a correlated pair is kept and the
output is byte-identical across runs (mirroring CPP's redundancy ordering).
Because `np.corrcoef` is undefined (NaN) on constant columns, constant features
are detected by zero peak-to-peak range and always retained by
`prune_by_correlation`; variance pruning is the step that removes them, which is
why the documented order is variance → correlation.

**D4 — Scale path by default, with an optional `X` passthrough.** Both methods
build `X` from `df_parts` (the scale `CPP.run` path) by default but accept a
pre-computed `X`, which covers `run_num` feature tables and lets a caller build
the matrix once and reuse it. No numerical-mode feature-matrix machinery is
added.

## Rejected alternatives

- **Re-expose CPP's in-run redundancy filter as a post-hoc method.** Rejected:
  it would repackage shipped logic and correlate scale vectors, not the user's
  data — the empirical filter is the part that is actually missing.
- **One strategy-dispatched method** (`prune(df_feat, strategy, param)`, à la
  `select_features`). Rejected: variance and correlation are a sequential
  pipeline, not mutually exclusive strategies; dispatch forces awkward disabling
  to isolate one filter and muddies the "feature selection" term reserved for
  TreeModel.
- **Standalone `aa.*` helpers.** Rejected: each new top-level symbol is a
  CONFIRM-FIRST `__init__.py` re-export for no discoverability gain over methods
  next to `feature_matrix` / `get_df_feat`.
- **A public `NumericalFeature.filter_variance` primitive** mirroring
  `filter_correlation`. Rejected: variance is a one-line `np.var` comparison not
  worth public surface; it stays a backend helper.
- **Plain `np.var > threshold` for constant detection.** Rejected: a constant
  column whose value is not exactly representable yields a tiny float epsilon,
  not `0.0`, so a `threshold=0` would keep it. Constant columns are snapped to
  zero variance by zero peak-to-peak range so `threshold=0` removes *exactly*
  the constant features (the issue KPI).

## Consequences

- New backend helper `_backend/num_feat/filter_variance.py::filter_variance_`;
  two new `SequenceFeature` methods; two example notebooks under
  `examples/feature_engineering/`.
- `CONTEXT.md` gains a **feature pruning** / **variance pruning** / **empirical
  correlation pruning** vocabulary block plus a pipeline relationship bullet.
- The pre-existing CPP in-run redundancy filter is unchanged; the two correlation
  surfaces now coexist by design and are documented as complementary.
