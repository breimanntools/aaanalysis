# ADR-0063 — SequenceFeatureTransformer: leak-free CPP selection as a scikit-learn transformer

Status: Accepted — 2026-07-16

## Context

`SequenceFeature.feature_matrix` already returns a plain numeric `X` that drops into any scikit-learn
estimator, but feature **selection** (`CPP.run` → `df_feat`) had to be run **once up front**, outside
the cross-validation. There was no scikit-learn-native way to put CPP selection *inside* a
`Pipeline` / `cross_val_score` without leaking the test fold into the selection — the last unbuilt
piece of the `aaanalysis.pipe` ergonomics epic (issue #241; supersedes the closed #24). The existing
sklearn-compat integration test explicitly deferred "leak-free per-fold reselection" to its own issue.

## Decision

Add **`SequenceFeatureTransformer(BaseEstimator, TransformerMixin)`** in `feature_engineering/`,
exposed as a public `aa.*` symbol (abbreviation `sft`). It wraps the existing
`get_df_parts` → `CPP.run` (+ optional `CPP.simplify`) → `feature_matrix` chain across `fit` /
`transform`:

- `fit(X, y)` runs CPP selection on the **training data only** and stores `features_` / `df_feat_`;
- `transform(X)` applies those **same** features → the numeric matrix `X`.

Inside a `Pipeline` / `cross_val_score`, selection re-runs per training fold, so the test fold never
influences which features are chosen. `X` is a `df_seq` (sequence info) or a pre-built `df_parts`.
Selection is binary (one `label_test` vs one `label_ref`).

### Reconciling the scikit-learn contract with the house style

The scikit-learn clone contract requires `__init__` to **only** assign parameters to same-named
attributes (no validation/coercion). The house "Validate block at the top of every public method"
therefore moves out of `__init__` and into `fit` (and `transform`); learned state carries a trailing
underscore. `fit(X, y=None)` keeps the sklearn signature but raises if `y is None` (selection is
supervised — also declared via `__sklearn_tags__().target_tags.required = True`).

### Estimator checks — a curated subset, not full `check_estimator`

The input is a **sequence DataFrame of strings**, not a validated numeric 2-D array, so scikit-learn's
numeric-array fuzzing (`check_array`, NaN/dtype checks) in the full `check_estimator` does not apply.
`__sklearn_tags__` sets `no_validation = True`. The tests run the **contract-relevant** checks
(`check_no_attributes_set_in_init`, `check_parameters_default_constructible`), clone / `get_params`
round-trips, `fit`-returns-self, and — the real point — an **integration** test proving leak-free
behaviour inside `Pipeline` + `cross_val_score`.

### `get_feature_names_out` / `set_output`

`get_feature_names_out` returns the selected CPP feature ids (the output column names), which enables
`set_output(transform="pandas")` for a labelled DataFrame output. Without it, `set_output` is gated
off by scikit-learn.

## Consequences

- **New public surface (minor bump):** `SequenceFeatureTransformer` in `aaanalysis/__init__.py`
  `__all__`; abbreviation `sft`. It is **core** (scikit-learn is already a hard dependency) — no
  `pro` gating.
- **New abbreviation family:** `sft` is the first `*Transformer`; not in any existing type-suffix
  group (`*p` preprocessor / `*_plot`). Recorded in the class-abbreviation registry + style guide.
- Reuses `SequenceFeature` / `CPP` directly (no new dedicated `_backend/`, no cross-class backend
  import); additive, no change to existing behaviour.

## Alternatives considered

- **A functional `aap`-style helper** rather than a transformer — rejected: it would not compose in a
  scikit-learn `Pipeline` / `cross_val_score`, which is the whole point (leak-free selection as a
  pipeline step).
- **Running the full `check_estimator`** via `parametrize_with_checks` — rejected: its numeric-array
  assumptions do not fit a sequence-DataFrame input; the curated subset + the leak-free integration
  test cover the contract that actually matters here.
