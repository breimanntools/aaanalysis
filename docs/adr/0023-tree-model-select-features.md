# ADR-0023 — `TreeModel.select_features` is a post-fit method, not a new selector class

Status: Accepted — 2026-06-06

## Context

`TreeModel` (in `explainable_ai/`) is documented as a tool for *Monte Carlo
estimates of feature importance and predictions*. Feature selection was present
but buried: `fit(use_rfe=False)` exposed Recursive Feature Elimination as a
boolean flag, with no discoverable, named selection surface. The RFE path
(`_backend/tree_model/tree_model_fit.py`) also hardcodes a plain
`RandomForestClassifier` and ignores the configured `list_model_classes`, so the
only "selection" the class offered was hard to find and not actually driven by
the user's models.

The package has rich vocabulary for windows, scoring, embeddings, structure, and
annotation, but **no feature-selection vocabulary at all** — the concept did not
exist as a named thing, only as a flag. The question was whether to introduce a
first-class multi-strategy tree-based feature selector, and in what shape.

A key constraint shaped the answer: scikit-learn already owns the generic
feature-selection zoo (`VarianceThreshold`, univariate `SelectKBest`, L1/Lasso
selection, generic `SequentialFeatureSelector`). AAanalysis should expose only
the selection that is *native to what `TreeModel` uniquely produces* — the Monte
Carlo `feat_importance` / `feat_importance_std` and the per-round `is_selected_`
masks — not re-wrap sklearn.

## Decision

**D1 — Selection is a method on `TreeModel`, not a new class or subpackage.**
A single `select_features(df_feat, strategy, param) -> df_feat` method. No new
public symbol is added to `aaanalysis/__init__.py` (`TreeModel` is already
exported), so this is additive and outside the CONFIRM-FIRST API surface.

**D2 — Three tree-native strategies, dispatched by a `STRATEGY_*` constant.**
`top_k` (keep the `param` highest-`feat_importance` features), `threshold` (keep
features with `feat_importance ≥ param`), and `frequency` (keep features chosen
in `≥ param` fraction of the per-round `is_selected_` masks). Constants and a
`LIST_*_STRATEGIES` live in `utils.py`. Strategies that sklearn already provides
generically are deliberately excluded.

**D3 — RFE stays the `fit`-time engine and is *not* a `select_features`
strategy.** RFE is an iterative re-fit loop; it cannot be "read off" a fitted
model the way the three strategies can. It remains `fit(use_rfe=True)`, where it
populates `is_selected_`. The `frequency` strategy is how a user *consumes* RFE's
per-round output. The `select_features` docstring cross-references `fit` for
recursive elimination. `use_rfe` is untouched — no deprecation cycle.

**D4 — `select_features` is post-fit and df_feat-in / df_feat-out.** It reads
`self.feat_importance` (top_k / threshold) or `self.is_selected_` (frequency),
requires `fit` first (`ValueError` if unfitted), and returns a row-filtered
`df_feat` aligned to feature order — the same alignment contract as
`add_feat_importance`. The boolean mask itself is not returned; it already lives
on `fit`'s `is_selected_`. `frequency` without `fit(use_rfe=True)` selects every
feature (masks are uniformly `True`), so it emits a `RuntimeWarning` rather than
silently no-op'ing.

**D5 — One polymorphic `param` knob per strategy.** `param: Union[int, float,
dict]` — a numeric scalar whose admissible type is fixed by the strategy
(`top_k`→int, `threshold`→float, `frequency`→float in (0,1]); a `dict` is
reserved as forward-capacity for a future multi-knob strategy. This mirrors
`sample_synthetic`'s polymorphic generation-strategy parameter. String thresholds
(`"mean"` / `"median"`, sklearn's `SelectFromModel` idiom) are **not** accepted —
numeric only.

## Rejected alternatives

- **A separate `TreeFeatureSelector` class (or a `feature_selection/`
  subpackage).** Cleanest responsibility split on paper, but it duplicates the
  tree-fitting machinery, adds a new public class (CONFIRM-FIRST), and the
  selection logic is a thin read off signals `TreeModel` already holds. Rejected:
  not enough substance to justify a new class; the method keeps the importance
  signal and its consumer co-located.
- **Move RFE out of `fit` into `select_features` (deprecate `use_rfe`).**
  Conceptually tidiest — `fit` would only fit + compute importance, all selection
  in one method. Rejected: `use_rfe` is public on a semver-strict v1.x API, so
  this needs a deprecation cycle and changes the meaning of `fit`'s
  `is_selected_`. Not worth it to relocate a working engine.
- **A "simple RFE" strategy inside `select_features`.** Considered, but an RFE
  without the iterative re-fit is mathematically identical to `top_k` — it would
  just rank the already-computed importance and cut. Naming that "RFE" would be
  misleading. Rejected: `top_k` already covers it honestly; real RFE keeps its
  re-fit loop in `fit`.
- **Return a boolean mask (or mask + optional df_feat).** A mask composes
  directly with `eval(list_is_selected=...)`. Rejected in favor of df_feat-in /
  df_feat-out for consistency with `add_feat_importance` and the df_feat-centric
  workflow; the mask is still available as `fit`'s `is_selected_`.
- **A wide signature of explicit per-strategy params (`n`, `threshold`,
  `min_freq`).** House style elsewhere, but it widens with every new strategy and
  most params are unused per call. Rejected for the single polymorphic `param`,
  which has precedent in `sample_synthetic`.
- **String thresholds (`"mean"` / `"median"`).** The familiar `SelectFromModel`
  idiom. Rejected to keep `param` validation numeric-or-dict and avoid string
  magic.

## Consequences

- A new "Feature selection vocabulary" section in `CONTEXT.md` coins the
  previously-absent terms (**feature selection**, **selection strategy**), with
  *Avoid* notes separating them from CPP feature *engineering*, `output_mode`,
  and `is_preselected`.
- The pre-existing RFE wart (hardcoded `RandomForestClassifier` ignoring
  `list_model_classes`) is **not** addressed here — it is orthogonal to the new
  surface and left for a separate decision.
- New backend module `_backend/tree_model/tree_model_select.py`
  (`select_top_k_` / `select_by_threshold_` / `select_by_frequency_`) returning
  masks the frontend applies to `df_feat`; new `examples/tm_select_features.rst`.
