# ADR-0041 — `aaanalysis.pipe` pipeline conventions and the core golden pipelines

Status: Accepted — 2026-06-24

Extends [ADR-0040](0040-golden-pipelines-convenience-api.md) (the `aaanalysis.pipe`
naming/tier/exposure conventions). It **refines ADR-0040 D3** (the return contract) and
fixes the cross-cutting conventions plus the catalogue of core pipelines, settled in a
grill-with-docs session.

## Context

ADR-0040 established `aaanalysis.pipe` (`aap`) as the stateless convenience facade and
set the naming (`verb_noun`), the Ends/Means tiering, and the documentation exposure. As
the first Ends were designed, three things sharpened that ADR-0040 left open or set
provisionally: the **return shape** (D3 had a `(primary, secondary)` 2-tuple, too narrow
once a pipeline yields a result *and* an evaluation *and* one or more figures), how the
**figures** are surfaced, and which pipelines form the **core workflow** users should
reach for first.

## Decision

**D1. Uniform return triple `(results, figs, evals)`.** Every End returns a 3-tuple —
the primary result, the figure(s), then the evaluation(s) — superseding ADR-0040's
2-tuple. Each slot is a **bare object or `None`** (no dicts, no nested tuples), so it
unpacks plainly:

    df_feat, ax, df_eval = aap.find_features(df_seq, labels)
    model,   _,  df_eval = aap.predict_labels(df_feat, df_parts, labels)

A pipeline that produces several figures still returns the **primary** one in `figs`
(e.g. the feature map) and creates the rest as ordinary pyplot figures, so a single
`plt.show()` renders **all** of them. `figs=None` when `plot=False`; `evals=None` when a
pipeline does no evaluation. This keeps the call site trivial while letting a pipeline
emit auxiliary plots.

**D2. The four core pipelines (the workflow spine).** These are the headline Ends, in
the order a user walks them — and each is the executable spine of exactly one protocol
(D5):

- **`obtain_samples`** — build the training set. The parameters *describe the user's
  sampling situation* (positives, candidate pool, role/strategy); it returns the optimal
  sample set plus a small held-out validation. Wraps `AAWindowSampler`; the
  reliable-negatives (PU) path requires a `df_feat`. *Sampling is a first-class first
  pipeline, not an afterthought.*
- **`find_features`** — identify the determinant features (and draw the feature map). A
  CPP **AutoML** pipeline (D3). Wraps `CPPGrid` + `CPP` + `CPPPlot` + `TreeModel`.
- **`predict_labels`** — train and cross-validate a predictor from the features. Wraps
  `SequenceFeature.feature_matrix` + `TreeModel`.
- **`explain_features`** — per-residue SHAP explanation + SHAP-coloured map (*pro*).
  Wraps `ShapModel`.

Further Ends (`design_mutations`, `evaluate_models`) and the Means (reliable negatives,
scale reduction, embeddings — flags on an End by default, per ADR-0040 D2) follow the
same conventions but are not part of the four-step spine.

**D3. `find_features` is a CPPGrid + model-CV AutoML pipeline.** It does not run a single
CPP; it sweeps configurations and selects the best by cross-validated model performance.
Three grades — **`fast`** (single default run), **`balanced`** (sweep `n_filter`),
**`exhaustive`** (sweep `list_parts` × `n_filter`) — name profiles of the underlying
levers (scale breadth `n_explain`, splits `n_split_max`, `n_filter`, `simplify`
strategy). A bounded `kws` dict overrides any single lever (unknown keys raise). Selection
uses a model (`svm` default; `rf`/`logreg`) with `cv=5` and a metric (`balanced_accuracy`
default) — all parameters — because `CPPGrid.eval`'s `avg_ABS_AUC` is monotone in
`n_filter` and cannot pick it. `CPPGrid`'s built-in smart slice computes CPP once at the
maximum `n_filter` and exactly `head(n)`-slices the rest, so the sweep is cheap; the cost
is one CV model per config.

**D4. The evaluation-grid plot adapts to the sweep's dimensionality.** `find_features`
emits a `viridis` evaluation plot of the per-config CV scores whose form follows the
number of swept axes: **1-D** (e.g. `balanced`'s `n_filter`) → a line/bar; **2-D** (e.g.
`exhaustive`'s parts × `n_filter`) → a single heatmap; **3–4-D** → **faceted
small-multiples** (one heatmap panel per level of the extra axes). This is the honest
answer to "a heatmap is 2-D but the sweep can be 4-D": never collapse silently — facet.

**D5. One pipeline ↔ one protocol.** Each golden pipeline is the executable spine of
exactly one narrative protocol under `protocols/protocol<N>_*.ipynb`, and each protocol
is anchored by one pipeline. This keeps the convenience layer and the documented
workflows in lockstep.

## Consequences

- The call site is uniform and trivial across the whole namespace; `plt.show()` "just
  works" for every pipeline, including auxiliary plots.
- `predict_labels` (the renamed, experimental `predict`) returns `(model, None,
  df_eval)`; existing experimental callers update to the triple — acceptable while the
  namespace is experimental and out of `__all__` (ADR-0040 D5).
- `find_features` carries real compute (N CV models per call on the higher grades); the
  smart `n_filter` slice keeps the *feature* cost to one CPP run, and the grade names set
  expectations.
- A new public plotting concern (the eval grid) lives inside `aap`, not as a `*Plot`
  class, so it is outside the ADR-0039 plot-method return contract (it returns via the
  pipeline's `figs` slot).

## Out of scope

- Per-pipeline parameter lists and the `obtain_samples` "situation" parameters / the
  eval-grid layout details — pipeline-local, fixed in each implementation PR.
- Promoting `aaanalysis.pipe` into the top-level `__all__` — still deferred and
  CONFIRM-FIRST (ADR-0040 D5).
- The machine-readable tool contract / MCP layer — ProtXplain's (ADR-0038).
