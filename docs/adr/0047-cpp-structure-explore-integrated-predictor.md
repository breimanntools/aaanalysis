# ADR-0047 — `CPPStructurePlot.explore`: integrated per-site predictor with `output=` dispatch

Status: Accepted — 2026-06-28

## Context

`CPPStructurePlot` (pro, `aaanalysis/feature_engineering_pro/`) already renders
per-residue CPP / CPP-SHAP impact onto a 3D structure through three methods:
`map_structure` (a `StructureView`), `plot_combined` (a `CombinedView`,
structure beside the feature map), `plot_linked` (a `LinkedView`, the
self-contained hover-linked HTML), and `interactive` (an ipywidgets explorer).

All four are *render* surfaces: the caller must already hold a `df_feat` whose
`col_imp` column carries the signed per-feature impact. `interactive` goes one
step further — it takes a user **`predictor`** callable `(sequence, p1) ->
df_feat` and re-runs it per site — but the user still has to *write* that
callable, wiring `CPP` + `ShapModel` by hand. The deployed cleavage app does the
whole loop for the user: given training data + a model it predicts per site,
explains the prediction, and paints it, with the feature map and the structure
linked.

Epic #288 asks for that integrated loop in the package, behind one entry point,
with a user-selectable output type (`widget` / `html` / `static`). Building it
forced three decisions that a future reader would otherwise find surprising.

## Decision

Add one method, **`CPPStructurePlot.explore(df_feat, df_seq, labels, sequence,
model="rf", *, output="widget", ...)`**, that builds a built-in predictor from
the data and dispatches to the existing render methods by `output=`. Three
choices underpin it:

1. **A Plot method fits models.** `explore` (and the built-in predictor behind
   it) fits a prediction estimator and a `ShapModel`, even though the house rule
   is "Plot classes visualize; logic classes fit." We accept this *local*
   exception because `CPPStructurePlot` is already an orchestrator (it drives
   `CPPPlot.feature_map`, structure parsing, and AlphaFold fetches) and the
   epic's value is precisely the one-call loop. The fitting logic lives in
   `_backend/cpp_struct/`; the frontend method only validates and delegates, so
   the frontend-trusts-backend split is preserved. The custom `predictor=`
   callback is kept as the escape hatch — pass it and `df_seq`/`labels`/`model`
   are ignored.

2. **Per-site SHAP impact is a refit, not a reuse.** The shipped `ShapModel`
   has **no out-of-sample explainer**: `fit(X, labels)` computes SHAP values for
   the *training* matrix and stores `self.shap_values`; `add_feat_impact` only
   reads stored rows. The documented way to explain a newly-predicted sample is
   to add it with a soft label equal to its predicted probability and refit with
   `fuzzy_labeling=True`. So the built-in predictor refits per site, using
   `fuzzy_aggregation="interpolate", n_rounds=1` — the exact two-fit estimate —
   on the small (N × fixed-feature) matrix. This mirrors `SeqOpt`'s
   `mode="impact"`, which refits every generation. The **predicted probability**
   itself *is* fit-once (one estimator fit on the training matrix, `predict_proba`
   per site is instant); only the impact refits. "Fast per-site" therefore means
   a small-matrix refit, documented honestly — not a persistent-explainer reuse,
   which `ShapModel` does not offer.

3. **`model=` drives prediction; `ShapModel` keeps its defaults.** `model=`
   accepts a name (`"rf"` / `"svm"` / `"log_reg"`, the `ut.MODEL_*` constants),
   a scikit-learn estimator, or a list — the same `Union[str, BaseEstimator,
   List]` convention as `aap.find_features`. It selects the estimator used for
   the per-site predicted probability / soft label. The `ShapModel` that
   computes the impact is constructed with **its own defaults** (TreeExplainer +
   RandomForest/ExtraTrees). Consequence: the prediction model and the
   explanation model can differ (e.g. predict with SVM, explain with the default
   tree ensemble); this is stated in the docstring.

The `output=` selector dispatches over the same built-in inputs and the same
per-residue impact so all three outputs agree:

- `output="widget"` → `interactive` (live per-site refit; needs a kernel +
  ipywidgets).
- `output="html"` → `plot_linked` (self-contained file at `path`).
- `output="static"` → `plot_combined` (a `CombinedView`).

Because `html` / `static` have no live kernel, the built-in predictor runs once
for the chosen target (a residue-site window, a TMD+JMD domain, or the whole
protein — selected through the same `df_seq` target conventions as
`SequenceFeature.get_df_parts`) and bakes that result in. Live multi-site
re-prediction inside the HTML stays out of scope here (Stage C of the epic).

## Rejected alternatives

- **Add an out-of-sample explainer to `ShapModel`** (persist the fitted
  estimator + explainer, expose `explain(X_new)`) for a true "fit once."
  Rejected for this issue: it is a new public surface on a pro class, only some
  SHAP explainers support it cleanly, and it complicates `ShapModel`'s
  averaged-over-rounds-and-models design. The per-site refit on a small matrix
  is fast enough and uses only shipped API. Revisit if per-site latency proves
  unacceptable.
- **A standalone builder returning the `(sequence, p1) -> df_feat` callable**,
  keeping `interactive` unchanged and the user wiring it. Cleaner plot/logic
  separation, but it does not satisfy the epic's "one entry point does the whole
  loop" — the user still composes the call. The escape-hatch `predictor=` keeps
  that compositional path available for power users.
- **`output=` on `interactive` itself.** Rejected: `interactive` returning a
  static figure when `output="static"` is a naming mismatch and varies the
  return type by argument. A dedicated `explore` keeps each lower-level method's
  name honest about its return type.
- **A string-name registry on `TreeModel` / `ShapModel`.** Rejected as
  cross-cutting scope creep: the name shorthand lives in the `pipe` convenience
  layer; the core explainable classes stay class-only. `explore` reuses the
  `pipe` convention rather than spreading it.

## Consequences

- One new public method on a pro class; no change to the four existing render
  methods or to `ShapModel` / `TreeModel`.
- The per-site refit cost scales with the training-set size and the chosen
  model; tree models (the default) stay fast, non-tree models force
  `KernelExplainer` and are slower per site (documented, not warned at runtime).
- The integrated `df_feat` for a site is test-anchored to equal a manual
  feature-matrix + `ShapModel.add_feat_impact` computation for that window.
