# ADR-0024 — `feature_map` gains SHAP support via the `shap_plot` toggle, not issue #63's `stack_by` / `df_imp`

Status: Accepted — 2026-06-06

## Context

Issue #63 ("Stacked bar charts for SHAP-based feature importance in
`plot_feature_map`") asked that the feature-map's cumulative importance bars
reveal *direction* — per-class / per-sample contribution — instead of one flat
gray cumulative bar, so users get SHAP-level granularity inside the existing
figure. Its *literal* proposal was a new `stack_by="class"|"group"` enum plus a
new long-format `df_imp(feature, class, importance)` frame, with a per-class
color palette and an extra class legend.

By the time #63 was implemented, the `CPPPlot` family had converged on a uniform
**`shap_plot`** convention. Three sibling methods — `ranking`, `profile`,
`heatmap` — already expose `shap_plot: bool`, switching between group-level
**feature importance** (`feat_importance`, gray) and sample-level **feature
impact** (`feat_impact_'name'`, signed red/blue), validated by the shared
`check_col_imp` / `check_col_val` helpers and colored by
`ut.COLOR_SHAP_POS` / `ut.COLOR_SHAP_NEG`. `feature_map` was the only member of
the family missing the toggle. The decision was which API to grow.

## Decision

**D1 — Implement #63 by adding `shap_plot` to `feature_map`, mirroring
`profile` / `heatmap`.** Same parameter name, same `check_col_imp` /
`check_col_val(sample_mean_dif=True)` validation, same `feat_impact_'name'` /
`mean_dif_'name'` column contract, same red/blue colors and diverging SHAP
colormap. No new public symbol; additive and outside the CONFIRM-FIRST surface.

**D2 — The bars stack by impact *sign*, not by class.** In SHAP mode the
per-position (top) and per-subcategory (right) bars sum positive impact (red,
stacked up/right) and negative impact (blue, stacked down/left) from a zero
baseline — the horizontal/vertical analogues of `_plot_cpp_shap_profile`. The
sign *is* the decomposition; this is consistent with every other SHAP plot in
the package.

**D3 — Markers stay a pure magnitude channel.** The heatmap `■` overlays encode
`abs(impact)` via the existing `imp_ths` size buckets and stay black; sign is
already carried by the diverging cell color and the bar color. Because feature
importance is always `≥ 0`, feeding `abs(val)` into the marker logic is
byte-identical for the non-SHAP path — no `shap_plot` branch is needed there.

**D4 — `shap_plot=False` is byte-identical to the previous output.** Every SHAP
behavior sits behind the flag; the gray cumulative bars, annotations, and CPP
colormap are unchanged by default, satisfying #63's hard acceptance criterion.

## Rejected alternatives

- **#63's literal `stack_by` enum + long-format `df_imp` frame.** It would bolt
  a *second*, inconsistent SHAP API onto a class that already has one, introduce
  a brand-new DataFrame schema, and force users to reshape `ShapModel`'s
  multi-column output. Rejected: one SHAP convention across `CPPPlot` beats two;
  the existing `feat_impact_'name'` columns already carry the decomposition.
- **Stack by user-defined *class* with a categorical color palette + a class
  legend.** More information per bar in principle, but it conflicts with the
  scale-category `dict_color` legend the feature map already owns, and the
  red/blue sign encoding is what every sibling plot uses. Rejected for
  consistency; class-vs-class comparison is served by rendering one impact
  column per class.
- **Recolor the heatmap markers by sign (red/blue).** Considered for vividness;
  rejected as redundant with the cell color and prone to low contrast on
  saturated cells — magnitude is the one thing the cell color does *not* already
  show.
- **Auto-darken the heatmap background in SHAP mode (as `heatmap` does).**
  Left out to keep `feature_map`'s `facecolor_dark` default behavior unchanged;
  can be revisited if low-impact cells read as too faint.

## Consequences

- New "Explainability (CPP-SHAP) vocabulary" section in `CONTEXT.md` coins
  **feature importance** (unsigned, group-level) vs **feature impact** (signed,
  sample-level) and the **shap_plot** toggle, with *Avoid* notes separating the
  two axes.
- Backend `cpp_plot_feature_map.py` threads `shap_plot` through
  `plot_feature_map` into both bar helpers; the marker helper switches to
  magnitude unconditionally.
- The `examples/cpp_plot_feature_map.ipynb` notebook gains a `shap_plot=True`
  cell; the narrative tutorial may follow in a separate PR.

## Out of scope

- A standalone per-class palette / legend (the rejected #63 shape) — revisit only
  if a concrete multi-class request appears.
- Tutorial-level narrative for CPP-SHAP feature maps (separate PR).
