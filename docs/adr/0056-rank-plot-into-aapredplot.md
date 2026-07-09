# ADR-0056 — Per-protein rank scatter moves into `AAPredPlot.predict_group(kind="rank_scatter")`

Status: Accepted — 2026-07-09

Supersedes: ADR-0006 (per-protein rank plot ships as a standalone `aa.plot_rank`).

## Context

ADR-0006 shipped the per-protein rank scatter (max-score-per-protein, group sets, threshold
lines) as a **standalone** `aa.plot_rank` in `aaanalysis/plotting/_plot_rank.py`. Its two
justifications were: (1) the chart had **no logic-class twin** — nothing in the library produced
the per-protein prediction scores it visualizes; and (2) it paired symmetrically with the
standalone `aa.metrics.comp_*` functions.

Both premises predate the `prediction` subpackage. Since ADR-0006, `AAPred` / `AAPredPlot` landed
as the general **evaluate-and-deploy** path: `AAPred.predict` scores raw sequences at the
whole-protein / domain / window level — it **is** the logic class that produces the per-protein
scores the rank scatter plots. And `AAPredPlot.predict_group` is already the *group-level,
across-samples* prediction-figure dispatcher (`kind` in `hist` / `ranking` / `scatter` / `cutoff`
/ `clustermap`). So the chart now has both a logic-class twin and a natural home; the loose
`aa.plot_rank` is the only chart-drawing function in an otherwise helpers-only `plotting` package
(the anomaly ADR-0006 itself flagged).

A naming hazard: `AAPredPlot.predict_group(kind="ranking")` already exists — the per-candidate
**leaderboard bars** (a different figure: per-sample `df_pred`, horizontal bars, cut-off lines).
The rank scatter is per-**protein** (max score) and a scatter. `ranking` and `rank` would be
near-homograph names for two distinct figures.

## Decision

**D1 — Fold the rank scatter into `AAPredPlot.predict_group` as `kind="rank_scatter"`.** The
drawing logic moves to `prediction/_backend/aa_pred/aa_pred_plot_rank_scatter.py`; the public entry
is the existing `predict_group` dispatcher (returns `(fig, ax)` via `ut.FigAxResult`). One new
public param, `group_order`, is added to `predict_group`; `col_score` / `col_group` / `dict_color`
/ `thresholds` (drawn as horizontal score lines here) / `marker_size` are reused from the shared
signature. `col_group` is required for this kind.

**D2 — Name it `rank_scatter`, not `rank`,** so it reads as a distinct figure from the existing
`kind="ranking"` bars rather than a one-letter variant of it.

**D3 — Remove the standalone `aa.plot_rank`** from the public API (and from `aaanalysis/plotting/`).
This is a breaking removal, but v1.1.0 is **unreleased**, so no published API is broken.

## Rejected alternatives

- **Keep `aa.plot_rank` as a delegating alias.** Leaves the plotting package with a chart-drawing
  function and two public entry points for one figure; since 1.1.0 has not shipped, a clean removal
  is available with no deprecation cost.
- **`kind="rank"`.** Collides visually with the existing `kind="ranking"` (leaderboard bars).
- **A dedicated `AAPredPlot.predict_rank` method.** Breaks the single `predict_group` dispatcher
  pattern for the across-samples figures.

## Consequences

`plotting/` returns to holding **only** stateless helpers. ADR-0006's metrics-symmetry rationale
weakens, but the `aa.metrics.comp_*` functions remain standalone (they are numeric and have no plot
class). The per-protein rank scatter now sits next to the predictor whose output it checks, and the
`ranking` (bars) vs `rank_scatter` (scatter) distinction is explicit in the `kind` name.
