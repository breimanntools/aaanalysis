# ADR-0006 — Per-protein rank plot ships as a standalone `aa.plot_rank`, not a `*Plot` method

Status: Superseded by ADR-0056 — 2026-07-09

> **Superseded.** The premise below ("this chart has no logic-class twin") no longer holds:
> the `AAPred` / `AAPredPlot` subpackage now owns the evaluate-and-deploy path, so the
> per-protein rank scatter moved to `AAPredPlot.predict_group(kind="rank_scatter")` and the
> standalone `aa.plot_rank` was removed. See ADR-0056. The original decision is kept below for
> the record.

## Context

Every chart in the library is a method on a `*Plot` class that mirrors a logic
class via `.eval` (`CPPPlot`, `AALogoPlot`, `dPULearnPlot`, …); the `aaanalysis/plotting/`
package holds **only** stateless helpers (`plot_get_clist`, `plot_settings`,
`plot_legend`, …) — zero chart-drawing functions. A heavy CPP user asked for the
"run4-style" per-protein rank plot (max-score-per-protein scatter, group sets,
threshold lines), the single most useful sanity check for a deployed predictor.

This chart has no logic-class twin: it visualizes per-protein *prediction scores*
and group membership, not a `df_feat`. And `CPPPlot.ranking()` already exists — it
ranks *features* from `df_feat`, a different input and axis.

## Decision

**D1 — Ship it as a standalone `aa.plot_rank`** in `aaanalysis/plotting/_plot_rank.py`,
exported top-level. It pairs symmetrically with the standalone `aa.metrics.comp_*`
functions (score vector + groups in, `(fig, ax)` out) that landed in the same effort.

**D2 — Colors default from the locked sample palette**, overridable via `dict_color`:
canonical group names (`substrate`→`COLOR_POS`, `non-substrate`→`COLOR_NEG`,
`hold-out`→`COLOR_REL_NEG`), with `plot_get_clist` as the fallback for other groups.

## Rejected alternatives

- **A method on a new `RankPlot`/`ScorePlot` class.** A whole class with a `.eval`
  for one chart, with no logic-class counterpart to mirror — ceremony without the
  pairing the `*Plot` convention exists to provide.
- **Add `CPPPlot.protein_ranking()`.** `CPPPlot` is `df_feat`/feature-centric; this
  input is per-protein scores. Conceptual mismatch, and a near-name collision with
  the existing `CPPPlot.ranking()` (feature ranking) that would confuse users.

## Consequences

A documented, deliberate exception to "the plotting module holds helpers only" —
justified by symmetry with the standalone metrics. Future readers see one loose
chart function and this ADR explains why it isn't a `*Plot` method.
