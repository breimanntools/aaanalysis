# ADR-0044 — `find_features` staged sensitivity search and multi-objective selection

Status: Accepted — 2026-06-24

## Context

ADR-0041 D3 defined `aap.find_features` as a CPPGrid sweep selected by a single model
cross-validation. The shipped v1 (PR #258) realised that as "CV-score every cell of the
Part × Split × Scale × `n_filter` grid, then pick the simplest config within 1 % of the best
balanced-accuracy". That works but has three limits: (1) it CV-scores the **full** grid, which
does not scale once the Scale axis grows (interpretability tiers **and** performance `top60`
sets); (2) it is single-objective (one metric, one model); and (3) it gives no signal about
**which lever actually matters**. This ADR refines D3 into a staged, interpretable, multi-objective
protocol. v1.1.0 is unreleased, so the changes (including the parameter rename) ship without a
deprecation cycle.

## Decision

- **D1 — `search` replaces `optimization`.** The effort grade is named `search`
  (`"fast" | "balanced" | "exhaustive"`). "Optimization" is reserved for the `SeqOpt`
  sequence directed-evolution chain (CONTEXT.md), so reusing it for feature-search effort was an
  overload. Free rename (unreleased).

- **D2 — Three-stage search.** Feature *generation* is separated from *post-processing*:
  1. **Sensitivity (Stage 1).** Run the full Cartesian **Part × Split × Scale** at a fixed
     reference `n_filter` via `CPPGrid`; cross-validate every cell.
  2. **Refinement of the dominant axis (Stage 2).** Sweep the single highest-impact axis ×
     `n_filter`, the other two axes pinned at the stage optimum (`n_filter` slices for free).
  3. **Feature-set refinement (Stage 3).** `CPP.simplify` then RFE on the winner, each kept only
     if it is not Pareto-dominated by the pre-refine winner.

- **D3 — Axis impact = normalized marginal-mean spread.** An axis's impact is the `max − min`
  spread of its **marginal-mean** CV score across its levels (averaging over the other axes),
  computed **per metric, normalized to `[0, 1]`, then averaged across metrics** — no privileged
  "primary" metric. The largest-impact axis is the **dominant axis**. The non-dominant axes are
  pinned at the stage's selected (Pareto-then-simplest) cell.

- **D4 — Per-stage Pareto selection.** Each stage selects its winner independently by
  **Pareto-optimality across all metrics**, tie-broken by **simplest** (fewest features, then
  smallest `n_filter`). Multiple models → the per-cell score is the **average of per-model CV
  means**; multiple metrics → **all** metrics are computed in one `cross_validate` (multi-scorer)
  pass, so the Pareto front is nearly free. The returned `df_feat` is the final stage's winner.

- **D5 — Pareto-dominated refine keep-rule.** Stage 3 keeps a `simplify`/RFE result iff it is not
  Pareto-dominated by the pre-refine winner (≥ on every metric, or a trade). RFE's
  no-headroom guard skips the fit when the winner already maxes every metric.

- **D6 — Runtime-budgeted grids in a registry.** `fast` is a **single** configuration (no search;
  **byte-identical** to the explicit CPP chain — the parity contract is preserved). `balanced`
  targets ≈ 10 min (Split + Scale + `n_filter`); `exhaustive` runs longer (also Part + Scale
  breadth, including the orthogonal `top60` performance sets, and a finer `n_split_max` step). The
  exact level lists live in the module's `_MODES` registry and are freely tunable.

- **D7 — Enriched `df_eval` contract.** One `mean`/`std` column **per metric**, plus `stage`,
  `is_pareto` (Pareto-optimal within its stage), `rank`, and `is_selected` (the single winner).

- **D8 — Publication eval figures via `ax.eval`.** `plot_eval` (refines #251's `aap.plot_eval`)
  **decomposes** the structural sweep into a set of **separate** publication-ready figures: the two
  most-informative axes (largest marginal-mean impact) form each 2D `viridis` heatmap, the
  least-informative axis is the slice (one figure per level), all sharing one color scale with the
  winner starred — plus a marginal-impact bar panel and an `n_filter` panel. `find_features` attaches
  the list to the returned feature-map `Axes` as **`ax.eval`** (empty for `fast`), preserving the
  uniform `(df_feat, ax, df_eval)` triple while letting users save each figure individually. This
  amends ADR-0041's return slot (the `figs` slot may carry auxiliary figures on the primary `Axes`)
  and supersedes #251's single faceted figure for the find_features path.

## Rejected alternatives

- **One-axis-at-a-time (OAT) sensitivity** (additive cost ≈ `|Part|+|Split|+|Scale|`). Cheaper, but
  misses axis interactions; full Cartesian was chosen for thoroughness, with runtime as the budget.
- **A single global Pareto front** over every evaluated config, or a single scalar (weighted/knee)
  selection. Rejected for **per-stage** Pareto: each stage answers a self-contained question.
- **CPPGrid's own `avg_ABS_AUC` ranking.** Monotone in `n_filter`, so it cannot pick `n_filter` —
  a model CV is required.
- **Sweeping `interpret_grade` (1–10) alongside `top_explain_n` (5–60).** They are the same
  interpretability ranking at two resolutions; sweeping both is redundant. The orthogonal second
  scale axis is the performance-ranked `top60_n`.

## Consequences

- Selection is multi-objective and interpretable (the dominant lever is reported), at the cost of
  more CV than v1 in `exhaustive`. `fast` is unchanged and still parity-tested.
- `df_eval` widens (per-metric columns); downstream readers must not assume a fixed `cv_bacc_*`
  schema. The eval-grid plot (separate issue) consumes this richer table.
- Refines ADR-0041 D3; does not change the uniform `(results, figs, evals)` return triple.

## Out of scope

- The eval-grid sweep **figure** (its own issue). The `df_eval` table here is its data source.
- `SeqOpt` directed-evolution "optimization" (the reserved sense), tracked separately.
