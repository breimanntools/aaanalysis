# ADR-0049 — CPP profile y-axis is non-negative; three plot-rendering "fixes" are rejected

Status: Accepted — 2026-07-04

## Context

A July 2026 code audit flagged three plot-rendering expressions as "bugs" (an
always-zero `min`, a bool-vs-str comparison that is always true, and a `weight_bold`
branch that leaves axis-line widths untouched) and drafted mechanical "fixes" for
each. **All three change the rendered figure against the plot's intent and are
unwanted.** None reached `master` — they were reverted before the batch PR (#346)
merged — but this ADR records the decision so they are never re-attempted.

The lead case is the profile y-axis, which exposes the core mistake: treating a
plot expression as a mechanical simplification without checking the **domain meaning
of the axis**.

- **CPP profile** (`shap_plot=False`, default) shows aggregated feature importance,
  which is **non-negative** — it lives in `[0, max]` with the baseline at 0.
- **CPP-SHAP profile** (`shap_plot=True`) shows **signed** SHAP contributions, centred
  on 0, where symmetric headroom is appropriate (handled by `_scale_ylim`).

## Decision

The following three drafted changes are **rejected permanently. Do not reconsider.**
`master`'s current expressions stay as-is; each carries an inline comment pointing to
the rationale (in plain language, per the no-ADR-refs-in-code rule).

1. **`profile` y-axis padding** — `y_space = min(0, (ymax - ymin) * 0.25)` must **not**
   become `max(0, …)`. `min(0, …)` is a deliberate no-op that keeps the CPP profile at
   `[0, max]`; `max(0, …)` adds symmetric 25 % padding, pushing an all-positive profile
   to `0 - 0.25·range < 0` and drawing empty **negative** y-space beneath it — wrong,
   because CPP importance is never negative.
2. **`feature_map` importance-bar ticks** — `show_only_max = add_imp_bar_top != "long"`
   must **not** be re-keyed to `imp_bar_label_type`. The current behaviour (only the max
   tick labelled on the cumulative-importance bar) is the intended, uncluttered look;
   the change relabels every default feature-map figure.
3. **`plot_settings(weight_bold=True)` spine/tick widths** — the bold branch must **not**
   gain `axes.linewidth` / tick-width settings. `weight_bold` intentionally affects font
   and label weight only; thickening spines/ticks changes every bold-styled figure.

## Consequences

- Audits and reviews must treat any change to a **rendered figure** as output-affecting
  and validate it against the plot's **visual intent** (importance is non-negative;
  tick and weight choices are deliberate) — never as a mechanical simplification of an
  "always-true" / "always-zero" expression.
- The three expressions carry inline comments so they are not re-flagged as dead code.
- If a genuine improvement to any of these is ever wanted, it must be driven by an
  explicit design decision (and figure review), not an audit "cleanup".
