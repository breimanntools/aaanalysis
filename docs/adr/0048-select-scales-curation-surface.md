# ADR-0048 — `AAclust.pre_select_scales` for metadata-only scale exclusion

Status: Accepted — 2026-06-29

## Context

The γ-secretase use case builds five study scale sets by hand: exclude whole
AAontology categories / specific subcategories from `df_cat`, intersect with the
classified scales, then redundancy-reduce to 100 % subcategory coverage. In the
notebook this was two helper functions over raw pandas plus a verbose
`AAclust.filter_coverage` call. The maintainer asked how to express this more
cleanly as package API.

Two genuinely different operations were conflated in that helper code:

1. **Metadata selection** — an arbitrary include/exclude over `df_cat`
   `category` / `subcategory`. This does **no clustering**; it is a boolean mask
   on metadata. The package had no public verb for it (the only category-aware
   load knob is `load_scales(unclassified_out=True)`).
2. **Redundancy reduction** — already owned by AAclust: `select_scales`
   (threshold / `n_clusters` driven, `df_scales` in / `df_scales` out) and
   `filter_coverage` (subcategory-coverage driven, list-of-ids out).

An earlier draft folded op 1 (and a coverage mode) **into** `select_scales`,
giving a single multi-mode curation verb. That was rejected by the maintainer in
favour of **separation of concerns**: keep each method doing one thing, and add a
dedicated metadata-only verb. `load_scales` was also rejected as the home (it
already carries six selectors and the arbitrary-exclusion case is unlike its
cumulative-tier `top_explain_n`). ADR-0025 had earlier rejected a standalone
`aa.filter_scales`, but that was for the interpretability *relabeling* transform
(now `CPP.simplify`), a different operation from arbitrary metadata subsetting,
so it does not foreclose this.

## Decision

**D1 — A dedicated `AAclust.pre_select_scales` method.** `df_scales` in,
`df_scales` out; it excludes AAontology categories (`cat_out`) / subcategories
(`subcat_out`) via `df_cat` and does nothing else — no clustering, no coverage.
It lands on AAclust (not `load_scales`, not a standalone `aa.filter_scales`)
because it sits in the AAclust scale-curation workflow and reads naturally as the
pre-step before `select_scales` / `filter_coverage` on an instance the user
already holds. The `pre_` prefix marks it as a preparation step.

**D2 — `select_scales` and `filter_coverage` are unchanged.** The metadata
filter is *not* merged into either; each method keeps one responsibility. The
three compose explicitly: `pre_select_scales` (metadata) → `select_scales`
(threshold) or `filter_coverage` (coverage). A multi-mode `select_scales` (a
`min_coverage` switch that neuters `min_th`) was built and then rejected — the
silent param-neutering and the duplication with `filter_coverage` were not worth
the one-call convenience.

**D3 — `cat_out` / `subcat_out` naming.** The `cat`/`subcat` stems match the
codebase shorthand (`df_cat`, `ut.COL_CAT`, `ut.COL_SUBCAT`, `build_cat`); the
`_out` suffix matches the existing `unclassified_out` exclusion idiom. Exclusion
only (no include-lists) for now — the "keep interpretable" axis is already
`top_explain_n`. `df_cat` defaults to the bundled categories; both params accept
a single string or a list. Unknown names raise `ValueError` (a silently-no-op
typo is a correctness trap).

**D4 — Returns a `df_scales` DataFrame**, preserving column order, with the
excluded scales dropped. Callers needing ids use `list(df_scales_pre)`.

## Rejected alternatives

- **Fold metadata exclusion + a coverage mode into `select_scales`.** Built
  first, then rejected for separation of concerns: a `min_coverage` switch that
  silently neuters `min_th` violates least surprise, and the coverage path
  duplicates `filter_coverage`.
- **More `load_scales` params** (`cat_out` / `subcat_out` on the loader).
  Rejected — `load_scales` already carries six selectors, and this is a curation
  step on an AAclust instance.
- **Standalone `aa.filter_scales`.** Rejected — same name ADR-0025 declined; a
  pure-metadata filter does not earn a top-level symbol when it reads naturally
  as an AAclust pre-step.
- **An include direction (`cat`/`subcat` keep-lists) now.** Deferred — the paper
  only excludes, and `top_explain_n` already covers "keep interpretable."

## Consequences

- `AAclust` gains `pre_select_scales` (new `1.1.0` method) and a
  `check_match_df_cat_cat_out` frontend helper. `select_scales` and
  `filter_coverage` are untouched.
- The γ-secretase notebook's `scales_excluding` helper collapses to one
  `pre_select_scales(...)` call, composed with `filter_coverage` for coverage.
- `CONTEXT.md` coins **scale pre-selection**, pinning `cat_out` / `subcat_out`
  against `cat_remove` / `subcat_remove` drift.
- A new `examples/aac_pre_select_scales.ipynb` (executed) backs the method's
  `Examples` include.
