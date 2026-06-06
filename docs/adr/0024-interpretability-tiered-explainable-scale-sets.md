# ADR-0024 — Interpretability-tiered "explainable" scale sets in `load_scales`

Status: Accepted — 2026-06-06

## Context

`aa.load_scales` already ships **performance-ranked** redundancy-reduced scale
sets: `top60` / `top60_n` selects one of 60 AAclust scale sets ranked by
predictive performance ([Breimann24a]). There was no way to load a set chosen
for **interpretability** — "give me only the most interpretable physicochemical
properties."

A new curation (`subcat_selections.xlsx`) rates each of the 74 AAontology
subcategories on `interpretability` (1–10, 1 = most interpretable) and assigns a
cumulative inclusion tier `top_subcat` ∈ {5,10,…,60,62,67}, derived from
unsupervised clustering of the subcategories combined with expert domain
knowledge (no separate publication). The goal: let users load **simplified, more
interpretable scale sets** and optionally redundancy-reduce them — the
explainability-axis sibling of `top60`.

A second, related capability — simplifying an *arbitrary* user-supplied scale set
— was scoped out (it will live in `CPP`).

## Decision

**D1 — A pre-filter on `load_scales`, not a new public symbol or class.** Two
optional, backward-compatible parameters: `top_explain_n` (the interpretability
tier) and `top_explain_min_th` (optional AAclust reduction). `load_scales` is
already exported; no `aaanalysis/__init__.py` change.

**D2 — "explain"-branding; selector stem matches the `df_cat` column.**
`top_explain_n` / `top_explain_min_th`; the per-scale tier column is `top_explain`
and the rating column `interpretability`. The xlsx-era name `top_subcat` is
dropped in favour of `top_explain` so the public selector and the column share a
stem. Distinct branding from performance-ranked `top60`.

**D3 — The unit is the subcategory; the default keeps all member scales.**
`top_explain_n=n` keeps every scale whose subcategory has `top_explain <= n` (the
7 `Unclassified (...)` subcategories are `NaN` and always excluded). With
`top_explain_min_th=None` (default) this returns *all* member scales — no
redundancy reduction. Redundancy reduction is left orthogonal to AAclust so the
two compose.

**D4 — Tier ceiling capped at 60.** Valid tiers are {5,10,…,60}. The xlsx steps
62 (graph-based measures) and 67 (principal components) are the explicitly
*least* interpretable subcategories; exposing them through a feature branded
*explain* is self-contradictory. Their true `top_explain` value is still stored
in `df_cat` for transparency, but the selector validator rejects them.

**D5 — `top_explain_n` and `top60_n` are mutually exclusive** (`ValueError` if
both set); `top_explain_min_th` requires `top_explain_n`. Two ranking axes, one
selector at a time.

**D6 — `top_explain_min_th` uses pre-computed, per-tier, dual-grid AAclust
selections.** For each (tier × `min_th` ∈ {0.3,…,0.9} × `just_aaindex` ∈
{False,True}) the generator pre-selects the pool, then runs `AAclust().fit(min_th=…)`
with default settings and a fixed seed, storing the medoid `scale_id`s. 12 × 7 ×
2 = 168 settings. Clustering is **per tier** (medoids are tier-optimal, *not*
nested across tiers — chosen over cluster-once-then-subset) and on **dual grids**
(AAindex-only sets are independently clustered after dropping LINS/KOEH, not
derived by post-filtering the full grid), so `just_aaindex` results are correct.

**D7 — `df_cat` carries the two new columns only under a tier selection.** The
default `load_scales(name='scales_cat')` schema is unchanged (backward
compatible). `interpretability` / `top_explain` are used internally for filtering
and dropped from every non-`top_explain_n` return; they ride along only when
`top_explain_n` is set.

**D8 — Compact, internal storage; no new loadable `name=`.** The selections live
in `aaanalysis/_data/top_explain.tsv` as one row per setting with a delimited
`scale_ids` string (not a dense 586-wide 0/1 matrix), minimising bundled-data
overhead. `top_explain` is **not** added to `NAMES_SCALE_SETS`; the file is an
internal selection table that drives scale filtering, never a user-facing
dataset. `subcat_selections.xlsx` is a dev-time source (under `dev_scripts/`),
not shipped.

## Rejected alternatives

- **One representative scale per subcategory.** Would fold a medoid-selection
  policy into an interpretability feature and overlap AAclust. Rejected — the
  unit is the subcategory; reduction stays AAclust's job (D3).
- **A dedicated `aa.filter_scales` / standalone transform now.** The enabling
  work is the `df_cat` columns + tier selector; once those exist, simplifying an
  arbitrary set is a documented filter. The general transform is deferred to
  `CPP`.
- **Cluster-once-then-subset (nested tiers).** Cheaper (14 runs) and gives nested
  sets, but the medoids are chosen against the full pool, not the tier. Rejected
  for per-tier clustering (D6).
- **Single AAclust grid + post-filter for `just_aaindex`.** Simplest, but a
  cluster whose medoid was a LINS/KOEH scale silently loses its representative.
  Rejected for dual grids (D6); residual coverage gaps are documented as a Note.
- **Dense 168×586 boolean-mask TSV** (mirroring `top60.tsv`). Rejected for the
  compact delimited-id format on the user's "as small as possible" directive
  (D8).
- **Exposing `name='top_explain'`** as a loadable matrix view like `top60`.
  Rejected — the mask is internal; `df_cat` plus the selector params are the
  whole surface (D8).
- **Always adding the two columns to every `df_cat`.** Rejected — would change
  the long-standing default schema and break consumers that assume the five
  AAontology columns (D7).

## Consequences

- `aaanalysis/_data/scales_cat.tsv` gains `interpretability` + `top_explain`
  columns (per-subcategory values broadcast over scales); a new
  `aaanalysis/_data/top_explain.tsv` holds the 168 pre-computed selections. Both
  are regenerated by `dev_scripts/dev_build_top_explain.py` from
  `subcat_selections.xlsx`.
- A new "Scale-set vocabulary" section in `CONTEXT.md` coins **explainable scale
  set**, **interpretability tier**, **interpretability rating**, and
  **top_explain_min_th**, separating the interpretability axis from `top60`'s
  performance axis.
- With `top_explain_min_th` set, the reduced set is **not guaranteed** to cover
  every subcategory in the tier (AAclust reduction + `just_aaindex` post-filter
  can drop a subcategory's only representative) — documented in the `load_scales`
  Notes.
- The pre-computation is reproducible (fixed seed); the bundled artifacts are
  deterministic.
