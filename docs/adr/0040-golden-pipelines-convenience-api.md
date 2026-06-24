# ADR-0040 — Golden pipelines: the `aaanalysis.pipe` (aap) convenience API

Status: Accepted — 2026-06-24

Implements the user-facing convenience-facade half of
[ADR-0038](0038-agentic-readiness-boundary.md) (AAanalysis owns convenience; the
machine-readable tool contract is ProtXplain's). This ADR fixes the naming schema,
the Ends/Means model, the return/state contracts, and the documentation exposure for
the `aaanalysis.pipe` namespace, so the pipelines added over time stay consistent.

## Context

AAanalysis exposes object-oriented **primitives** (`CPP`, `TreeModel`, `ShapModel`,
`SequenceFeature`, `AAclust`, …) with every parameter exposed. Getting a result still
means hand-wiring the recurring `load → parts → CPP → model → explain → plot` chain
each time, which is a high floor for first use and gives coding agents no single,
stable entry point.

The fix is a second, opt-in API — `import aaanalysis.pipe as aap` — of **golden
pipelines**: one function each for the workflow users run most often. This mirrors
Matplotlib's two interfaces (the explicit `Axes`/`Figure` API and the `pyplot`
façade): reach for `aap` like `plt` for the common case, drop to the primitives for
full control. A first pipeline (`aap.predict`, now `predict_labels` under D1) shipped
in #244; this ADR sets the conventions before the namespace grows and goes public.

## Decision

**D1. Naming schema = `verb_noun`.** Every golden pipeline is `verb_noun`, consistent
with every existing module-level function in the package (`load_dataset`,
`comp_seq_sim`, `scan_motif`, …) — there is no bare-verb precedent. The verb encodes
the role: result-verbs (`find`, `predict`, `explain`, `design`, `evaluate`) are Ends;
producer-verbs (`get`, `select`, `sample`, `embed`) are Means. (The #244 seed
`aap.predict` is renamed `aap.predict_labels` under this schema; it is experimental
and unreleased, so the rename is hard, no deprecation.)

**D2. Tiered model — "golden pipeline" means an End.**
- **Ends** are deliverables: one call returns the thing the user wanted. These are the
  golden pipelines: `find_features`, `predict_labels`, `explain_features` (*pro*),
  `design_mutations`, `evaluate_models`.
- **Means** are producers of an *input* to an End (reliable negatives, a reduced scale
  set, sampled windows, embeddings). A Means is a **flag** on the End it feeds by
  default (e.g. `find_features(dpulearn=True, subcategories=…)`); it is promoted to a
  standalone `aap` producer only on genuine standalone need, and is never *both* a flag
  and a function for the same canonical path.

**D3. Return contract = `(primary, secondary)`.** Each End returns a 2-tuple, primary
deliverable first. A plotting End returns `(data, ax)` with `ax` a Matplotlib `Axes`
(Figure via `ax.figure`) and `ax=None` when `plot=False` (always a 2-tuple, never a
bare value). A modelling End returns `(model, df_eval)`. `aap` **adopts whatever plot
return-contract #133 settles** rather than inventing its own — the plot return shape
is extracted at the wrapper boundary, not hard-coded deep in `aap`.

**D4. Thin, stateless, byte-identical.** Pipelines add no algorithm of their own;
defaults are **byte-identical** to the explicit primitive path (a parity test pins
each one). They are stateless (no `pyplot`-style global figure/among-call state),
return plain numpy/pandas, thread `random_state`/`n_jobs`, and add **no new
dependency** (sklearn is already core; torch stays the `[embed]` extra). Ends accept
`df_seq` **or** `df_parts`.

**D5. Documentation exposure (the pyplot standard).** `aaanalysis.pipe` is documented
like `matplotlib.pyplot`: a dedicated **API-reference page** (autosummary of `aap.*`),
a **narrative "Golden pipelines" guide** stating the two-interface model (primitives
for control, `aap` for quick results), each pipeline cross-linked to the primitives it
wraps with an example notebook under `examples/pipe/`. The namespace ships
**experimental** and stays **out of `aaanalysis.__all__`** (used only via
`import aaanalysis.pipe as aap`) until it stabilizes — so it is not yet on the
top-level CONFIRM-FIRST API surface.

**D6. Catalog (the planned pipelines).** Built as their primitive is ready; not all at
once:
- *Ends:* `find_features` (CPP), `predict_labels` (TreeModel), `explain_features`
  (ShapModel, *pro*), `design_mutations` (AAMut/SeqMut), `evaluate_models`
  (metrics + TreeModel).
- *Means (flag-first):* reliable negatives (dPULearn → `dpulearn=`), scale reduction
  (AAclust → `subcategories=`), `sample_windows` (AAWindowSampler), `embed_sequences`
  (EmbeddingPreprocessor, *pro* `[embed]`).
- *Future (pipeline lands with its primitive):* `map_structure` (#119/#120), `evolve`
  (#57), `active_learn` (#60), `estimate_uncertainty` (#16/#53).

## Consequences

- A consistent, future-proof namespace: a new pipeline picks a `verb_noun`, an Ends/Means
  role, and the return contract — no per-pipeline naming debate.
- The experimental, out-of-`__all__` status keeps the top-level public surface (and its
  semver weight) unchanged while the convenience API matures.
- `aap`'s plot return is coupled to #133; the two must land their plot-return shape in the
  same release.
- "Golden pipeline" stays a crisp term (a deliverable), so the docs can advertise a small
  headline set rather than a sprawling function list.

## Out of scope

- The machine-readable tool contract / MCP / verb-orchestration layer — ProtXplain's
  (ADR-0038).
- Promoting `aaanalysis.pipe` into `aaanalysis.__all__` / the top-level API — a later,
  CONFIRM-FIRST decision once the API is stable.
- The sklearn `SequenceFeatureTransformer` (a `feature_engineering` interop class, not an
  `aap` pipeline) — tracked in #241, decided separately.
