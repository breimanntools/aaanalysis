# ADR-0042 — SeqMut model-guided prediction shift (ML-guided directed evolution)

Status: Accepted — 2026-06-24

## Context

Issue #57 ("ML-guided directed evolution", v1.2) is the optimization layer the protein-design
chain (#37 → #57 → #59 → #60) was decomposed toward. ADR-0027 built #37 as the **measurement +
minimal single-objective** layer on `AAMut`/`SeqMut` and explicitly **deferred the model-based
prediction delta** to #57, calling it "out of scope" and `SeqMut` "deterministic and model-free".
Issue #58 ("hit refinement") was folded into #57 as a duplicate, so residue-level prioritisation
also lands here.

Two questions had to be settled before code:

1. **Where does the ML-guided layer live** — a new optimizer class, or `SeqMut` itself? ADR-0027
   and the glossary had drawn `SeqMut` as model-free, so a new class looked consistent. But the
   maintainer's framing is that *`SeqMut` is exactly the place that should work with ML models* —
   "essentially giving the change of prediction score per mutation and position" — while `AAMut`
   stays the deterministic, scale-only primitive.
2. **What the headline artifact is.** The requested output is a **mutation-scan heatmap**: 20
   substitutions × positions, each cell colored by the model prediction shift, with the wild-type
   prediction in the title — i.e. the model's per-position sensitivity, not an abstract library.

## Decision

**D1 — `SeqMut` becomes *optionally* model-aware; `AAMut` stays deterministic.** A fitted
classifier is bound at construction (`SeqMut(model=..., target_class=...)`). With no model (the
default), `SeqMut` is byte-for-byte the model-free ΔCPP tool of ADR-0027. With a model, every
scoring method additionally reports the **prediction shift** `delta_pred`. No new public class is
introduced; `AAMut` is untouched. This supersedes ADR-0027 D2's "model-free / out of scope" wording
for `SeqMut` only.

**D2 — `delta_pred = (P_target(mut) − P_target(wt)) · 100`, duck-typed on `predict_proba`.** The
model is trusted to expose `predict_proba`; the engine reuses the wild-type and mutant CPP feature
matrices it already builds for ΔCPP, so no extra heavy computation. `TreeModel` returns the
positive-class `(pred, pred_std)` tuple — the std drives the heatmap's "score ± std" title — while a
scikit-learn classifier returns the 2-D probability matrix and `target_class` selects the column
(default: positive class). This keeps `protein_design` in **core** (no SHAP dependency forced).

**D3 — `SeqMut` *scores*; all *search/optimization* lives in a separate `SeqOpt` class.** `SeqMut`
adds `combine` — scoring explicit **combined variants** (several point mutations applied to one
sequence, one score each) — and "feature-consistent" substitutions are simply the top scorers per
position (scale-direction filtering left as a future option). The *search* layer (greedy directed
evolution, and multi-objective / Pareto and active-learning selection) is **not** on `SeqMut`; it is
the forthcoming `SeqOpt` optimizer (its own issue/ADR), keeping the scoring-vs-optimization boundary
clean. `SeqOpt` reuses model-bound `SeqMut` as its fitness engine.

**D4 — The plots carry the design story.** `SeqMutPlot.mutation_landscape` renders the model-guided
mutation-scan heatmap (diverging `delta_pred`, parts-colored sequence bar, wild-type-prediction
title), falling back to the model-free `delta_cpp` when no model was bound. `variant_impact` ranks
combined variants as a bar chart and `epistasis` maps pairwise non-additivity.

## Rejected alternatives

- **A dedicated optimizer class** (`SeqOpt`/`SeqDesign`) leaving `SeqMut` model-free. Rejected by
  the maintainer: the prediction-shift-per-mutation *is* the natural job of the sequence mutator;
  a second class would split one cohesive surface and duplicate the ΔCPP engine.
- **A `model=` argument per call** rather than at construction. Rejected: the model interprets a
  fixed `df_feat`/`target_class`, so binding it once (with `target_class`) keeps every call clean
  and mirrors how the model is conceptually paired with the feature set.
- **Putting search/optimization (greedy stacking, multi-objective) on `SeqMut`.** Rejected:
  `SeqMut` is the *scoring* surface; mixing automated search into it blurs the boundary. Explicit
  combination (`combine`) stays (it is scoring, not search), but greedy/multi-objective directed
  evolution moves to a dedicated `SeqOpt` optimizer (its own issue) that reuses `SeqMut` as fitness.
- **Scale-direction ("feature-consistent") filtering of allowed substitutions now.** Deferred:
  top-N-by-score ships the minimal useful behavior; the explicit `to_aa` alphabet already lets a
  user constrain substitutions.

## Consequences

- #57 (and the absorbed #58) is delivered; the forthcoming `SeqOpt` (#59 / #60) builds its search
  on `delta_pred` + `combine`.
- `delta_pred` / `wt_pred` / `wt_pred_std` join the `df_seqmut_scan` schema (optional columns);
  `df_seqmut_variant` is a new documented schema.
- The public `SeqMut` / `SeqMutPlot` surface grows (new constructor args, `combine`,
  `variant_impact`, `epistasis`); changing it later is a breaking change.
- ADR-0027 is amended (not superseded wholesale): its #37↔#59/#60 boundary and `AAMut` decisions
  stand.
