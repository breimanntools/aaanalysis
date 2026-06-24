# ADR-0027 — Protein design (AAMut/SeqMut): scope boundary and model-free ΔCPP

Status: Accepted — 2026-06-11 (amended 2026-06-24 — see *Amendment* below and ADR-XXXX)

## Amendment (2026-06-24) — SeqMut becomes optionally model-aware (#57)

D2 below established a **model-free** ΔCPP for #37 and deferred the model-based prediction
delta to "#57 (ML-guided) territory ... out of scope." Issue #57 has now landed and **places
that ML-guided layer on `SeqMut` itself** (not a new class): constructing `SeqMut(model=...)`
binds a fitted classifier so each mutation also carries a model **prediction shift** `delta_pred`
= `(P_target(mut) − P_target(wt))·100`, and two new verbs (`combine`, `evolve`) stack mutations.
The model-free ΔCPP path is unchanged and remains the default. `AAMut` stays deterministic. So D2's
"model-free, out of scope" wording is **superseded for SeqMut** by ADR-XXXX; D1/D3/D4/D5 stand.

## Context

Issue #37 ("Finish AAmut & SeqMut for Protein Design", v1.2) asked to turn the
four empty `aaanalysis/protein_design/` stubs (`AAMut`, `AAMutPlot`, `SeqMut`,
`SeqMutPlot`, all raising `NotImplementedError`) into a working CPP-guided
mutation/design module. #37 is the **gate** for the protein-design chain — #57
(ML-guided directed evolution), #59 (multi-objective optimization), #60 (active
learning) — which the issue handoff records as *blocked-by #37*.

The tension: #37's own issue body lists a "Design strategies" section ("suggest
mutations toward desired CPP shifts", "optimize sequences toward a target
signature") and an "Evaluation" section that read almost verbatim like #57/#59/#60.
Implemented literally, #37 would collapse the whole chain into one PR and leave the
downstream issues nothing to build. The handoff explicitly flagged that the
grilling "must settle what 'done' means before the design chain builds on it."

A second fork: the only prior mutation code in the repo (`dev_scripts/dev_cpp.py:
mutation_pred`) scores mutations by the **change in a trained RandomForest's
predicted probability** — a model-based ΔCPP. #59's acceptance criteria, however,
state "no black-box optimization methods used."

## Decision

**D1 — #37 is the primitive + measurement layer plus a *minimal single-objective*
suggestion; the strategy/optimization layer is #57–#60.** In scope: apply
mutations (single/batch/region), measure ΔCPP, rank by `|ΔCPP|`, the residue-level
substitution landscape (`AAMut`), stable-vs-disruptive evaluation, and a minimal
target-shift `suggest`. Deferred to the chain: mutation-library generation,
multi-objective/Pareto weighting, and uncertainty/active-learning selection. The
line is **descriptive/measurement (here) vs prescriptive/optimization (there)** —
ranking *by raw magnitude* is sorting a measurement and stays here; ranking
*toward a target via an optimizer/library* is #57/#59.

**D2 — ΔCPP is feature-space and model-free.** `ΔCPP = X_mut − X_wt` over a
`df_feat`'s features (`X` via `SequenceFeature.feature_matrix`), aggregated to
`delta_cpp = Σ|ΔX|`. No trained model is involved. The `dev_cpp.py` model-based
path is recognized as #57 (ML-guided) territory and is out of scope.

**D3 — The suggestion target is the `mean_dif` direction already in `df_feat`.**
`SeqMut.suggest` ranks by `shift_score = Σ sign(mean_dif) · ΔX` (optionally
weighted by `feat_importance`/`abs_auc`), i.e. movement toward the test-class
profile. The target needs no labels, no extra model, and no separate target
vector — it reuses what CPP already computed — though an explicit `target`
feature-vector override is accepted for later multi-objective use.

**D4 — AAMut is CPP-agnostic; all CPP coupling lives in SeqMut.** `AAMut` operates
only on `df_scales` (substitution impact per scale); `SeqMut` is the sole class
that consumes `df_seq` + `df_feat`. This keeps the reusable physicochemical
primitive testable on its own terms and the CPP coupling in one place.

**D5 — Free-form classes, descriptive verbs; module stays core.** Neither class
subclasses the `Tool`/`Wrapper` ABCs (like `AAWindowSampler`/`SequenceFeature`):
`AAMut.run`/`eval`, `SeqMut.mutate`/`scan`/`suggest`/`eval`. The stub `fit` is
dropped — neither class wraps a model. No heavy/fragile dependency is introduced,
so `protein_design` stays in **core** (no `pro` gate).

## Rejected alternatives

- **Implement #37's issue body verbatim** (incl. optimization workflow).
  Rejected: collapses #57–#60 into one PR and defeats the gate decomposition the
  handoff established.
- **Model-based ΔCPP** (score mutations by a trained classifier's predicted-
  probability delta, as in `dev_cpp.py:mutation_pred`). Rejected for #37: violates
  the #59 "no black-box" constraint and is the substance of #57; the model-free
  feature-space delta is interpretable and dependency-light. The model-based view
  remains available to the downstream chain.
- **Recompute a class centroid from labels as the suggestion target**, or
  **require a user-supplied target vector**. Rejected as the default: the first
  duplicates what `df_feat.mean_dif` already encodes and needs labels at SeqMut
  time; the second has no sensible default for the common case. The `mean_dif`
  direction is the zero-input default; an explicit `target` override covers the
  rest.
- **Subclass `Tool` (`run`/`eval`) or keep the stub `fit`.** Rejected: SeqMut has
  three distinct verbs (mutate/scan/suggest) and neither class has fitted model
  state, so the ABC ceremony and the `fit` name mislead; free-form descriptive
  verbs match the sequence-utility siblings.

## Consequences

- #57/#59/#60 are unblocked and build on `SeqMut.suggest` / the ΔCPP engine.
- The public API surface (`AAMut.run`/`eval`, `SeqMut.mutate`/`scan`/`suggest`/`eval`)
  is the contract the design chain depends on; changing it later is a breaking change.
- Docs-only follow-up (#107): graduate Protein Design from the Experimental to the
  Stable API grouping now that the classes are implemented.
