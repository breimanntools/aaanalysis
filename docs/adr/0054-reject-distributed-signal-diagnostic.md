# ADR-0054 — Reject the distributed-signal (joint-vs-marginal lift) diagnostic

Status: Accepted — 2026-07-08

## Context

CPP ranks and filters features **one at a time** by a marginal discriminative statistic (adjusted
AUC, mean difference). A recurring worry (raised as issue #341, a child of the epic #336 usability
sweep) is that this per-feature filter is **blind to distributed signal**: a set of features that are
each individually weak but *jointly* separate the classes gets discarded, because no single one of
them clears the marginal threshold.

A prototype was built on a local branch to make that gap measurable: a `comp_block_lift` metric —
the cross-validated AUC of a feature *block* trained jointly, minus the best single-feature AUC in
that block — plus a small `CPP` hook to compute it and ~260 lines of tests. It was never pushed
(no PR); it sat as held work pending two unresolved decisions: whether to publicly re-export
`aa.comp_block_lift` (a CONFIRM-FIRST `__init__.py` change) and how to define "blocks" and bound the
cross-validation cost.

## Decision

**D1. Do not pursue the distributed-signal diagnostic.** The idea is discarded, not deferred. The
prototype branch is dropped (never merged, never pushed); the local work is deleted. Issue #341 is
closed as not-planned, pointing here.

**D2. CPP's filter stays marginal / per-feature by design.** Joint and higher-order effects are the
responsibility of the **downstream modeling layer**, where they already surface — TreeModel feature
importance and `ShapModel` SHAP values both attribute a fitted model's use of a feature *in the
context of the others*, which is exactly the "does this feature matter jointly?" question. CPP's role
is the interpretable, position-resolved marginal signal; conflating the two would blur that contract.

## Rejected alternatives

- **Ship `comp_block_lift` (the prototype).** It introduces a multivariate/block-evaluation concept
  into a package whose value proposition is *interpretable per-feature* physicochemical profiling.
  The public-API cost (a CONFIRM-FIRST re-export) was never justified by a concrete user request, and
  the design was unsettled: how blocks are chosen, the cross-validation budget per block, and the
  lift threshold were all open. A metric users cannot act on cleanly is worse than no metric.
- **Fold block evaluation into CPP's filter itself** (keep jointly-strong blocks that the marginal
  filter would drop). This would make the filter's output depend on a model and a CV split, so the
  same `df_parts` + scales would no longer yield a deterministic, model-independent `df_feat` — a
  regression of CPP's reproducibility and interpretability guarantees.
- **Keep it as a held/independent issue.** Carrying unmerged prototype code and an open design
  question indefinitely is a maintenance and confusion cost; recording the decision and removing the
  code is cleaner than an ever-open "maybe someday" branch.

## Consequences

- No `comp_block_lift` (or any block/multivariate lift metric) in the public API or the `metrics`
  subpackage. Users needing joint-effect evidence use TreeModel importance or `ShapModel`.
- If a concrete, well-scoped use case for distributed-signal detection appears later, it starts fresh
  from a new issue and a new ADR that supersedes this one — not from the discarded prototype.
