---
status: proposed
---

# Introduce `CPP.run_num` and a parallel `_filters_num/` backend

## Context

`CPP.run` (in `aaanalysis/feature_engineering/_cpp.py`) is the core feature-engineering algorithm. Two pressures pushed a redesign:

1. **Numerical input.** Users want CPP to operate on per-residue numerical representations (PLM embeddings, DSSP one-hots, PTM dummies) directly — not only on amino-acid sequences mediated by a `(20, n_scales)` lookup table. The existing `_filters/_assign.py` pipeline already *internally* converts sequences to numerical (n, L) arrays before splitting, so the natural change is to swap the value source rather than re-order operations.

2. **Performance.** `_filters/_stat_filter.py` re-runs the per-sample split-position computation for every scale (the inner loop is `for scale: for part: ...`). `_filters/_add_stat.py` then recomputes feature values from scratch via `get_feature_matrix_`, duplicating work `pre_filtering_info` already did. Together these dominate run time on realistic workloads (n_samples ~ 1k, n_scales ~ 600).

## Decision

Add a new method `CPP.run_num(df_seq, dict_num=None, df_scales=None, df_cat=None, …)` and a parallel backend folder `aaanalysis/feature_engineering/_backend/cpp/_filters_num/` mirroring `_filters/` stage-for-stage. Both paths coexist; `_filters/` is *not* modified.

Key contracts:

- **Bit-identical parity in seq-only mode.** With `dict_num=None` and the same scales, `CPP.run_num` must produce a `df_feat` equal to `CPP.run`'s output to the byte — enforced by a parity test fixture in `tests/unit/cpp_tests/`.
- **Per-residue staging via `dict[part] = (n_samples, L_part_max, D)` float32 tensors** with NaN padding for variable-length parts; downstream stats use `np.nanmean`. The (A vs B) head-to-head between this `dict[part]` shape and a single `(n, n_parts, L_global, D)` 4D tensor is benched in `dev_scripts/`; the loser is removed after one PR cycle.
- **Streaming pre-filter.** `_filters_num/_stat_filter.py` keeps per-sample feature values for the survivors of the `std_test < max_std_test` mask and hands them to `_filters_num/_add_stat.py`, which no longer calls `get_feature_matrix_`. Eliminates the duplicate per-feature split work.
- **`n_batches` partitions over D**, not scales.
- **Redundancy filter ported verbatim** from `_filters/_redundancy_filter.py`; future vectorization (precomputed overlap + correlation matrices) is captured as a `# DEV:` note inline rather than rebuilt now (see [[feedback-minimal-first]]).

## Considered alternatives

- **Polymorphic `CPP.run(values=...)`.** Rejected: would force a single method to dispatch on input type and obscure the head-to-head dev workflow.
- **Replace `CPP.run` in place.** Rejected: burns the head-to-head capability the moment the PR merges; parity regressions can hide forever.
- **New class (`CPPNumerical` / `CPPEmbed`).** Rejected: doubles the public surface, drift between class internals is a real maintenance cost.
- **Constructor-level mode switch.** Rejected: couples mode to instance identity; one CPP object cannot drive both `run` and `run_num` for parity tests.
- **Rewrite `_filters/` in place with a generalized tensor seam.** Rejected: head-to-head profiling and parity testing require both pipelines to be runnable on the same input — modifying `_filters/` while building the new path destroys the baseline.

## Consequences

- Source-tree growth (~one folder mirror) during the dev window. Acceptable; folder is small and clearly named.
- Two places to maintain stage signatures during dev. Mitigated by the parity test fixture catching contract drift immediately.
- Long-term: once `_filters_num/` is proven on realistic workloads and the head-to-head shows clear wins, `_filters/` may be removed and `CPP.run` may delegate to `_filters_num/`. That step is a separate decision (and a separate ADR), not part of this one.
- The legacy design sketch at `docs/source/design/run_embed.md` is superseded by this ADR. The vocabulary in `CONTEXT.md` ("Numerical-mode CPP vocabulary") replaces the `run_embed`/`dict_emb_part` terminology with `run_num`/`dict_num`.

## Amendment (PR2, 2026-05-21): precision unification in `_filters/_assign.py`

Original "Considered alternatives" rejected "rewriting `_filters/` in place" to preserve the head-to-head baseline. PR2 introduced a tightly-scoped exception: changed `scale_matrix` from `np.float32` to `np.float64` in `_filters/_assign.py` (one-line dtype change, ~no behavioral effect on round(3)-published `df_feat` columns).

The rationale: legacy `CPP.run` used **dual precision** internally — pre-filter stats came from float32-scaled values (`_filters/_assign.py` storage), while `add_stat` recomputed via `get_feature_matrix_` at full precision (Python `dict_scale` lookup). The cached-matrix streaming pre-filter in `_filters_num/` cannot simultaneously match *both* precisions in a single pass. Unifying `_filters/_assign.py` to float64 collapses the dual-precision design into a single precision, lets `_filters_num/` match exactly with one pass, and is a strict precision improvement for `CPP.run` users (drift absorbed by the existing `round(3)` step before `df_feat` is published).

This is the only modification to `_filters/`; it is not a rewrite. The head-to-head baseline remains intact.
