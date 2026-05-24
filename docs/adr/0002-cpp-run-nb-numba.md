---
status: superseded by ADR-0003
---

# Add Numba-accelerated `CPP.run_nb` behind the `pro` extra

## Context

`CPP.run_num` already has two fast paths: the bit-exact Python implementation in `_filters_num/_get_feature_matrix_fast.py` (~3.6× vs legacy at n=1000) and the hand-rolled-pairwise Cython kernel in `_filters_num_c/_inner.pyx` (~7–11× vs legacy at single-thread). The Cython path requires the user to run `python setup_inner.py build_ext --inplace`; users who skip that step silently fall back to the Phase-C Python path and miss the speedup.

Two pressures push a third backend:

1. **Install ergonomics.** A Numba JIT path replaces the manual compile step with `pip install aaanalysis[pro]`. First call pays a ~5–20s warmup; subsequent calls (with `@njit(cache=True)` in Stage 2) skip it.
2. **Headroom beyond Cython.** The Cython kernel re-gathers `arr_2d = scale_matrix_f64[aa_idx, scale_idx]` per feature (`_get_feature_matrix_c.py:107`), wasting work across the ~50 splits per `(part, scale_idx)` group. Numba's nogil region naturally hosts a cross-feature batch over splits sharing one gather, plus `prange` over the feature axis — both unavailable to the per-feature Cython dispatch loop without a deeper rewrite.

## Decision

Add `CPP.run_nb(df_seq, dict_num=None, …)` and a parallel backend folder `aaanalysis/feature_engineering/_backend/cpp/_filters_num_nb/` mirroring `_filters_num_c/` stage-for-stage. The Numba path coexists with the existing Cython and pure-Python paths; **none of `_filters/`, `_filters_num/`, `_filters_num_c/` is modified by this work**.

Stage the work explicitly:

- **Stage 1 (this PR): verbatim port.** Six kernels (Segment / Pattern-N / Pattern-C × {mean, nanmean}) re-implement `_inner.pyx`'s hand-rolled pairwise summation in `@njit` Python — same 8-way unrolled `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))` tree, same recursive halving for `n > 128`, same `np.round(_, 5)` applied at the Python boundary. PeriodicPattern falls back to the Phase-C Python path, mirroring Cython. Per-feature kernel calls + joblib threading retained unchanged from `_get_feature_matrix_c.py`. Goal: parity + the A-win (install ergonomics).
- **Stage 2 (follow-up PR): batching + prange + cache.** Sort features by `(part, scale_idx)`, gather `arr_2d` once per group, iterate splits inside one nogil region. Replace joblib threading with `numba.prange` over the feature axis (NOT the reduction axis). Enable `@njit(cache=True)`. Goal: the B-win over Cython.
- **Stage 3 (later): PeriodicPattern Numba kernel.** Inline `_get_list_periodic_pattern_pos` into a fourth `@njit` kernel. Closes the coverage gap inherited from Cython.

Key contracts:

- **Bit-identical parity.** `np.array_equal(X_legacy, X_nb)` per a new `tests/unit/cpp_tests/test_get_feature_matrix_nb_parity.py`. End-to-end `pd.testing.assert_frame_equal(df_legacy, df_nb, check_exact=True)` per `test_run_nb_parity.py`. Both skipped cleanly if `numba` is not installed.
- **Compile flags fixed.** `@njit(fastmath=False, boundscheck=False, error_model="numpy")`. `fastmath=True` reorders FP ops and would break ULP parity. `np.round(_, 5)` stays at the Python boundary because `np.mean` inside `@njit` does NOT match numpy's pairwise summation — the hand-rolled kernel produces the unrounded pairwise sum, then a single `np.round` pass matches `_inner.pyx:202`.
- **Pro-extras gating.** `numba` is added to `[project.optional-dependencies].pro` (CONFIRM-FIRST per CLAUDE.md §2). `CPP.run_nb` lazily imports the backend inside the method; `ImportError` whose `e.name == "numba"` raises a friendly `ImportError("…install with: pip install 'aaanalysis[pro]'")`; any other `ImportError` re-raises unchanged. Pure-Python install still gets `CPP.run` and `CPP.run_num` without error; `CPP.run_c` requires the Cython compile step; `CPP.run_nb` requires the `pro` extra.

## Considered alternatives

- **Verbatim-port + batch-restructure in one shot.** Rejected: bigger diff, larger review surface, and Stage 2's gains over Cython are unverified — the Cython kernel may already be near memory-bandwidth-bound, in which case batching saves only Python boundary overhead. Cleaner to land Stage 1 first and use its perf number as the apples-to-apples Numba-vs-Cython baseline before optimizing.
- **Replace Cython with Numba.** Rejected for now: removes the no-extras compiled path and kills the three-way head-to-head between Python / Cython / Numba. Reconsider in the auto-dispatch follow-up once Stage 2 proves the Numba path on realistic workloads.
- **Auto-dispatch inside `run_num` from day one (`numba > cython > python`).** Rejected: hides which backend ran during profiling, makes parity-test attribution harder. Auto-dispatch is a follow-up after Stage 2.
- **`backend="numba"|"cython"|"auto"` kwarg on `run_num`.** Rejected: implementation detail leaking into the user API; the `run_c` / `run_nb` method pattern is already established and head-to-head-able.
- **`prange` over the reduction axis.** Rejected: parallel reduction is order-non-deterministic and would break bit-exact parity. `prange` is restricted to the feature axis only (each feature's reduction stays sequential).
- **Inline `np.round(_, 5)` into the @njit kernel.** Rejected: numba's `@njit`'d `np.round` is not guaranteed to match numpy's banker's-rounding bit-for-bit; keeping the round at the Python boundary costs one full-array pass and buys zero risk, matching the Cython kernel's structure.

## Consequences

- Public CPP surface gains `run_nb`. Combined with existing `run`, `run_num`, `run_c`, that's a four-method surface during the dev window. The eventual auto-dispatch follow-up collapses it.
- `numba` becomes a `pro`-extra dependency, classified as "heavy" in `.claude/rules/pro-core-boundary.md`. Approval gated by CLAUDE.md §2 CONFIRM-FIRST list.
- First call to `CPP.run_nb()` pays a ~5–20s JIT warmup. Documented in the method docstring; Stage 2's `cache=True` makes it a one-time-per-Numba-version cost.
- ADR-0001's parity guarantee for `CPP.run_num` is unaffected — that path is not touched.
- Long-term: once Stage 2 proves Numba > Cython on realistic workloads, `CPP.run_num` may auto-dispatch (numba > cython > python) and `run_c` / `run_nb` become explicit-backend escape hatches. That step is a separate ADR, not part of this one.

## Amendment (PR3, 2026-05-23): bigger wins lived outside the kernel

After Stage 1 landed, profiling showed the kernel layer (`get_feature_matrix_nb_`)
was only ~6.5% of `CPP.run_nb` time at n=200 — Stage 2's planned kernel batching
would have hit diminishing returns. We pivoted from the original Stage-2 scope
(in-kernel batching + `prange` + `cache=True`) toward larger non-kernel targets
identified by profile.

**Landed in PR3:**

- **Path 1: Numba `add_stat_num`.** `_filters_num_nb/_stats_nb.py` adds
  bit-exact `@njit` versions of `scipy.stats.rankdata(method='average')`,
  the vectorized Mann-Whitney U test, and the rank-sum AUC formula —
  `add_stat_num_nb` is wired through `CPP.run_nb` via a new
  `add_stat_func` parameter on `cpp_run_num_single` / `_batch`.
  Bit-exact parity: the @njit `rankdata_axis0` reproduces scipy's
  average-rank-for-ties exactly; `scipy.stats.norm.sf` stays at the
  Python boundary as the bit-exact reference for the closed-form normal
  CDF approximation. Result: 1.27× over Cython at n=200, 1.09× at n=500,
  1.01× at n=1000 (the win shrinks because `pre_filtering_info_num`
  dominates at large n and is not touched by this PR).

- **Path 3: gather amortization** (part of original Stage 2). Features
  are grouped by `(part, scale_idx)` before kernel dispatch;
  `np.ascontiguousarray(scale_matrix_f64[aa_idx, scale_idx])` is gathered
  once per group instead of once per feature. Same per-feature
  arithmetic, same kernels, bit-exact. Memory delta drops to 0 MB at
  n=500/n=1000 (vs Cython's +114 / +136 MB), since the gather buffer is
  reused across the ~50 splits per group at default `split_kws`.

**Attempted, reverted:**

- **Path 2: vectorized redundancy filter.**
  `_redundancy_filter_nb.py` precomputes the `(n_pre_filter, n_pre_filter)`
  overlap / subset / correlation matrices upfront so the greedy loop's
  per-pair check becomes O(1) numpy lookups. **Reverted** — the legacy
  filter exits the greedy loop at `n_filter=100` accepted features, so it
  only does ~100 × small-constant pair checks; the vectorized version's
  upfront precompute over all 4,400 pre-filter survivors was a 44×
  regression (7.46s vs legacy 0.17s at n=200 × 586) and allocated ~1 GB.
  The file is kept (no deletions per CLAUDE.md §0) but is dead code
  until / unless rewritten as a streaming per-row precompute.

- **`@njit(cache=True)`.** Enabled during Path 3, immediately segfaulted
  the bench on numba 0.65.1 + Python 3.14. Reverted to `cache=False`.
  Cold-start cost stays at ~0.5–1.0s per fresh process — acceptable for
  the foreseeable future; revisit when numba ships Python 3.14 cache
  support.

- **`numba.prange` over the feature axis.** Not attempted. Kernel layer
  is small enough at the target workload sizes that joblib threading
  remains sufficient.

**Stage 3 (PeriodicPattern @njit kernel) status unchanged** — still deferred.

**Next biggest target** identified by profile: `_pre_filtering_info_num` is
~33% of `run_nb` time at n=200 and is the bottleneck at n=1000. It uses
already-vectorized numpy; further Numba gain there would require
restructuring the streaming chunk loop. Out of scope for this ADR.
