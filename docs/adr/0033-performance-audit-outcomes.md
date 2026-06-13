# ADR-0033 — Whole-library performance audit: accepted wins and rejected optimizations

Status: Accepted — 2026-06-13

## Context

A whole-library performance audit (June 2026) scored every public class / method
and the significant backend functions for speed and memory headroom — roughly 80
symbols across all subpackages. It produced a long candidate table that drove the
performance work: same-output optimization batches (#169, #172, #174, #175, #180,
#186), a committed benchmark suite (#187), and a numerical-equivalence tolerance
policy (#188 → ADR-0032).

Two findings shaped the whole effort and are the reason this ADR exists:

1. **The audit's percentage estimates were systematically unreliable.** Roughly half
   the high-`%` candidates delivered ~no measurable gain once actually benchmarked
   (the CPP core was already optimized; several "hot" loops were not hot, or were
   I/O- / numpy- / IPC-bound rather than Python-loop-bound).
2. **The largest *estimated* wins were output-changing** — they reorder floating-point
   reductions or change discrete decisions — so a same-output bar excludes them by
   construction; some are also simply not worth it once measured.

Without a record, these dead ends will be re-attempted. This ADR captures what we
accepted, and — the part that earns the ADR — **what we tried, rejected, and must
not redo**, with the reason for each.

## Decision

- **D1. Same-output first.** Ship an optimization only if its output is byte-identical,
  or — under ADR-0032 — numerically-equivalent (`np.allclose`, `atol=1e-10`) with every
  discrete decision (labels, selected features / medoids, kept / dropped sets) unchanged.
  Byte-identical is the default.
- **D2. Benchmark every candidate in isolation before claiming a win.** Audit estimates
  are not evidence. A candidate that benches ≤ ~1.1× is dropped and recorded below, not
  shipped. (Caveat: back-to-back in-process timing is contaminated — a heavy old-loop
  warps the next measurement; trust **isolated single-function** timings.)
- **D3. Output-changing optimizations go through ADR-0032's tiers** with a pinned
  regression anchor (the ADR-0015 pattern), as their own tier-declared PR.
- **D4. Measurement is committed** (#187 / #193): a benchmark + regression-guard suite
  over the hot entry points, so future perf work is data-driven rather than estimated.
- **D5. Per-function "×" figures are isolated micro-benchmarks, not end-to-end deltas.**
  Real impact is concentrated in window-sampling, structure-encoding, and bulk-fetch
  workflows; the common CPP feature-extraction path was already fast and is largely
  unchanged.

### Accepted wins (merged, each pinned by an equivalence test)

| Win | Module | Output | Micro-bench | PR |
|---|---|---|---|---|
| `filter_redundancy` / `filter_similarity_to_test` | AAWindowSampler | identical (integer ratio) | ~33× / ~35× | #169 |
| `_dist_to_medoids` (einsum row-wise Pearson) | AAclust (`select_proteins`) | identical | ~46× | #169 |
| `kullback_leibler_divergence_` (per-feature chunked) | dPULearn.eval | bit-exact | ~2.2× | #169 |
| `candidate_centers_` / `_scan_protein_` / `score_window_pwm_` | AAWindowSampler | identical | ~44× / ~12× / ~1.5× | #172 |
| `encode_one_hot` (vectorized scatter) | SequencePreprocessor | byte-identical | ~3× | #174 |
| `encode_pdb` CA-CA contact count | StructurePreprocessor | byte-identical | ~50× | #175 |
| `encode_pdb` alignment cache | StructurePreprocessor | byte-identical | ~12× | #183 |
| AlphaFold/UniProt pool + opt-in concurrency | data_handling_pro | identical data | wall-clock | #190 |
| `encode_disulfide`, `_plddt` reuse, DSSP aligner reuse | StructurePreprocessor | identical | 16–48× / 2.2× | #194 |
| `filter_correlation_`, redundancy `df_cor`→numpy, `_greedy_simplify_` copy-drop | feature_engineering | identical | ~3.3× / ~3.4× | #195 |
| `AAMut.comp_substitution_impact`, `get_sliding_aa_window` | protein_design / data_handling | identical | ~19× / ~1.8× | #196 |

Infrastructure: benchmark + regression suite (#193); tolerance policy ADR-0032 (#191).

The "×" above are **speed** micro-benchmarks. **Memory** was touched only where noted:
`_greedy_simplify_` drops a per-feature `X.copy()` (memory-only); the `encode_pdb`
alignment / contact caches trade a little memory for speed; the pro-IO path streams.
Most wins are pure compute (loop vectorization) with neutral memory.

## Performance relative to v1.0.3 (last released stable)

Two **distinct** efforts separate v1.0.3 from current `master` — do not conflate them:

1. **The CPP performance overhaul (post-1.0.3) — the order-of-magnitude gain, and NOT part
   of this sweep.** v1.0.3 and earlier predate the Cython feature-matrix kernel, macOS-safe
   threaded `n_jobs`, scale / AA-index caching, and scale / sample batching. Per the
   CHANGELOG, 1.0.3 could show **hour-long, low-CPU `CPP.run`**; the overhaul replaced that
   (v1.0.2 already noted "3–5× faster `CPP.run`"; the kernel + batching go well beyond),
   and CPP is now **memory-bounded** by streaming / chunking instead of materializing the
   full sample×scale×split tensor. This is the dramatic **efficiency *and* memory** gain
   versus 1.0.3.
2. **This same-output sweep (ADR-0033) — incremental.** Its wins sit mostly in **v1.1-new
   subsystems** (`AAWindowSampler`, `StructurePreprocessor`, embeddings) that have **no
   v1.0.3 counterpart**, plus targeted wins on pre-existing code (dPULearn KLD,
   `filter_correlation_`, AAclust `_dist_to_medoids`). Mostly speed; memory as noted above.

**No precise end-to-end ×-factor vs 1.0.3 is asserted here.** It would require benchmarking
1.0.3 against `master`, and most swept symbols did not exist in 1.0.3, so a fair comparison
is only meaningful on the shared core (`CPP.run`, `AAclust.fit`,
`SequenceFeature.feature_matrix`) — where the gain is the overhaul's, not this sweep's. The
committed benchmark suite (#193) measures current `master` going forward; 1.0.3 has neither
the suite nor the v1.1 surface, so any 1.0.3 number must come from an explicit benchmark run,
not an estimate (D2).

## Rejected alternatives — DO NOT RE-ATTEMPT

### (a) Benchmarked → no measurable gain (the audit over-estimated these)

| Candidate | Why it was dropped |
|---|---|
| FASTA writer `iterrows`→vectorized | I/O + string-format bound, **~1.0× even at 100k rows** |
| TreeModel CV parallelization | joblib spawn / IPC overhead — **~1.0× at fine *and* coarse grain** |
| `load_scales` / `load_dataset` caching | already `@lru_cache`d at the read layer |
| dPULearn AUC parallelization | `auc_adjusted_` already self-parallelizes by default |
| `encode_integer` | already fast (≈0.009 s / 5k seqs); not a hot path |
| `encode_pae` O(L²) loops | already numpy-bound per row — broadcasting was a wash |
| `_compute_centers` / `_position_aa_freq` | one-time / not hot |
| `comp_seq_cons` | marginal **and** unexported dead code (no caller/test) |
| `AAlogo.get_df_logo_info` | no single-call redundancy; caching = cross-method API change |
| `_retrieve_tmd_aligned` | `str.pad` measured **slower** than the `apply` it would replace |
| `_get_aa_window_odd/even` | string-slice bound, not append-bound |
| `SeqMut.build_scan_plan` | `iterrows`-bound; the append is a wash |
| `EmbeddingPreprocessor.encode_` streaming | streaming min/max equivalent but not faster; streaming quantile/sigmoid(std) ≠ full-corpus value (out of scope) |

### (b) Output-changing → disqualified, or correct but not worth it

| Candidate | Verdict |
|---|---|
| **AAclust binary-search `k`** | **Empirically disqualified** even under ADR-0032 T3. `min_cor(k)` is **non-monotonic** (KMeans clusters at `k` vs `k+1` are not nested → multiple threshold crossings), so binary search is fundamentally unsafe *and* no safe sublinear search exists. Measured: speedup **0.8×–4.1×, ≈1× on the real 586-scale case** (the audit's "35–50%" did not hold); `k` differed in 8/9 cases; **medoid-scale Jaccard as low as 0.14** (~86% of representative scales change). This is the tolerance policy *working* — it blocked a tempting-but-wrong change. |
| ShapModel 4D → rolling-mean | Correct T2 (running mean = same numbers, `allclose`) but **memory-only**: SHAP/KernelExplainer compute dominates wall-clock by orders of magnitude → no measurable time win → fails the "benchmark shows the win" gate. |
| Sparse one-hot encoding | A return-dtype/structure change (dense → sparse) — an **API change**, not a numerical-equivalence question; risky for a public encoder. |
| `_build_base_matrix_` → `get_feature_matrix_fast_` | The fast path uses a **float32** scale matrix vs float64 here → not byte-identical (drift). |
| `_adjust_non_canonical_aa` vectorization | Narrowing the char scan changes the regex set; the "remove" branch can hit a regex-metacharacter edge → not safely identical. |
| `_compute_medoids` correlation-once | Full-matrix `corrcoef` vs per-cluster reorders sums → can flip the `argmax` (different medoid). |
| `dPULearn.get_neg_via_pca` reorder | Vectorizing the per-PC selection can flip tie-breaks → different identified negatives. |
| `SeqMut` float32 ΔX | dtype change → different output. |

## Consequences

- Future performance work **consults this ADR (and ADR-0032) first** and does not
  re-open the rejected candidates without new evidence.
- Any output-affecting optimization must declare its ADR-0032 tier and add a
  regression anchor (ADR-0015 pattern).
- New same-output candidates are gated by D2 (isolated benchmark) before a PR.

## Out of scope (still open, genuine same-output wins)

Tracked in #186, not yet implemented in any PR:

- `_eligible_candidates_` (CPP `_simplify.py`) — drop the per-feature `.items()` over the
  ~586-scale `df_cor` Series; hoist `interp`/`cor` to numpy. ~66× (the standout).
- `encode_pdb` per-entry coordinate/residue cache — one structure walk + best-chain pick
  shared across the ~13 encoders; needs golden tests on the P1/P2/AF_TINY fixtures. ~12×.
