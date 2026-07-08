# ADR-XXXX — CPP bootstrap / stability feature selection

Status: Accepted — 2026-07-08

## Context

`CPP.run` / `CPP.run_num` do a **single-pass** univariate selection: score every candidate feature
by its marginal statistic (adjusted AUC, mean difference) on the one training sample, pre-filter,
then greedily reduce redundancy to `n_filter`. The selected list is therefore a function of the
specific sample. A resampling study (the Federle 10-benchmark analysis referenced on issue #368)
found that single-pass selection is **unstable** — the mean pairwise Jaccard between selections on
bootstrap resamples is only ~0.05–0.22 — so a single feature list over-interprets sampling noise.
The same study found that resampling **only the reference/negative group** (fixing the positives) is
markedly more stable than resampling both, and that stability selection buys **reproducibility, not
accuracy** (a strict "stable-only" threshold actually hurts, because CPP signal is distributed over
~150 weak features).

Issue #368 asks for an opt-in stability-selection mode that resamples, re-selects, ranks features by
**selection frequency**, and exposes that frequency, without changing the default output.

Two design questions were settled with the maintainer before implementation:

1. **Where the mode lives.** `run` and `run_num` (and the newer `run_composit`) already duplicate the
   whole selection-filter parameter block; adding bootstrap parameters to each method would
   triplicate an already-duplicated block. Bootstrapping is a *cross-cutting selection behaviour*,
   and it needs the RNG seed that already lives on the constructor (`random_state`).
2. **What bootstrap actually decides.** Bootstrapping should select *stable candidates*; the **full
   dataset stays authoritative** for statistics and filtering. A feature that looks good in a
   subsample but fails the full-data `max_std_test` or redundancy check must still be dropped.

## Decision

**D1. The bootstrap *mode* is configured on `CPP.__init__`; the output *size* stays per-run.**
Four constructor parameters — `n_bootstrap` (default `0` = off), `resample`
(`"reference"` default / `"both"` / `"test"`), `bootstrap_frac` (default `0.8`), and `n_freq`
(default `50`) — configure stability selection once, and `run` / `run_num` apply it uniformly.
`n_filter` (the final output size) remains a per-call argument. "How to select robustly" is object
config; "how many to keep" is per call.

**D2. Bootstrap is a thin wrapper (candidate generator), not a new algorithm.** With
`n_bootstrap>0`, `run` / `run_num` route to `cpp_run_bootstrap`, which:

- **Phase 1 (resampled candidate generation).** Repeat the existing single-pass selection
  (`cpp_run_single`) `n_bootstrap` times, each on a bootstrap resample of the rows — **sampled with
  replacement** per group at `round(bootstrap_frac · n_group)` rows; `resample` chooses which
  group(s) are resampled and which are passed through unchanged. Tally how often each feature is
  selected; `selection_frequency = count / n_bootstrap`. The `n_freq` most-frequently-selected
  features (union across rounds) are the stable candidate set.
- **Phase 2 (full-data authoritative).** Recompute statistics for the candidates on the **complete**
  test + reference set, then let CPP's own filters decide the output: the `max_std_test` pre-filter
  threshold and the redundancy filter (`max_overlap` / `max_cor` / `n_filter`). The result is ordered
  by `abs_auc` exactly like a normal run and carries an extra `selection_frequency` column appended
  after `positions`.

**D3. Default is byte-identical.** `n_bootstrap=0` keeps the existing single-pass dispatch untouched,
so the default output (and the perf A/B output digest) is unchanged; the feature is purely additive
and opt-in.

**D4. Reproducibility.** The resampling RNG is seeded from the constructor `random_state`
(`np.random.default_rng`), so a fixed `random_state` reproduces the frequencies and the output
bit-for-bit; `random_state=None` is truly random, per the package contract.

**D5. Scope + interaction.** Bootstrap is wired into `run` and `run_num` (the positional-selection
methods the evidence is about). It is **not** combinable with the memory-batching modes
(`n_batches` / `n_sample_batches`) — each bootstrap round runs single-pass — and that combination
raises a `ValueError`. `run_composit` / `run_aac` (a different, composition feature type) are out of
scope for this change; the constructor placement future-proofs wiring them later.

## Rejected alternatives

- **Sub-sample without replacement (Meinshausen–Bühlmann stability selection).** A defensible
  alternative that matches `bootstrap_frac<1` literally, but the maintainer chose classic
  bootstrap-with-replacement per group; `bootstrap_frac` scales the per-group draw size.
- **Rank the final output by `selection_frequency`.** Rejected: frequency is a *stability filter*
  (which candidates survive), not the ranking key. The output is ordered by `abs_auc` like a normal
  run so `df_feat` composes unchanged with downstream models and plots.
- **Round-averaged statistics.** Rejected: reporting stats averaged over the resampled subsets would
  diverge from a normal run's semantics. The full dataset is authoritative for every statistic and
  every filter; only `selection_frequency` comes from the rounds.
- **Bootstrap parameters per `run*` method.** Rejected: triplicates the already-duplicated
  selection-filter block and cannot reuse the constructor `random_state`.

## Empirical validation (DOM_GSEC / gamma-secretase)

Measured on the bundled `DOM_GSEC` dataset (80 sequences, 40 substrate / 40 non-substrate; 50
scales; `resample="reference"`). "Run-to-run Jaccard" is the Jaccard overlap of the selected feature
set between two different `random_state`s at the same setting — the direct measure of how
reproducible the selection is. Higher is more stable.

- **Single-pass selection is highly unstable.** Two single-pass CPP selections on two data resamples
  overlap by a Jaccard of only **0.06** — a different sample gives a largely different feature list.
- **Bootstrapping stabilises monotonically with rounds** (per-round cost is linear, ~constant per
  round):

  | n_bootstrap | run-to-run Jaccard | wall-clock |
  |---|---|---|
  | 5  | 0.22 | 1x |
  | 10 | 0.31 | 2x |
  | 20 | 0.53 | 4x |
  | 50 | 0.66 | 10x |

  Even 5 rounds already lifts reproducibility ~4x over single-pass; 20 rounds is a practical
  stability/cost knee (~9x the single-pass overlap at 4x the cost of 5 rounds).
- **`bootstrap_frac=0.8` is the empirical optimum.** Sweeping the per-group resample fraction at
  `n_bootstrap=20`, run-to-run Jaccard **peaks at 0.8** (0.53) and *declines* on either side
  (0.5 -> 0.27, 0.7 -> 0.41, **0.8 -> 0.53**, 0.9 -> 0.49, 1.0 -> 0.42): too small a fraction starves
  each round of data, while a full resample makes the rounds too similar to average out sampling
  noise. This validates the chosen default.
- **Optimum for this dataset:** `n_bootstrap` around 20, `bootstrap_frac=0.8`, `resample="reference"`
  — the shipped defaults (bar `n_bootstrap`, which is off by default and the user opts into).

## Consequences

- `df_feat` gains an optional `selection_frequency` column (registered in `DICT_DF_FEAT` as
  non-required; appended by `sort_cols_feat` after `positions`), present only when `n_bootstrap>0`.
- Cost scales ~`n_bootstrap ×` a single run (rounds run serially with the existing inner `n_jobs`
  parallelism over scales); ~20–50 rounds is enough for the top-150 ranking to converge.
- Stability selection improves reproducibility/interpretability, **not** predictive accuracy — the
  docstrings say so, and mandate fold-internal use to stay leakage-safe. It is complementary to
  ADR-0054 (marginal-by-design filter; joint effects belong to the downstream model).
