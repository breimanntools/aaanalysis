# ADR-0055 — CPP bootstrap / stability annotation

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
2. **Whether bootstrap changes the selection or only annotates it.** Resolved in favour of a thin
   *annotate* wrapper (D2): bootstrap reports a per-feature `selection_frequency` on the ordinary
   full-data run; it does not change which features are selected (`n_filter` stays the criterion).

## Decision

**D1. A boolean gate turns the mode on; the tuned config carries good defaults; the output *size*
stays per-run.** A boolean `bootstrap` (default `False` = off) is the switch, in front of three tuned
constructor parameters — `n_bootstrap` (default `20`), `resample` (`"reference"` default / `"both"` /
`"test"`), and `bootstrap_frac` (default `0.8`). `bootstrap=True` applies them; `bootstrap=False`
ignores them and runs the single pass. Separating the *switch* (a bool) from the *round count*
(`n_bootstrap`) is deliberate: with a single `n_bootstrap=0`-means-off integer the tuned value (`20`)
could never be the default (the default must be the off state), so the good number would live only in
the user's memory. `n_filter` (the final output size) remains a per-call argument. "How to select
robustly" is object config; "how many to keep" is per call.

**D2. Bootstrap is a thin wrapper that ANNOTATES the ordinary run — it does not change the
selection.** With `bootstrap=True`, `run` / `run_num` / `run_composit` loop the *ordinary* selection
on `n_bootstrap` bootstrap resamples of the rows (via fresh `CPP(bootstrap=False)` instances, so it
literally re-uses the public run methods rather than a parallel backend pipeline) to tally each
feature's `selection_frequency = count / n_bootstrap`, then return the **ordinary full-data run** with
a `selection_frequency` column appended after `positions`. The selected feature list is exactly that
of a non-bootstrap run (`n_filter` is the selection criterion); bootstrapping adds the per-feature
stability annotation, it does **not** restrict the candidate pool or change which features are
selected. Resampling is **with replacement** per group at `round(bootstrap_frac · n_group)` rows;
`resample` chooses which group(s) are resampled and which pass through unchanged.

**D3. Default is byte-identical.** `bootstrap=False` keeps the existing single-pass dispatch untouched,
so the default output (and the perf A/B output digest) is unchanged; the feature is purely additive
and opt-in.

**D4. Reproducibility.** The resampling RNG is seeded from the constructor `random_state`
(`np.random.default_rng`), so a fixed `random_state` reproduces the frequencies bit-for-bit;
`random_state=None` is truly random, per the package contract.

**D5. Scope + interaction.** The wrapper is method-agnostic and is wired into `run`, `run_num`, **and**
`run_composit`. It is **not** combinable with the memory-batching modes (`n_batches` /
`n_sample_batches`) — each round runs a full single pass — and that combination raises a `ValueError`.
The redundant `run_aac` convenience method (a pure `run_composit(composition="aac")` alias, added in
the same unreleased cycle) is **removed** here — `run_composit` covers it.

## Rejected alternatives

- **A dedicated backend pipeline that changes the selected list (candidate-restriction / stability
  selection proper).** An earlier draft made bootstrap a candidate generator: candidates = features
  ever selected across rounds, then a full-data filter cuts to `n_filter`, producing a
  stability-*informed* (different, more reproducible) list. Rejected in favour of the thin annotate
  wrapper: the maintainer wanted bootstrap to be a simple wrapper over the existing run methods, not a
  parallel "method for itself." **Accepted trade-off:** the annotate design does **not** make the
  feature *list* more reproducible (the list equals a normal run) — it reports which of the selected
  features are reproducible. Making the list itself more robust would require the rejected
  candidate-restriction.
- **A `selection_frequency` cut-off (a `min_freq` threshold, or a top-N-by-frequency count) as the
  feature selector.** Rejected: `n_filter` (the ordinary filter) is the selection criterion. A
  frequency threshold empirically **over-prunes** (sweep below), and a top-N-by-frequency count is
  dataset-size-dependent.
- **`n_bootstrap=0`-means-off as the only switch (no `bootstrap` bool).** Rejected: it forces the
  round count and the on/off state to share one parameter, so the tuned count (`20`) can never be the
  default. A boolean gate lets the *on* state carry the good defaults.
- **Sub-sample without replacement (Meinshausen–Bühlmann stability selection).** A defensible
  alternative matching `bootstrap_frac<1` literally, but the maintainer chose classic
  bootstrap-with-replacement per group.
- **Round-averaged statistics.** Rejected: the returned stats are the normal full-data run's; only
  `selection_frequency` comes from the rounds.
- **Bootstrap parameters per `run*` method.** Rejected: triplicates the already-duplicated
  selection-filter block and cannot reuse the constructor `random_state`.

## Empirical validation (DOM_GSEC / gamma-secretase)

Measured on the bundled `DOM_GSEC` dataset (80 sequences, 40 substrate / 40 non-substrate; 50 scales;
`resample="reference"`).

- **Motivation — single-pass selection is sample-fragile.** Two single-pass CPP selections on two data
  resamples overlap by a Jaccard of only **~0.06**: a different sample gives a largely different list.
  Knowing *which* of a run's selected features are reproducible under resampling is therefore useful —
  which is exactly what `selection_frequency` reports.
- **The list does not change; the annotation does.** Because bootstrap returns the ordinary full-data
  run, the selected features are deterministic (identical across `random_state`s). What varies with
  more rounds is the *precision* of the `selection_frequency` estimate — ~20 rounds is a practical
  sweet spot, ~50 converges it; cost is ~linear in `n_bootstrap`. `bootstrap_frac=0.8` is the
  conventional sub-sample size and a robust default.
- **Why frequency is reported, not thresholded.** A prototype `min_freq` threshold was swept at
  `n_bootstrap=20` (threshold / features kept / cross-seed set Jaccard): `0.0` -> 100 / 0.58,
  `0.2` -> 86 / 0.54, `0.3` -> 48 / 0.46, `0.5` -> 8 / 0.27, `0.7` -> 3 / 0.25, `≥0.8` -> 0. A strict
  threshold collapses CPP's distributed signal to a handful of *borderline* features that are **less**
  reproducible — so frequency is annotated, never used to filter.
- **Shipped defaults:** `bootstrap=True` with `n_bootstrap=20`, `bootstrap_frac=0.8`,
  `resample="reference"`.

## Consequences

- `df_feat` gains an optional `selection_frequency` column (registered in `DICT_DF_FEAT` as
  non-required; appended by `sort_cols_feat` after `positions`), present only when `bootstrap=True`.
  The other columns and the selected rows are byte-identical to a normal run.
- Cost scales ~`n_bootstrap ×` a single run (each round constructs a fresh `CPP` and calls the public
  run method on a resample; rounds run serially with the existing inner `n_jobs` parallelism).
- Bootstrap annotates **interpretability / trust**, **not** the list's robustness or predictive
  accuracy — the docstrings say so, and mandate fold-internal use to stay leakage-safe. Complementary
  to ADR-0054 (marginal-by-design filter; joint effects belong to the downstream model).
- `run_aac` is removed (redundant alias of `run_composit(composition="aac")`).
