# ADR-0007 — `CPPGrid`: a class that sweeps across configurations on a threads-default backend

Status: Accepted — 2026-05-31

## Context

A heavy user runs CPP sweeps at the scale of "13 scales × 15 JMD × 7 n_features ×
4 classifiers × 3 strategies = 16 380 configs". Two costs dominated: (1) joblib
re-creates a process pool per `CPP.run`/`feature_matrix` call, and (2) **dataframe
serialization** to workers on every call. The feedback asked for a `CPP.run_many(...)`
entry point backed by a *persistent `loky` process pool*.

## Decision

**D1 — A public `CPPGrid` class (Tool template), not a `CPP.run_many` classmethod.**
A class is idiomatic here (the library is class-based; `CPP` itself is a `Tool`),
owns the pool/cache lifecycle, and gives `.eval` an obvious future home for the
compare/select-best harness.

**D2 — Parallelize *across* configurations; each config runs serially** (`n_jobs=1`),
avoiding nested oversubscription.

**D3 — Default `backend="threads"`** (loky opt-in). Threads share `df_seq`/`df_scales`
in-process → **zero serialization** (the actual measured bottleneck), and sidestep
the Python 3.14 / macOS `__main__`-guard spawn footgun on the reporter's own platform.
The inner compute is GIL-releasing numpy/Cython.

**D4 — Four stage-grouped param dicts** (`params_parts`/`params_split`/`params_scales`/
`params_cpp`), each feeding exactly one pipeline stage; a `list` value is a swept axis
(Cartesian product); `df_cat` is resolved internally from each `df_scales`.

**D5 — Lightweight, reconstructable `df_params`** (object axes as position index, scalar
axes literal) plus `n_warnings`/`n_errors` counts derived from `last_filter_stats`.

**D6 — Smart sweeping: never re-run CPP for `n_filter`, cache parts/splits.**
Configurations that differ *only* in `n_filter` run CPP **once at the largest** value;
smaller values are exact `head(n)` slices. This is correct because CPP's redundancy
filter is a **greedy top-down pass over a candidate superset**, so the kept top-`n` is
invariant to the (n_filter-scaled) pre-filter size — verified byte-identical to
independent runs. `df_parts` are built once per parts-config and `split_kws` once per
split-config (neither depends on `df_scales`/filter knobs/`n_filter`), then reused across
the grid; the D3 content-hash scale LRU covers `df_scales` reuse. Measured ~14× on a
15-value `n_filter` sweep. Per-combo `n_warnings` uses `n_after_redundancy` (not the
max-run's `n_final`) as the shortfall basis, so a sliced member warns exactly as an
independent run at its own `n_filter` would.

## Rejected alternatives

- **`CPP.run_many` classmethod.** A lone classmethod is a poor fit in a class-based
  library and has no natural home for the downstream comparison harness.
- **Persistent `loky` process pool as the default** (the feedback's instinct). It
  amortizes pool *creation* but still re-serializes `df_scales`/`df_parts` to every
  worker on each dispatch — the very cost the user flagged — and its process spawn is
  what triggers the Py3.14/macOS footgun. Threads fix the real cause for free; loky
  stays an opt-in for genuinely pure-Python-bound configs.
- **A single flat `param_ranges` dict** keyed by literal parameter names. Workable, but
  the four stage-grouped dicts remove all routing ambiguity (which knob feeds which
  stage) at no cost.
- **Live `warnings.catch_warnings` capture for `n_warnings`.** Not thread-safe (it
  mutates global filter state); concurrent combos under the threads default would
  corrupt counts. Derived deterministically from `last_filter_stats` instead.

## Out of scope

The multi-seed CV / select-best-by-objective "comparison harness" stays an example /
tutorial; only the generalizable sweep *runner* is upstreamed.
