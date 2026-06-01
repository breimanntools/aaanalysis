# ADR-0008 — Unified `n_jobs` contract: `None` means optimized

Status: Accepted — 2026-05-31

## Context

The documentation claimed `n_jobs=None` was "optimized automatically", but
`check_n_jobs` passed `None` straight through to joblib, which treats it as a single
worker — so a user typing `n_jobs=None` for "library default" got **zero speedup** (a
silent no-op). Only `n_jobs=-1` engaged all cores. A heavy user measured the resulting
1× and flagged the gap. The fix changes observable behavior, so it warrants a record.

## Decision

**D1 — One contract everywhere:** `1` = serial, `-1` = all cores (`os.cpu_count()`),
`N>1` = exactly N, `None` = **optimized** via `ut.resolve_n_jobs(n_jobs, n_work)` =
`min(cpu_count, max(n_work // 10, 1))` — small jobs stay serial to avoid spawn overhead.

**D2 — A global `options['n_jobs']`** (default `"off"`) overrides the per-call value when
set, mirroring `verbose` / `random_state`.

**D3 — Centralize the formula.** The three hot paths (two CPP feature-matrix builders +
`_utils/metrics.py:auc_adjusted_`) route through the single `resolve_n_jobs`.

## Rejected alternatives

- **`None → -1` (all cores).** Over-parallelizes tiny jobs, paying spawn/serialization
  overhead that exceeds the work; the `n_work`-scaled `resolve_n_jobs` avoids this.
- **`None → 1` plus a louder docstring only.** Leaves the no-op in place and just warns
  about it — the feedback explicitly wanted the optimized behavior.
- **In-library detection of the missing `__main__` guard** (the Py3.14/macOS spawn
  footgun). Too magic / fragile; addressed with a loud docs note instead.

## Consequences

`None` now does real work (a behavior change, semver-relevant). The `__main__`-guard
spawn footgun on Python 3.14 / macOS remains the user's responsibility, documented in
the `n_jobs` docstrings and the parallelism tutorial note.
