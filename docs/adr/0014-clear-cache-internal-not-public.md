# ADR-0014 — Demote cache eviction to an internal utility; drop public `CPP.clear_cache()`

Status: Accepted — 2026-06-03

Supersedes (in part): ADR-0009 (its **D2** — the public `CPP.clear_cache()`
classmethod). ADR-0009's **D1** (the internal content-hash LRU) stands unchanged.

## Context

ADR-0009 added an internal content-hash LRU (`maxsize=32`) for the float64
scale-matrix and exposed a public `CPP.clear_cache()` classmethod (D2) to evict
it. In practice the eviction knob is almost never needed: the LRU is
**self-bounding** — once 32 distinct `df_scales` have been seen it evicts the
oldest automatically, and correctness never depends on eviction (the cache is
keyed by content, so a changed `df_scales` is simply a new key, never a stale
hit). The only real use is freeing held matrices *eagerly* in a long-running
process that churns through many distinct scale sets — a power-user / memory
concern, not an everyday operation. Exposing it publicly invites the question
"when do I call this?" with the honest answer "you almost never do."

The classmethod is **unreleased**: it is `versionadded:: 1.1.0` and 1.1.0 has
not shipped (current PyPI is 1.0.x; next release is 1.1.0 per ADR-0010). So
removing it now carries **no deprecation cost** — the same window ADR-0011 used.

## Decision

**D1 — Remove the public `CPP.clear_cache()` classmethod.** The eviction
capability remains as the existing module-level internal utility
`clear_scale_lookup_cache()` in
`feature_engineering/_backend/cpp/_filters/_get_feature_matrix_fast.py`, which
tests and power users can call directly. No public API surface for it.

**D2 — Do not auto-evict on any CPP event.** The LRU's `maxsize` *is* the
automatic bound. An automatic clear (e.g. at end of `run`) would evict exactly
the tensors the cache exists to reuse across `CPPGrid` sweeps and repeated
constructions, defeating ADR-0009 D1.

## Rejected alternatives

- **Keep `CPP.clear_cache()` public.** Harmless but adds a knob users cannot
  reason about; since it is unreleased there is no cost to removing it now.
- **Auto-clear on a trigger.** Rejected per D2 — undermines the cache.

## Consequences

- Public API of `CPP` loses `clear_cache` (no semver impact; unreleased).
- `clear_scale_lookup_cache()` stays as the internal escape hatch; its tests
  (`test_scale_lookup_cache.py`) are repointed from `aa.CPP.clear_cache()` to
  the utility, and the former `TestCPPClearCache` class is renamed
  `TestClearScaleLookupCache`.
- Backend docstrings that referenced `:meth:`CPP.clear_cache`` now point at the
  internal utility.
