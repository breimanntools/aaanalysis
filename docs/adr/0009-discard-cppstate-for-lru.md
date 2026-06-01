# ADR-0009 — Reuse scale tensors via an internal content-hash LRU, not a public `CPPState`

Status: Accepted — 2026-05-31

## Context

`CPP.__init__` rebuilds the scale lookup / float64 scale-matrix on every construction.
In a sweep where many configurations share the same `(df_scales, df_cat)`, this rebuild
is pure waste. The feedback proposed a user-facing `CPPState(df_scales, df_cat)` object
passed into `CPP(state=...)` to cache the heavy bits across instances.

## Decision

**D1 — An internal module-level LRU** (`functools.lru_cache`, `maxsize=32`) keyed on a
**content hash** of `df_scales` (`_ScalesKey`: columns + index + `values.tobytes()`),
built once and reused across every `CPP` that sees an equal `df_scales` — including
`CPPGrid` combos. No public type, no constructor path to thread through.

**D2 — `CPP.clear_cache()` classmethod** to evict the process-wide cache explicitly for
long-running processes cycling many distinct scale sets.

## Rejected alternatives

- **A public `CPPState` object + `CPP(state=...)`** (the feedback's proposal, which the
  maintainer explicitly disliked). It adds a new public type and an alternate
  construction path users must learn and thread through; two distinct-but-equal
  `df_scales` objects would also miss the cache. The content-hash LRU delivers the same
  reuse transparently — equal content hits the cache regardless of object identity —
  with zero API surface.

## Consequences

Scale-tensor reuse is automatic across `CPP` instances and `CPPGrid` configurations that
share a scale set; the only public addition is the `clear_cache()` escape hatch.
