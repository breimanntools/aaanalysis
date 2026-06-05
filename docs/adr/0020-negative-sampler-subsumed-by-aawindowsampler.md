# ADR-0020 — Issue #66 `NegativeSampler` is subsumed by `AAWindowSampler`

Status: Accepted — 2026-06-05

## Context

Issue #66 (`prio:1`, `type:feature`, topic:data) asked for a general-purpose
`NegativeSampler` class: same-/different-protein and synthetic sampling, an
explicit N/U/Control **role** taxonomy, a unified output schema, a composable
filter pipeline, per-call seeds, and a `sample_benchmark_set` multi-arm
orchestrator.

By the time #66 was scheduled, `AAWindowSampler` (`seq_analysis/_aa_window_sampler.py`,
`.. versionadded:: 1.1.0`, **unreleased** — the package is at 1.0.3) already
shipped the substance of that request: `sample_same_protein` (role `Negative`),
`sample_different_protein` (role `Unlabeled`), `sample_synthetic` (role
`Control`), `sample_motif_matched`, the `ut.ROLE_*` taxonomy, the eight-column
`segments` schema, distance-band / context / motif / identity filters that
compose in a documented order, and per-call `seed` + constructor `random_state`.
A gap analysis against #66's task list found only two genuine, additive gaps:
`sample_benchmark_set` and a `custom_filter` hook. Everything else was either
already present (`treat_as` ≡ the existing `role=` kwarg; `provenance` ≡ the
existing `strategy` column) or deliberately deferred (structure filters, YAML
round-trip, a `SamplingFilters` dataclass, markov synthetic, cpp/embedding
similarity, a separate `df_pos` table).

Building `NegativeSampler` as specified would have created a second public class
whose `sample_same_protein` / `sample_different_protein` / `sample_synthetic`
surface is near-identical to `AAWindowSampler`'s, forcing a permanent
"which one do I use?" question on every user.

## Decision

**D1 — No `NegativeSampler` class.** `AAWindowSampler` is the negative/reference
sampler. #66 is resolved by adding its two genuine gaps directly to
`AAWindowSampler`, then closing #66 as solved.

**D2 — Add `sample_benchmark_set(df_seq, arms, seed)`.** `arms` is a
`Dict[name -> {"method": <strategy-tag>, **kwargs}]` over the four strategy
tags; it runs each arm with a deterministic per-arm sub-seed
(`np.random.SeedSequence`), concatenates the `segments` outputs, and adds an
`arm` column. No automatic cross-arm dedupe — every row is preserved.

**D3 — Add a constructor-level `custom_filter`.** A
`(window, entry, source_position) -> bool` keep-predicate that composes in the
per-window iterative filter pipeline of every `sample_*` method; the sanctioned
escape hatch for the deferred structure/domain decoy rules.

## Rejected alternatives

- **New `NegativeSampler` that wraps/composes `AAWindowSampler`** (the original
  kickoff recommendation). Rejected: two public classes with the same three
  `sample_*` methods is a lasting source of user confusion, and — because
  `AAWindowSampler` is unreleased — there is no stable-API reason that forced a
  wrapper rather than evolving the class in place.
- **Rename `AAWindowSampler` → `NegativeSampler`.** Rejected: "window/segment
  sampling" is the broader, accurate identity (`sample_motif_matched` is not
  negative sampling), and the name is already woven through CONTEXT.md, tests,
  and sibling issues.
- **Add a `provenance` column / `SamplingFilters` dataclass / `treat_as`
  kwarg** verbatim from the issue. Rejected as redundant: `strategy` + `arm`
  already carry provenance; the filters are already exposed as method kwargs;
  `role=` already does what `treat_as` would.

## Consequences

- #66 is closed as solved (pending maintainer sign-off), with the deferred
  items (structure filters, cpp/embedding similarity, markov, YAML) recorded
  as out-of-scope / future follow-ups rather than silently dropped.
- The `arm` column is added to the segments output only via
  `sample_benchmark_set`; `ut.COLS_SEGMENTS` (the eight-column schema) is
  unchanged, so existing schema assertions still hold.
- Reversing this — shipping a real `NegativeSampler` later — would be a new
  public class (semver-additive), not a breaking change, so the decision is
  recoverable if a distinct config-first frontend is ever wanted.
