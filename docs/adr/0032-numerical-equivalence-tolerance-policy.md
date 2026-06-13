# ADR-0032 — Numerical-equivalence tolerance policy for output-affecting optimizations

Status: Accepted — 2026-06-13

## Context

The library-wide performance effort (#169, #172, #174, #175, #180, #183) held a
strict **byte-identical / same-output** bar: an optimization shipped only if its
output was numerically identical *and* every discrete decision (labels, selected
features/medoids, kept/dropped sets) was unchanged. That bar was the right
default and shipped a series of safe wins, but it **structurally excludes the
largest remaining algorithmic wins** — ones that are statistically equivalent
yet not bit-identical:

- **AAclust binary-search `k`** — the single biggest estimated win in the audit
  (~35–50 % off the dominant clustering loop); changes which `n_clusters` is
  found in edge cases.
- **Sparse one-hot encoding** — changes the output dtype/structure.
- **`ShapModel` 4D → rolling-mean aggregation** — reorders sums (ULP drift).
- Vectorized correlation/distance reformulations that are `allclose` but not
  bit-identical.

There was no documented policy for *when* an output-affecting optimization is
acceptable or *what evidence* lets it land, so these were simply blocked. This
ADR defines that policy. It does not itself change any numerical output — it sets
the bar each future optimization PR must clear.

## Decision

**D1 — Three equivalence tiers, strictest first.** Every optimization declares
the tier it lands under (in its PR description). A change must land at the
**strictest tier it can satisfy**; reaching for a looser tier than necessary is
rejected at review.

| Tier | Definition | When it applies |
|---|---|---|
| **T1 — Byte-identical** | Output is bit-for-bit identical to the prior implementation on the equivalence dataset(s). | The current default — vectorizing a loop with the same formula, hoisting an invariant, caching a deterministic result. |
| **T2 — Numerically-equivalent** | `np.allclose(new, old, atol=1e-10, rtol=0)` on all numerical outputs **AND** identical *discrete decisions* (labels, selected features/medoids, kept/dropped/ranked sets). | ULP-level reorderings (einsum/BLAS reductions, rolling-mean aggregation) and `allclose` correlation/distance reformulations that leave every decision intact. |
| **T3 — Statistically-equivalent** | Outputs differ and discrete decisions *may* differ, but documented quality metrics (clustering quality, downstream AUC, kept-feature overlap) stay within an agreed band on canonical datasets. | Reserved for genuinely algorithmic changes (binary-search `k`). Not a fallback for a change that *could* meet T2. |

**D2 — Required evidence per tier.** A PR may not claim a tier without the
evidence for it, captured in a throwaway validation harness (gitignored
`dev_scripts/`, the established `perf_*_validate.py` pattern — inline the
original impl, assert equivalence, benchmark old-vs-new) **and** distilled into a
committed regression anchor (D3):

| Tier | Equivalence test | Dataset(s) | Tolerance | Committed anchor |
|---|---|---|---|---|
| **T1** | exact array/object equality (`==`, `array_equal`) | the optimization's own unit fixtures | none (exact) | existing unit/parity tests suffice; no new anchor required |
| **T2** | `np.allclose(atol=1e-10, rtol=0)` on values **+** equality on every discrete-decision artifact | a canonical bundled slice (`DOM_GSEC` or the function's domain fixture) | `atol=1e-10, rtol=0` | a frozen anchor on the decision artifact (feature/medoid identity, label vector) |
| **T3** | documented quality metric(s) within band | the **named canonical dataset(s)** the band was agreed on | the **agreed band**, stated numerically in the PR + anchor (e.g. ΔAUC ≤ 0.005, kept-feature Jaccard ≥ 0.95) | a frozen anchor pinning the metric to its band on the canonical cell |

**D3 — Anchors extend the ADR-0015 pattern; they do not replace its CPP
anchor.** Every T2/T3 optimization adds (or extends) an exact-value regression
anchor following ADR-0015: `@pytest.mark.regression`, `skipif` off the canonical
cell (Linux + floor Python; `AAA_RUN_REGRESSION=1` forces it locally), frozen
values captured once and re-frozen **only** on an intentional, reviewed change.
A T2 anchor freezes the discrete-decision artifact and an `atol=1e-10` value; a
T3 anchor freezes the quality metric and asserts it stays within the documented
band. Anchors run in the **non-gating nightly** (`mutation_nightly.yml`), never
in the blocking matrix — exact/banded values are canonical-cell-specific and a
3rd-decimal platform drift must not block merges (ADR-0015 D2).

**D4 — One optimization per PR, tier declared.** Each previously-excluded
optimization (AAclust binary-search `k`, sparse one-hot, `ShapModel` rolling
aggregation, …) lands in **its own PR** that (a) names its tier, (b) links its
validation harness output and benchmark, and (c) commits its anchor. The
reviewer acceptance checklist lives in `CONTRIBUTING.rst` (and the
`agentic_engineering.md` quality gates).

## Rejected alternatives

- **Keep the byte-identical-only bar.** Simplest and safest, but permanently
  forecloses the biggest algorithmic wins (binary-search `k`) for no reason
  other than the absence of a policy. Rejected — the wins are real and the risk
  is controllable with a regression anchor.
- **A single global tolerance (`atol`) for all changes.** One number cannot
  serve both a ULP reordering (wants tight `atol`, identical decisions) and an
  algorithmic change (wants a *metric band*, accepts different decisions).
  Collapsing them either blocks T3 or silently waves through decision changes
  under a value tolerance. Rejected in favour of explicit tiers.
- **Allow T3 changes without a committed anchor (benchmark + review only).**
  A one-time review cannot catch *future* drift; the whole point of unlocking an
  algorithmic change is that its quality must stay banded forever. Rejected —
  every T2/T3 change carries a committed anchor (D3).
- **Let a change pick the loosest tier it merely passes.** A T1-capable change
  declared as T3 loses signal — a later real regression hides inside the band.
  Rejected: land at the strictest tier the change satisfies (D1).

## Consequences

- The excluded queue (AAclust binary-search `k`, sparse one-hot, `ShapModel`
  rolling aggregation) is unblocked, each as its own tier-declared PR (D4).
- The nightly accumulates one anchor per landed T2/T3 optimization alongside the
  ADR-0015 CPP anchor; all run on the canonical Linux/floor-Python cell and are
  re-frozen only on intentional, reviewed change.
- `CONTRIBUTING.rst` / `CONTRIBUTING_COPY.rst` carry the reviewer acceptance
  checklist; `docs/guides/agentic_engineering.md` references this ADR from its
  quality gates; `CONTEXT.md` defines the three tier terms.

## Out of scope

- The same-output optimization queue (#180, #186 same-output items) — those land
  under T1 with no policy change.
- API / behavior changes unrelated to numerical equivalence.
- Building the benchmark-suite infrastructure (companion perf issue); this ADR
  governs *equivalence evidence*, not how throughput is measured.
