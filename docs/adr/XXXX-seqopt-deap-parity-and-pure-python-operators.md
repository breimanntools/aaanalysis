# ADR-XXXX — SeqOpt: full pure-Python EA operator set + DEAP parity (ship ours)

Status: Proposed — 2026-06-25

## Context

ADR-0043 introduced `SeqOpt` (pro, SHAP-guided multi-objective directed evolution) with a
pure-Python NSGA-II core, and laid out a **parity-first** plan: prototype against DEAP, reimplement,
and decide ship-ours-vs-depend-on-DEAP from a comparison. That first PR deliberately **deferred**
the DEAP-dependent half (the `deap` dev dependency was held), shipping only the NSGA-II core
validated by hand-computed golden tests. Two gaps remained:

1. **Capability coverage.** Only the NSGA-II selection core + the basic operators were implemented.
   The DEAP→SeqOpt mapping in #261 lists a broader in-scope set (varOr, eaSimple, constraints,
   Hall of Fame, convergence) that was not yet pure-Python.
2. **The parity evidence.** No DEAP oracle, parity test, or comparison existed, so the
   "ship ours vs depend on DEAP" question was unanswered with data.

The maintainer asked to close both: cover **all** discussed capabilities in pure Python (not only
NSGA-II), with DEAP added **only as a temporary dev/test oracle**.

## Decision

**D1 — The full DEAP-mapped EA operator set is pure-Python.** Beyond the NSGA-II core (fast
non-dominated sort, crowding, DCD mating, (mu+lambda)), `SeqOpt` now implements, all DEAP-free at
runtime: **varAnd / varOr** variation, **(mu+lambda) / (mu,lambda) / eaSimple** survival,
**crossover** (uniform / one- / two-point) and **substitution / shift** mutation over mutation-sets,
**constraints** with DeltaPenalty / ClosestValidPenalty semantics, a single-objective **Hall of
Fame** (`hall_of_fame_`) beside the Pareto archive, and **hypervolume / spread / convergence**
metrics. Exposed via `run` params `variation`, `survival`, `constraints`, `penalty`, `hof_size` and
`eval`'s `ref_front`.

**D2 — Two engines, identical results.** `engine="exact"` is the pure-Python kernel whose crowding
formula matches DEAP's `assignCrowdingDist` (`nobj·span` normalization); `engine="fast"` vectorizes
the O(n²) non-dominated sort with numpy and returns a **numerically identical** front (same survivor
list, not just set). `fast` is purely a speed option.

**D3 — DEAP is a dev/test-only parity oracle.** `deap` is added to the **`[dev]`** extra; the
shipped runtime never imports it. Parity is asserted at the **selection-primitive** level on
synthetic fitness sets (the algorithm-agnostic, robust place to compare): our `fast_non_dominated_sort`,
`crowding_distance` and `select_nsga2` reproduce DEAP's `sortNondominated` / `assignCrowdingDist` /
`selNSGA2` — identical non-dominated rank (incl. heavy ties), identical crowding values + ordering
(within `atol`), and identical selNSGA2 survivor **profile** (equivalent up to arbitrary crowding
ties on the partial front). The bar is **equivalence, not byte-identity** (per ADR-0043).

**D4 — Ship ours.** The Phase-C comparison (`.github/scripts/seqopt_deap_comparison.py`) benchmarks
ours-`exact` / ours-`fast` / DEAP across `pop_size × n_objectives`: all three are correctness-
identical, and `engine="fast"` is **faster than DEAP** (e.g. ~14 ms vs ~102 ms at 500×3) while
keeping the runtime dependency-free. Decision: **ship the pure-Python implementation** (`fast` for
speed, `exact` as the RNG-matched reference); DEAP stays dev-only.

## Rejected alternatives

- **Depend on DEAP at runtime.** Rejected by D4: ours is faster and dependency-free.
- **Byte-exact end-to-end parity** (matched RNG stream through the whole evolve loop). Rejected:
  brittle and unnecessary — operator-level equivalence on fitness sets is the robust, sufficient bar.
- **Leave the operator set NSGA-II-only.** Rejected by the maintainer: the full discussed capability
  set must be pure-Python.

## Consequences

- `deap` joins `[dev]` (test/benchmark only). New `SeqOpt.run` params + `hall_of_fame_` attribute +
  `eval` `ref_front`/`convergence` grow the pro surface (changing it later is breaking).
- `crowding` values are now DEAP-normalized (`nobj·span`); a re-scaling of the informational
  `crowding` column (selection decisions unchanged) — the ADR-0043 golden value is updated.
- Parity + comparison are reproducible (`test_seqopt_deap_parity.py`, `seqopt_deap_comparison.py`).
- Builds directly on ADR-0043 (its parity-first plan is now delivered); does not supersede it.
