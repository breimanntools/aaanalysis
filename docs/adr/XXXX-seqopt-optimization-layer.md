# ADR-XXXX — SeqOpt optimization layer (SHAP-guided, fuzzy-labeled multi-objective directed evolution)

Status: Proposed — 2026-06-24

## Context

The protein-design chain (#37 → #57 → #59 → #60) decomposed into a measurement layer
(ADR-0027, `AAMut`/`SeqMut`) and a model-guided scoring layer (ADR-0042, `SeqMut` becomes
optionally model-aware via `delta_pred`). Both ADRs **explicitly deferred search/optimization**
to a separate `SeqOpt` class: ADR-0042 D3 states *"`SeqMut` scores; all search/optimization lives
in a separate `SeqOpt` class … `SeqOpt` reuses model-bound `SeqMut` as its fitness engine."* This
ADR settles `SeqOpt` (#261), delivering the substance of #59 (multi-objective optimization) and
setting up #60 (active learning).

Three things had to be settled before code, and the maintainer's framing moved the design well
beyond the issue's original "pure-Python core NSGA-II reusing `SeqMut.combine`" wording:

1. **What actually guides the search.** Real directed evolution can't enumerate the mutation-set
   space (positions × 20 AAs, in sets up to `n_mut_max`). The maintainer's design prunes it with
   **residue-level model attribution**: each round, the strongest-impact residues are mutated. The
   attribution source is `ShapModel` (SHAP `feat_impact`), **refit every round** so the guidance
   tracks the moving population — which forces a SHAP dependency onto the search path.
2. **How newly generated variants get labels.** Generated variants are unlabeled, but the per-round
   `ShapModel` refit needs labels. They are assigned **fuzzy labels** = their own model prediction
   score in `[0, 1]` (the shipped `ShapModel.fit(fuzzy_labeling=True)` path, designed precisely "to
   explain newly predicted samples, where the class label is set to the prediction probability").
3. **What "parity with DEAP" means.** The NSGA-II selection core is a reimplementation; DEAP is the
   oracle. Byte-for-byte identity (matched float-summation order) is brittle and research-grade;
   the maintainer accepted **equivalence** instead.

## Decision

**D1 — `SeqOpt`/`SeqOptPlot` are `pro`, in a new `aaanalysis/protein_design_pro/` subpackage.**
Because SHAP attribution is the central engine (not an optional add-on), `SeqOpt` imports
`ShapModel` directly and is gated behind the `pro` extra exactly like `ShapModel`
(`shap` is already in `_EXTRA_MODULES["pro"]`). This does **not** contradict ADR-0042 D2's "no SHAP
dependency forced": that decision keeps `SeqMut` and the `protein_design` **core** module SHAP-free;
the new optimizer lives in a separate `*_pro` module. `SeqMut` (core) remains the fitness engine and
is imported by `SeqOpt`.

**D2 — Two guidance modes, named after their `df_feat` attribution column.**
- `mode="impact"` (default, headline): per-round `ShapModel` refit under **fuzzy labeling** →
  fresh per-residue `|feat_impact|` → an **adaptive NSGA-II** population evolves the Pareto front.
- `mode="importance"`: static per-residue `feat_importance` (from `df_feat`, no SHAP, no refit) →
  positions walked **highest-importance-first** in a deterministic **greedy** search. The cheap,
  fully reproducible baseline.

**D3 — NSGA-II is reimplemented pure-Python; DEAP is a dev/test-only parity oracle; the bar is
equivalence, not byte-identity.** The selection core (fast non-dominated sort → non-dominated
`rank`, crowding distance, DCD mating selection, `(mu+lambda)` survival) is ours. On fixed seeds it
must reproduce a DEAP `selNSGA2` reference with **identical Pareto-front membership and identical
rank/crowding ordering**; objective values match within `atol=1e-9`. We do **not** assert byte-
identical `df_pareto` serialization. DEAP is added to the **`[dev]`** extra only — the shipped
runtime never imports it.

**D4 — One wild-type per `run`; the population is a set of mutation-set variants.** `run` validates
`df_seq` has exactly one entry. A variant's genome is a sparse `{pos: to_aa}` map (distinct
positions, size `1..n_mut_max`, positions inside the scannable JMD-N+TMD+JMD-C span, `to_aa` from
the alphabet). Crossover unions parent positions and inherits each per-position substitution from
one parent; mutation re-points / adds / removes / shifts a position, importance-weighted; both
repair to `≤ n_mut_max`. Multi-entry batching is a documented non-goal (a future loop), as are
NSGA-III / SPEA2 / Bayesian optimization.

**D5 — Fitness reuses the `SeqMut`/CPP feature matrices; objectives are a typed spec.**
`objectives` is a list of `(name, "max"/"min", source)` with `source ∈ {"delta_pred" (+target_class),
"delta_cpp", "shift_score", "n_mut", callable(df_variant)->array}`; ≥2 objectives for a Pareto run.
Per generation the whole population is scored in one feature-matrix pass (memoized on the canonical
genome) to bound cost. Output `df_pareto`: `entry`, `variant`, `n_mut`, `sequence_mut`, one column
per objective, `rank` (front index), `crowding`.

**D6 — `eval` reports Pareto-quality metrics; the plot pair returns `FigAxResult`.**
`eval(df_pareto, ref_point=None)` → `hypervolume`, `n_front`, `spread`. `SeqOptPlot.pareto_front`
(2-D/3-D objective scatter colored by `rank`) and `SeqOptPlot.hypervolume` (per-generation trace)
both return `ut.FigAxResult`. Full `random_state`/`seed` threading (constructor + per-call, into
`ShapModel` too); two same-seed `run`s yield identical `df_pareto`.

## Rejected alternatives

- **`SeqOpt` in core with a duck-typed importance guide** (no `shap` import inside it). Rejected by
  the maintainer: SHAP-guided refit-every-round is the reason the class exists, so it is `pro` and
  uses `ShapModel` internally rather than pretending the dependency is optional.
- **Byte-exact DEAP parity** (matched `random.Random` stream + float-summation order, exact
  `assert_frame_equal`). Rejected: brittle and likely the dominant cost; equivalence (identical
  front membership + rank/crowding ordering, values within `atol`) is a credible parity claim and
  robust to numpy/Python float-order differences. The issue's byte-exact KPIs are amended to this.
- **Putting search/optimization on `SeqMut`.** Already rejected by ADR-0042 D3; `SeqMut` stays the
  scoring surface and the fitness engine, search stays here.
- **Per-generation single SHAP refit over the whole fuzzy-labeled population.** Off `ShapModel`'s
  "exactly one fuzzy label among a balanced reference" optimum (it warns otherwise); the per-variant
  / top-k refit path is preferred for attribution fidelity, bounded by a `guide`-budget cap.
- **A single DEAP runtime dependency** (ship DEAP instead of reimplementing). Rejected: keeps the
  shipped runtime free of a heavy EA dependency; DEAP stays dev/test-only as the parity oracle.

## Consequences

- New `pro` surface: `aa.SeqOpt` / `aa.SeqOptPlot`, gated like `ShapModel`; changing it later is a
  breaking change. `protein_design_pro` joins the backend-import-hygiene `DEDICATED_OWNERS` map.
- `df_pareto` / `df_seqopt_eval` join `DICT_DF_SCHEMAS`; new EA domain terms enter `CONTEXT.md`
  (population, generation, Pareto front, non-dominated rank, crowding distance, hypervolume,
  fuzzy-labeled SHAP guidance).
- `deap` is added to the **`[dev]`** extra (test/benchmark only); `shap` is reused (already `pro`).
- #59 is delivered (maintainer decides whether to close it); #60 builds on this engine.
- ADR-0042 is cross-referenced (its `SeqOpt`-is-forthcoming wording is now realized), not superseded.
