# ADR-0031 — Integration & e2e test tiers: scope, taxonomy, and merge-gating

Status: Accepted — 2026-06-13

Relates to: the `testing.md` and `sharp-edges.md` path-scoped rules; ADR-0015
(CPP regression anchor) and ADR-0016 (coverage policy).

## Context

`tests/integration/` and `tests/e2e/` had been **deferred to v2**
(`sharp-edges.md`, `testing.md`: "do not add integration tests proactively").
The only file in either tree was `tests/integration/test_online_fetches.py`, a
`network`-marked live-endpoint contract — not a cross-component test. Two facts
made the deferral costly:

- **nbmake is not in any CI workflow.** The protocol notebooks under
  `protocols/` run end to end *by hand only*; they assert nothing and gate
  nothing. The tutorial/example nbmake job described in older notes is a
  local-only gate.
- **The CPP regression anchor (ADR-0015) is the only seeded multi-component
  pipeline, and it is `@pytest.mark.regression`** — it runs only in the
  non-gating nightly on the canonical cell.

Net: there was **no blocking, assertion-bearing, multi-component test in CI**.
Unit tests mock the seams; the contracts *between* components (CPP→TreeModel,
dPULearn→TreeModel, AAclust→scales→features, sampler→logo, design→features,
encoder→consumer, fasta round-trip) were asserted nowhere.

A recurring question when adding higher tiers is "how many, given ~4k unit
tests?" The naive test-pyramid ratio (70/20/10) would imply ~1100 integration
tests, which is wrong here: e2e runs the real CPP pipeline (seconds each, on the
full matrix), so redundant input-validation re-tests would be slow,
drift-flaky, and add nothing the unit layer already covers.

## Decision

- **D1 — Stand up two real tiers that gate merges.** `tests/integration/`
  (cross-component *seams*) and `tests/e2e/` (full protocol-mirroring
  *workflows*) are default-selected: the blocking job already runs
  `-m "not regression"`, so both tiers run on every push/PR. New `integration`
  and `e2e` markers are registered in `tests/pytest.ini` for selection/clarity,
  **not** for deselection.
- **D2 — Count is bounded by seams × workflows, not by unit-test volume.** Each
  distinct component seam and each protocol notebook is covered **once**. The
  initial suite is ~12 integration seams and 10 e2e workflows.
- **D3 — Per-tier division of labor: don't re-test at a higher tier what a
  lower tier covers.**

  | Aspect | Unit | Integration | E2e |
  |---|---|---|---|
  | Job | per-method correctness | seam contract holds | workflow → right artifact |
  | Positive | per-param, hypothesis-fuzzed | happy path | happy path |
  | Negative | per-param invalid-input sweep | **composition failures only** | minimal: degenerate dataset → clear error |
  | Hypothesis | per-argument strategies | **pipeline invariants / metamorphic** | **reproducibility** (seed→same artifact) |

- **D4 — Higher-tier negatives are composition failures only.** Failures that
  emerge *when components meet* and are invisible to either component's unit
  tests (shape/label-set mismatch at a seam, a PU dataset with no unlabeled
  rows, a sampler that returns no windows feeding a logo, single-class labels
  into `CPP.run`). Per-parameter invalid-input negatives stay in the unit layer
  (the frontend `# Validate` block); re-running them at e2e is the anti-pattern
  this ADR forbids.
- **D5 — Higher-tier hypothesis is invariants/metamorphic + reproducibility,
  with small `max_examples` (3–5).** Each example runs the real pipeline, so
  broad per-argument fuzzing belongs to units. Examples: `len(X) == n_samples`;
  same seed → identical `df_feat` / predictions / PU carve; row-permutation of
  `df_seq` ⇒ same *set* of top features.
- **D6 — Structural/range assertions, never frozen exact values.** These tiers
  run on the full CI matrix where 3rd-decimal drift would flake; exact-value
  freezing remains the regression anchor's job (ADR-0015). Assert shapes, schema
  columns, metrics in `[0,1]`, monotonic ranking, finiteness.
- **D7 — One shared seeded spine.** `tests/_pipeline.py` defines the
  load→parts→CPP→feature-matrix builders once; `tests/integration/conftest.py`
  exposes them as session fixtures and e2e imports the functions directly, so a
  seam's call pattern is defined in exactly one place.
- **D8 — Core-only, offline.** No `pro`/`embed` imports and no `network`
  dependency in these tiers; where a protocol cell uses a pro feature
  (protocol9's `ShapModel`) the core path (`TreeModel.add_feat_importance`) is
  substituted.

## Consequences

- CI now has a blocking, assertion-bearing pipeline test for every documented
  workflow and every cross-component seam — the protocol notebooks finally have
  checked analogues even though nbmake is not in CI.
- The `testing.md` taxonomy table and the `sharp-edges.md` deferral are updated
  to match; "integration/e2e deferred to v2" is no longer true.
- Reviewers have a bright line for higher-tier scope: a new integration test is
  justified only by a *new seam* or a *new composition failure mode*, not by a
  parameter that the unit layer already sweeps.
- This supersedes the v2 deferral in `sharp-edges.md` for integration/e2e only;
  lint/type-check tiers (ruff/mypy/pre-commit) remain v2-deferred per ADR-0016.

## External practice

The tier division follows the standard test pyramid: many fast unit tests, fewer
integration tests at component seams, fewest end-to-end tests over full
workflows (Cohn, *Succeeding with Agile*; Fowler, *TestPyramid* and
*IntegrationTest*; *Software Engineering at Google*, ch. 11–14 on test sizes and
scope). The "don't duplicate lower-tier coverage at a higher tier" rule and the
preference for metamorphic/invariant checks over re-fuzzing are drawn from the
same sources and from metamorphic-testing literature.
