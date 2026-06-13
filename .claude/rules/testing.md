---
paths:
  - "tests/**/*.py"
---

# Tests

## Stack
- `pytest` + `hypothesis`. Coverage via `pytest-cov`.
- Layout: `tests/unit/<class_or_topic>_tests/test_<class_or_method>.py`. One
  file per public method.
- **Three tiers** (ADR-0031): `tests/unit/` (per-method, the bulk),
  `tests/integration/` (cross-component *seams*), `tests/e2e/`
  (full protocol-mirroring *workflows*). All three are default-selected and
  **gate merges** (the blocking job runs `-m "not regression"`). See the
  per-tier taxonomy below before adding to integration/e2e.

## Tier taxonomy (what each tier tests — ADR-0031)
Count is bounded by distinct *seams* and *workflows*, **not** by unit-test
volume; cover each seam/workflow **once**. The governing rule: **don't re-test
at a higher tier what a lower tier already covers.**

| Aspect | Unit | Integration | E2e |
|---|---|---|---|
| Job | per-method correctness | seam contract holds | workflow → right artifact |
| Positive | per-param, hypothesis-fuzzed | happy path | happy path |
| Negative | per-param invalid-input sweep | **composition failures only** (NOT input validation) | minimal: degenerate dataset → clear error |
| Hypothesis | per-argument strategies | **pipeline invariants / metamorphic** | **reproducibility** (seed→same artifact) |

- **Integration negatives are composition failures** — failures invisible to
  either component's unit tests (shape/label-set mismatch at a seam, PU dataset
  with no unlabeled rows, sampler returns no windows → empty logo, single-class
  labels into `CPP.run`). Do **not** re-run per-parameter invalid-input
  negatives there — that is the unit layer's job (frontend `# Validate` block).
- **Keep inputs tiny + seeded**, assert **structural/range** artifacts (shapes,
  schema columns, metrics in `[0,1]`, finiteness) — **never frozen exact
  values** (that is the regression anchor's job; e2e runs the full matrix where
  3rd-decimal drift would flake). Hypothesis here uses small `max_examples`
  (3–5) because each example runs the real pipeline.
- **Shared seeded builders** live in `tests/_pipeline.py` (one definition of the
  load→parts→CPP→matrix spine), exposed to integration as session fixtures in
  `tests/integration/conftest.py` and imported directly by e2e.
- `tests/integration/test_online_fetches.py` is a separate, `network`-marked
  live-endpoint tier (deselected by default); it is *not* the cross-component
  integration tier above.

## File header (current style)
Each test file currently opens with:
```python
"""This is a script to test <Class>.<method>()."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")
```
- **Hypothesis deadlines are disabled (`deadline=None`) suite-wide.** CI runs the
  tests under **`pytest-xdist -n auto`** (parallel), where shared-CPU contention
  makes per-example wall-clock deadlines flake (a 1.5s-deadline test ran 2.4s
  purely from a co-scheduled worker). Wall-clock timing is not a reliable gate
  under parallelism, so all `register_profile`/`@settings` use `deadline=None`.
  Performance/correctness is guarded by the **CPP regression anchor** (nightly),
  not by these deadlines. New tests: `register_profile("ci", deadline=None)`.
- A `tests/conftest.py` autouse fixture resets `aa.options` to defaults
  around every test — the `aa.options["verbose"] = False` line at the top of
  every test file is drift to fix on touch.
- A session-scoped `_warm_matplotlib` autouse fixture + the `Agg` backend in
  `conftest.py` pay matplotlib's font-cache / first-figure cost once up front
  (kept from #83; harmless now that deadlines are off).
- **`HYPOTHESIS_DEADLINE` was a no-op and has been removed.** Hypothesis reads
  `HYPOTHESIS_PROFILE`, not `HYPOTHESIS_DEADLINE`, and nothing in the suite read
  it — so the `=10000000` in `main.yml` / `test_coverage.yml` /
  `mutation_nightly.yml` did nothing and was deleted (#83). Deadlines are fully
  determined by per-file `register_profile` + per-test `@settings`; CI still
  sets `MPLBACKEND: Agg` (matching the `conftest.py` backend pin).

## Test classes per file
Two classes per public method, no exceptions:

- `Test<Method>` — **normal cases, one parameter per test method.** Each
  parameter of the target function gets its own positive test (using
  `@given(...)` from hypothesis) and its own negative test (looping over
  invalid values asserting `pytest.raises(ValueError)`). Aim for **≥10
  positive and ≥10 negative tests** in this class.
- `Test<Method>Complex` — **combinations and edge interactions.** Tests
  that intentionally cross multiple parameters. **≥5 positive and ≥5
  negative tests.**

Targets per public method: **≥30 unique tests total**, written as
complete, runnable code — no `TODO`, no `pass`, no placeholder bodies.
When generating new tests, follow the structure of an existing test file
in the same subpackage as the template.

## Test quality (not just line coverage)
Line coverage proves a line *ran*, not that a wrong line would *fail* a test.
For the **scientific core** (CPP feature values + filtering/ranking, AAclust,
`SequenceFeature` / `NumericalFeature`, scale lookup) add, **inline in the same
per-method file** (do not spawn a parallel correctness-suite tree):

- **Golden values** — a `Test<Method>GoldenValues` class asserting *hand-computed
  expected numbers* on a tiny input (feature names, positions, aggregations,
  AUCs), not just "runs without error".
- **Property tests** — `@given(...)` checks of invariants (frequencies sum to 1,
  `len(X) == n_sequences`, scores sorted descending, returned features satisfy
  the overlap/correlation thresholds, same seed → same output).
- **Property-based testing with hypothesis is the house standard.** Every
  positive test in `Test<Method>` parametrizes its argument with a
  `hypothesis.strategies` variant rather than a single hardcoded value, unless
  the argument is a small categorical enum (e.g. `mode="global"`). Use
  `@given(arg=some.integers(...) / some.floats(...))` plus
  `@settings(max_examples=5, deadline=None)` (small `max_examples` keeps the
  suite fast under xdist). Example:
  `@settings(max_examples=5, deadline=None)` /
  `@given(n=some.integers(min_value=2, max_value=5))`.
- **Error-message tests** — `pytest.raises(ValueError, match="…")` so users get
  the tailored message, not a cryptic pandas/sklearn traceback. When a guard is
  reachable (e.g. a part-based `df_seq` reaching a `'sequence' not in columns`
  check that `check_df_seq` lets through), cover it here — its message is a
  feature.

**Reproducibility tests** — for every stochastic public method (`dPULearn`,
`TreeModel`, `AAclust`, `AAWindowSampler`), assert `fit(seed=0) == fit(seed=0)`
inline (aligns with `reproducibility.md`).

## Test tiers & markers
Register markers in `tests/pytest.ini` `markers =`; only stand up markers
actually in use — **do not** create empty `benchmark` tiers.

- `regression` — a frozen-value anchor (see below).
- `slow` — opt-in heavy tests, deselectable in fast CI runs.
- `integration` — cross-component seam test (see the tier taxonomy above);
  default-selected, so it gates merges.
- `e2e` — full protocol-mirroring workflow; default-selected, so it gates
  merges. Add the marker module-wide with `pytestmark = pytest.mark.integration`
  (or `e2e`).

### CPP regression anchor (ADR-0015)
`tests/unit/cpp_tests/test_cpp_regression.py` runs a seeded `DOM_GSEC`
mini-pipeline and asserts the **top-feature identity + frozen `auc.round(3)`**.
It is `@pytest.mark.regression` and **`skipif` off the canonical cell**
(Linux + floor Python; `AAA_RUN_REGRESSION=1` forces it locally) because
exact-value freezing is only reproducible on a fixed environment. The **blocking
CI runs `-m "not regression"`** (`main.yml`, `test_coverage.yml`); the anchor
runs/re-verifies in the **non-gating nightly** (`mutation_nightly.yml`) so a
3rd-decimal drift never blocks merges. Frozen values are re-frozen only on an
intentional, reviewed change. It extends the unit-level parity-test precedent —
it is **not** the deferred integration tier.

### Mutation testing
`mutmut` is an **opt-in dev tool + non-gating nightly CI job** (not a hard
gate). Use surviving mutants on the core filters/validators to find weak tests,
then convert them into golden/property assertions above.

## Coverage gate (ADR-0016)
- Measure the **package only** — `--cov=aaanalysis` / `codecov.yml`
  `ignore: ["tests/**"]`. Never `--cov=./` (that counts the test files and
  inflated the badge to ~95% vs the honest ~90%).
- **Ratchet, don't flip absolutes on at today's level.** Enabled now (green at
  ~90%): Codecov `project: {target: auto}` (no regression) + `patch: 90%`, plus
  a CI backstop `--cov-fail-under=88`. Climb toward `project 95%`, a `cpp_core`
  component (`_filters/**`) at 100%, `patch 95%` — turn the hard numbers on
  per-component only once reached; raise the CI floor in tracked steps.
- **100% core is honest, not gamed.** Reach it with real tests; remove only
  *proven-dead* lines (prove no input satisfies the upstream validator while
  failing the guard — e.g. `encode_dssp.py:37`); reachable guards get tests.
  `# pragma: no cover` only with an inline justification.
- **No lint/type-check CI gate** — ruff/mypy/pre-commit stay deferred to v2
  (`sharp-edges.md`); re-evaluated and upheld.
- README carries a coverage badge (codecov).

## Notebook execution
- CI runs `pytest --nbmake --nbmake-timeout=120 tutorials/ examples/` on
  Linux + py 3.12 (single config — no need to multiply across the matrix).
- If a notebook errors, CI fails. PRs that touch the public API must update
  the affected notebooks in the same PR.
