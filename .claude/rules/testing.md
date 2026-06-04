---
paths:
  - "tests/**/*.py"
---

# Tests

## Stack
- `pytest` + `hypothesis`. Coverage via `pytest-cov`.
- Layout: `tests/unit/<class_or_topic>_tests/test_<class_or_method>.py`. One
  file per public method.
- `tests/integration/` and `tests/e2e/` exist but are nearly empty —
  integration / e2e coverage is **deferred to v2**. Do not add integration
  tests proactively.

## File header (current style)
Each test file currently opens with:
```python
"""This is a script to test <Class>.<method>()."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")
```
- Per-file `register_profile("ci", deadline=...)` blocks **stay**. Deadlines
  may vary across files; do not centralize them in `conftest.py`.
- A `tests/conftest.py` autouse fixture resets `aa.options` to defaults
  around every test — once that lands, the `aa.options["verbose"] = False`
  line at the top of every test file becomes drift to fix on touch.
- Don't remove `HYPOTHESIS_DEADLINE=10000000` from CI.

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
actually in use — **do not** create empty `integration`/`benchmark` tiers
(integration/e2e stay deferred to v2, below).

- `regression` — a frozen-value anchor (see below).
- `slow` — opt-in heavy tests, deselectable in fast CI runs.

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
