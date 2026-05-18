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

## Coverage gate
- CI runs `pytest --cov=aaanalysis --cov-fail-under=70`.
- Initial floor is 70%; tighten in CHANGELOG-tracked steps. PRs that **raise**
  the floor are encouraged.
- README carries a coverage badge (codecov or coveralls).

## Notebook execution
- CI runs `pytest --nbmake --nbmake-timeout=120 tutorials/ examples/` on
  Linux + py 3.12 (single config — no need to multiply across the matrix).
- If a notebook errors, CI fails. PRs that touch the public API must update
  the affected notebooks in the same PR.
