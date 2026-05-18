# Workflow: running locally and git / PR

Not path-scoped. Referenced from the root CLAUDE.md.

## Running things locally

```bash
# install dev environment
uv pip install -e ".[dev]"

# run all tests
pytest tests -x -vv --durations=20

# run a single file or method
pytest tests/unit/aa_window_sampler_tests/test_aa_window_sampler_synthetic.py -x -vv
pytest tests/unit/aaclust_tests/test_aaclust_fit.py::TestAAclustFit::test_valid_min_th -x -vv

# coverage (matches CI gate)
pytest --cov=aaanalysis --cov-fail-under=70 tests

# run notebooks (matches CI gate)
pytest --nbmake --nbmake-timeout=120 tutorials/ examples/

# build docs
cd docs && make html

# quick prototyping (inspection harness)
python dev_scripts/dev_aa_window_sampler.py
```

## Git / PR workflow

- Branch names: `fix/...`, `feat/...`, `doc/...`, `refactor/...`.
- Keep PRs small and focused; rebase on `master`.
- Never merge if `pytest tests` fails on Linux *or* Windows. CI runs the
  full matrix py 3.11–3.14.
- Update `CONTRIBUTING.rst`, any affected `examples/*.ipynb`, and (once it
  exists) `CHANGELOG.md` in the same PR; tutorials may follow in a separate
  PR but never on a different release.
- Do not push or force-push without explicit approval (hard rule §0 in the
  root CLAUDE.md — scoped per-action, not per-session).
