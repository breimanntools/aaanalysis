---
paths:
  - "pyproject.toml"
---

# Dependencies and pyproject.toml

Editing `pyproject.toml` is **CONFIRM-FIRST** (see CLAUDE.md §10).

## Pin policy

- **Conservative**: library deps (numpy, pandas, sklearn, scipy, matplotlib,
  seaborn, joblib, etc.) → `>=` lower bound only, no upper.
- Tooling / docs deps (sphinx, numpydoc, docutils, black, etc.) → `>=`
  lower bound; accept latest.
- Treat exact `==` pins as defects unless paired with a code comment
  explaining the constraint.
- Adding a new dependency: justify in the PR description and use a real lower
  bound.

## Project / Poetry duality

- The file currently carries both `[project]` (PEP 621) and `[tool.poetry]`
  metadata.
- **`[project]` is canonical.** All metadata and dependency edits go in
  `[project]`. **Never edit `[tool.poetry]`.** It is legacy and will drift
  naturally.
- v2 target: drop `[tool.poetry]` entirely and switch the build backend to
  `hatchling`. Do not migrate prematurely.

## Releasing

- Version bump in `pyproject.toml` (`[project] version = "..."`). Keep
  `[project]` and (legacy) `[tool.poetry]` in sync until the Poetry block is
  removed in v2.
- Tag `vX.Y.Z`; the `build_wheels.yml` workflow builds and publishes to
  TestPyPI then PyPI.
- Pro extras are versioned with the core; bumping a pro dep requires a new
  minor at minimum.
