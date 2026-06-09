# v2 follow-ups

Deferred work that doesn't fit in any current PR but needs to be tracked
somewhere visible. Each entry will be promoted to a real GitHub Issue (or
a dedicated PR) when the v2 milestone starts.

Per `.claude/rules/sharp-edges.md`, v2 also drops `[tool.poetry]` from
`pyproject.toml`, switches build backend to `hatchling`, adds `ruff` /
`pre-commit` / type checker — these are stand-alone work items not
captured below.

---

## V2-1: Migrate dev workflow from Poetry to uv

**Status:** deferred to v2.
**Refs:** ADR-0001 (`docs/adr/0001-cpp-backend-architecture.md`).

### Context

ADR-0001 switched the **build backend** from `poetry-core` to `setuptools` to
enable Cython compilation. The **dev workflow** (`poetry install`,
`poetry.lock`, `poetry add`) is unchanged — Poetry still works against the
new setuptools backend.

This entry tracks the deferred follow-up: replacing Poetry with `uv` as the
maintainer/contributor workflow tool. End-user `pip install aaanalysis` and
`uv add aaanalysis` already work today regardless — this is purely about
the maintainer/contributor experience.

### Motivation

- **Speed:** uv's resolver is ~10–100× faster than Poetry's.
- **Standards alignment:** uv reads PEP 621 `[project]` natively; less
  custom config drift.
- **Single tool:** uv covers install, lock, build (via the configured
  backend), publish, venv management.
- **Trend:** uv has become the default in modern Python tooling (2024+).

### Scope

- Delete `poetry.lock`, generate `uv.lock` via `uv lock`.
- `pyproject.toml`: optionally move `[tool.poetry.group.dev.dependencies]`
  (if any) to `[dependency-groups]` (PEP 735, uv-native). Leave
  `[tool.poetry]` legacy block alone per
  `.claude/rules/dependencies-and-pyproject.md` — it disappears in the
  separate hatchling migration.
- `.github/workflows/main.yml` and `test_coverage.yml`: replace
  `poetry install` → `uv sync`, `poetry run pytest` → `uv run pytest`.
- `CONTRIBUTING.rst`: update install instructions.
- Verify `pip install -e .` and `uv pip install -e .` both still work for
  contributors who prefer either.

### Out of scope

- End-user installation (`pip install aaanalysis`, `uv add aaanalysis`) —
  already works.
- Build backend (`setuptools`) — stays per ADR-0001.
- Switch to hatchling — separate v2 decision per
  `.claude/rules/dependencies-and-pyproject.md`.

### Estimated effort

~30–80 lines diff across `pyproject.toml`, CI workflows, and
`CONTRIBUTING.rst`.

---

## How to use this file

When v2 work starts, each `V2-N` entry above becomes a real GitHub Issue.
Pasting the body into `gh issue create --title "V2-N: ..." --body-file -`
is the intended workflow.

Until then: edits welcome. Add new `V2-N` sections as discrete work items
get identified. Don't conflate them with `CPP_RUN_NUM_BACKLOG.md` (which
is the per-phase ledger for the CPP backend rewrites specifically) or
ADRs (which are decision records, not work items).
