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
**Refs:** ADR-0003 (`docs/adr/0003-drop-numba-setuptools-cibuildwheel.md`).

### Context

ADR-0003 switched the **build backend** from `poetry-core` to `setuptools` to
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
- Build backend (`setuptools`) — stays per ADR-0003.
- Switch to hatchling — separate v2 decision per
  `.claude/rules/dependencies-and-pyproject.md`.

### Estimated effort

~30–80 lines diff across `pyproject.toml`, CI workflows, and
`CONTRIBUTING.rst`.

---

## V2-2: Promote `run_num` to public; auto-dispatch Cython inside `run` and `run_num`

**Status:** planned for the release after ADR-0003 ships (one cibuildwheel
release cycle of soak time first).
**Refs:** ADR-0001, ADR-0003.

### Context

After ADR-0003 ships, the technical state is:

- **Public:** `CPP.run` (legacy backend `_filters/`), `CPP.eval`.
- **Internal / undocumented:** `CPP.run_num` (Python `_filters_num/`),
  `CPP.run_c` (Cython `_filters_num_c/`).

All three `run*` methods produce *bit-identical* `df_feat` on seq-mode
inputs (PR2 amendment to ADR-0001 + PR3 parity). The split is currently
implementation-flavored, not API-flavored.

### Decision (planned for v2)

1. **Promote `CPP.run_num` to a public method.** Different mental model
   from `run`: per-call `df_seq` (and optional `dict_num` for numerical
   inputs like PLM embeddings / DSSP one-hots / PTM dummies). Documented
   docstring, included in `aaanalysis/__init__.py` exposure.
2. **`CPP.run` keeps its constructor-bound-`df_parts` mental model** but
   its body is rewritten to route through the same fast pipeline as
   `run_num`. Output bit-identical to today's `run` (already verified).
3. **Both `run` and `run_num` auto-dispatch internally**: Cython
   (`_filters_num_c/`) if the compiled `.so` is importable, else
   Python (`_filters_num/`). No `backend=` kwarg, no `run_c` on the
   surface.
4. **`CPP.run_c` deleted entirely.** Was never public; no deprecation
   cycle, no wrapper. Its function survives as the internal default
   fast path picked by the auto-dispatch helper.
5. **Legacy `_filters/` folder deleted** as part of the same PR, since
   no public method routes through it anymore. The `filtering` /
   `filtering_info_` helpers (~50 lines) are lifted into
   `_filters_num/_redundancy_filter.py` as their canonical home before
   `_filters/` is removed.

### Public surface after this change

| Method | Mental model |
|---|---|
| `cpp.run(labels, ...)` | "I configured CPP with `df_parts` upfront; compute features." Constructor-bound input. |
| `cpp.run_num(df_seq=, dict_num=, ...)` | "I have `df_seq` (and optionally `dict_num` for numerical input) for this call; compute features." Per-call input. |
| `cpp.eval(...)` | Unchanged. |

### Why not in PR4

- Wait for one cibuildwheel release cycle so wheel-build problems
  surface before the Cython path silently becomes the default for
  `run`.
- Side-effect parity for `cpp.run` needs verification before shipping:
  output is bit-exact, but warnings, exception types, and
  `ut.print_out` content from the new path may differ from legacy
  `_filters/`. New test required:
  `test_run_side_effect_parity.py` — asserts same warnings emitted,
  same exception types for malformed input, same verbose log content
  vs the current `run` behavior.
- Auto-dispatch hides which backend ran. Add an INFO-level
  `print_out` line on first use of the Python fallback so it's not
  invisible — something like `"CPP using Python kernel (compiled
  extension not available)"`.

### Out of scope for this entry

- Adding a `backend=` kwarg or env var (e.g.
  `AAANALYSIS_FORCE_BACKEND=python`). Not needed for users; the
  Cython path is bit-exact with Python, so there's nothing to A/B in
  production. If needed for debugging, add later as a private kwarg.
- Adding `list_parts=` subsetting to `run`/`run_num` (subset of
  `self.df_parts` columns per call). Useful but independent — track
  separately if/when needed.

### Estimated effort

- ~50 lines in `_cpp.py` (gut `run`'s body, route through
  `cpp_run_num_single`; same for `run_num`).
- ~30 lines in `cpp_run_num.py` (new `_pick_feature_matrix_builder()`
  helper; remove obsolete numba-era plumbing if not already gone).
- ~50 lines lifted from `_filters/_redundancy_filter.py` →
  `_filters_num/_redundancy_filter.py`.
- ~80 lines of side-effect parity tests for `run`.
- File deletions (CLAUDE.md §0 — needs explicit per-file approval):
  `_filters/_assign.py`, `_filters/_stat_filter.py`,
  `_filters/_pre_filter.py`, `_filters/_add_stat.py`,
  `_filters/_progress.py`, `_filters/_redundancy_filter.py`,
  `_filters/__init__.py`, and the `_filters/` folder itself.

Total: ~3–4 hours focused work, plus one release of cibuildwheel
soak before starting.

---

## How to use this file

When v2 work starts, each `V2-N` entry above becomes a real GitHub Issue.
Pasting the body into `gh issue create --title "V2-N: ..." --body-file -`
is the intended workflow.

Until then: edits welcome. Add new `V2-N` sections as discrete work items
get identified. Don't conflate them with `CPP_RUN_NUM_BACKLOG.md` (which
is the per-phase ledger for the CPP backend rewrites specifically) or
ADRs (which are decision records, not work items).
