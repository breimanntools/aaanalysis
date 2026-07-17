# Workflow: running locally and git / PR

Not path-scoped. Referenced from the root CLAUDE.md.

## Running things locally

```bash
# install dev environment
uv pip install -e ".[dev]"

# run all tests
pytest tests -x -vv --durations=20

# sub-minute smoke gate — the critical CPP path end to end
# (load -> CPP.run -> TreeModel.fit). Fast local sanity before the full suite;
# finishes in a few seconds. Non-blocking (no dedicated CI job).
pytest -m smoke -c tests/pytest.ini

# run a single file or method
pytest tests/unit/aa_window_sampler_tests/test_aa_window_sampler_synthetic.py -x -vv
pytest tests/unit/aaclust_tests/test_aaclust_fit.py::TestAAclustFit::test_valid_min_th -x -vv

# coverage (matches CI gate: test_coverage.yml uses --cov-fail-under=88)
pytest --cov=aaanalysis --cov-fail-under=88 tests

# branch coverage + the branch/line gate (issue #84). --cov-branch reports a
# branch-rate; the helper parses coverage.xml and fails below the committed
# line/branch gates (do NOT use --cov-fail-under with --cov-branch: it would
# gate the combined number and silently redefine the line floor).
pytest tests -m "not regression" --cov=aaanalysis --cov-branch --cov-report=xml -n auto -c tests/pytest.ini
python .github/scripts/check_branch_coverage.py

# parameter coverage meta-test (fast, no coverage run needed)
pytest tests/unit/api_tests/test_param_coverage.py -x -vv -c tests/pytest.ini

# run notebooks — LOCAL gate only (nbmake is NOT in blocking CI; re-run + re-commit
# outputs before every push; RTD renders committed outputs but does not execute them)
pytest --nbmake --nbmake-timeout=120 tutorials/ examples/

# perf benchmark suite (issue #187) — opt-in [bench] extra, runs in the perf
# nightly only (NOT the blocking matrix; wall-clock is noisy). Needs the plugin:
#   uv pip install -e ".[dev,pro,bench]"
pytest tests/benchmarks --benchmark-json=perf_run.json -c tests/pytest.ini
python .github/scripts/check_perf_regression.py perf_run.json   # compare vs baseline
# refresh the committed baseline (do it on CI's runner class via the perf-nightly
# workflow_dispatch with refresh_baseline=true, then commit the artifact):
python .github/scripts/check_perf_regression.py perf_run.json --update

# build docs
cd docs && make html

# quick prototyping (inspection harness)
python dev_scripts/dev_aa_window_sampler.py
```

## Dev gotchas (repo-wide)

- **Run with `-c tests/pytest.ini`.** It carries the `filterwarnings` that
  silence the deliberate D5b/D7 CPP advisories on tiny fixtures and registers
  the `regression` / `slow` markers. Without it those advisories surface as
  noise and the markers warn as unknown.
- **`aa.load_dataset(name=..., n=N)` returns `2N` rows** (n per class). To get
  labels use `df_seq["label"].to_list()`, not `len(df_seq)`-based assumptions.
- **Coverage is measured on the package only** — `--cov=aaanalysis`, never
  `--cov=./` (that counts the test files and inflates the number). See ADR-0016.
- **A push/PR to `master` triggers up to 8 workflows**, in two groups. Feature-branch
  *pushes* trigger none (CI is gated to master push/PR). **Verify against
  `.github/workflows/*` before quoting this — it drifts.**
  - **Code-gated (6)** — each carries `paths-ignore: ['docs/**', '**/*.md',
    '**/*.rst']`: **Unit Tests** (`main.yml`), **CodeQL** (`codeql_analysis.yml`),
    **Integration & E2E Tests** (`integration_e2e.yml`), **Perf (A/B regression gate)**
    (`perf_nightly.yml`), **Version Guard** (`version_guard.yml`), and **Type Check
    (advisory)** (`pyright.yml` — reports, never gates).
  - **Always-run (2)** — deliberately **no `paths-ignore`**, so they run on *every*
    master push including docs-only: **Test Coverage** (`test_coverage.yml`; the
    Codecov badge must stay current) and **Packaging** (`packaging.yml`; required
    checks must not be "skipped-but-required", ADR-0060).

  The exact-value CPP regression anchor (`-m regression`) runs only in the nightly,
  not the blocking matrix (ADR-0015); the Perf gate benchmarks
  current-vs-latest-release on one runner (ADR-0037); Version Guard asserts the
  pyproject version is ahead of the latest PyPI release.
- **Two more are path-*scoped* the other way** (they fire only for their paths):
  **ADR hygiene** (`adr_hygiene.yml`, `paths: ['docs/adr/**']` — `check_adrs.py`:
  duplicate-number / cross-ref / index gate) and **Visual Regression**
  (`visual_regression.yml`, PR-only, scoped to `aaanalysis/**/*.py` +
  `tests/unit/plotting_tests/**`).
- **A docs-only push does NOT skip everything.** It skips all 6 code-gated workflows
  but still runs the 2 always-run ones — so a docs-only push that **doesn't** touch
  `docs/adr/**` triggers **two** (Test Coverage + Packaging); one that **does** touch
  an ADR triggers **three** (+ ADR hygiene). Don't tell the user "all of CI will run"
  for a docs-only change — nor that nothing will.

## Git / PR workflow

- Branch names: `fix/...`, `feat/...`, `doc/...`, `refactor/...`.
- **One isolated `git worktree` per task.** Pair every `git switch -c` with a
  `git worktree add` so concurrent task streams never share one working tree /
  `HEAD` — sharing a checkout lets commits land on the wrong branch and
  uncommitted work bleed across tasks. Do the edits in the worktree, commit,
  push, then `git worktree remove`. **If you're running parallel agents this is
  mandatory:** without a per-task worktree a concurrent agent's commit/push can
  land *between* your `git status` and your commit. So even in a shared checkout,
  re-check `git status` immediately before staging, commit **explicit pathspecs
  only** (never a blind `git add -A` / `git commit -a`), and never commit, revert,
  or discard changes you did not make — stop and surface unexpected edits instead.
- Keep PRs small and focused; rebase on `master`.
- Never merge if `pytest tests` fails on Linux *or* Windows. CI runs the
  full matrix py 3.10–3.14 on Linux; Windows brackets min+max (3.10 + 3.14),
  with the full Windows range in the nightly.
- **Issue lifecycle / `Closes #NN`.** A closing keyword
  (`Closes`/`Fixes`/`Resolves #NN`) auto-closes the issue on merge to `master`
  when it appears in **either the PR body or the merge commit
  message**. To **keep an issue open through a merge, remove the keyword from
  the PR body before merging** — editing only the commit message is not enough.
- Update `CONTRIBUTING.rst`, any affected `examples/*.ipynb`, and (once it
  exists) `CHANGELOG.md` in the same PR; tutorials may follow in a separate
  PR but never on a different release.
- Do not push or force-push without explicit approval (hard rule §0 in the
  root CLAUDE.md — scoped per-action, not per-session).
- Full agentic workflow + all quality gates: `docs/guides/agentic_engineering.md`.
