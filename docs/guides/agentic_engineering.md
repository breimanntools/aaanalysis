# Agentic engineering protocol

How AAanalysis is developed with AI-assisted ("agentic") workflows — and, just as
importantly, the **automated gates every change must pass before a human merges it**.
The durable artifacts of this process (ADRs, `CONTEXT.md`, `CLAUDE.md` / `.claude/rules`,
and the code) are the single sources of truth; per-issue planning notes are ephemeral and
not committed.

## Workflow (step by step)

1. **Pick the issue.** No skill required. Optionally clean up or generate the issue wording
   first with `/triage` or `/to-issues` (house style: `docs/guides/issue_style_guide.rst`).
2. **`/grill-with-docs` — the highest-leverage step.** A custom command that sharpens the
   spec against the *live* codebase and refreshes `CONTEXT.md` / ADRs **before any code is
   written**. The closest built-in, `/init`, only (re)generates codebase docs — it does not
   do the adversarial spec-vs-reality pass. Keep grill for the real work; use `/init` only as
   a one-time bootstrap when `CONTEXT.md` does not exist yet.
3. **Branch + isolated worktree.** `git switch -c <type>/<slug>` off `master` (plain git).
   **Always pair it with `git worktree add` so each task gets its own checkout** — concurrent
   streams then cannot contaminate each other (see Process notes → *Isolated worktrees*).
4. **Implement.** Plan mode / the goal skill. Use a structured plan for multi-file or
   architectural changes; drop to plain edits for trivial diffs. Plan mode is preferable when
   you want to approve the approach before commits land.
5. **Push → open PR.** `gh` / `git`. A PR needs ≥1 commit; push a scaffolding commit and open
   a **draft PR early** so CI + the Read the Docs (RTD) preview run while you build.
6. **Automated review gate (machine-enforced, not eyeballed).** Run `/review` (reviews the PR
   diff) and `/security-review` (scans the pending diff for vulnerabilities). The quality gates
   below must be green first. **Never merge red** — automated checks *gate* the human review,
   they do not replace it.
7. **Refine on the same branch.** Back to plan / goal mode; push more commits (the PR and the
   RTD preview update automatically).
8. **Keep current.** Periodically merge `master` → branch (plain git). A good fit for the
   `/schedule` skill: auto-**sync** each morning so you wake to a synced branch or an early
   conflict warning. **Schedule the sync only — never an auto-merge to master** (a scheduled
   job can sync when clean but cannot resolve conflicts; it should just flag you).
9. **Manual review → merge.** No skill, and that is correct: read the RTD preview + PR diff
   (informed by the step-6 findings) and make the call with `gh pr merge`.
10. **Delete the branch.** Plain git, with permission (root `CLAUDE.md` §0).

## Quality gates — how AAanalysis checks each

> **Rule: never merge red.** These automated checks gate the human review.

| Gate | "Green" means | AAanalysis mechanism |
|---|---|---|
| **Tests** | full matrix passes | `.github/workflows/main.yml` ("Unit Tests"): `pytest tests -m "not regression" -x -n auto` on **Ubuntu py3.10–3.14** + **Windows py3.10 & 3.14** (Windows brackets min+max; the full Windows range and the `-m regression` exact-value CPP anchor run in the nightly — ADR-0015). |
| **Coverage** | **≥ 88 %** line coverage, package-only | `.github/workflows/test_coverage.yml`: `pytest … --cov=aaanalysis --cov-fail-under=88` (+ Codecov `patch` / `project` / `project/cpp_core`). Measured on the package only (`--cov=aaanalysis`, never `--cov=./`) — ADR-0016. |
| **Docs** | RTD builds; API + examples render | `docs/readthedocs.org` check: Sphinx + nbsphinx. `docs/source/conf.py` runs `export_example_notebooks_to_rst` with `nbsphinx_execute='never'` — it **renders committed notebook outputs, it does not execute them**. |
| **Docstrings** | numpydoc shape, named `Returns`, per-method `Examples` include, no doc-vs-signature drift | the `/docstrings` skill: `check_docstrings.py`, `doc_signature_drift.py`, `check_example_notebooks.py` under `.claude/skills/docstrings/scripts/`. **Local gate** (not yet a CI job). |
| **Notebooks execute** | every `examples/` + `tutorials/` notebook runs clean with embedded outputs | `pytest --nbmake --nbmake-timeout=120 examples/ tutorials/`. **Local gate only — NOT in blocking CI.** Re-run and re-commit outputs before every push (see Process notes). |
| **Architecture** | matches `CONTEXT.md` / ADRs; no cross-class backend imports or layering violations | machine: `tests/unit/api_tests/test_backend_import_hygiene.py` (runs inside `pytest tests`). Spec / ADR conformance is human + `/grill-with-docs`. |
| **Parameter coverage** | every public parameter is exercised by name in tests | `tests/unit/api_tests/test_param_coverage.py` — **landing via PR #111**; not yet on master. |
| **Lint (errors)** | no syntax errors / undefined names | `.github/workflows/codeql_analysis.yml` ("code-quality" job): `flake8 . --select=E9,F63,F7,F82`. |
| **Style / types (full)** | black (88) / isort / flake8 (88) / mypy clean | **manual at review** — no pre-commit, ruff, or type-checker in CI (v2 target; `.claude/rules/sharp-edges.md`). |
| **Security** | CodeQL clean | `.github/workflows/codeql_analysis.yml` ("Analyze" job). No separate dependency-scan PR gate yet — worth adding. |
| **Issue linkage** | the PR's lifecycle keyword is set per policy | `Closes #NN` in the **PR body** (see Process notes → *Issue lifecycle*). |

## Process notes (hard-won)

- **Isolated worktrees per task.** Create a `git worktree` per branch so two concurrent
  streams never share one working tree / `HEAD`. Sharing a single checkout caused commits to
  land on the wrong branch and uncommitted work to bleed across tasks. A worktree also lets you
  build and verify one branch without disturbing another in-flight branch — do the edits in the
  worktree, commit, push, then `git worktree remove`.
- **Issue lifecycle — `Closes #NN`.** GitHub auto-closes a referenced issue on merge to the
  default branch when a closing keyword (`Closes` / `Fixes` / `Resolves #NN`) appears in **either
  the PR body or the merge (squash) commit message**. To **keep an issue open through a merge,
  remove the keyword from the PR body before merging** — fixing only the commit-message text is
  not enough. To auto-close, keep `Closes #NN` in the PR body.
- **Notebooks are a local-only gate.** Because nbmake is not in blocking CI, a broken example
  surfaces only on RTD (as wrong/un-rendered output) or in a local run. Always run
  `pytest --nbmake examples/ tutorials/` and commit fresh outputs before pushing.
