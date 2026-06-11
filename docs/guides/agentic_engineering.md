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
4. **Implement.** Plan mode. Use a structured plan for multi-file or architectural changes;
   drop to plain edits for trivial diffs. Plan mode is preferable when you want to approve the
   approach before commits land.
5. **Push → open PR.** `gh` / `git`. A PR needs ≥1 commit; push a scaffolding commit and open
   a **draft PR early** so CI + the Read the Docs (RTD) preview run while you build.
6. **Automated review gate (machine-enforced, not eyeballed).** Run `/review` (reviews the PR
   diff) and `/security-review` (scans the pending diff for vulnerabilities). The quality gates
   below must be green first. **Never merge red** — automated checks *gate* the human review,
   they do not replace it.
7. **Refine on the same branch.** Back to plan mode; push more commits (the PR and the RTD
   preview update automatically).
8. **Keep current.** Periodically merge `master` → branch (plain git). A good fit for the
   `/schedule` skill: auto-**sync** each morning so you wake to a synced branch or an early
   conflict warning. **A scheduled job syncs only — it must never resolve conflicts or merge a
   branch to `master` unattended; it just flags you.** (PR auto-merge in step 9 is a different,
   safe mechanism — GitHub completes it only when checks are green *and* the branch merges cleanly.)
9. **Arm auto-merge.** Once the step-6 review is green and you've read the RTD preview + PR diff,
   enable GitHub-native auto-merge: `gh pr merge --auto --squash`. GitHub then merges the moment
   every required check passes and the branch is conflict-free, so **"never merge red" still
   holds** — a red check blocks the merge instead of completing it. For a hard human gate on a
   given PR, skip `--auto` and merge manually once green.
10. **Auto-fix red CI.** If GitHub Actions reports a failure (whether or not auto-merge is armed),
    pull the failing logs (`gh run view --log-failed`, or `gh run watch` to follow live),
    reproduce locally, fix **forward on the same branch**, and push. Armed auto-merge re-arms
    itself and completes once the re-run is green — no need to re-issue the merge. Disarm with
    `gh pr merge --disable-auto` if you need to hold the PR.
11. **Delete the branch.** Plain git, with permission (root `CLAUDE.md` §0).

## Quality gates — how AAanalysis checks each

> **Rule: never merge red.** These automated checks gate the merge. With
> `gh pr merge --auto` GitHub enforces it for you — it completes the merge only once every
> required check is green and the branch is conflict-free.

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
- **Auto-merge + auto-fix loop.** `gh pr merge --auto --squash` is the default finish: it is
  safe because GitHub merges only on all-green + conflict-free, preserving *never merge red*. When
  a check goes red, **fix forward on the same branch** (`gh run view --log-failed` → reproduce
  locally → push) — the armed auto-merge needs no re-issuing and completes on the green re-run.
  Use `gh pr merge --disable-auto` to hold a PR. Arming auto-merge is still a publish action, so
  it needs the per-action go-ahead in root `CLAUDE.md` §0, exactly like a manual merge.
- **Notebooks are a local-only gate.** Because nbmake is not in blocking CI, a broken example
  surfaces only on RTD (as wrong/un-rendered output) or in a local run. Always run
  `pytest --nbmake examples/ tutorials/` and commit fresh outputs before pushing.
