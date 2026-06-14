# agentic-engineering — reference (deep detail)

Companion to [SKILL.md](SKILL.md). The canonical protocol is
`docs/guides/agentic_engineering.md` (single source of truth). This file holds the
deep detail — hazards, the delegation model, and the drift-prone CI commands — so
SKILL.md stays a lean step-by-step driver.

## Delegated skills (orchestration model)

This skill is an **orchestrator**: it owns the lifecycle (issue → worktree → PR →
review → merge → cleanup) and delegates each specialist gate to another local
sub-skill. The slash commands in SKILL.md are those sub-skills, **not** shell or
GitHub commands:

| Sub-skill | Gate it owns |
|---|---|
| `/grill-with-docs` | spec-vs-code reality check + `CONTEXT.md` / ADR refresh (step 2) |
| `/review` | PR-diff review for correctness, regressions, missed ripple effects |
| `/security-review` | vulnerability scan of the pending diff |
| `/code-review high` / `ultra` | deeper / cloud quality review for substantial diffs |
| `/simplify` | complexity / reuse cleanup |
| `/docstrings` | public-API & docstring gate (numpydoc shape, drift) |
| `/github-issues`, `/triage`, `/to-issues` | issue selection / wording (step 1) |
| `/schedule` | the background `master`→branch sync (sync-only) |

**Failure rule (why it matters).** If a required sub-skill is **unavailable, fails,
or returns an inconclusive result, stop and surface the missing gate** — do not
silently continue. Otherwise the orchestrator behaves as if `/security-review` or
`/docstrings` *passed* when it never actually ran, and a real gate is skipped under
the appearance of green. Continue only after the user explicitly approves a manual
fallback (you run the equivalent check by hand) or an explicit skip of that gate.
The human permission gates (push, PR, merge, deletion, CONFIRM-FIRST files) override
every delegated result regardless.

## Git-stash hazard (why isolation = worktrees, never stash)

**Never fake isolation with `git stash` (this caused a real incident).** Isolation comes from
`git worktree add` — *never* from stashing. The shared stash stack is **global, not per-branch**:
a bare `git stash pop` / `git stash apply` operates on whatever is on top, which may be a
**pre-existing stash you did not create** (an earlier session's or teammate's WIP). Worse, a
*failed* `git stash push` (e.g. a pathspec that doesn't match a **staged deletion** — `git rm`
leaves nothing in the worktree to match) creates **no** stash and returns non-zero; a following
`git stash pop` then silently applies *someone else's* stash, producing conflicts and apparent
data loss. Rules:

- **Don't stash to "move work onto a fresh base."** To land uncommitted edits on a clean base,
  either `git worktree add ../wt-<slug> -b <type>/<slug> origin/master` and make the edits there,
  or `git switch -c <branch> origin/master` (git carries clean working changes across the switch).
  If the switch reports a conflict, **stop and surface it** — do not stash to force it.
- **If you must stash:** run `git stash list` first; `git stash push -m <msg> -- <explicit paths>`
  and **verify it was actually created** (check the exit status *and* that `git stash list` grew);
  then pop **by explicit ref you confirmed is yours** (`git stash pop stash@{0}`), never a bare
  `git stash pop`. Always check a git command's exit status before running the next one.

## CI specifics drift — verify, don't trust

The *Local gate commands* block below and the guide's quality-gates table name concrete
jobs/thresholds (`codeql_analysis.yml` "code-quality" job runs the docstring checkers; coverage
≥ 88% in `test_coverage.yml`; push-to-`master` triggers the Unit Tests / Coverage / CodeQL /
Integration & E2E workflows). These are accurate as of writing but **rot when CI changes** —
confirm against `.github/workflows/*` and `gh pr checks <n>`, and treat
`docs/guides/agentic_engineering.md` as the source of truth. SKILL.md intentionally carries no
threshold numbers — it points here and tells you to verify the live configuration first.

## Local gate commands (reproduce CI before fixing)

Run with `-c tests/pytest.ini` (carries `filterwarnings` + markers). Coverage is on the package
only (`--cov=aaanalysis`, never `--cov=./`). The thresholds/job names below are the drift-prone
ones — verify them live per the section above before claiming a gate's status.

```bash
pytest tests -m "not regression" -x -n auto -c tests/pytest.ini          # Unit Tests gate (fast)
pytest --cov=aaanalysis --cov-fail-under=88 tests                        # Coverage gate (≥88%)
pytest tests/unit/api_tests/test_param_coverage.py -x -vv -c tests/pytest.ini   # Param coverage
flake8 . --select=E9,F63,F7,F82                                          # Lint (errors) — CodeQL job
cd docs && make html                                                     # Docs build (RTD)
pytest --nbmake --nbmake-timeout=120 examples/ tutorials/               # Notebooks — LOCAL gate only
```

`/docstrings` runs the docstring checkers (`check_docstrings.py`, `doc_signature_drift.py`,
`check_example_notebooks.py`). The first two are **blocking CI** in `codeql_analysis.yml`
("code-quality" job); `check_example_notebooks` runs there **advisory (non-blocking)** until the
remaining notebook param-coverage gaps are cleared. All three also run locally via the skill.
Notebooks are a **local-only** gate (nbmake is not in blocking CI) — re-run and re-commit fresh
outputs before every push.
