---
name: agentic-engineering
description: Drive a change through the AAanalysis agentic-engineering protocol step by step — pick/sharpen the issue, branch into an isolated worktree, implement, open a draft PR, run the automated review + quality gates, keep the branch synced, then arm GitHub-native auto-merge with a fix-forward loop on red CI. Use when the user wants to start work on an issue, "walk me through the workflow", take a change from issue to merge, or wants the auto-merge / auto-fix loop driven for them. The canonical protocol lives in docs/guides/agentic_engineering.md; this skill executes it.
---

# AAanalysis agentic-engineering — drive a change to merge

Step-by-step driver for the protocol in **`docs/guides/agentic_engineering.md`** (the single
source of truth — read it first; this skill executes it). The durable artifacts (ADRs,
`CONTEXT.md`, `CLAUDE.md` / `.claude/rules`, the code) are authoritative; per-issue planning
notes are ephemeral and never committed.

Work the steps in order. Stop and surface to the user at each gate rather than barrelling
through — especially the CONFIRM-FIRST surfaces and the §0 hard rules below.

## Hard rules (override everything here)

From root `CLAUDE.md` §0 — authorization is **per-action, never per-session**:

- **Never delete/rename a file without explicit permission** (includes `git worktree remove` of
  a tree with uncommitted work, branch deletion, `git mv`).
- **Never push or publish without explicit permission** — `git push`, `gh pr create`,
  `gh pr merge` (including arming `--auto`), `gh release create`. Ask again for the next one.
- **CONFIRM-FIRST files** (CLAUDE.md §2): `pyproject.toml`, `aaanalysis/__init__.py`, anything in
  `aaanalysis/_data/`, `.github/workflows/*`, `config.py`, `template_classes.py`, and any rename/
  delete of a symbol in `__all__`. Ask before editing.

## Steps

1. **Pick the issue.** Optionally sharpen the wording first with `/triage` or `/to-issues`
   (house style: `docs/guides/issue_style_guide.rst`). For "what next?", `/github-issue-handoff`
   produces a prioritized, parallelization-aware plan.
2. **`/grill-with-docs` — the highest-leverage step.** Sharpen the spec against the *live*
   codebase and refresh `CONTEXT.md` / ADRs **before any code is written**. Do not skip to
   implementation on anything non-trivial. (`/init` only bootstraps codebase docs when
   `CONTEXT.md` is missing — it is not a substitute for the adversarial spec-vs-reality pass.)
3. **Branch + isolated worktree.** `git switch -c <type>/<slug>` off `master`, **always paired
   with `git worktree add`** so concurrent streams never share a checkout (`fix/`, `feat/`,
   `doc/`, `refactor/`). Do the edits in the worktree; remove it (with permission) when done.
4. **Implement.** Use plan mode for multi-file or architectural changes so the approach is
   approved before commits land; drop to plain edits for trivial diffs. Honor the path-scoped
   rules in `.claude/rules/` that auto-load for the files you touch.
5. **Push → draft PR early.** A PR needs ≥1 commit: push a scaffolding commit and open a **draft
   PR** so CI + the Read the Docs preview run while you build. (Push needs §0 go-ahead.)
6. **Automated review gate.** Run `/review` (PR diff) and `/security-review` (vulnerability scan).
   The quality gates in the canonical doc's table must be green first. **Never merge red** — these
   gate the human review, they do not replace it.
7. **Refine on the same branch.** Push more commits; the PR and RTD preview update automatically.
8. **Keep current.** Periodically merge `master` → branch. Good fit for `/schedule`: an auto-**sync**
   each morning. A scheduled job **syncs only** — it must never resolve conflicts or merge a branch
   to `master` unattended; it just flags you.
9. **Arm auto-merge.** Once step-6 is green and you've read the RTD preview + PR diff, enable
   GitHub-native auto-merge: `gh pr merge --auto --squash`. GitHub merges the moment every required
   check passes and the branch is conflict-free, so **"never merge red" still holds** — a red check
   blocks the merge instead of completing it. Skip `--auto` and merge manually for a hard human gate.
   *Arming auto-merge is a publish action → needs §0 go-ahead.*
10. **Auto-fix red CI (fix-forward loop).** If GitHub Actions reports a failure (armed or not):
    - `gh run watch` to follow live, or `gh run view --log-failed` for the failing logs.
    - **Reproduce locally** before fixing — run the failing gate (see *Local gate commands*).
    - Fix **forward on the same branch** and push. Armed auto-merge re-arms itself and completes
      on the green re-run; do not re-issue the merge. Use `gh pr merge --disable-auto` to hold it.
    - Loop until green. Don't paper over a real failure to force a merge — surface it.
11. **Delete the branch + worktree.** Plain git, **with permission** (§0).

## Local gate commands (reproduce CI before fixing)

Run with `-c tests/pytest.ini` (carries `filterwarnings` + markers). Coverage on the package only.

```bash
pytest tests -m "not regression" -x -n auto -c tests/pytest.ini          # Unit Tests gate
pytest --cov=aaanalysis --cov-fail-under=88 tests                        # Coverage gate (≥88%)
pytest tests/unit/api_tests/test_param_coverage.py -x -vv -c tests/pytest.ini   # Param coverage
flake8 . --select=E9,F63,F7,F82                                          # Lint (errors) — CodeQL job
cd docs && make html                                                     # Docs build (RTD)
pytest --nbmake --nbmake-timeout=120 examples/ tutorials/               # Notebooks — LOCAL gate only
```

`/docstrings` runs the docstring checkers (`check_docstrings.py`, `doc_signature_drift.py`,
`check_example_notebooks.py`) — a local gate, not yet a CI job.

## Notes

- **Fix forward, never merge red.** Auto-merge is safe precisely because GitHub only completes it
  on all-green + conflict-free. The auto-fix loop turns a red check into another commit, not a
  reason to override the gate.
- **Issue lifecycle.** To auto-close on merge, keep `Closes #NN` in the **PR body**. To keep the
  issue open through a merge, **remove the keyword from the PR body** — editing only the commit
  message is not enough.
- **Notebooks are a local-only gate** (nbmake is not in blocking CI). Re-run
  `pytest --nbmake examples/ tutorials/` and commit fresh outputs before every push.
