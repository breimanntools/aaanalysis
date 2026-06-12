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

Seven steps in three phases. **Full rationale + the quality-gates table live in
`docs/guides/agentic_engineering.md`** — read it first; this checklist drives it.

**Prepare**

1. **Pick & sharpen the issue.** Optionally `/triage` or `/to-issues` for wording (house style:
   `docs/guides/issue_style_guide.rst`); `/github-issue-handoff` for a prioritized "what next?" plan.
2. **`/grill-with-docs`** — sharpen the spec against the *live* codebase and refresh `CONTEXT.md` /
   ADRs **before any code**. Don't skip on anything non-trivial. (`/init` only bootstraps codebase
   docs when `CONTEXT.md` is missing — not a substitute for the adversarial spec-vs-reality pass.)
3. **Branch + isolated worktree.** `git switch -c <type>/<slug>` off `master`, **always paired with
   `git worktree add`** (`fix/`, `feat/`, `doc/`, `refactor/`) so concurrent streams never share a
   checkout. Even in a shared checkout, re-check `git status` before committing, stage **explicit
   pathspecs only** (never a blind `git add -A` / `git commit -a`), and never commit, revert, or
   discard changes you did not make — stop and surface unexpected edits instead.

**Build**

4. **Implement + open a draft PR early.** Plan mode for multi-file / architectural changes (approve
   the approach before commits land); plain edits for trivial diffs. Honor the auto-loaded
   `.claude/rules/` for the files you touch. A PR needs ≥1 commit: push a scaffolding commit and open
   a **draft PR** so CI + the RTD preview run while you keep pushing to refine. (Push → §0 go-ahead.)
   **Before a substantive push, run the fast local unit gate** (`pytest tests -m "not regression" -x
   -n auto -c tests/pytest.ini`) — far faster than a red-CI round-trip. **No change is done until you
   walk the Ripple checklist below** — the code edit usually lands with its mirrors *in the same PR*.
5. **Automated review gate.** Run `/review` (PR diff) + `/security-review` (vulnerability scan); for a
   substantial diff also `/code-review high` (or `ultra` for the deep cloud review) + `/simplify`, and
   `/docstrings` when public API or docstrings change. The canonical doc's quality gates must be green.
   **Never merge red** — these gate the human review, they do not replace it. *Meanwhile:* periodically
   sync `master` → branch (`/schedule` fits) — **sync only**, never resolve conflicts or merge to
   `master` unattended; it just flags you.

**Merge & clean up**

6. **Arm auto-merge; fix-forward on red.** Once green and you've read the RTD preview + PR diff:
   `gh pr merge --auto --squash` (*a publish action → §0 go-ahead*). GitHub merges only on all-green
   + conflict-free, so *never merge red* holds. On red CI: `gh run watch` / `gh run view --log-failed`
   → **reproduce locally** (see *Local gate commands*) → fix **forward on the same branch** and push;
   armed auto-merge re-arms and completes on the green re-run. `gh pr merge --disable-auto` to hold;
   skip `--auto` for a hard human gate. Don't paper over a real failure to force a merge — surface it.
7. **Clean up — gated on merge + a green `master` (with permission, §0).** Trigger off **merge
   state, not a CI run**: wait until `gh pr view <n> --json state,mergedAt` shows `MERGED`, then
   let the push-triggered `master` workflows pass (confirms the squash is clean). An intervening
   push from any session just re-runs checks; armed auto-merge waits for the new green, so the
   trigger survives. Confirm no work is lost (`git branch --merged master` lists it; worktree
   `git status --porcelain` empty), then `git switch master` → `git worktree remove <path>`
   (uncommitted work needs `--force` → also permission) → `git worktree prune` →
   `git branch -d <branch>`; remote head branch auto-deletes on squash-merge if the repo setting
   is on, else `git push origin --delete <branch>` (push → §0). A scheduled/unattended job may
   *flag* "ready to clean up" but never deletes on its own (§0).

## Ripple checklist (no change is done until its mirrors are in sync)

A code edit almost always lands with its mirrors **in the same PR** (tutorials may trail, but never
a different release). The package is heavily cross-referenced and meta-tested, so walk this when you
touch code — full rationale + exact paths are in the guide's *Propagate every change* section:

- **Docstrings** (numpydoc; citations → `references.rst`) — `/docstrings`, now blocking CI.
- **Public API** — `aaanalysis/__init__.py` `__all__` (CONFIRM-FIRST); API ref + autosummary follow.
- **Examples** — `examples/<abbr>_<method>.ipynb` (one per method, included in the docstring); cover
  every param, re-run with executed outputs.
- **Tutorials** — `tutorials/*.ipynb`; **Protocols** — `protocols/protocol<N>_*.ipynb` (workflow changes).
- **Tests** — the change's unit tests **+** cross-file meta-tests: `test_param_coverage.py`,
  `test_class_abbreviation_registry.py`, backend-import-hygiene, extras/stub parity.
- **Cheat sheet** — `docs/_cheatsheet/content.py` (single source → regen html/pdf; public symbols only).
- **Tables** — `docs/source/index/tables*.rst` via `create_tables_doc.py` (scales/datasets changes).
- **Release notes** — `docs/source/index/release_notes.rst` (the changelog; *Unreleased* section).
- **Contributing** — `CONTRIBUTING.rst` **+** its port `docs/source/index/CONTRIBUTING_COPY.rst`.
- **Glossary / ADRs** — `CONTEXT.md`; `docs/adr/NNNN-*.md`. **Conventions** — `CLAUDE.md` / `.claude/rules/*`.
- **Build / deps** — `pyproject.toml` / `config.py` (both CONFIRM-FIRST).

Most surface late (stale cheat sheet, red meta-test, wrong RTD render), not in the fast unit job.

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
`check_example_notebooks.py`). The first two are **blocking CI** in `codeql_analysis.yml`
("code-quality" job); `check_example_notebooks` runs there **advisory (non-blocking)** until the
remaining notebook param-coverage gaps are cleared. All three also run locally via this skill.

## Notes

- **Fix forward, never merge red.** Auto-merge is safe precisely because GitHub only completes it
  on all-green + conflict-free. The auto-fix loop turns a red check into another commit, not a
  reason to override the gate.
- **Issue lifecycle.** To auto-close on merge, keep `Closes #NN` in the **PR body**. To keep the
  issue open through a merge, **remove the keyword from the PR body** — editing only the commit
  message is not enough.
- **Notebooks are a local-only gate** (nbmake is not in blocking CI). Re-run
  `pytest --nbmake examples/ tutorials/` and commit fresh outputs before every push.
