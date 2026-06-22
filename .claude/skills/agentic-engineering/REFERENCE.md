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

## Merge method — merge commits, never squash

**Land every PR as a merge commit (`gh pr merge --auto --merge`). Do not pass `--squash`.** Two
reasons, both learned the hard way:

- **Squash rewrites history into a new SHA**, so the branch's own commits are never reachable from
  `master`. That makes `git branch --merged master` omit the branch and `git branch -d` refuse it
  ("not fully merged") — the cleanup detection in step 8 goes blind, branches pile up across parallel
  sessions, and you're forced into `git branch -D` (force) + `git diff master...<branch>` workarounds
  that are easy to get wrong. Merge commits keep the branch reachable, so `--merged` / `-d` just work.
- **Squash silently collapses the individual commits** (a scaffold + a fix + a style commit become
  one), losing the per-commit narrative the reviewer and `git log` rely on.

The merge **method is its own explicit decision** — never fold `--squash`/`--merge` into the wording
of the step-6 "skip review → auto-merge" option. A real incident: a PR was squash-merged because the
flag rode along inside an auto-merge option the user picked for its *review-skip* meaning. If the user
has not stated a method, ask; the standing default for this repo is `--merge`.

## ADR numbering under parallel sessions

The ADR number is the one field that can't be derived safely from a local checkout. Each
concurrent branch computes "next = highest committed `NNNN` + 1" against its *own* base, and
because the other branches' ADRs aren't on `master` yet, several pick the **same** number — or,
once some merge out of order, the sequence gains a gap. A real incident left `0034`–`0036` written
on unmerged branches and missing from the committed `INDEX.md`, with a fourth session about to
reuse `0034`.

The fix is **late allocation against live state**, plus letting git surface the rest:

- **Draft number-less.** Title `# ADR-XXXX — <decision>`, filename `docs/adr/XXXX-<slug>.md`. A
  slug-keyed file never collides on *path* with another session's draft, and `XXXX` makes an
  un-numbered ADR trivial to grep for before merge.
- **De-dup the decision, not just the number.** Before writing, `git fetch origin --prune` then
  scan open PRs/branches (`gh pr list`; grep their diffs for `docs/adr/`) for an ADR covering the
  same decision — settling it in step 1 via `/grill-with-docs` is what stops two sessions opening
  rival ADRs for one decision.
- **Allocate as the last step before the ADR's PR merges.** Rebase on a freshly-fetched
  `origin/master`, compute `max + 1` over **both** committed `docs/adr/NNNN-*.md` **and** numbers
  claimed in open PRs, rename the file + title, then regenerate the table:
  `python .claude/skills/agent-readiness-audit/scripts/check_adrs.py --write-index`.
- **Let the collision become a conflict.** Two ADRs racing for the same number both edit the same
  `INDEX.md` row region and both add a sequential filename, so a genuine duplicate shows up as a
  merge conflict on the second PR rather than landing silently — resolve it by taking the next free
  number, never by force.
- **Never renumber a *merged* ADR.** Its number is now referenced and the rename deletes a path
  (§0). If a duplicate already merged, supersede or correct forward with the maintainer's go-ahead.

`check_adrs.py` now **fails on a duplicate number** (`ADR-DUP-NUMBER` defect; an un-numbered
`XXXX-<slug>.md` draft is advisory). It is gated in CI by `.github/workflows/adr_hygiene.yml`,
which is scoped `paths: ['docs/adr/**']` so it actually runs on ADR-only PRs — the CodeQL /
code-quality job carries `paths-ignore: docs/**` and would skip exactly those. Number-last
drafting is still the *prevention*; the checker + the `INDEX.md` merge conflict are the backstop.
Gaps are deliberately not flagged (an ADR on an unmerged branch leaves a normal hole).

## Session self-titling (telling concurrent sessions apart)

Several sessions run this repo at once, each in its own worktree, and by default every terminal
tab looks identical (`aaanalysis — <generic> — …`), so you cannot tell which tab is which task. A
machine-local hook fixes this: after a session has used ~1% of its context window it sets the
terminal tab title to a descriptive label and keeps it current.

- **Format:** `<topic> · PR#<n> · ADR<nnnn>` — e.g. `adr-parallel-fix · PR#233 · ADR0038`.
  - *topic* = the branch slug with its `doc/` / `feat/` prefix stripped (the human-authored "what
    this is about"); falls back to the worktree/repo dir name on `master`/detached.
  - *PR#* = the open PR for the branch (one `gh` call, cached once found).
  - *ADR* = new `docs/adr/NNNN-*.md` files introduced on the branch (added-vs-`origin/master` plus
    still-untracked).
- **When:** a `Stop` hook re-evaluates each turn past the ~1% gate but only re-writes the title
  when the label actually changes — so it is silent except when the branch, PR, or ADR first
  appears (effectively "set once, then refreshed on meaningful change"). It writes only to
  `/dev/tty` and always exits 0.
- **Why it matters for this protocol:** the title is only as descriptive as the branch slug, so the
  step-2 worktree command should name the branch for the task (`feat/<slug>` / `doc/<slug>`), not a
  throwaway. The PR/ADR segments fill in automatically as step 4 (PR) and any ADR land.
- **Mechanism (local, not committed):** `~/.claude/set_session_title.py` invoked by a global `Stop`
  hook in `~/.claude/settings.json`. Tunables: `CLAUDE_TITLE_THRESHOLD_CHARS` (default 8000 ≈ 1% of
  a 200k window), `CLAUDE_TITLE_MIN_TURNS`. A global-settings edit needs a `/hooks` reload (or a
  restart) to take effect in an already-running session; new sessions pick it up automatically.

## Parallel-session hazards (concurrent streams, one repo)

Several sessions/agents work this repo at the same time. Almost every "what was already done?"
confusion traces to **one session acting on stale knowledge of another's merges/deletes**. The
mitigations SKILL.md's *Parallel sessions* section summarizes, in full:

- **Isolation is per-task worktrees, full stop.** One `git worktree` (or separate clone) per
  concurrent stream — never a shared checkout / `HEAD`. A shared tree lets a concurrent commit/push
  land *between* your `git status` and your commit (observed: an unrelated refactor was committed by
  another process mid-task). If you must share a tree, it is safe only for strictly serial work.
- **Live state is derived, never remembered.** Session memory, a prior turn's `gh pr list`, or a
  screenshot are all instantly stale. Before any status claim / commit / merge / cleanup:
  `git fetch origin --prune` (drops remote-tracking refs for branches another session already
  merged+deleted — the usual "ghost branch that looks ahead/unmerged") **and** a fresh
  `gh pr list` / `gh pr view <n> --json state,mergedAt`.
- **Squash made this worse; merge commits make detection honest.** Under squash, a merged branch
  looked unmerged to git, so "is this forgotten work?" was unanswerable from git alone and the
  screenshot-driven guesses were wrong. With merge commits, `git branch --merged master` is the
  truth, and `prune_merged_branches.py` cross-checks PR state on top.
- **Only ever act on the branch/worktree you created.** Enumerate others with `git worktree list` +
  `gh pr list`; an unexpected branch/worktree/edit is someone else's in-flight work — **stop and
  surface it**, never commit/revert/delete/clean it. A non-empty `git diff master...<branch>` on a
  branch with **no PR** is *forgotten work* — flag it for a human, never delete it.
- **Cleanup runs from any session and is idempotent.** Parallel auto-merges land after the opening
  session ends, so `python .github/scripts/prune_merged_branches.py` (report-only by default,
  `--apply` to delete) is the reconcile tool — run it from any session; it is PR-state-driven and
  never touches FORGOTTEN no-PR branches. A scheduled/unattended job may *flag* but never deletes (§0).

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
