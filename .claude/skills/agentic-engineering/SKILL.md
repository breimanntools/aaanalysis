---
name: agentic-engineering
description: Autonomously drive an entire issue or a complete new feature from spec to merge through the AAanalysis agentic-engineering protocol — sharpen the issue, branch into an isolated worktree, implement, push and open the PR early so CI runs, run the automated review + quality gates, keep the branch synced, then hold at the human-review gate where the user picks a manual PR-review loop or skips (skip = post an approving review comment, then arm GitHub-native auto-merge), with a fix-forward loop on red CI. Use when the user wants to start work on an issue, "walk me through the workflow", take a change from issue to merge, or wants the auto-merge / auto-fix loop driven for them. NOT for typos, one-line fixes, or trivial local edits — make those directly. The canonical protocol lives in docs/guides/agentic_engineering.md; this skill executes it.
---

# AAanalysis agentic-engineering — drive a change to merge

Step-by-step driver for the protocol in **`docs/guides/agentic_engineering.md`** (the single
source of truth — read it first; this skill executes it). Durable artifacts (ADRs, `CONTEXT.md`,
`CLAUDE.md` / `.claude/rules`, the code) are authoritative; per-issue planning notes are ephemeral.

**When to use:** to autonomously drive an **entire issue or a complete new feature** from spec to
merge (a substantial bug-fix issue counts). **Not** for typos, one-line fixes, or trivial local
edits — make those directly.

## Happy path (read this first)

`⛔` = stop for an explicit §0 permission; each is its **own** ask (push ≠ PR ≠ merge ≠ each cleanup deletion).

1. Sharpen the issue → **`/grill-with-docs`** (refresh `CONTEXT.md` / ADRs) **before any code**.
2. Worktree: `git fetch origin && git worktree add ../wt-<slug> -b <type>/<slug> origin/master`
3. Implement (honor auto-loaded `.claude/rules/`) + walk the **Ripple checklist** below.
4. Fast local gate → ⛔ push scaffold commit → open a **draft PR** (this starts CI + the RTD preview).
5. **`/review`** + **`/security-review`** (+ `/code-review high`, `/simplify`, `/docstrings` for big/API diffs). **Never merge red.**
6. ⛔ **Human gate** — user picks **(a)** manual review loop or **(b)** skip → post an approving comment.
7. ⛔ `gh pr merge --auto --merge`; fix-forward on red (armed auto-merge completes on the green re-run).
8. ⛔ After the PR is **MERGED** + `master` is green: clean up (worktree remove · `branch -d` · remote delete — three separate asks).

> **Parallel sessions are the norm here.** Before any status claim, commit, merge, or cleanup, refresh
> live state (`git fetch origin --prune` + `gh pr list` / `gh pr view`) and act **only** on the
> branch/worktree *you* created — see [Parallel sessions](#parallel-sessions-dont-let-concurrent-streams-collide).

## Delegated skills

The slash commands here (`/grill-with-docs`, `/review`, `/security-review`, `/code-review`,
`/simplify`, `/docstrings`, `/github-issues`, `/triage`, `/to-issues`, `/schedule`) are **local
sub-skills this skill orchestrates** — not shell or GitHub commands. If a required sub-skill is
**unavailable, fails, or returns an inconclusive result, stop and surface the missing gate**;
continue only after the user approves a manual fallback or an explicit skip. Never proceed as if a
gate ran when it did not. (Why: [REFERENCE.md](REFERENCE.md) → *Delegated skills*.)

## Hard rules (override everything here)

From root `CLAUDE.md` §0 — authorization is **per-action, never per-session**:

- **Never delete/rename a file without explicit permission** (includes `git worktree remove` of a
  tree with uncommitted work, branch deletion, `git mv`).
- **Never push or publish without explicit permission** — `git push`, `gh pr create`,
  `gh pr merge` (including arming `--auto`), `gh release create`. Ask again for the next one.
- **CONFIRM-FIRST files** (CLAUDE.md §2): `pyproject.toml`, `aaanalysis/__init__.py`, anything in
  `aaanalysis/_data/`, `.github/workflows/*`, `config.py`, `template_classes.py`, and any rename/
  delete of a symbol in `__all__`. Ask before editing.

## Steps (detail behind the happy path)

Eight steps in three phases. **Full rationale + the quality-gates table live in
`docs/guides/agentic_engineering.md`** — read it first; this drives it.

**Prepare**

1. **Pick & sharpen the issue.** Optionally `/triage` or `/to-issues` for wording (house style:
   `docs/guides/issue_style_guide.md`); `/github-issues` for a prioritized "what next?" plan.
2. **`/grill-with-docs`** — sharpen the spec against the *live* codebase and refresh `CONTEXT.md` /
   ADRs **before any code**. Don't skip on anything non-trivial. (`/init` only bootstraps codebase
   docs when `CONTEXT.md` is missing — not a substitute for the adversarial spec-vs-reality pass.)
3. **Branch + isolated worktree.** Use the canonical command (no ambiguity):

   ```bash
   git fetch origin
   git worktree add ../wt-<slug> -b <type>/<slug> origin/master   # fix/ feat/ doc/ refactor/
   cd ../wt-<slug>
   ```

   A worktree (not `git stash`) is how concurrent streams stay isolated. Even in a shared checkout,
   re-check `git status` before committing, stage **explicit pathspecs only** (never a blind
   `git add -A` / `git commit -a`), and never commit, revert, or discard changes you did not make —
   stop and surface unexpected edits instead. **Never fake isolation with `git stash`** (a real
   incident — the stash stack is *global*, not per-branch). Hazard analysis + safe-stash rules:
   [REFERENCE.md](REFERENCE.md) → *Git-stash hazard*.

**Build**

4. **Implement + open a draft PR early.** Plan mode for multi-file / architectural changes; plain
   edits for trivial diffs. Honor the auto-loaded `.claude/rules/` for files you touch. Push a
   scaffolding commit and open a **draft PR** so CI + the RTD preview run while you keep pushing
   (push → §0). **Open the PR *before* the human gate (step 6), not after** — that is what starts
   CI/RTD. **Before a substantive push, run the fast local unit gate** (see *Local gates*). **No
   change is done until you walk the Ripple checklist** — code lands with its mirrors *in the same PR*.
5. **Automated review gate.** `/review` (PR diff) + `/security-review`; for a substantial diff also
   `/code-review high` (or `ultra`) + `/simplify`, and `/docstrings` when public API or docstrings
   change. The guide's quality gates must be green. **Never merge red** — these gate the human
   review, they don't replace it. *Meanwhile:* periodically sync `master` → branch (`/schedule`
   fits) — **sync only**, never resolve conflicts or merge to `master` unattended.

**Review, merge & clean up**

6. **Human review gate — the PR is already up; the user picks.** CI/Actions + RTD are already running
   (confirm `gh pr checks <n>`). This is human judgement on the *content* — not a push decision (that
   happened in step 4). Do **not** advance on your own; surface the fork and **wait**:
   - **(a) Manual review (iterate).** Address each comment by refactoring **forward on the same
     branch** (Ripple checklist + `.claude/rules/`), re-run the fast local gate, push (each re-push →
     §0), report per comment. **Loop** until the user says review is done; only then step 7.
   - **(b) Skip → approve + auto-merge.** Post a short **approving review comment** so the skip is
     recorded, then go to step 7.
   Recommend (a) for substantial/architectural diffs, (b) for trivial ones — but **never assume.**
7. **Arm auto-merge; fix-forward on red.** After step 6 clears (on skip, after the approving comment)
   and you've read the RTD preview + diff: **`gh pr merge --auto --merge`** (publish → §0). The merge
   **method** is its own explicit choice — don't fold it into the step-6 skip option (why `--merge`:
   [REFERENCE.md](REFERENCE.md) → *Merge method*). GitHub merges only on all-green + conflict-free, so
   *never merge red* holds. On red: `gh run view --log-failed` → reproduce locally → fix **forward on
   the same branch** → push; armed auto-merge completes on the green re-run. `gh pr merge --disable-auto`
   to hold. Don't paper over a real failure to force a merge.
8. **Clean up — gated on merge + a green `master` (with permission, §0).** Trigger off **merge state,
   not a CI run**: wait until `gh pr view <n> --json state,mergedAt` shows `MERGED`, then let the
   push-triggered `master` workflows pass. Because PRs land as **merge commits**, the
   branch's commits stay reachable from `master`, so `git branch --merged master` lists it and a plain
   **`git branch -d <branch>`** deletes it safely (no `-D` force, no `git diff` workaround needed).
   First `git fetch origin --prune` to drop stale remote-tracking refs another session's merge left
   behind. As separate §0 asks: `git switch master` → `git worktree remove <path>` (uncommitted work
   needs `--force` → also permission) → `git worktree prune` → `git branch -d <branch>`; remote head
   auto-deletes only if the repo's "automatically delete head branches" setting is on, else
   `git push origin --delete <branch>` (push → §0). **Canonical tool:**
   `python .github/scripts/prune_merged_branches.py` (PR-state-driven; report-only by default, `--apply`
   to delete, never touches FORGOTTEN no-PR work) — prefer it over ad-hoc deletes and run it from *any*
   session, since parallel auto-merges often land after the opening session ended. A scheduled/unattended
   job may *flag* "ready to clean up" but never deletes on its own (§0).

## Parallel sessions (don't let concurrent streams collide)

Multiple sessions/agents run against this repo at once. The failure mode (a real, repeated incident):
one session acts on **stale knowledge** of what another already merged, deleted, or is mid-edit — so
branches pile up, "is this forgotten work?" gets misjudged, and cleanup touches the wrong thing.
Discipline:

- **One worktree per task, always** (step 3) — never share a checkout / `HEAD`. Mandatory with
  parallel agents: in a shared tree a concurrent commit/push can land *between* your `git status` and
  your command.
- **Derive live state; never trust session memory or a screenshot.** Before any status claim, commit,
  merge, or cleanup, run `git fetch origin --prune` **and** `gh pr list` / `gh pr view <n> --json
  state,mergedAt`. `--prune` drops stale remote-tracking refs another session's merge+delete left
  behind (the usual source of "ghost" branches that look unmerged/ahead).
- **Act only on the branch/worktree *you* created.** Discover what else is in flight with
  `git worktree list` + `gh pr list`. If you find a branch, worktree, or working-tree edit you did not
  make, **stop and surface it** — never commit, revert, delete, or "clean up" another stream's work.
- **Re-check immediately before the act**, not only at the start — re-run `git status` +
  `gh pr view` right before staging / merging / deleting.
- **Cleanup is idempotent + PR-state-driven** (`prune_merged_branches.py`), safe to run from *any*
  session; parallel auto-merges routinely land after the opening session ended, so end-of-session
  cleanup alone can't keep up. A scheduled/unattended job may *flag* but never deletes (§0).

Full hazard analysis: [REFERENCE.md](REFERENCE.md) → *Parallel-session hazards*.

## Ripple checklist (no change is done until its mirrors are in sync)

A code edit almost always lands with its mirrors **in the same PR** (tutorials may trail, but never
a different release). Full rationale + exact paths: the guide's *Propagate every change* section.

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

## Local gates & CI

The exact commands, job names, and thresholds (fast unit gate, coverage ≥ floor, param coverage,
lint, docs build, notebooks) live in **[REFERENCE.md](REFERENCE.md) → *Local gate commands***. CI
job names and thresholds drift — **verify the live configuration** (`.github/workflows/*`,
`gh pr checks <n>`) before claiming any gate's status; treat the guide as the source of truth.

## Notes

- **Merge with `gh pr merge --auto --merge`** — the method is its own explicit decision, never bundled
  into the step-6 skip-review option. Rationale: [REFERENCE.md](REFERENCE.md) → *Merge method*.
- **Worktrees, not `git stash`, for isolation.** See step 3 / [REFERENCE.md](REFERENCE.md).
- **Parallel sessions: derive live state, touch only your own branch.** See *Parallel sessions* above.
- **Fix forward, never merge red.** Auto-merge is safe precisely because GitHub only completes it
  on all-green + conflict-free; the auto-fix loop turns a red check into another commit.
- **Issue lifecycle.** To auto-close on merge, keep `Closes #NN` in the **PR body**. To keep the
  issue open, **remove the keyword from the PR body** — editing only the commit message is not enough.
- **Notebooks are a local-only gate** (nbmake is not in blocking CI). Re-run and commit fresh
  outputs before every push.
