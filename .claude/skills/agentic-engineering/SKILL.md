---
name: agentic-engineering
description: Autonomously drive an entire issue or a complete new feature from spec to merge through the AAanalysis agentic-engineering protocol — sharpen the issue, branch into an isolated worktree, implement, push and open the PR early so CI runs, run the automated review + quality gates, keep the branch synced, then hold at the human-review gate where the user picks a manual PR-review loop or skips (skip = post an approving review comment, then arm GitHub-native auto-merge), with a fix-forward loop on red CI. Use when the user wants to start work on an issue, "walk me through the workflow", take a change from issue to merge, or wants the auto-merge / auto-fix loop driven for them. NOT for typos, one-line fixes, or trivial local edits — make those directly. The canonical protocol lives in docs/guides/agentic_engineering.md; this skill executes it.
---

# AAanalysis agentic-engineering — drive a change to merge

Lean driver for the protocol in **`docs/guides/agentic_engineering.md`** (the single source of truth —
read it first; this skill executes it). Deep detail — delegation model, git-stash + parallel-session
hazards, merge-method rationale, CI/local-gate commands — lives in **[REFERENCE.md](REFERENCE.md)**.
Durable artifacts (ADRs, `CONTEXT.md`, `CLAUDE.md` / `.claude/rules`, the code) are authoritative;
per-issue planning notes are ephemeral.

**When to use:** to autonomously drive an **entire issue or a complete new feature** from spec to
merge (a substantial bug-fix issue counts). **Not** for typos, one-line fixes, or trivial local
edits — make those directly.

## Happy path

`⛔` = stop for an explicit §0 permission; each is its **own** ask (push ≠ PR ≠ merge ≠ each cleanup deletion).

1. Sharpen the issue → **`/grill-with-docs`** (refresh `CONTEXT.md` / ADRs) **before any code**.
2. Worktree: `git fetch origin && git worktree add ../wt-<slug> -b <type>/<slug> origin/master`, then `cd` in.
3. Implement (honor auto-loaded `.claude/rules/`) + walk the **Ripple checklist** below.
4. Fast local gate (REFERENCE.md → *Local gate commands*) → ⛔ push scaffold commit → open a **draft PR** (starts CI + RTD).
5. **`/review`** + **`/security-review`** (+ `/code-review high`, `/simplify`, `/docstrings` for big/API diffs). **Never merge red.**
6. ⛔ **Human gate** — user picks **(a)** manual review loop (iterate forward on the branch) or **(b)** skip → post an approving comment.
7. ⛔ `gh pr merge --auto --merge`; fix-forward on red (armed auto-merge completes on the green re-run).
8. ⛔ After the PR is **MERGED** + `master` green: clean up (see *Cleanup* below).

> **Parallel sessions are the norm here.** Before any status claim, commit, merge, or cleanup, refresh
> live state (`git fetch origin --prune` + `gh pr list` / `gh pr view`) and act **only** on the
> branch/worktree *you* created — surface anything you didn't make, never touch it. Full hazards:
> [REFERENCE.md](REFERENCE.md) → *Parallel-session hazards*.

## ADRs under parallel sessions (don't collide on a number)

Concurrent branches each draft an ADR and grab "the next number" from their *own* stale
checkout — so two land as `0034`, or a gap appears (a real incident left `0034`–`0036`
unindexed). The durable rule lives in `docs/adr/README.md` → *Conventions*; in execution:

- **Settle the decision in step 1** (`/grill-with-docs`). First `git fetch origin --prune` and
  scan in-flight work for an ADR on the *same* decision (`gh pr list`; grep open PR diffs for
  `docs/adr/`) — don't open a rival ADR for one another session already owns.
- **Draft number-less:** title `# ADR-XXXX —`, file `docs/adr/XXXX-<slug>.md`. Never bake a real
  number in while implementing — local state is stale the moment another session merges.
- **Number it last,** as the PR is about to merge: rebase on a fresh `origin/master`, take one
  past the max across committed ADRs **and** open PRs, rename file + title, regenerate
  `docs/adr/INDEX.md` (`check_adrs.py --write-index`). Let the index row / sequential filename
  collide as a git conflict — that's the safety net; resolve by taking the next free number.
- **Never renumber a *merged* ADR** (rename = new path, §0). Detail:
  [REFERENCE.md](REFERENCE.md) → *ADR numbering under parallel sessions*.

## Hard rules (override everything here)

From root `CLAUDE.md` §0/§2 — authorization is **per-action, never per-session**:

- **Never delete/rename a file without explicit permission** (incl. `git worktree remove` of a tree
  with uncommitted work, branch deletion, `git mv`).
- **Never push or publish without explicit permission** — `git push`, `gh pr create`, `gh pr merge`
  (incl. arming `--auto`), `gh release create`. Ask again for the next one.
- **CONFIRM-FIRST files** (CLAUDE.md §2): `pyproject.toml`, `aaanalysis/__init__.py`,
  `aaanalysis/_data/*`, `.github/workflows/*`, `config.py`, `template_classes.py`, any `__all__` symbol rename/delete.

## Delegated skills

`/grill-with-docs`, `/review`, `/security-review`, `/code-review`, `/simplify`, `/docstrings`,
`/github-issues`, `/triage`, `/to-issues`, `/schedule` are **local sub-skills this skill orchestrates**
— not shell/GitHub commands. If one is **unavailable, fails, or is inconclusive, stop and surface the
missing gate**; never proceed as if it passed. Detail: [REFERENCE.md](REFERENCE.md) → *Delegated skills*.

## Cleanup (step 8 — gated on merge + a green `master`, §0)

Trigger off **merge state, not a CI run**: wait until `gh pr view <n> --json state,mergedAt` shows
`MERGED`, then let the push-triggered `master` workflows pass. `git fetch origin --prune` first (drops
stale remote-tracking refs another session's merge left behind). Because PRs land as **merge commits**,
`git branch --merged master` lists the branch and a plain **`git branch -d <branch>`** deletes it
safely. As separate §0 asks: `git switch master` → `git worktree remove <path>` (`--force` if
uncommitted → also permission) → `git worktree prune` → `git branch -d <branch>`; remote head
auto-deletes if the repo setting is on, else `git push origin --delete <branch>` (push → §0).
**Canonical tool:** `python .github/scripts/prune_merged_branches.py` (PR-state-driven, report-only by
default, `--apply` to delete, never touches FORGOTTEN no-PR work) — run it from *any* session, since
parallel auto-merges land after the opening session ends. Unattended jobs may *flag* but never delete (§0).

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
- **Glossary / ADRs** — `CONTEXT.md`; a new `docs/adr/NNNN-*.md` (settle it in step 1; draft it
  number-less and number it last — see *ADRs under parallel sessions*). **Conventions** — `CLAUDE.md` / `.claude/rules/*`.
- **Build / deps** — `pyproject.toml` / `config.py` (both CONFIRM-FIRST).

Most surface late (stale cheat sheet, red meta-test, wrong RTD render), not in the fast unit job.

## Local gates & CI

The exact commands, job names, and thresholds live in **[REFERENCE.md](REFERENCE.md) → *Local gate
commands***. CI job names and thresholds drift — **verify the live configuration** (`.github/workflows/*`,
`gh pr checks <n>`) before claiming any gate's status; treat the guide as the source of truth.

## Notes

- **Merge with `gh pr merge --auto --merge`** — method is its own explicit decision, never bundled into
  the step-6 skip option. Why: [REFERENCE.md](REFERENCE.md) → *Merge method*.
- **Worktrees, not `git stash`, for isolation; one per task.** [REFERENCE.md](REFERENCE.md) →
  *Git-stash hazard* / *Parallel-session hazards*.
- **Sessions self-title for at-a-glance disambiguation.** After ~1% of context each session sets
  its terminal tab title to `<topic> · PR#<n> · ADR<nnnn>` — where *topic* is the branch slug —
  and refreshes it when the PR/ADR appear, so concurrent worktrees are tellable apart. So **give
  the branch a descriptive slug** at step 2: that slug *is* the session's topic. (Local mechanism:
  a global `Stop` hook running `~/.claude/set_session_title.py`; details in
  [REFERENCE.md](REFERENCE.md) → *Session self-titling*.)
- **Fix forward, never merge red.** GitHub completes auto-merge only on all-green + conflict-free.
- **Issue lifecycle.** Keep `Closes #NN` in the **PR body** to auto-close; remove it there to keep the
  issue open (the commit message alone isn't enough).
- **Notebooks are a local-only gate** (not in blocking CI). Re-run + commit fresh outputs before every push.
