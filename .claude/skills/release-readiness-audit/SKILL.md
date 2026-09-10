---
name: release-readiness-audit
description: Validates and resolves the P0 and P1 findings in this repository's release-readiness audit without expanding their scope. Use when preparing a release, reviewing `RELEASE_AUDIT.md`, verifying release blockers, or implementing validated release-audit findings.
---

# Release-readiness audit

Validate the release audit against the repository before acting on it. Treat the
audit as a set of claims to verify, not a backlog to implement unquestioned.
Work through validated blockers in priority order and keep the changes tightly
scoped to release readiness.

## Quick start

1. Read `RELEASE_AUDIT.md` and the repository instructions it relies on.
2. Verify every P0 and P1 finding against current code, tests, documentation,
   configuration, and Git history where relevant.
3. Remove findings that the evidence does not support, preserving or adding a
   concise explanation when it would help prevent the finding from returning.
4. Use the local `agentic-engineering` skill to implement each validated P0,
   then each validated P1, one at a time.

If `RELEASE_AUDIT.md` does not exist, stop and report that prerequisite. Do not
substitute a similarly named planning document without the user's direction.

## Audit workflow

1. **Establish the baseline.** Read `CLAUDE.md`, applicable `.claude/rules/`,
   and the entire audit. Refresh repository state before deciding whether a
   finding is still open.
2. **Verify each P0/P1 finding.** Record the evidence: affected paths, a
   reproducer or failing check when applicable, user-visible or release impact,
   and why the proposed remediation fits the current design. Verify before
   claiming a finding is stale, fixed, duplicated, or valid.
3. **Prune unsupported findings.** Remove only findings that are not justified
   by the current repository. Do not weaken a valid finding because its fix is
   inconvenient. Keep P2 and lower findings out of the implementation queue
   unless the user separately requests them.
4. **Order the validated work.** Resolve all P0 items before P1 items. Within a
   priority, choose the smallest independent item that does not hide a larger
   blocker. Do not batch unrelated findings into one change.
5. **Implement one finding.** Load and follow `agentic-engineering` for the
   individual finding, including its isolation, review, and explicit permission
   gates. Before code changes, add or update a focused test when a test can
   demonstrate the defect or regression. Keep the implementation minimal and
   avoid unrelated refactors.
6. **Verify and document.** Run the relevant targeted tests and the applicable
   lint, type, documentation, or build checks. Update documentation when
   behavior, public APIs, configuration, or release instructions change. Mark
   the finding resolved only with the exact evidence that proves it.
7. **Reassess before continuing.** Confirm the next finding is still valid
   after the preceding change. Stop when all validated P0/P1 findings are
   resolved, or when a blocker needs a human decision.

## Decision boundaries

- Stop and report when a proposed fix needs an architectural decision, a public
  API or compatibility choice, a new dependency, a data-format change, or a
  change outside the finding's stated release risk. Explain the options and
  evidence; do not choose silently.
- Honor all repository confirmation requirements, including the per-action
  permissions for pushes, pull requests, merges, deletions, renames, and
  CONFIRM-FIRST files. This skill does not grant those permissions.
- Do not claim a release is ready merely because P0/P1 findings were processed;
  report the checks actually run and any remaining risks or unverified areas.

## Completion report

Report separately:

- findings removed as unsupported, with brief evidence;
- validated P0 and P1 findings, their resolution status, and verification;
- any architecture decisions or approval gates that stopped progress; and
- remaining release risks, including findings outside P0/P1.
