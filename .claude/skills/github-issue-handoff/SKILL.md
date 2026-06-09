---
name: github-issue-handoff
description: Audit all open GitHub issues for the aaanalysis package against its scope and coding standards (CLAUDE.md + .claude/rules/sharp-edges.md), flagging each as ready / needs-revision / conflicts / already-addressed, detecting overlaps that must not be developed in parallel, and writing a prioritized step-by-step implementation plan to docs/guides/handoff_github_issues.md. Use when the user wants to review, triage, audit, or scope GitHub issues, refresh the issue handoff, decide what to implement next, or plan parallel work across issues.
---

# GitHub issue handoff

Produce/refresh `docs/guides/handoff_github_issues.md`: a scope-and-standards audit of every
open issue plus a prioritized, parallelization-aware implementation plan. The output is
a *handoff* — another session (or a parallel one) should be able to pick any lane and run.

## Workflow

1. **Fetch.** Run the bundled script to get every open issue with its full body:
   `python .claude/skills/github-issue-handoff/scripts/fetch_issues.py` (add `--limit N`
   if there are many; `--json` for raw). Do not hand-list issues — always fetch fresh.
2. **Load the baselines** (so verdicts are grounded, not guessed):
   - Scope + standards: `CLAUDE.md`, `.claude/rules/sharp-edges.md`, and skim the other
     `.claude/rules/*.md`. See [REFERENCE.md](REFERENCE.md) for the scope statement and the
     standing reject/defer checklist distilled from these.
   - Already-addressed signal: `git log --oneline -40`, `docs/adr/`, and a quick `git grep`
     for the symbol/feature an issue names. **Verify before claiming "Done"** — read the
     commit/code, don't trust the title.
3. **Classify each issue** with one verdict from the taxonomy in REFERENCE.md
   (✅ Ready / 🔄 Revisit / ⏸️ Defer-v2 / ❌ Reject / ☑️ Done-or-Partial), plus one line each:
   scope fit, standards note, already-addressed note, and a *complementary* implementation
   hint that adds detail beyond the issue body (files to touch, reuse opportunities, the
   decision the issue glosses over). Cite the exact rule for every Reject/Defer.
4. **Map overlaps.** Group issues that touch the same code/feature into overlap clusters
   and mark them **serialize — do not develop in parallel** (e.g. a perf cluster all editing
   the CPP feature loop). Also flag true duplicates and copy-paste bodies.
5. **Order + lane it.** Produce a numbered implementation order (prio:1 → prio:3, respecting
   dependencies and quick wins) and group independent work into parallel *lanes* that never
   touch the same files.
6. **Write `docs/guides/handoff_github_issues.md`** using the template in REFERENCE.md. If it already
   exists, update it in place (preserve any human-added notes; refresh the audit + counts).

## Rules

- The handoff is opinionated and honest: a Reject/Defer without a cited rule is invalid; a
  "Done" without a verified commit/ADR is invalid.
- Read-only by default — this skill audits and plans; it does not edit code or close issues
  unless the user explicitly asks.
- Keep the package scope front-of-mind: AAanalysis is interpretable, CPP-centered, sequence-
  based protein prediction with a `pro` extra for heavy deps — work that balloons the
  dependency surface or belongs in the downstream ProtXplain is *Revisit*, not *Ready*.

See [REFERENCE.md](REFERENCE.md) for the scope statement, verdict taxonomy, reject/defer
checklist, and the handoff file template.
