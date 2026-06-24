---
name: github-issues
description: Audit all open GitHub issues for the aaanalysis package against its scope and coding standards (CLAUDE.md + .claude/rules/sharp-edges.md), flagging each as ready / needs-revision / conflicts / already-addressed, detecting overlaps that must not be developed in parallel, and writing a prioritized step-by-step implementation plan to docs/guides/handoff_github_issues.md. Issues are the primary lens; pull requests are mapped in as secondary context — an "Issue ↔ PR activity" table (issues on the left, their PR(s) on the right, — when an issue has no PR) plus per-issue in-flight-PR notes so each issue and its PR are shown together. Use when the user wants to review, triage, audit, or scope GitHub issues or PRs, refresh the issue handoff, see what work is in flight, decide what to implement next, or plan parallel work across issues.
---

# GitHub issue handoff

Produce/refresh `docs/guides/handoff_github_issues.md`: a scope-and-standards audit of every
open issue plus a prioritized, parallelization-aware implementation plan. The output is
a *handoff* — another session (or a parallel one) should be able to pick any lane and run.

## Workflow

> **Issues are the spine; PRs are secondary context.** The handoff is organized around issues; PRs
> are mapped in to show *what work is in flight or just landed* against each issue.

1. **Fetch issues.** Run the bundled script to get every open issue with its full body:
   `python3 .claude/skills/github-issues/scripts/fetch_issues.py` (add `--limit N`
   if there are many; `--json` for raw). Do not hand-list issues — always fetch fresh.
2. **Fetch PRs (secondary).** Pull open + recently-merged PRs and map each to its issue(s):
   - `gh pr list --state open --json number,title,headRefName,files`
   - `gh pr list --state merged --limit 30 --json number,title,mergedAt,closingIssuesReferences`
   - Map via `closingIssuesReferences` (auto-close keyword). For a PR with **no** closing keyword
     (e.g. an "Addresses #NN" body), attribute it to the issue / program item it advances (from the
     title/body) — note it as `(#NN)` in parens vs a closing `#NN ✅`.
   - Flag **in-flight** PRs whose changed files overlap an open issue's file-path → that issue is
     *being worked* and its lane is occupied; surface in the per-issue audit + overlap clusters.
   - See REFERENCE.md → *PR ↔ Issue mapping*.
3. **Load the baselines** (so verdicts are grounded, not guessed):
   - Scope + standards: `CLAUDE.md`, `.claude/rules/sharp-edges.md`, and skim the other
     `.claude/rules/*.md`. See [REFERENCE.md](REFERENCE.md) for the scope statement and the
     standing reject/defer checklist distilled from these.
   - Already-addressed signal: `git log --oneline -40`, `docs/adr/`, and a quick `git grep`
     for the symbol/feature an issue names. **Verify before claiming "Done"** — read the
     commit/code, don't trust the title.
4. **Classify each issue** with one verdict from the taxonomy in REFERENCE.md
   (✅ Ready / 🔄 Revisit / ⏸️ Defer-v2 / ❌ Reject / ☑️ Done-or-Partial), plus one line each:
   scope fit, standards note, already-addressed note, **any open/merged PR on its row** (issue +
   PR shown together), and a *complementary* implementation hint beyond the issue body (files to
   touch, reuse, the decision it glosses over). Cite the exact rule for every Reject/Defer.
5. **Map overlaps.** Group issues that touch the same code/feature into overlap clusters and mark
   them **serialize — do not develop in parallel** (e.g. a perf cluster editing the CPP feature
   loop); fold in any **in-flight-PR** file collisions. Flag true duplicates and copy-paste bodies.
6. **Order + lane it.** Produce a numbered implementation order (prio:1 → prio:3, respecting
   dependencies, in-flight PRs, and quick wins) and group independent work into parallel *lanes*
   that never touch the same files.
7. **Write `docs/guides/handoff_github_issues.md`** using the template in REFERENCE.md — including the
   secondary **Issue ↔ PR activity** table (issues on the left, their PR(s) on the right, **—** when an
   issue has no PR). If it exists, update in place (preserve any human-added notes; refresh counts).

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
