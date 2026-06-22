# Architecture Decision Records

An ADR records a **decision and the alternatives we rejected** — the one thing
code cannot show. It is not a build journal and not a changelog.

## When to write one

Write an ADR only when all three hold:

1. **Hard to reverse** — changing your mind later is costly.
2. **Surprising without context** — a future reader will ask "why this way?".
3. **A real trade-off** — there were genuine alternatives and you picked one.

If any is missing, skip it. A renamed symbol, a bugfix, a dependency bump, a
docstring tweak — none of these earn an ADR.

## Template

```markdown
# ADR-NNNN — <short decision title>

Status: Accepted — YYYY-MM-DD

## Context
What forced a decision (1–2 short paragraphs). Cite code/files only as
background; never as a contract.

## Decision
What we chose. Numbered D1, D2, … in ascending order.

## Rejected alternatives
What we did NOT do, and why. This is the part that earns the ADR.

## Consequences   (optional)
What follows from the decision.

## Out of scope   (optional)
Explicitly deferred work.
```

## Conventions (what keeps ADRs from rotting)

- **No volatile metadata.** No `Branch:`, no commit hashes, no
  "commit N of M", no per-commit `pytest` verification blocks, no
  commit-history sections. That belongs in commit messages and PR
  descriptions, which are the right home for build process.
- **Number last, against live state (parallel-safe).** The number is the racy part: several
  branches draft ADRs at once and each picks `max + 1` from its *own* stale checkout, so two land
  as the same number or leave a gap. Draft number-less — title `# ADR-XXXX —`, file
  `docs/adr/XXXX-<slug>.md` — and assign the real number only when the ADR's PR is about to merge,
  after `git fetch origin --prune`, as one past the highest number across **committed ADRs *and*
  open PRs**. Regenerate `INDEX.md` then; the index row + sequential filename make a genuine
  duplicate surface as a merge conflict. Never renumber a *merged* ADR (rename = new path, which
  needs deletion permission).
- **One header format:** an inline `Status:` line — never YAML frontmatter.
  Status values: `Accepted`, or `Superseded by ADR-MMMM`.
- **Immutable once Accepted.** Do not edit a decision in place. To reverse one,
  write a new ADR that supersedes it and flip the old one's status.
- **Delete only when fully obsolete** — and only with the maintainer's explicit
  go-ahead (repo hard rule on file deletion).
- **Code must never reference an ADR.** ADRs change; shipped code must not point
  at them. Put the rationale inline in the code comment instead. ADRs may cite
  code for background, but that citation is context, not a guarantee of
  currency.
- **Keep it concise.** Decision + rejected alternatives. Tables for surveys are
  fine; narration is not.
