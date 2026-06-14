# agent-readiness-audit — reference (portable methodology)

Rubric, front-door template, and apply discipline. **Repo-agnostic** — every fact
about *this* codebase (subpackage list, won't-flag ledger, doc pointers, worked
example, per-repo status) lives in
[aaanalysis_conventions.md](aaanalysis_conventions.md). The deterministic half is
`scripts/check_agentic_docs.py`; this file is the *judgment* half a checker can't see.

---

## 1. Agent-friendliness rubric

What makes a package legible to an agent, by leverage. **Only B and C are
tool-backed**; A, D, F are reviewed by hand — report them as judgment, never as a
script result. The per-repo ✓/⚠/🔒 status snapshot lives in the conventions file.

- **A. Root navigational map (mental model)** — does a root doc state what each
  subpackage does *and how they connect* (the inter-module call/data-flow graph)?
  Plus a glossary, decision log (ADRs), and auto-loading per-area rules. The
  **dataflow map** (`docs/module_map.md`) is curated (semantic flow can't be derived
  from imports — frontends are decoupled through the `utils` barrel); its *coverage*
  is tool-backed (`check_module_map.py`: `MAP-MISSING`, `MAP-MISSING-SUBPKG`,
  `MAP-STALE-SUBPKG`), its *correctness* is judgment.
  - **A′. Decision-log hygiene** — ADR status lines well-formed, cross-refs resolve,
    supersession reciprocal, no ADR referenced from code, and the overview table
    current. *Tool: `check_adrs.py` (`ADR-NO-STATUS`, `ADR-BAD-STATUS`,
    `ADR-XREF-DANGLING`, `ADR-IN-CODE`, `ADR-INDEX-MISSING/STALE`,
    `ADR-SUPERSEDED`).* Whether an ADR's *content* is outdated is judgment; ADRs are
    **append-only** (supersede, never delete — deletion needs explicit go-ahead).
- **B. Per-subpackage front-door** — a module docstring in every public
  `__init__.py`: one-line purpose · key public objects · sibling relationship ·
  pointers. *Tool: `PKG-NO-DOCSTRING`.*
- **C. Public-API contract** — `__all__` present, equals the relative re-exports
  (0 missing / 0 stale), no `_private` leak; top-level `__all__` reconciles with the
  union of subpackage `__all__`s + any dynamic `append` stubs. *Tool: `ALL-MISSING`,
  `ALL-EXPORT-MISMATCH`, `PRIVATE-LEAK`, `TOPLEVEL-ALL-DRIFT`.* Public-API deltas are
  semver-relevant — surface, don't silently apply.
- **D. Type legibility** — every public parameter + return annotated;
  `Optional[...]` for `None`-defaults. Presence lets an agent reason without
  executing. *Judgment (grep public signatures); running mypy/ruff is out of scope.*
- **E. Docstrings as contract** — owned by the `docstrings` skill (numpydoc, named
  `Returns`, per-method `Examples`, doc-vs-signature drift). Cross-reference, don't
  re-specify.
- **F. Tight feedback loops** — single-test/module run path, a documented
  **sub-minute smoke path** (verify without a full pipeline), and errors that teach
  (a precise message → one-step self-correct). *Judgment.*
- **G–J (verify, don't churn)** — one-way module boundaries / no tangled
  cross-imports (G); a predictable skeleton and file/naming convention (H);
  golden-path runnable examples (I); a reproducibility/seed contract (J).

## 2. Checker codes

| Code | Severity | Meaning |
|---|---|---|
| `PKG-NO-DOCSTRING` | Defect | package `__init__.py` has no module docstring |
| `ALL-MISSING` | Defect | no `__all__` |
| `ALL-EXPORT-MISMATCH` | Defect | `__all__` ≠ relative re-exports (lists missing + stale) |
| `PRIVATE-LEAK` | Defect | a `_`-prefixed name in `__all__` |
| `TOPLEVEL-ALL-DRIFT` | Advisory | root `__all__` ≠ union(subpackage `__all__`) + appends |
| `MAP-MISSING` | Defect | `docs/module_map.md` (dataflow map) absent |
| `MAP-MISSING-SUBPKG` | Defect | a public subpackage is not covered by the map |
| `MAP-STALE-SUBPKG` | Defect | the map roster names a removed subpackage |

`0 defects` = the front-door contract holds. The checker is `ast`-based and
**import-free**, so it runs without the package's optional extras installed.

## 3. Front-door docstring template

A package `__init__` docstring is **not** a feature-module docstring. The
repo-specific opener convention, carve-out, and a worked example are in the
conventions file (§5). Generic shape:

```
"""
<Subpackage one-liner>.

Public objects: <names — must equal this subpackage's __all__>.
<How it connects to siblings — what it consumes / feeds>.

See <conventions/rules> for conventions, <glossary> for domain terms, <ADR> for decisions.
"""
```

"Public objects" must equal `__all__` (the checker enforces `__all__` ↔ re-exports;
keep the prose list in step).

## 4. Apply-mode discipline (generic)

- A package `__init__.py` is a sensitive surface — show the diff, get explicit
  per-file approval, never a blind multi-file write. (Repo's exact CONFIRM-FIRST
  list: conventions file §6.)
- A public-API (`__all__`) change carries semver weight — adding a symbol is a minor
  bump; rename/remove needs a deprecation path. Touch `__all__` only to fix a genuine
  `ALL-EXPORT-MISMATCH`; surface deltas separately.
- After writing docstrings, the docs build is the final gate the checker can't see —
  run it and confirm it still succeeds.
- Defer docstring *prose* quality to the `docstrings` skill; this skill owns
  *existence + `__all__` sync + navigability*.

## 5. Audit-report template (use verbatim for a consistent output)

Every audit ends with this shape, so results are comparable across runs:

```markdown
# Agent-readiness audit — <repo> @ <commit>

**Structural (deterministic):** check_agentic_docs <N defects>; check_adrs <M defects>.
> 0 defects = structural integrity (front-door exists + `__all__` synced + ADR hygiene),
> NOT prose/semantic correctness — docstring quality is the `docstrings` skill's job.

## Per-subpackage scorecard
| subpackage | front-door (B) | __all__ (C) | type hints (D, judgment) | pointers |
|---|---|---|---|---|
| <name> | ✅/❌ PKG-NO-DOCSTRING | ✅/❌ | ✅/⚠/— | ✅/⚠ |

## Repo-level (judgment)
- A. Mental-model map: <present / ⚠ missing edges / drift>
- F. Sub-minute smoke path: <documented? where?>

## Deliberate — NOT flagged (won't-flag ledger)
<list the sharp edges encountered, each marked "deliberate — see sharp-edges.md">

## Prioritized fixes
1. <issue> — <leverage> — <judgment | apply-mode (CONFIRM-FIRST)>
```

## 6. Bootstrapping a new repo (design settled; build at promotion)

The conventions file (`aaanalysis_conventions.md` here) is the *only* repo-specific
file. When this skill is taken to another package, create that repo's conventions by
the **hybrid** path — derive what's derivable, interview the rest:

1. **Derive (skill):** run `check_agentic_docs.py <pkg>` to enumerate the public
   subpackages; scan for the repo's doc homes (`CLAUDE.md`/`AGENTS.md`, `docs/adr/`,
   `CONTEXT.md`, a docstring guide); detect the ADR status format. These fill the
   pointer table, subpackage list, and ADR-conventions section.
2. **Interview (grill-with-docs):** settle the *judgment* parts — the won't-flag
   ledger (what is a *deliberate* sharp edge in this repo?), the front-door voice /
   template, and the dataflow map.

**Status: design only.** Under the current project-scoped model this is moot for
aaanalysis. Build a generic `conventions.template.md` + this workflow the first time
the skill runs on a real second repo — not speculatively. (At that point the
hard-coded `aaanalysis_conventions.md` references also become a generic discovered
path; see the scoping note — promote to a global skill only once proven here.)

## 7. Known coverage gaps (honest limits)

- **Structural, not semantic.** A subpackage can pass every code while carrying a
  *misleading* docstring — the checker proves the front-door exists and `__all__` is
  synced, nothing about whether the prose is true. Always state this in the report.
- **Prose `Public objects:` line drift — `PUBLIC-OBJECTS-PROSE-DRIFT` (decided: always-on
  defect; build with the first front-door).** §3 requires the human-read
  `Public objects:` line to equal `__all__`; today only `__all__` ↔ re-exports is
  enforced, so the prose could drift. Decision: add this as an **always-run defect** in
  `check_agentic_docs.py` (parse the `Public objects:` line, expand `X(+Plot)` →
  `X, XPlot`, set-compare to `__all__`) — built alongside the first front-door (the
  `metrics` pilot) so every audit thereafter guards it. The front-door's *other* claims
  (purpose, siblings, pointers) stay judgment, not regex.
- **Inter-module map (A) is curated, coverage-checked only.** `docs/module_map.md`
  holds the dataflow map; `check_module_map.py` proves every public subpackage is
  covered (and the roster has no dead entry) but **cannot verify the edges are
  correct** — that stays human judgment. An import-graph generator was deliberately
  *not* built: this repo's frontends are decoupled through the `utils` barrel (≈3
  cross-subpackage edges total), so a derived import graph would miss the real flow.
