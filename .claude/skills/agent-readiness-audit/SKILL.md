---
name: agent-readiness-audit
description: Audit and improve a Python package for agent-friendliness, applying the highest-leverage fix — a per-subpackage front-door = the PEP 257 module docstring in __init__.py (purpose, public objects, sibling links, pointers) plus an __all__ that matches its re-exports. Deterministic half is an ast-based checker (scripts/check_agentic_docs.py — PKG-NO-DOCSTRING, ALL-MISSING, ALL-EXPORT-MISMATCH, PRIVATE-LEAK, TOPLEVEL-ALL-DRIFT); the broader rubric (root call/data-flow map, type-hint consistency, feedback loops) is reviewed by hand, NOT scored by a tool. Audit-first: propose diffs, apply only on confirmation; __init__.py / __all__ edits are CONFIRM-FIRST and semver-relevant. All repo-specific facts (subpackage list, won't-flag ledger, doc pointers, front-door template) live in aaanalysis_conventions.md so they can be edited or swapped. Use when the user wants to make the codebase more agent-friendly, add or refresh subpackage front-door / module docstrings, audit __all__ / public-API legibility, build the inter-module mental-model map, or work the module-front-door task (issue #69).
---

# agent-readiness-audit — make a package legible to agents

An agent-friendly codebase is a *legible* one with *tight feedback loops*. This
skill audits a package against that bar and applies the highest-leverage fix: a
per-subpackage **front-door** = the `__init__.py` module docstring + a synced
`__all__`. It uses docstrings, **not READMEs** (PEP 257; the per-dir agent niche is
already filled by auto-loading `.claude/rules/`).

> **Two deliberate layers.** The **deterministic** layer is the checker (proves the
> front-door *exists* and `__all__` is in sync). The **judgment** layer is the
> [REFERENCE.md](REFERENCE.md) rubric (call-graph, type-hint consistency, feedback
> loops) — **reviewed by hand, not scored by a script**; don't claim a tool result
> for it. **All repo-specifics live in [aaanalysis_conventions.md](aaanalysis_conventions.md)**
> (a swappable pointer file). The `docstrings` skill owns docstring *prose style*.

## Quick start

```bash
SK=.claude/skills/agent-readiness-audit/scripts
python3 $SK/check_agentic_docs.py aaanalysis      # front-door: __init__ docstring + __all__
python3 $SK/check_adrs.py                          # decision-log hygiene (docs/adr/)
python3 $SK/check_adrs.py --write-index            # (re)generate docs/adr/INDEX.md table
python3 $SK/check_module_map.py                    # inter-module dataflow map coverage (docs/module_map.md)
```

`0 defects` = every public subpackage has a front-door docstring with an `__all__`
in sync, and the decision log is mechanically clean. Output splits **Defects**
(exit != 0) from **Advisory** (never fail), mirroring the `docstrings` checker.

**Front-door codes** (`check_agentic_docs.py`):
- `PKG-NO-DOCSTRING` — package `__init__.py` has no module docstring (empty front-door).
- `ALL-MISSING` — no `__all__`.
- `ALL-EXPORT-MISMATCH` — `__all__` ≠ the relative re-exports (0 missing / 0 stale).
- `PRIVATE-LEAK` — a `_`-prefixed name in `__all__`.
- `TOPLEVEL-ALL-DRIFT` (advisory) — top-level `__all__` ≠ union of subpackage
  `__all__`s + the dynamic pro/dev `append` stubs.

**ADR-hygiene codes** (`check_adrs.py`):
- `ADR-NO-STATUS` / `ADR-BAD-STATUS` — missing/malformed `Status:` line.
- `ADR-XREF-DANGLING` — an `ADR-NNNN` reference with no such ADR (catches a
  reference left stale after an ADR is removed).
- `ADR-IN-CODE` — source code references an ADR (the repo convention forbids it).
- `ADR-INDEX-MISSING` / `ADR-INDEX-STALE` — the overview table `docs/adr/INDEX.md`
  is absent or out of date (regenerate with `--write-index`).
- `ADR-SUPERSEDED` / `ADR-SUPERSEDE-ASYMMETRIC` (advisory) — superseded ADRs surfaced
  for review; non-reciprocal supersession.

**Module-map codes** (`check_module_map.py` — validates the *curated* dataflow map):
- `MAP-MISSING` — `docs/module_map.md` absent.
- `MAP-MISSING-SUBPKG` / `MAP-STALE-SUBPKG` — a public subpackage is missing from the
  map, or the roster names one that no longer exists (`--write-roster` to resync).
  The map is **curated by hand** (semantic dataflow can't be auto-derived); the
  validator only proves coverage, not that the edges are correct.

(`PKG-NO-DOCSTRING` is deliberately distinct from the `docstrings` skill's
`INIT-NO-DOCSTRING`, which means a class `__init__` *method* lacking a docstring.)
Whether an ADR's **content** is outdated is judgment (rubric A), not a checker code.

## Three modes

1. **Audit (default, read-only).** Run the checker, then score each subpackage on
   the deterministic codes and review the judgment rubric by hand. Output a
   prioritized report and **state the deliberate sharp edges as "not flagged"**
   (the won't-flag ledger in the conventions file) so the scan reads as honest.
2. **Apply (CONFIRM-FIRST).** Draft front-door docstrings from the *actual*
   re-exports + sibling relationships using the conventions-file template; show the
   diff; edit `__init__.py` **only on explicit per-file confirmation**. Treat any
   `__all__` change as semver-relevant — surface it separately, never silently.
   Defer deep docstring prose to the `docstrings` skill.
3. **Refresh.** Re-run both checkers after code/ADR changes; regenerate any
   front-door whose public API drifted from `__all__`, and re-run
   `check_adrs.py --write-index` so the ADR overview table stays current.

## Rules

- **Audit-first, never bulk-overwrite.** A confident-but-stale docstring misleads
  an agent worse than none.
- **Honor the won't-flag ledger** (conventions file → derived from
  `sharp-edges.md`); name deliberate choices, don't "fix" them.
- **Type hints, map, smoke-path = judgment, not tooling.** Only the front-door /
  `__all__` codes are deterministic. Don't report an un-tooled dimension as if a
  script produced it.
- **Link, don't duplicate** the authoritative homes.
- **ADRs are append-only.** Flag superseded/contradictory ADRs and fix stale
  references; never delete an ADR — supersede it (new ADR + flip old status).
  Deletion needs the maintainer's explicit go-ahead (`docs/adr/README.md` + hard rule).
- **Verify before asserting.** A claimed gap is a candidate until checked against source.

## See also

- [REFERENCE.md](REFERENCE.md) — portable rubric, front-door template, apply discipline.
- [aaanalysis_conventions.md](aaanalysis_conventions.md) — all repo-specifics (swappable).
- `docstrings` skill — docstring prose + numpydoc + doc-vs-signature drift.
- Issue **#69** (module front-door) and **#126** (inter-module mental model).
