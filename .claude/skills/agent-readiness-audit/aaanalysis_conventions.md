# AAanalysis conventions (for `agent-readiness-audit`)

**This is the only repo-specific file in the skill.** [SKILL.md](SKILL.md) and
[REFERENCE.md](REFERENCE.md) are portable methodology; everything that ties the
skill to *this* codebase lives here, and it is mostly **pointers to the
authoritative docs** (so they stay the single source of truth). To retarget the
skill at another package, replace this file.

> Rule of thumb: if a fact about the repo could change, it belongs here as a
> pointer — never hard-coded into SKILL.md / REFERENCE.md / the checker.

---

## 1. Authoritative docs (pointers — do not duplicate, link)

| Topic | Source of truth |
|---|---|
| Scope, hard rules, CONFIRM-FIRST surfaces | `CLAUDE.md` |
| Repo tree, class templates, extras | `.claude/rules/repo-layout.md` |
| Accepted sharp edges (the won't-flag basis) | `.claude/rules/sharp-edges.md` |
| Code conventions, type hints, module skeleton | `.claude/rules/code-conventions.md` |
| Frontend/backend split, validation | `.claude/rules/frontend-backend.md` |
| Pro/core boundary, `missing_feature_stub` | `.claude/rules/pro-core-boundary.md` |
| Public-API stability / semver | `.claude/rules/api-stability.md` |
| Errors / warnings / logging | `.claude/rules/errors-warnings-logging.md` |
| Reproducibility (`random_state`/`seed`) | `.claude/rules/reproducibility.md` |
| Domain glossary | `CONTEXT.md` |
| Architecture decisions | `docs/adr/` (0001–0033) |
| Internal dataflow map (curated Mermaid) | `docs/module_map.md` (GitHub; RTD version via #106) |
| Docstring prose style | `docs/source/index/docstring_guide.rst` (+ the `docstrings` skill) |

## 2. Public subpackages the checker scans

`feature_engineering`, `data_handling`, `seq_analysis`, `pu_learning`,
`explainable_ai`, `plotting`, `metrics`, `protein_design`, and the `*_pro` /
dev siblings `data_handling_pro`, `seq_analysis_pro`, `explainable_ai_pro`,
`show_html`. Internal dirs (`_utils/`, `_backend/`, `_data/`) are skipped by the
leading-underscore rule. The top-level `aaanalysis/__init__.py` appends the 7
pro/dev symbols via `__all__.append(...)` inside `try/except ImportError`; the
`TOPLEVEL-ALL-DRIFT` check parses those.

## 3. Won't-flag ledger (deliberate sharp edges — name, never "fix")

**Source of truth: `.claude/rules/sharp-edges.md` + the root hard rules.** The audit
must name these as deliberate and never propose fixing them — this is what keeps the
scan honest. Distilled:

- `utils.py` ~1500-line god-module (v2 refactor).
- No CI type checker / `ruff` / `pre-commit` / mypy config (out until v2).
- Bare `ValueError`/`RuntimeError`; no `AAanalysisError` base.
- Positional-list backend rows (no NamedTuple/dataclass until v2).
- `[project]` + `[tool.poetry]` duality in `pyproject.toml`.
- No `SECURITY.md`.
- The rejected perf optimizations (ADR-0032 / ADR-0033).

Out of scope here, owned elsewhere: docstring prose (`docstrings` skill),
refactoring (`improve-codebase-architecture`), issue triage (`github-issues`).

## 4. Per-repo rubric status (snapshot — re-check on audit)

Against the REFERENCE rubric: **A** root map ✓ except the inter-module call/data-flow
graph ⚠ (the #126 mental-model); **B** front-doors ⚠ (all 12 `__init__.py` empty
today); **C** `__all__` ✓ but unguarded → the checker guards it; **D** type hints —
convention exists, presence is hand-reviewed (no tool); **E** docstrings ✓ (owned by
`docstrings`); **F** ⚠ a documented sub-minute smoke path; **G–J** ✓ strong (verify,
don't churn).

## 5. Front-door template + the carve-out

A package `__init__` docstring is **not** a feature-module docstring, so it does
**not** open with `"""This is a script for ..."""` (that skeleton in
`code-conventions.md` is for `_<feature>.py`). **Known tension:** `code-conventions.md`
states the opener applies to "every module" with no carve-out — an agent could
"correct" a front-door to the script opener. Apply-mode should add a one-line
exemption to `code-conventions.md` (or `docstring_guide.rst`) when it first writes
front-doors. The opener is **not** tool-enforced (`check_docstrings.py` does not check
module-docstring openers), so there is no automated conflict today.

Template (defer prose to `docstring_guide.rst`):

```
"""
<Subpackage one-liner, in the package's voice>.

Public objects: <Class>(+Plot), <func>, ...   # must equal this subpackage's __all__.
<How it connects to siblings — what it consumes / feeds>.

See ``.claude/rules/<relevant>.md`` for conventions, ``CONTEXT.md`` for domain terms
(<the glossary terms this subpackage owns>), ADR-<NNNN> for <the decision>.
"""
```

Worked example (`feature_engineering`):

```
"""
Feature engineering: scale-based amino-acid features for interpretable prediction.

Public objects: AAclust(+Plot), SequenceFeature, NumericalFeature, CPP, CPPGrid, CPPPlot.
Consumes scale sets from ``data_handling.load_scales``; CPP features (``df_feat``)
feed ``explainable_ai.TreeModel`` and ``protein_design``.

See ``.claude/rules/code-conventions.md`` / ``frontend-backend.md`` for conventions,
``CONTEXT.md`` for domain terms (df_parts, part, split, scale), ADR-0001 for the CPP backend.
"""
```

For a `*_pro` subpackage, name the gating dependency and cross-link its core sibling
(`pro-core-boundary.md`).

## 6. ADR (decision-log) conventions

Source of truth: `docs/adr/README.md`. Enforced by `scripts/check_adrs.py`.

- **Status line** (line ~3, inline, never YAML): `Status: Accepted — YYYY-MM-DD` or
  `Status: Superseded by ADR-MMMM`; partial supersession noted in a parenthetical.
- **Append-only / immutable once Accepted.** Reverse a decision with a *new* ADR and
  flip the old status — do **not** edit a decision in place. **Delete only when fully
  obsolete and only with the maintainer's explicit go-ahead** (repo hard rule). The
  checker surfaces superseded ADRs as *advisory* (visibility), never auto-deletes.
- **Code must never reference an ADR** — put rationale inline in the comment. The
  checker's `ADR-IN-CODE` enforces this (currently 4 known violations:
  `utils.py:204,383`, `_get_feature_matrix_fast.py:79,98` — fix on touch; the two in
  `_get_feature_matrix_fast.py` are inside docstrings, which the house style also forbids).
- **Overview table:** `docs/adr/INDEX.md` (a separate file from `README.md`) holds the
  auto-generated table between `ADR-INDEX:START/END` markers. Regenerate with
  `check_adrs.py --write-index`; `ADR-INDEX-STALE` fires if it drifts. README stays the
  prose conventions doc; INDEX is the machine-maintained roster.

## 7. Apply-mode: CONFIRM-FIRST + semver here

`aaanalysis/__init__.py` and every subpackage `__init__.py` are CONFIRM-FIRST
(CLAUDE.md §2). An `__all__` change is semver-relevant (`api-stability.md`): adding a
symbol = minor bump; rename/remove needs a deprecation shim. Touch `__all__` only to
fix a genuine `ALL-EXPORT-MISMATCH`; surface any delta separately. Final gate after
writing docstrings: `cd docs && make html` must still succeed.
