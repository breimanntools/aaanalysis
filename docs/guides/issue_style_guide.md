# AAanalysis issue style guide

How to write a GitHub issue for this repo. Derived from the strongest existing
issues (**#61**, **#62**) and the coding standards in `CLAUDE.md` +
`.claude/rules/`. The skill `github-issues` and any agent creating issues
should follow this.

**Core idea.** A good issue is a *contract*: a reader can **scope it, build it,
and verify it done** without asking a follow-up. The two load-bearing parts are
**Requirements** (what to build) and **KPIs** (how we know it's done —
*measurably*). An issue with vague acceptance criteria is not done-able; it is a
note.

---

## Required sections

1. **Title** — imperative, specific, names the symbol/area.
   - ✅ `Add optional GPU/CPU parallelization to CPP.fit (graceful CPU fallback)`
   - ❌ `Performance improvements` · `New scale selection`

2. **Labels** — exactly one `prio:1|2|3`, one `topic:core|data|performance|XAI`,
   one `type:feature|bug|dcos`.

3. **## Problem** — the gap *and why it costs something now*. Quantify the pain
   where you can (#62: "O(n_seq × n_scales × n_positions) … proteome scale …
   prohibitive runtimes"). 1–2 short paragraphs. Don't just say "X is missing" —
   say what its absence prevents.

4. **## Goal** — one sentence: the target outcome **plus the key constraint**
   (e.g. "…with graceful CPU fallback and no new required dependencies").

5. **## Requirements** (a.k.a. Tasks) — checkboxed, concrete, **grouped under
   sub-headings when there are more than ~5**. Each item names the file/symbol
   where known and respects the standards checklist below. Avoid bare verbs
   ("improve", "support", "integrate") without a concrete object and method.

6. **## KPIs / Acceptance criteria** — **measurable, binary pass/fail**, with at
   least one quantified. This is the section most current issues get wrong.
   - ✅ "≥5× speedup on 10,000 sequences vs single-core; CPU/GPU equal within tol"
   - ✅ "default output byte-identical to current (regression-tested)"
   - ✅ "covered by ≥1 end-to-end test; example notebook runs under the nbmake CI gate"
   - ✅ "generated sequences pass the composition-similarity check"
   - ❌ "outputs easily usable externally" · "significant improvement" ·
     "clear and minimal" · "works with CPP features" (not testable)

## Recommended sections

7. **## Scope / non-goals** — what's explicitly out, and the **core-vs-`pro`** /
   **ProtXplain** boundary (heavy deps → `pro` or downstream, per
   `pro-core-boundary.md`).

8. **## Dependencies** — `blocked-by #N`, `relates ADR-XXXX`. Make ordering
   visible (e.g. "#29/#33 depend on the #18 output schema").

9. **## Standards checklist** (AAanalysis-specific — tick what applies):
   - [ ] Frontend/backend split honored; validation block; backend trusts frontend
   - [ ] **CONFIRM-FIRST surface touched?** (`pyproject.toml`, `__init__.py`,
         `__all__`, `_data/`, `.github/workflows/`, `config.py`,
         `template_classes.py`) — call it out explicitly
   - [ ] numpydoc docstring (named `Returns`, per-method `Examples` include)
   - [ ] tests (unit; property/golden where apt); reproducibility
         (`random_state`/`seed` contract)
   - [ ] no `print()` (use `ut.print_out`); bare `ValueError`/`RuntimeError`;
         no `aaanalysis._utils.*` imports outside `utils.py`
   - [ ] new public symbol → re-export in `__init__.py` (CONFIRM-FIRST) +
         `pro`/core gating if heavy

## Size rule

**One issue = one deliverable with its own KPI.** If an issue needs more than
one independent acceptance test, split it. TODO-dump *buckets* (the closed
#34/#38/#39/#41) are an anti-pattern — they hide many deliverables behind a
title and can never be "done".

## Copy-paste template

```markdown
## Problem
<the gap and what it costs now; quantify if possible>

## Goal
<one sentence: outcome + key constraint>

## Requirements
- [ ] <concrete, names a file/symbol>
- [ ] ...

## KPIs / Acceptance criteria
- [ ] <measurable, binary; at least one quantified>

## Scope / non-goals
- <what's out; name the layer explicitly — core vs `aaanalysis[pro]` (open, heavy
  optional deps) vs ProtXplain (closed agent/decision layer). Never write
  "ProtXplain / pro": they are two different boundaries.>

## Dependencies
- blocked-by #N · relates ADR-XXXX

## Standards checklist
- [ ] frontend/backend · CONFIRM-FIRST? · numpydoc · tests · no-print
```
