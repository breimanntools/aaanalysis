# ADR-0035 — Input validation stays `ut.check_*`; no Pydantic; agent-typed contracts live in ProtXplain

Status: Accepted — 2026-06-22

## Context

An agent-readiness review proposed typed I/O contracts (Pydantic
`CPPRequest`/`CPPResult`), a high-level "verb" facade (`run_cpp`,
`run_prediction`, …), and a generated MCP / JSON-schema tool layer, to make the
package legible to LLM agents.

The package already covers the substance of that goal with its own primitives:
every public method opens with a sklearn-style **Validate block** of hand-written
helpers (`ut.check_number_range`, `ut.check_df_seq`, `ut.check_str_options`, …)
that raise bare `ValueError` / `RuntimeError` in the canonical
`"'<name>' (<got>) should be <expected>"` format; feature output is a `df_feat`
DataFrame with a documented column convention (`ut.LIST_COLS_FEAT` +
`sort_cols_feat`). A documented organizing principle already splits the work:
**AAanalysis ships interpretable scientific primitives; ProtXplain orchestrates
them** for agents (see `docs/guides/handoff_github_issues.md`, issue #26).

## Decision

D1. **Input validation remains `ut.check_*`.** Public methods keep the
    sklearn-style Validate block. We do **not** adopt Pydantic — or any
    model-validation framework — for input contracts, in v1 **or** v2.
    `ut.check_*` already provides runtime validation plus the house error-message
    format, with zero added dependency, and is the family-wide convention.

D2. **No Pydantic / pandera dependency in AAanalysis.** Pydantic validates
    scalar/nested models, not DataFrame columns; pandera (the DataFrame
    equivalent) is likewise not adopted. The `df_feat` output contract is
    expressed as a **data dictionary** — column name · dtype · nullability ·
    value-semantics — published in the docs, anchored on the existing
    `ut.LIST_COLS_FEAT` / `sort_cols_feat` order, and guarded by one
    schema-stability test (issue #26). A new public schema *accessor* is deferred
    until a concrete consumer needs to import it (strict-semver caution).

D3. **Agent-facing typed contracts live downstream in ProtXplain.** Any
    `CPPRequest`/`CPPResult` Pydantic models, JSON-schema emission,
    MCP / function-calling tool definitions, and the verb-selection / ranking /
    decision logic belong to ProtXplain, where agents call in. AAanalysis's only
    obligation to that layer is a **stable, documented `df_feat` contract**;
    ProtXplain pins to the documented column-name strings.

## Rejected alternatives

- **Pydantic for input contracts in AAanalysis.** Duplicates the existing
  `ut.check_*` validation, adds a dependency, introduces `pydantic.ValidationError`
  (conflicting with the bare-`ValueError`/`RuntimeError` rule), and reverses the
  standing "no typed records until v2" position — all for a benefit (JSON-schema
  emission for agent tools) needed only where agents call in (ProtXplain).
- **pandera for runtime `df_feat` schema enforcement.** A new dependency to
  enforce what a documented data dictionary + one stability test already cover.
  Deferred unless a concrete OSS-side consumer needs runtime enforcement.
- **A "verb" facade + MCP layer inside AAanalysis (OSS ships agent-ready tools).**
  Pushes orchestration-adjacent surface and semver weight into the primitives
  package, partly duplicates `CPPGrid`, and contradicts the primitives-vs-
  orchestration boundary. Verbs / MCP / agent logic live in ProtXplain.

## Consequences

- The published `df_feat` data dictionary becomes a stability contract
  (test-guarded); changing it is a semver event.
- "Agent-readiness" for AAanalysis means **type hints + docstrings + a
  documented, tested `df_feat` contract** — not a typed-model framework.
- ProtXplain owns the Pydantic models and tool schemas; no new public symbol is
  required in AAanalysis yet.

## Out of scope

- The OOD / uncertainty "refuse-to-predict" gate, counterfactuals, and
  candidate-ranking — all ProtXplain-scoped.
- Whether to ever expose a programmatic `df_feat` schema accessor in AAanalysis
  (revisit only when ProtXplain needs programmatic import).
