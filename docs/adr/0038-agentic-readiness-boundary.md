# ADR-0038 — Agentic-readiness boundary: usability/improvability is ours, agent integration is ProtXplain's

Status: Accepted — 2026-06-23

Refines [ADR-0035](0035-validation-ut-check-not-pydantic.md) — it does **not**
supersede it. ADR-0035's decisions (D1 input validation stays `ut.check_*`; D2 no
Pydantic/pandera dependency; D3 agent-typed contracts live in ProtXplain) stand
unchanged. This ADR sharpens *which* layer is withheld from AAanalysis, because a
later review read ADR-0035 as withholding all agent-facing work — including
user-facing convenience — which was never the intent.

## Context

ADR-0035 placed typed I/O contracts, a "verb" facade, and the MCP / JSON-schema
tool layer in ProtXplain, keeping AAanalysis to interpretable primitives plus a
documented `df_feat` contract. A follow-up agentic-readiness review (a
grill-with-docs session over external feedback) surfaced two genuinely
uncaptured needs that ADR-0035's wording did not clearly place:

1. A second, high-level **convenience API** so *users* get results in one call
   (an mpl/`pyplot`-style facade over the existing primitives), and a
   sklearn-compliant transformer so CPP composes in `sklearn.pipeline.Pipeline`.
2. A clarified statement of **what "agentic readiness" means for this package**,
   so improvability/usability work is not mistaken for the withheld
   agent-integration layer.

The ambiguity was the word "agent": ADR-0035 withholds the layer where *external
agents call in as a tool*, not the work that makes the package legible, usable,
and improvable — which serves both human users and the coding agents that
*improve* the package.

## Decision

**D1. The withheld layer is *agent integration*, not usability or
improvability.** ADR-0035 keeps the machine-readable tool contract, MCP server,
verb/tool schemas, and selection/ranking/decision/OOD logic in ProtXplain. It
does **not** withhold user-facing convenience or internal legibility from
AAanalysis.

**D2. Two agent audiences, drawn explicitly.** Agents that *improve* AAanalysis
are served **in** AAanalysis — through type hints, documented data contracts,
consistent class templates, tests, and legible primitives. Agents that *use*
AAanalysis as a tool are served by **ProtXplain** — through the MCP /
machine-readable tool contract. Usability and improvability stay here;
tool-integration goes downstream.

**D3. The border is the machine-readable tool contract.** Human- and
sklearn-idiomatic convenience (a stateless `aaanalysis.pipe` facade of "golden
pipelines"; a sklearn-compliant `SequenceFeatureTransformer`) is **AAanalysis**.
The MCP server, JSON/tool schemas, and verb orchestration are **ProtXplain**.

**D4. Ownership split.**
- **AAanalysis owns:** legible scientific primitives; user-facing convenience
  facades / golden pipelines (`aaanalysis.pipe`); plain numpy/pandas outputs that
  feed sklearn and torch equally; maximal improvability for coding agents.
- **ProtXplain owns:** the machine-readable tool contract; the MCP server;
  verb/tool schemas; selection, ranking, decision, and OOD/uncertainty logic.

## Consequences

- A new `aaanalysis.pipe` (`ap`) convenience namespace and a
  `SequenceFeatureTransformer` are **in scope** for AAanalysis, as a stateless,
  thin facade whose defaults are byte-identical to the explicit primitive path
  (no new algorithm, no new required dependency; torch stays the `[embed]`
  extra). They are tracked as their own feature work, not built here.
- "Agentic readiness" for AAanalysis means: legible/typed/contracted/**improvable**
  primitives plus user-facing convenience — **not** an in-package agent-tool
  framework, which remains ProtXplain's.
- Science/product tracks (structure-XAI, XAI-eval, the design bridge) are a
  separate roadmap, unaffected by this boundary.

## Out of scope

- The `aaanalysis.pipe` API and transformer implementation (tracked separately).
- Anything ADR-0035 already placed in ProtXplain (typed request/result models,
  MCP/tool schemas, ranking/decision/OOD) — unchanged.
