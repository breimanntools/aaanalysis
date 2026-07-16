# ADR-0038 — The AAanalysis ↔ ProtXplain border: three layers, one direction

Status: Accepted — 2026-06-23 (consolidated 2026-07-16; absorbs and removes the former validation/Pydantic border ADR)

This is the **single authoritative statement of the border**. Earlier boundary
decisions were spread across a separate validation/Pydantic ADR (now removed, its
decisions folded in as D10/D11), this ADR's original 2026-06-23 body
(improve-vs-use), and border asides in ADR-0036 (typing), ADR-0040 / ADR-0041
(`aaanalysis.pipe`), ADR-0062 (`find_features`) and ADR-0063 (sklearn
transformer). Those ADRs keep their own API and science decisions; **their border
claims are superseded by this one.**
Where an older ADR's wording disagrees with the vocabulary in D13, this ADR wins.

## Context

AAanalysis is the open scientific package; **ProtXplain** is the closed downstream
agent, decision and execution system. Two years of decisions kept re-litigating the
same question — "does this belong here or downstream?" — because the border was
stated in three places and in words that do not survive contact:

1. **"The border is the machine-readable tool contract"** (the old D3) is
   ambiguous: Python signatures, type hints, `py.typed` and DataFrames are *already*
   machine-readable. Read literally, that phrase withholds ordinary Python
   legibility, which was never the intent.
2. **"pro" and "ProtXplain" get conflated.** Issue and guide wording such as
   "ProtXplain / pro" implies one boundary where there are two independent ones:
   `aaanalysis[pro]` is **open source** and its boundary is *dependency weight*;
   ProtXplain is **closed** and its boundary is *agent decision logic*.
3. **"Selection/ranking" and "OOD/uncertainty logic"** were assigned wholesale to
   ProtXplain, which contradicts shipped, accepted ADRs: AAanalysis openly performs
   feature selection, CPP-configuration search (ADR-0044/0062), Pareto optimization
   (ADR-0043), bootstrap stability ranking (ADR-0055), repeated-CV model comparison
   (ADR-0061) and reliability/uncertainty estimation.

A second external agentic-readiness review (2026-07) confirmed the architecture is
sound but scored the *written* border 8.3/10 on exactly these four ambiguities. It
also corrected an earlier recommendation: a public capability registry, typed result
envelopes and an agent error hierarchy should **not** be added to AAanalysis — they
would blur the chosen border. This ADR states the border once, in words that hold.

## Decision

### D1. Three layers, not two

| Layer | Licence | Boundary criterion | Owns |
|---|---|---|---|
| **AAanalysis core** | open | lightweight, robust dependencies | scientific primitives, pipelines, sklearn interop |
| **AAanalysis[pro]** | **open** | heavy / fragile / narrow optional dependencies | the same science, gated by an optional extra |
| **ProtXplain** | closed | agent contract, decision and execution | tool intelligence, orchestration, governance |

`pro` is a **packaging** boundary, **not** the commercial one. A `*_pro` subpackage
is open source that happens to need `shap`, structure tooling or another heavy
extra; it is imported conditionally and stubbed when the extra is absent. Nothing
becomes closed by being `pro`, and nothing becomes ProtXplain's by being heavy.

### D2. What the border actually is

> **The border is the semantically complete, externally invokable autonomous-agent
> contract, together with its decision and execution layer.**

This replaces "the machine-readable tool contract". Specifically ProtXplain-side:
stable agent tool IDs; JSON Schema; MCP / function-calling definitions; task and
capability semantics; execution metadata; typed result envelopes; structured error
codes; selection and recovery policy; API exposure. **Ordinary Python introspection
— signatures, type hints, `py.typed`, docstrings, `__all__`, DataFrames — stays
open and is not "the contract" in this sense.**

### D3. The border is *not* "simple science open, clever science closed"

AAanalysis may contain arbitrarily sophisticated scientific capability: CPP and
feature engineering, dPULearn, predictors and evaluation, XAI, reliability and
uncertainty algorithms, clustering, protein-design and sequence-optimization
primitives, scientific metrics and plots. `SeqMut`, `SeqOpt`, `ModelEvaluator`,
nested feature selection and `SequenceFeatureTransformer` are all open by accepted
ADRs, even though each performs optimization, comparison or automated scientific
search. Making the open package artificially weak would damage adoption and create
no defensibility.

### D4. AAanalysis owns

- **Scientific capability** — every algorithm above, core or `pro`.
- **Human usability** — the ordinary class API; `aaanalysis.pipe` (`ap`); a small
  number of stable golden pipelines; sklearn-compatible transformers; static
  presets (`quick` / `balanced` / `thorough`); notebooks and protocols.
- **Static contracts for Python consumers** — honest type hints, `py.typed`,
  documented runtime validation, semver-stable signatures, documented DataFrame
  schemas, stable feature identifiers, deterministic seeds, plain exceptions,
  plain numpy / pandas outputs.
- **Scientific software quality** — tests and regression anchors, version truth,
  dependency and release security, optional provenance.

### D5. ProtXplain owns

- **Agent tool contract** — stable tool IDs, Pydantic request/result models, JSON
  schemas, MCP / function-calling definitions, typed result envelopes, structured
  error codes, schema migrations, remote-execution compatibility.
- **Capability intelligence** — machine-readable capability registry, biological
  task ontology, pre/postconditions, tool-compatibility graph, supported input
  modalities, resource and runtime estimates, benchmark evidence, known
  limitations, alternatives and fallbacks.
- **Decision layer** — user-intent interpretation, **tool and workflow** selection
  and ranking, OOD-based refusal, confidence thresholds, recovery after failure,
  human-in-the-loop questions, experiment recommendations.
- **Execution and governance** — MCP/API gateway, isolated execution, permissions,
  cost and compute budgets, cancellation and timeouts, artifact storage, audit
  trail, workflow versioning, proprietary tools, private data.

### D6. Uncertainty: measurement is open, action is closed

- **AAanalysis** calculates uncertainty: reliability models, confidence intervals,
  bootstrap stability, repeated CV, calibration metrics, scientific OOD /
  applicability scores, statistical evidence.
- **ProtXplain** acts on it: combining signals, deciding whether confidence
  suffices, refusal thresholds, fallback choice, escalation to a human.

> Uncertainty estimation is open science; uncertainty-driven action is closed
> product logic.

This narrows the old "ProtXplain owns OOD/uncertainty logic", which read far too
broadly and contradicted shipped reliability code.

### D7. Selection: scientific selection is open, tool/workflow selection is closed

- **AAanalysis** performs *selection inside a defined scientific method*: feature
  selection, scale ranking, model selection within an evaluation, hyperparameter
  and CPP-configuration search, mutation optimization, Pareto optimization,
  bootstrap stability ranking.
- **ProtXplain** performs *selection of the method*: which tool, which workflow,
  CPP vs embeddings vs structure vs an external tool, which preset given data size
  and budget, what to do next given intent, evidence, cost and constraints.

Always write **"tool/workflow selection and ranking"** when the closed layer is
meant. Bare "selection and ranking" is ambiguous and is not used in this repo.

> AAanalysis optimizes *inside* a defined scientific method. ProtXplain decides
> *which* methods and workflows to use.

### D8. Golden-pipeline boundary rule

`aaanalysis.pipe` stays open. To stop it drifting into an open orchestration
framework, every golden pipeline must satisfy **all** of:

1. one clear scientific objective;
2. stateless execution;
3. deterministic when seeded;
4. no network service;
5. no dynamic tool discovery;
6. no benchmark-conditioned method selection;
7. no agent-specific JSON contract;
8. no task-planning loop;
9. no hidden call to ProtXplain;
10. results reproducible through explicit primitive calls.

A pipeline that violates any of these is an **agent workflow** and belongs in
ProtXplain. `ap.find_features()` combining several primitives does not make it
orchestration: it is predetermined, stateless, human-callable and single-package.

### D9. Dependency direction is one-way, and ProtXplain adapts per version

```
ProtXplain  ───────►  AAanalysis        (allowed: pinned release)
AAanalysis  ───✕───►  ProtXplain        (never)
```

AAanalysis must **never** import, depend on, special-case or name ProtXplain in
its runtime code. ProtXplain depends on a **precisely pinned** AAanalysis release
and maintains an **adapter per supported version**, so AAanalysis changes do not
force a change to the external agent tool contract. The adapter — Pydantic
request/result models, exception→error-code translation, result normalization,
provenance enrichment, version mapping — is closed and lives downstream.

```
Human Python user
       │
       ▼
┌────────────────────────────────────────┐
│ AAanalysis core / AAanalysis[pro]      │  open
│ scientific classes · golden pipelines  │
│ sklearn transformers · numpy/pandas    │
│ documented schemas · provenance        │
│ ordinary Python exceptions             │
└───────────────────┬────────────────────┘
                    ▲  stable Python + data contract, pinned version
┌───────────────────┴────────────────────┐
│ ProtXplain AAanalysis adapter          │  closed
│ Pydantic request/result · error codes  │
│ result normalization · version mapping │
└───────────────────┬────────────────────┘
                    ▲  stable internal tool contract
┌───────────────────┴────────────────────┐
│ ProtXplain agent + execution layer     │  closed
│ capability registry · tool/workflow    │
│ selection · benchmark evidence · MCP   │
│ OOD/refusal/recovery · audit           │
└───────────────────┬────────────────────┘
                    ▲
              Scientific agent
```

### D10. The open contract AAanalysis owes the adapter

The adapter is only as reliable as what it pins. AAanalysis therefore commits to:

| Open contract | Why the adapter needs it |
|---|---|
| honest public type hints | safe adapter implementation |
| `py.typed` | static downstream checking |
| semver-stable signatures | version compatibility |
| stable `df_feat` schema | feature consumption |
| stable `PART-SPLIT-SCALE` feature ids | interpretation and visualization |
| stable prediction / evaluation tables | result normalization |
| stable per-residue score shapes | sequence-level explanation |
| deterministic seed semantics | reproducibility |
| documented warnings and failure behaviour | error mapping |
| provenance metadata | audit and traceability |
| contract tests | detect drift before release |

The documented output contract **spans more than `df_feat`**: it covers `df_eval`,
prediction score outputs, per-residue explanation outputs, mutation-scan tables,
stability columns, and golden-pipeline return slots. This stays an ordinary,
documented, test-guarded **scientific data contract** (a data dictionary anchored
on `ut.LIST_COLS_FEAT` / `sort_cols_feat`); ProtXplain's normalized result
envelope stays closed. Changing a contracted column is a semver event.

### D11. Input validation stays `ut.check_*`; no Pydantic / pandera, in v1 or v2

(Folded from the removed validation/Pydantic ADR.) Public methods keep the sklearn-style
Validate block of hand-written `ut.check_*` helpers raising bare `ValueError` /
`RuntimeError` in the canonical `"'<name>' (<got>) should be <expected>"` format.
No model-validation framework enters AAanalysis for input contracts or output
schema. Pydantic models, JSON-schema emission and MCP definitions are the
adapter's, and are not duplicated upstream.

### D12. Provenance is ours, thin and opt-in; typed result envelopes are not

Default outputs stay plain numpy / pandas. A small, **plain-`dict`, opt-in,
JSON-serializable** provenance record is in scope as part of the reproducibility
contract and must **not** change any existing return type. Its load-bearing field
is the **effective resolved seed** (after `options["random_state"]` → constructor
→ per-call resolution) plus deterministic-vs-stochastic status; package version,
dependency versions, git commit when resolvable, and an input hash ride along.
Recording the effective seed and software version is a general reproducibility
responsibility, so it belongs in the open substrate — it is table-stakes, trivially
copyable, and it makes the paper-pipeline Use Cases auditable. A mandatory typed
`result.data/schema/metrics/artifacts` envelope is **not** in scope; that is the
ProtXplain result contract.

### D13. Errors stay bare; classification is the adapter's job

No custom exception hierarchy, no `AAanalysisError`, no error-code taxonomy in
AAanalysis. The canonical message format is parseable enough for the adapter to map
a raised error to a structured `{code, stage, recoverable, ...}` at the boundary.
AAanalysis's contribution is keeping that format consistent and precise.

### D14. No public capability registry in AAanalysis; the registry is ProtXplain's moat

AAanalysis keeps the raw capability **facts** derivable from what it already ships —
`__all__`, honest signatures, docstrings, the tutorial "You will learn" boxes
(Tool / Input / Output / Best used for), pro-gating, the abbreviation registry — so a
downstream generator can build a manifest without re-implementing internal
conventions. AAanalysis ships **no** public `capabilities()` / `describe()`
accessor: a public symbol is a permanent semver promise and there is no OSS-side
consumer; it is deferred until one exists. The consolidated machine-readable
capability graph lives in ProtXplain, auto-regenerated from each pinned release.

### D15. The moat is accumulated intelligence, not the wrapper

Anyone can wrap an open package. Pydantic models, MCP definitions, JSON schemas, a
1:1 tool registry, wrappers around `ap.predict` and structured error conversion are
**not**, alone, a defensible moat — a competent competitor reproduces them quickly.
The durable ProtXplain moat is what accumulates *around* the tools: capability
ontology (what each method can and cannot scientifically answer), benchmark evidence
by dataset type / sample size / biological setting, the compatibility graph,
decision policies, recovery knowledge, execution infrastructure, proprietary tools
and pipelines, feedback data on what actually solves user problems, biological
context (ProtSpace, literature, families, experimental evidence), and the trust
layer. Improving AAanalysis therefore *strengthens* ProtXplain: a trusted, widely
learned open engine lets the closed layer spend its effort on high-value decisions
instead of repairing weak primitives.

### D16. Release, typing, supply-chain and governance maturity are ours, now

Version truth, a typing ratchet that gates, trusted publishing / supply-chain
provenance, one canonical release path, notebook execution in CI, dead-code removal
and lightweight governance (CODEOWNERS) are **not** border questions. They make the
primitives trustworthy for humans and coding agents alike, are in scope now, and are
tracked as issues rather than deferred to v2.

### D17. Vocabulary

| Use | Not |
|---|---|
| tool/workflow selection and ranking | selection and ranking (ambiguous — D7) |
| uncertainty measurement (ours) / uncertainty policy (theirs) | OOD/uncertainty logic (too broad — D6) |
| agent-consumable scientific pipeline | agent-facing golden pipeline (implies it *is* the agent layer) |
| `aaanalysis[pro]` — open, heavy optional deps | "ProtXplain / pro" (conflates two boundaries — D1) |
| the externally invokable agent contract | the machine-readable contract (Python already is — D2) |

`find_features` is **agent-consumable** — ProtXplain may consume its score — but it
is not itself agent integration.

### D18. The decision rule for every future feature

Ask in order:

1. **Is this a reusable scientific computation a direct Python user would
   reasonably need?** → yes: AAanalysis.
2. **Does it need heavy or fragile optional dependencies?** → yes:
   `aaanalysis[pro]`; no: core.
3. **Does it describe *when, why* or *in which workflow* a method should be
   used?** → ProtXplain.
4. **Does it expose a method to autonomous agents via MCP, JSON Schema or an
   external tool contract?** → ProtXplain.
5. **Does it combine evidence, cost, confidence, user intent or biological context
   to choose the next action?** → ProtXplain.

A feature may split: the scientific algorithm goes to AAanalysis (core or `pro`),
its selection and operational policy to ProtXplain.

| Capability | Home |
|---|---|
| Calculate prediction uncertainty | AAanalysis |
| Refuse a prediction because uncertainty is too high | ProtXplain |
| Compute feature stability | AAanalysis |
| Decide whether a result is stable enough for an experiment | ProtXplain |
| Run `SeqOpt` | AAanalysis |
| Decide whether to use `SeqOpt`, RFdiffusion or ProteinMPNN | ProtXplain |
| `SequenceFeatureTransformer` | AAanalysis |
| An MCP tool wrapping that transformer | ProtXplain |
| A `search="fast"` preset | AAanalysis |
| Dynamically selecting the preset from data size and cost budget | ProtXplain |
| `ModelEvaluator` and confidence intervals | AAanalysis |
| Ranking methods using accumulated benchmark evidence | ProtXplain |
| A SHAP implementation needing heavy dependencies | `aaanalysis[pro]` |
| Interpreting SHAP confidence and selecting the next workflow | ProtXplain |

## Rejected alternatives

- **"Machine-readable tool contract" as the border phrase.** Python signatures,
  type hints and docstrings are machine-readable; the phrase reads as withholding
  ordinary legibility. Replaced by D2.
- **A public capability registry / `capabilities()` accessor in AAanalysis.**
  Re-proposed by both reviews; the second retracted it. It blurs the border,
  creates a permanent semver promise with no OSS consumer, and gives away the one
  thing worth accumulating (D14/D15).
- **Mandatory typed result envelopes as the default output.** Breaks the plain
  numpy/pandas contract that lets outputs feed sklearn and torch equally, and pulls
  the ProtXplain result model upstream. Opt-in provenance (D12) covers
  reproducibility without changing return types.
- **A custom exception hierarchy / error-code taxonomy in AAanalysis.** Reverses
  the no-`AAanalysisError` rule for a benefit needed only at the boundary, where a
  thin adapter maps the canonical message to a code (D13).
- **Pydantic / pandera for I/O contracts, and a verb + MCP / tool-schema layer in
  AAanalysis** (carried over from the removed validation ADR's rejected
  alternatives; re-proposed by the 2026-07 review). Still rejected: duplicate
  validation, an added dependency, a
  conflict with bare exceptions, semver weight, and a border violation.
- **Withholding convenience to protect the moat** — deliberately making AAanalysis
  awkward to call. Damages adoption and creates no defensibility, because the moat
  was never the wrapper (D15).
- **Treating `pro` as the commercial boundary.** `pro` is open source; its criterion
  is dependency weight. Conflating the two mis-routes features that need `shap` into
  a closed layer where no user can read them (D1).
- **Assigning all "selection/ranking" and "OOD/uncertainty" to ProtXplain.**
  Contradicts shipped, accepted ADRs (find_features, SeqOpt, bootstrap stability,
  ModelEvaluator, reliability models). Narrowed by D6/D7.

## Consequences

- **One border ADR.** The former validation/Pydantic ADR is removed and folded in
  (D10/D11); the border asides in ADR-0036 / 0040 / 0041 / 0062 / 0063 are
  superseded by this ADR, while those ADRs keep their own API and science
  decisions. Its number is retired rather than reused, so an old citation cannot
  silently resolve to a different decision.
- **Wording debt to clear:** every "ProtXplain / pro" phrasing must be resolved to
  one of the three layers (D1), and D17's vocabulary applies to new issues, guides
  and ADRs.
- **Issue #26 widens:** the documented output contract spans `df_eval`, prediction
  scores, per-residue explanations, mutation-scan tables, stability columns and
  pipeline return slots — not `df_feat` alone (D10).
- **Nothing about the placement changes:** no MCP server, tool schema, verb
  orchestration, typed request/result model, capability registry, or tool/workflow
  decision logic enters AAanalysis.
- The target profile is deliberate: AAanalysis scores highly as an open scientific
  SDK and for coding-agent improvability, and *intentionally* scores ~5/10 on
  autonomous-agent integration. That gap is ProtXplain's product, not a defect.

## Out of scope

- ProtXplain's internal design (adapter layout, registry format, policy engine).
- The `aaanalysis.pipe` API and transformer implementations (ADR-0040/0041/0063).
- Per-item maturity work (D16), tracked as its own issues and ADRs.
