# ADR-0038 — The AAanalysis ↔ ProtXplain border: three layers, one direction

Status: Accepted — 2026-06-23 (consolidated 2026-07-16; absorbs and removes the former validation/Pydantic border ADR)

> **AAanalysis is the trusted open scientific engine. ProtXplain is the proprietary
> decision and execution system that knows how, when and why to use it.**
>
> AAanalysis should be agent-**legible** and agent-**improvable**, but not
> agent-**integrated**. ProtXplain turns AAanalysis into governed, selectable,
> composable agent tools.

This is the **single authoritative statement of the border**. The border was
previously distributed across a separate validation/Pydantic ADR (now removed, its
decisions folded in as D11/D13), this ADR's original 2026-06-23 body
(improve-vs-use), and border asides in ADR-0036, ADR-0040 / ADR-0041, ADR-0062 and
ADR-0063. Those ADRs keep their own API and science decisions and are mapped in
*Adjacent ADRs* below; **where their wording disagrees with this ADR, this ADR
wins.**

## Context

The architecture was never wrong — the *words* were. Three external
agentic-readiness reviews all confirmed the intended split (AAanalysis owns
scientific primitives, human-facing convenience, plain NumPy/pandas outputs,
sklearn compatibility, documentation, typing and maintainability; ProtXplain owns
the agent tool contract, MCP exposure, tool schemas, and tool/workflow selection,
ranking and decision logic). The latest scored the *written* border **8.3/10** and
traced every remaining point of confusion to four ambiguities in our own prose:

1. **"The border is the machine-readable tool contract."** Python signatures, type
   hints, `py.typed`, docstrings and DataFrames are *already* machine-readable. Read
   literally, the phrase withholds ordinary Python legibility — never the intent.
2. **`pro` and ProtXplain conflated.** Wording such as "ProtXplain / pro" in issue
   templates and guides implies one boundary where there are two independent ones.
   This is the largest remaining documentation ambiguity.
3. **"ProtXplain owns OOD/uncertainty logic."** Too broad: it contradicts shipped,
   accepted ADRs that put reliability models, bootstrap stability, repeated CV and
   confidence intervals openly inside AAanalysis.
4. **"ProtXplain owns selection and ranking."** Also too broad: AAanalysis openly
   performs feature selection, CPP-configuration search, Pareto optimization and
   stability ranking.

The same review also **retracted** an earlier recommendation: it would *not* now add
a public capability registry, typed result envelopes or an agent error hierarchy to
AAanalysis, because those blur the chosen border. Plain provenance here, full typed
contracts and capability intelligence downstream, is better aligned with the
strategy.

---

## Decision

### § A — The layers and the border

#### D1. Three layers, not two

| Layer | Licence | Boundary criterion | Owns |
|---|---|---|---|
| **AAanalysis core** | open | lightweight, robust dependencies | scientific primitives, pipelines, sklearn interop |
| **AAanalysis[pro]** | **open** | heavy, fragile or narrowly used optional dependencies | the same science, gated by an optional extra |
| **ProtXplain** | closed | agent contract, decision and execution | tool intelligence, orchestration, governance |

**`pro` is a packaging boundary, not the commercial boundary.** A `*_pro`
subpackage is open source that happens to need `shap`, external structure tooling,
causal/XAI libraries or another heavy extra; it is imported conditionally and, when
the extra is absent, replaced by a stub that raises an installation hint. Core and
pro methods may share schemas while producing scientifically different results
(ADR-0057 places some XAI in core and some in the open `explainable_ai_pro`
precisely on dependency burden). **Nothing becomes closed by being `pro`, and
nothing becomes ProtXplain's by being heavy.**

Never write "ProtXplain / pro". Resolve it to a specific layer:
*"`aaanalysis[pro]`, because of optional heavy dependencies"* or *"ProtXplain,
because it is agent/product orchestration."* A feature may split across both: the
scientific algorithm to `aaanalysis[pro]`, its selection and operational policy to
ProtXplain.

#### D2. What the border actually is

> **The border is the semantically complete, externally invokable
> autonomous-agent contract, together with its decision and execution layer.**

This replaces "the machine-readable tool contract". The border specifically
comprises: stable agent tool IDs; JSON schemas; task and capability semantics;
execution metadata; structured results and errors; MCP / function-calling
definitions; and selection and recovery policy. All ProtXplain.

**Ordinary Python introspection remains open** — signatures, type hints,
`py.typed`, docstrings, `__all__`, DataFrames. These are not "the contract" in this
sense. ADR-0036 already settled the corollary: keeping the OSS package type-opaque
provides no useful defensive advantage, and ProtXplain must be able to type-check
cleanly against AAanalysis.

#### D3. The border is **not** "simple science open, clever science closed"

AAanalysis may contain arbitrarily sophisticated capability, not merely low-level
primitives: interpretable feature engineering; supervised and PU learning;
prediction and model comparison; uncertainty and reliability estimation; scientific
XAI; statistical evaluation; nested cross-validation; mutation scoring;
evolutionary sequence optimization; sklearn-native transformers; and user-facing
one-call pipelines.

The later ADRs demonstrate this: `SeqMut` (ADR-0042), `SeqOpt` (ADR-0043/0045),
`ModelEvaluator` (ADR-0061), nested feature selection (ADR-0062) and
`SequenceFeatureTransformer` (ADR-0063) are all open, even though each performs
optimization, comparison or automated scientific search. A border drawn on
cleverness would make the open package artificially weak and would contradict
several accepted ADRs.

### § B — Ownership

#### D4. AAanalysis owns

**Scientific capability** — CPP and other feature engineering; dPULearn;
predictors and evaluation; XAI methods; reliability and uncertainty algorithms;
clustering and protein-design primitives; scientific metrics and plots.

**Human-facing usability** — the ordinary class API; `aaanalysis.pipe`; a small
number of stable golden pipelines; sklearn-compatible transformers; sensible
*static* presets such as `quick` / `balanced` / `thorough`; notebooks and
protocols; normal Python type hints.

**Scientific software quality** — tests and scientific regression anchors;
truthful versions and documentation; dependency and release security;
deterministic seed handling; optional provenance; stable documented DataFrame
contracts; plain, understandable exceptions; NumPy/pandas outputs.

#### D5. ProtXplain owns

**Agent tool contract** — stable tool IDs; Pydantic request and result models;
JSON schemas; MCP / function-calling definitions; typed result envelopes;
structured error codes; schema migrations; compatibility with remote execution.

**Capability intelligence** — the machine-readable capability registry; biological
task ontology; method pre- and postconditions; the tool-compatibility graph;
supported input modalities; resource and runtime estimates; evidence and benchmark
metadata; known limitations; alternative and fallback tools. AAanalysis may
document each method thoroughly; the *consolidated machine-readable capability
graph* stays downstream.

**Decision layer** — user-intent interpretation; **tool and workflow** selection;
ranking based on task, data and evidence; OOD detection as a *decision*; refusal
and escalation policy; confidence thresholds; recovery after failed execution;
human-in-the-loop questions; experiment recommendations.

**Execution and governance** — MCP/API gateway; isolated execution; permissions;
cost and compute budgets; cancellation and timeout handling; artifact storage; a
complete audit trail; workflow versioning; proprietary tool integration;
organizational knowledge and private data. These are not wrappers around Python
functions — they are the operational agent platform.

### § C — The four distinctions that keep being confused

#### D6. Uncertainty: measurement is open, action is closed

| AAanalysis calculates | ProtXplain decides |
|---|---|
| uncertainty estimation methods | which uncertainty method to use |
| confidence intervals | combining several confidence signals |
| calibration metrics | whether confidence is sufficient to continue |
| stability analyses | refusal thresholds |
| scientific OOD / applicability scores | choosing a fallback workflow |
| reliability models, statistical evidence | escalation to human review |

> **Uncertainty estimation is open science; uncertainty-driven action is closed
> product logic.**

This narrows the old "ProtXplain owns OOD/uncertainty logic", which read as
withholding the reliability algorithms AAanalysis already ships (ADR-0061, ADR-0055).

#### D7. Selection: scientific selection is open, tool/workflow selection is closed

| AAanalysis selects *inside* a method | ProtXplain selects *the method* |
|---|---|
| feature selection; scale ranking | which scientific tool to use |
| model selection within an evaluation | the overall workflow |
| hyperparameter search | CPP vs embeddings vs structure vs external tools |
| CPP-configuration search (ADR-0044/0062) | presets from data size, budget, rigor |
| mutation optimization (ADR-0042) | composing several tools end to end |
| Pareto optimization (ADR-0043) | changing strategy after intermediate results |
| bootstrap stability ranking (ADR-0055) | ranking alternative analyses for a user's task |

> **AAanalysis optimizes *inside* a defined scientific method. ProtXplain decides
> *which* methods and workflows should be used.**

Always write **"tool/workflow selection and ranking"** for the closed layer. Bare
"selection and ranking" is ambiguous and is not used in this repo.

#### D8. Golden pipeline vs agent workflow — and the boundary rule

`ap.find_features()` or `ap.predict_samples()` may combine several primitives. That
does not make them orchestration.

| A golden pipeline is | An agent workflow is |
|---|---|
| predetermined | selected dynamically |
| stateless | context-dependent |
| directly callable by a human | aware of goals and constraints |
| scientifically coherent | potentially multi-package |
| implemented entirely within one package | able to branch and recover |
| not dynamically planned | governed by confidence, permissions, budget |
| not selected from a capability graph | |
| not adapted through iterative reasoning | |

**The boundary rule.** `aaanalysis.pipe` stays open (ADR-0040/0041). To stop it
drifting into an open orchestration framework, every golden pipeline must satisfy
**all** of:

1. one clear scientific objective;
2. stateless execution;
3. deterministic behaviour when seeded;
4. no network service;
5. no dynamic tool discovery;
6. no benchmark-conditioned method selection;
7. no agent-specific JSON contract;
8. no task-planning loop;
9. no hidden call to ProtXplain;
10. results reproducible through explicit primitive calls.

Violate any one and it is an agent workflow: it belongs downstream.

#### D9. Convenience does not leak the agent layer

`aaanalysis.pipe` is **not** already too agent-compatible. A Python function such
as `ap.cpp_feature_map(...)` is easy for humans and easier for coding assistants,
but it does **not** tell an autonomous agent:

- when CPP is scientifically appropriate;
- whether embeddings would be better;
- whether the labels constitute a PU-learning problem;
- which optimization preset is justified;
- whether the sample size is sufficient;
- how much the computation costs;
- which downstream analysis is compatible;
- whether the result is out of distribution;
- whether the tool failed recoverably;
- what alternative workflow to try;
- whether human confirmation is required.

**That semantic and operational layer is the real agent interface** — and it is
entirely ProtXplain's. Therefore the package stays convenient. Deliberately making
AAanalysis difficult to call would damage adoption without creating any meaningful
defensibility.

### § D — The seam

#### D10. The dependency direction is one-way, and ProtXplain adapts per version

```
ProtXplain  ───────►  AAanalysis        (allowed: a precisely pinned release)
AAanalysis  ───✕───►  ProtXplain        (never)
```

AAanalysis must **never** import, depend on, special-case or name ProtXplain in its
runtime code. ProtXplain depends on a **precisely pinned** AAanalysis release and
maintains an **adapter per supported version**, so changes to AAanalysis do not
force a change to the external ProtXplain tool contract. Issue #26 already enforces
the direction by prohibiting ProtXplain-specific code and dependencies here.

```
Human Python user
       │
       ▼
┌─────────────────────────────────────────────┐
│ AAanalysis core / AAanalysis[pro] — open    │
│                                             │
│ scientific classes                          │
│ golden pipelines                            │
│ sklearn transformers                        │
│ pandas / NumPy outputs                      │
│ documented schemas                          │
│ reproducibility metadata                    │
│ ordinary Python exceptions                  │
└──────────────────────┬──────────────────────┘
                       ▲
                       │ stable Python + data contract
                       │ pinned AAanalysis version
┌──────────────────────┴──────────────────────┐
│ ProtXplain AAanalysis adapter — closed      │
│                                             │
│ Pydantic request/result models              │
│ schema normalization                        │
│ exception → error-code translation          │
│ provenance enrichment                       │
│ AAanalysis-version compatibility mapping    │
└──────────────────────┬──────────────────────┘
                       ▲
                       │ stable internal tool contract
┌──────────────────────┴──────────────────────┐
│ ProtXplain agent and execution layer        │
│                                             │
│ capability registry                         │
│ tool and workflow selection                 │
│ benchmarks and evidence                     │
│ MCP / API exposure                          │
│ OOD / refusal / recovery policy             │
│ execution, artifacts and audit              │
└──────────────────────┬──────────────────────┘
                       ▲
                       │
                 Scientific agent
```

#### D11. The open contract AAanalysis owes the adapter

The adapter is only as reliable as what it pins. AAanalysis therefore commits to:

| Open AAanalysis contract | Why ProtXplain needs it |
|---|---|
| honest public type hints | safe adapter implementation |
| `py.typed` (ADR-0036) | static downstream checking |
| semver-stable method signatures (ADR-0030) | version compatibility |
| stable `df_feat` schema | feature consumption |
| stable `PART-SPLIT-SCALE` feature ids | interpretation and visualization |
| stable prediction / evaluation tables (ADR-0022) | result normalization |
| stable per-residue score shapes | sequence-level explanation |
| deterministic seed semantics | reproducibility |
| documented warnings and failure behaviour | error mapping |
| provenance metadata (D13) | audit and traceability |
| contract tests | detect drift before release |

**None of this requires open Pydantic models, MCP definitions or a capability
registry.**

The documented output contract **spans more than `df_feat`**. Issue #26 pinned the
`df_feat` columns, the `PART-SPLIT-SCALE` feature-id format, per-residue score
shape, dtypes and a golden contract test; it should widen to the canonical
artifacts the later ADRs created: `df_eval`, prediction score outputs, per-residue
explanation outputs, mutation-scan tables, stability columns (ADR-0055), and
golden-pipeline return slots (ADR-0041). This stays an ordinary, documented,
test-guarded **scientific data contract** — a data dictionary anchored on
`ut.LIST_COLS_FEAT` / `sort_cols_feat`, not a typed model. ProtXplain's normalized
result envelope stays closed. Changing a contracted column is a semver event.

### § E — What stays out of AAanalysis

#### D12. Input validation stays `ut.check_*`; no Pydantic / pandera, in v1 or v2

Public methods keep the sklearn-style Validate block of hand-written `ut.check_*`
helpers raising bare `ValueError` / `RuntimeError` in the canonical
`"'<name>' (<got>) should be <expected>"` format. No model-validation framework
enters AAanalysis for input contracts or output schema. Pydantic models,
JSON-schema emission and MCP definitions belong to the adapter and are **not**
duplicated upstream.

#### D13. Provenance is ours, thin and opt-in; typed result envelopes are not

Default outputs stay plain numpy / pandas. A small, **plain-`dict`, opt-in,
JSON-serializable** provenance record is in scope and must **not** change any
existing return type. Its load-bearing field is the **effective resolved seed**
(after `options["random_state"]` → constructor → per-call resolution) plus the
run's deterministic-vs-stochastic status; package version, dependency versions, git
commit when resolvable, and an input hash ride along.

Recording the effective seed and software version is a **general reproducibility
responsibility**, so it belongs in the open substrate: it is table-stakes, trivially
copyable, not a moat, and it makes the paper-pipeline Use Cases auditable. A
mandatory typed `result.data/schema/metrics/artifacts` envelope is **not** in
scope — that is the ProtXplain result contract.

#### D14. Errors stay bare; error classification is the adapter's job

No custom exception hierarchy, no `AAanalysisError`, no error-code taxonomy in
AAanalysis. The canonical message format is parseable enough for the adapter to map
a raised error to a structured `{code, stage, recoverable, ...}` at the boundary.
AAanalysis's contribution is keeping that format consistent and precise.

#### D15. No public capability registry in AAanalysis

AAanalysis keeps the raw capability **facts** derivable from what it already ships —
`__all__`, honest signatures, docstrings, the tutorial "You will learn" boxes
(Tool / Input / Output / Best used for), pro-gating, the abbreviation registry — so
a downstream generator can build a manifest without re-implementing internal
conventions. AAanalysis ships **no** public `capabilities()` / `describe()`
accessor: a public symbol is a permanent semver promise and there is no OSS-side
consumer, so it is deferred until one exists. The consolidated capability graph
lives in ProtXplain, auto-regenerated from each pinned release.

#### D16. The moat is accumulated intelligence, not the wrapper

Anyone can build their own agent wrapper around an open package. The following are
**not**, alone, a strong moat — a competent competitor reproduces them quickly:
Pydantic models; MCP definitions; JSON schemas; a one-to-one tool registry;
wrappers around `ap.predict`; structured error conversion.

The strong ProtXplain moat is the **accumulated intelligence around the tools**:

- **capability ontology** — what every method can and cannot scientifically answer;
- **benchmark evidence** — performance by dataset type, sample size, biological setting;
- **compatibility graph** — which tools, inputs and outputs compose safely;
- **decision policies** — how to select among several plausible methods;
- **recovery knowledge** — what to do when an analysis is unstable, underpowered or fails;
- **execution infrastructure** — reproducible GPU/CPU workflows, artifacts, caching, audit;
- **proprietary tools and pipelines** — methods unavailable in the open package;
- **feedback data** — which workflows actually solve users' problems;
- **biological context** — ProtSpace, literature, protein families, experimental evidence;
- **trust layer** — confidence, OOD detection, validation, human approval.

The interface architecture is strong **only when the closed layer contains
substantially more than an MCP wrapper.**

### § F — Working rules

#### D17. Release, typing, supply-chain and governance maturity are ours, now

Version truth, a static-typing ratchet that actually gates, trusted publishing and
supply-chain provenance, one canonical release path (ADR-0060), notebook execution
in CI, dead-code removal, and lightweight governance (CODEOWNERS) are **not** border
questions. They make the primitives trustworthy for human users and coding agents
alike, are in scope **now** rather than deferred to v2, and are tracked as issues. An
item that refines an earlier ADR gets its own follow-up ADR.

#### D18. Vocabulary

| Use | Not |
|---|---|
| tool/workflow selection and ranking | "selection and ranking" (ambiguous — D7) |
| uncertainty **measurement** (ours) / uncertainty **policy** (theirs) | "OOD/uncertainty logic is ProtXplain's" (too broad — D6) |
| **agent-consumable** scientific pipeline | "agent-facing golden pipeline" (implies it *is* the agent layer) |
| `aaanalysis[pro]` — open, heavy optional deps | "ProtXplain / pro" (conflates two boundaries — D1) |
| the externally invokable **agent contract** | "the machine-readable contract" (Python already is — D2) |

ADR-0062 calls `find_features` an "agent-facing golden pipeline" because a
downstream agent may consume its score. The clearer term is **agent-consumable
scientific pipeline**: it is consumable *by* ProtXplain, but it is not itself the
agent-integration layer.

#### D19. The routing rule for every future feature

Ask in order:

1. **Is this a reusable scientific computation a direct Python user would
   reasonably need?** → yes: **AAanalysis**. No: continue.
2. **Does it require heavy or fragile optional dependencies?** → yes:
   **`aaanalysis[pro]`**; no: **AAanalysis core**.
3. **Does it describe *when*, *why* or *in which workflow* a scientific method
   should be used?** → **ProtXplain**.
4. **Does it expose a method to autonomous agents through MCP, JSON Schema or an
   external tool contract?** → **ProtXplain**.
5. **Does it combine evidence, cost, confidence, user intent or biological context
   to choose the next action?** → **ProtXplain**.

| Capability | Correct home |
|---|---|
| Calculate prediction uncertainty | AAanalysis |
| Refuse a prediction because uncertainty is too high | ProtXplain |
| Compute feature stability | AAanalysis |
| Decide whether the result is sufficiently stable for an experiment | ProtXplain |
| Run `SeqOpt` | AAanalysis |
| Decide whether to use `SeqOpt`, RFdiffusion or ProteinMPNN | ProtXplain |
| `SequenceFeatureTransformer` | AAanalysis |
| An MCP tool wrapping the transformer | ProtXplain |
| A `search="fast"` preset | AAanalysis |
| Dynamically selecting the preset from data size and cost budget | ProtXplain |
| `ModelEvaluator` and confidence intervals | AAanalysis |
| Ranking methods using accumulated benchmark evidence | ProtXplain |
| A SHAP implementation requiring heavy dependencies | `aaanalysis[pro]` |
| Interpreting SHAP confidence and selecting the next workflow | ProtXplain |

---

## Adjacent ADRs — what each contributes to the border

These are **not** border ADRs; each settles its own API or science question. They
are listed because each one *demonstrates* where the border falls, and because
their border wording is superseded by this ADR.

| ADR | Its own decision | What it settles about the border |
|---|---|---|
| [0022](0022-prediction-task-level-taxonomy.md) | prediction-task taxonomy (residue / domain / protein) | the level vocabulary and prediction outputs the adapter normalizes (D11) |
| [0026](0026-feature-pruning-empirical-not-scale-correlation.md) | empirical feature pruning, `df_feat`-in/out | scientific selection is ours (D7) |
| [0027](0027-protein-design-mutation-deltacpp-scope.md) | AAMut/SeqMut scope, model-free ΔCPP | a scope boundary drawn *inside* the open package (D3) |
| [0030](0030-changelog-and-deprecation-policy.md) | strict-semver deprecation policy | the signature stability the adapter pins (D11) |
| [0036](0036-type-hints-contract-checker-deferred-pyright.md) | ship `py.typed`, adopt pyright | type-opacity buys no defensibility; the moat is downstream (D2, D16) |
| [0039](0039-plot-return-contract.md) | uniform `(fig, ax)` return | plain returns, no envelope (D13) |
| [0040](0040-golden-pipelines-convenience-api.md) | the `aaanalysis.pipe` (ap) convenience API | convenience is ours; the tool contract is not (D8) |
| [0041](0041-pipe-pipeline-conventions.md) | pipe conventions + core golden pipelines | the pipeline return slots the contract spans (D8, D11) |
| [0042](0042-seqmut-model-guided-prediction-shift.md) | SeqMut model-guided prediction shift | model-aware design is open science (D3, D7) |
| [0043](0043-seqopt-optimization-layer.md) | SeqOpt multi-objective optimization | optimization *inside* a method is ours (D3, D7) |
| [0044](0044-find-features-search-protocol.md) | find_features staged search | configuration search is scientific selection (D7) |
| [0046](0046-predict-samples-multi-model-harness.md) | predict_samples comparison harness | model comparison is ours; tool comparison is not (D7) |
| [0053](0053-dpulearn-project-out-of-sample.md) | dPULearn.project | PU primitives are ours (D4) |
| [0055](0055-cpp-bootstrap-stability-selection.md) | CPP bootstrap / stability annotation | stability *measurement* is ours (D6); its columns are contracted (D11) |
| [0057](0057-v2-api-naming-and-future-xai-layer.md) | v2 naming + future XAI layer | core vs `pro` placement is dependency burden only (D1) |
| [0060](0060-packaging-install-from-wheel-gate.md) | packaging install-from-wheel gate | release trust the adapter pins against (D17) |
| [0061](0061-model-evaluator-repeated-cv-bootstrap-ci.md) | ModelEvaluator, repeated CV, bootstrap CIs | uncertainty *measurement* is ours (D6) |
| [0062](0062-find-features-nested-selection-scope.md) | nested selection scope | coined "agent-facing golden pipeline"; corrected to agent-consumable (D18) |
| [0063](0063-sequence-feature-transformer.md) | SequenceFeatureTransformer | sklearn interop is ordinary usability, not orchestration (D3, D4) |

Issue **#26** is the physical seam: it pins the open scientific data contract that
the closed adapter consumes (D10, D11).

## Rejected alternatives

- **"Machine-readable tool contract" as the border phrase.** Python signatures,
  type hints and docstrings are machine-readable; the phrase reads as withholding
  ordinary legibility. Replaced by D2.
- **Treating `pro` as the commercial boundary.** `pro` is open source; its criterion
  is dependency weight. Conflating the two mis-routes any feature needing `shap`
  into a closed layer where no user can read it. Rejected; see D1.
- **Assigning all "selection/ranking" and all "OOD/uncertainty" to ProtXplain.**
  Contradicts shipped, accepted ADRs (0042, 0043, 0044, 0055, 0061, 0062). Narrowed
  by D6/D7.
- **A public capability registry / `capabilities()` accessor in AAanalysis.**
  Proposed by an earlier review and explicitly retracted by the latest. It blurs the
  border, creates a permanent semver promise with no OSS consumer, and gives away
  the one thing worth accumulating. Rejected; see D15/D16.
- **Mandatory typed result envelopes as the default output.** Breaks the plain
  numpy/pandas contract that lets outputs feed sklearn and torch equally, and pulls
  the ProtXplain result model upstream. Rejected; opt-in provenance (D13) covers
  reproducibility without changing return types.
- **A custom exception hierarchy / error-code taxonomy in AAanalysis.** Reverses the
  no-`AAanalysisError` rule for a benefit needed only at the boundary, where a thin
  adapter maps the canonical message to a code. Rejected; see D14.
- **Pydantic / pandera for I/O contracts, and a verb + MCP / tool-schema layer in
  AAanalysis** (carried over from the removed validation ADR; re-proposed twice
  since). Still rejected: duplicate validation, an added dependency, a conflict with
  bare exceptions, semver weight, and a border violation.
- **Withholding convenience to protect the moat** — deliberately making AAanalysis
  awkward to call. Damages adoption and creates no defensibility, because the moat
  was never the wrapper. Rejected; see D9/D16.
- **Deferring release / typing / supply-chain maturity to v2.** Not algorithm or
  API-surface work; deferring lets version divergence and toothless gates persist.
  Rejected; see D17.

## Consequences

- **One border ADR.** The former validation/Pydantic ADR is removed and folded in;
  its number is **retired, not reused**, so an old citation cannot silently resolve
  to a different decision. The border asides in the adjacent ADRs above are
  superseded by this one.
- **The target profile is deliberate.** After the hardening plan AAanalysis should
  read as a strong open scientific SDK — high on scientific breadth, human
  usability, sklearn interoperability and (highest) coding-agent improvability —
  while scoring **~5/10 on autonomous-agent integration, intentionally**. That gap
  is not a defect: humans can use it directly, sklearn users can compose it, coding
  agents can understand and improve it, external developers can wrap it, but
  autonomous agents do not receive a complete discovery, selection, execution and
  recovery layer from the open package. Boundary clarity is the metric this ADR
  moves: **8.3/10 → ~9.5/10** once the wording corrections land.
- **Improving AAanalysis strengthens ProtXplain.** A more trusted scientific
  foundation, more researchers who know the API, more standardized outputs and more
  contributors improving the engine give ProtXplain a credible open anchor and let
  the closed layer spend its effort on high-value decisions instead of repairing
  weak primitives. The open and closed layers are not in tension.
- **Wording debt to clear:** every "ProtXplain / pro" phrasing resolves to one of the
  three layers (D1), and D18's vocabulary applies to new issues, guides and ADRs.
- **Issue #26 widens** beyond `df_feat` to the canonical artifacts in D11.
- **Nothing about the placement changes:** no MCP server, tool schema, verb
  orchestration, typed request/result model, capability registry, or tool/workflow
  decision logic enters AAanalysis.

> ProtXplain does not merely execute AAanalysis. It knows which capability to use,
> under which conditions, how to combine it with other tools, whether to trust the
> result, and what to do next.

## Out of scope

- ProtXplain's internal design (adapter layout, registry format, policy engine).
- The `aaanalysis.pipe` API and transformer implementations (ADR-0040/0041/0063).
- Per-item maturity work (D17), tracked as its own issues and ADRs.
