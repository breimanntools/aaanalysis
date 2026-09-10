# Additive roadmap — triage of the v1.2 → v2.X issue proposals

_Created 2026-07-18. A triage of a 53-item external suggestion (ChatGPT "additive
roadmap") against the live tracker, the coding standards (`CLAUDE.md` +
`.claude/rules/`) and the AAanalysis ↔ ProtXplain border ([ADR-0038](../adr/0038-aaanalysis-protxplain-border.md))._

**What this is.** An external review proposed 53 new issues to make AAanalysis
"scientifically complete, agent-ready and omics-interoperable". Rather than file 53
near-duplicates into a deliberately-curated tracker, every proposal was checked against
(a) the ~66 open issues, (b) the hard rules in `sharp-edges.md`, and (c) the border
ADR. The result: **22 filed** (net-new, single-deliverable, rule-clean), the rest
**deduplicated, folded, deferred, or routed to ProtXplain**.

**The border test that shaped this.** ADR-0038 already answers almost every boundary
question the proposals raise. The recurring seam (**D6/D7**, now generalised in the new
**D20**): AAanalysis owns the *measurement, diagnostic, evaluation metric, constraint
object and local record*; ProtXplain owns the *threshold-driven action, the stable code
taxonomy, the cross-project memory and the execution budget*. Three proposals that read
like AAanalysis features are ProtXplain under that seam and were **not** filed (see
*Routed to ProtXplain* below).

**Style note.** Filed issues follow `issue_style_guide.md` (Problem / Goal /
Requirements / measurable KPIs / Scope-non-goals / Dependencies / Standards checklist).
Per the hard rule, **no ADR is cited in any issue body**; the border rationale is stated
in plain language there. ADR cross-references live only here and in the ADR.

---

## Filed — 27 issues (#473–#499)

Priority mapping: the review's P0/P1 was re-mapped to the repo's `prio:1|2|3`. `prio:1`
is reserved for the maintainer's milestone-defining epics, so these additive contracts
are `prio:2` (load-bearing) or `prio:3` (supporting). The 8 tail issues carry the
`[Potential]` status blockquote.

### v1.2 — safe-design foundations

| # | Title | prio | Reframe / border |
|---|---|---|---|
| [#473](https://github.com/breimanntools/aaanalysis/issues/473) | Applicability-domain / OOD scoring | 2 | AD **scoring** is ours; the refusal/escalation *policy* is ProtXplain |
| [#474](https://github.com/breimanntools/aaanalysis/issues/474) | Selective-prediction risk–coverage **evaluation** | 3 | Reframed from an abstaining `predict_selective` (that refusal is ProtXplain) to the eval **metric** |
| [#475](https://github.com/breimanntools/aaanalysis/issues/475) | Shared `DesignConstraints` object | 2 | `ut.check_*`-validated params object, **not** Pydantic (D12) |
| [#476](https://github.com/breimanntools/aaanalysis/issues/476) | Candidate-lineage / design-provenance record | 3 | Plain-dict, like the #446 provenance record; **local** run only (campaign memory = ProtXplain) |
| [#477](https://github.com/breimanntools/aaanalysis/issues/477) | Golden-workflow failure-path contract tests | 2 | Bare `ValueError`/`RuntimeError` only; **no** exception hierarchy (D14) |

### v1.3 — leakage-safe evaluation

| # | Title | prio | Reframe / border |
|---|---|---|---|
| [#478](https://github.com/breimanntools/aaanalysis/issues/478) | Group / protein / family-aware CV splitters | 2 | Consumes supplied clusters; not a homology engine |
| [#479](https://github.com/breimanntools/aaanalysis/issues/479) | `aa.audit_leakage` diagnostic | 2 | Human-readable findings; block-or-approve *policy* is ProtXplain |
| [#480](https://github.com/breimanntools/aaanalysis/issues/480) | Probability calibration + reliability (Platt / isotonic, Brier, ECE) | 2 | Calibration **metrics** are ours (D6) |
| [#481](https://github.com/breimanntools/aaanalysis/issues/481) | External-validation protocol | 2 | Evaluate external evidence; sufficiency-for-claim is ProtXplain |
| [#482](https://github.com/breimanntools/aaanalysis/issues/482) | `AAPred.eval` parity across binary/multiclass/regression/PU | 3 | Task-def *choice* stays ProtXplain |
| [#486](https://github.com/breimanntools/aaanalysis/issues/486) | Deterministic parallel-execution conformance tests (`n_jobs` invariance) | 2 | Quality guard on shipped behaviour; guards the already-fixed #339 path (closed 2026-07-05); moved here from v1.5 |
| [#487](https://github.com/breimanntools/aaanalysis/issues/487) | **[Potential]** `DatasetCard` descriptive metadata | 3 | Plain-dict, extends `DICT_DF_SCHEMAS`; TaskCard (intended-use/governance) dropped as ProtXplain |
| [#488](https://github.com/breimanntools/aaanalysis/issues/488) | **[Potential]** Benchmark runner over shipped baselines | 3 | Runner, **not** a capability registry / auto-selector (D15) |

### v1.4 — multimodal interoperability

| # | Title | prio | Reframe / border |
|---|---|---|---|
| [#483](https://github.com/breimanntools/aaanalysis/issues/483) | Deterministic coordinate / identity mapping | 2 | From supplied maps; KB-resolution/agent-disambiguation is ProtXplain |
| [#484](https://github.com/breimanntools/aaanalysis/issues/484) | Multimodal-fusion evaluation protocol | 2 | Benchmarks combinations; production routing is ProtXplain |
| [#489](https://github.com/breimanntools/aaanalysis/issues/489) | **[Potential]** Preserve assay/batch/missingness metadata | 3 | No automatic batch correction |
| [#490](https://github.com/breimanntools/aaanalysis/issues/490) | **[Potential]** Proteomics matrix / long-table adapter | 3 | Consumes processed outputs; not MaxQuant/DIA-NN |
| [#491](https://github.com/breimanntools/aaanalysis/issues/491) | **[Potential]** Single-cell → protein aggregation bridge | 3 | scverse **consumer**, not a single-cell engine |

### v1.5 — scale & production data handling

| # | Title | prio | Reframe / border |
|---|---|---|---|
| [#485](https://github.com/breimanntools/aaanalysis/issues/485) | Chunked / out-of-core CPP feature generation | 2 | Local chunking; distributed exec is ProtXplain |
| [#492](https://github.com/breimanntools/aaanalysis/issues/492) | **[Potential]** Sparse / backed matrix support | 3 | Explicit failure over silent densification |
| [#493](https://github.com/breimanntools/aaanalysis/issues/493) | **[Potential]** Provenance-aware local caching / checkpoints | 3 | Local disk, no network; distributed cache is ProtXplain |
| [#494](https://github.com/breimanntools/aaanalysis/issues/494) | **[Potential]** Optional-dependency compatibility matrix (CI) | 3 | Infra; CONFIRM-FIRST (`workflows` + `pyproject`) |

### v2.X — architecture & multimodal (added 2026-07-18)

Filed after the "why nothing for v2.X?" review: the four genuinely-additive,
non-duplicate v2.X items, plus one epic holding the architecture-decision cluster (which
is ADR-first, not a build order). All `prio:3`, `[Potential]` except the epic.

| # | Title | topic | Reframe / border |
|---|---|---|---|
| [#495](https://github.com/breimanntools/aaanalysis/issues/495) | **[Potential]** Dataset-shift / drift diagnostics | core | Detects shift; retrain/recalibrate *decision* is ProtXplain |
| [#496](https://github.com/breimanntools/aaanalysis/issues/496) | **[Potential]** Scientific lineage graph | core | Lineage of one analysis; cross-project linking is ProtXplain; extends #476/#446 |
| [#497](https://github.com/breimanntools/aaanalysis/issues/497) | **[Potential]** Experimental-feedback result schema | data | Open record of one experiment; cross-project memory is the ProtXplain moat |
| [#498](https://github.com/breimanntools/aaanalysis/issues/498) | **[Potential]** Cross-modal explanation alignment | XAI | Aligns/reports; collapsing to one conclusion is ProtXplain; needs #483 |
| [#499](https://github.com/breimanntools/aaanalysis/issues/499) | **epic:** v2 unified architecture | core | Holds TaskSpec/DataSpec · typed result objects · namespaces · migration · extension protocols — ADR-first, plain-dict, no capability registry / typed envelope |

---

## Not filed as a standalone issue — verdicts on the remaining 27 proposals

### Duplicates of existing issues (11) — enrich the existing issue instead

| Proposal | Already covered by |
|---|---|
| N13-04 Learning curves / data-sufficiency | **#93** Learning-curve utility |
| N14-05 MuData adapter | **#273** scverse/AnnData `[omics]` adapters |
| N14-08 Cross-modal identity resolver | folded into **#483** (same concern) |
| N15-05 Performance budgets / benchmark matrix | perf A/B gate + benchmark suite already shipped |
| N20-08 Core/optional capability boundary | `pro-core-boundary.md` + ADR-0038 **D1** already formalize it |
| N20-10 Public-API typing + schema conformance | **#242** pyright burn-down + **#442** pyright ratchet |
| N2X-01 XAI scientific evaluation benchmark | **#55** XAI evaluation framework |
| N2X-02 Explanation stability & uncertainty | **#53** Uncertainty-aware XAI |
| N2X-03 Ontology-grounded concepts | **#48** Concept-based explanations |
| N2X-04 Counterfactual validity | **#49** Example-based (prototypes + counterfactuals) |
| N2X-08 Cell-state context adapter | **#273** `[omics]` adapters (v1.4) |

### Routed to ProtXplain (5) — border conflict, not AAanalysis-side

| Proposal | Why it's ProtXplain |
|---|---|
| N20-03 Exception hierarchy (`AAanalysisError`) | **D14** + `sharp-edges.md`: no custom exception base — hard rule |
| N13-08 Scientific-warning **code taxonomy** | **D14/D11/D20**: AAanalysis keeps *documented* warnings; the stable code→policy map is the adapter |
| N15-08 Workflow resource estimation | **D5**: "resource and runtime estimates" is capability-intelligence |
| N20-05 Static capability descriptors | **D15**: no public `capabilities()`/`describe()` accessor in AAanalysis |
| N2X-07 Evidence-strength / claim-level annotations | **D5/D16**: evidence intelligence is the ProtXplain moat |

### Folded into an existing epic (4) — comment there, no new issue

| Proposal | Fold into |
|---|---|
| N12-05 Propagate uncertainty into design outputs | the open half is #473 (AD) + #474 (risk–coverage) feeding #475/#476; overlaps #16/#354 |
| N14-01 Unified representation-adapter protocol | **#210** ecosystem-integration epic |
| N14-09 Omics–protein integration reference workflow | **#109** Use Cases (a notebook, not a feature) |
| N15-07 Backed AnnData / Zarr / Parquet I/O | **#33** export formats + **#273** AnnData I/O |

### v2.X architecture — now resolved (was "deferred 11")

The 2026-07-18 v2.X review filed the four additive items as own issues and rolled the
architecture-decision cluster into one epic:

- **Filed as own issues:** `N2X-06` → #495 (drift) · `N20-07` → #496 (lineage graph) ·
  `N2X-11` → #497 (experimental-feedback) · `N2X-05` → #498 (cross-modal alignment).
- **Rolled into epic #499** (ADR-first, plain-dict, no capability registry / typed
  envelope): `N20-01` TaskSpec/DataSpec · `N20-02` typed result objects (post-migration) ·
  `N20-04` harmonize namespaces · `N20-09` migration/deprecation tooling · `N20-06`
  extension protocols.
- **Still deferred, no ticket yet (2):** `N2X-09` perturbation-evidence adapter
  (speculative) · `N2X-10` extension conformance suite (follows the #499 extension-protocol
  child).

---

## Tally

| Verdict | Proposals | New issues |
|---|---:|---:|
| Filed as its own issue (#473–#498) | 26 | 26 |
| Rolled into the v2 architecture epic (#499) | 5 | 1 |
| Duplicate of an existing issue | 11 | — |
| Routed to ProtXplain (border/rule) | 5 | — |
| Folded into an existing epic | 4 | — |
| Still deferred, no ticket yet | 2 | — |
| **Total** | **53** | **27** |

The 27 new issues are **#473–#499** (milestones v1.2–v2.X). v1.1 stays at 0 open and
ships as planned; #486 (v1.3) is unblocked — #339 is already fixed (closed 2026-07-05) —
and #477 is the first v1.2 hardening.

**Border codification.** The recurring measurement-vs-action seam across the v1.2–v1.3
issues is now recorded once as **ADR-0038 § G / D20**, with the two rejected
re-proposals (abstaining predictor; scientific-warning code taxonomy) added to that
ADR's *Rejected alternatives*.
