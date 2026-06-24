# GitHub issues — implementation handoff

> **Refresh — 2026-06-24.** The **agentic-readiness program is well underway** (PRs #220–#249). Since the
> 2026-06-23 snapshot: **#133 closed** (PR #246 — ADR-0039, one uniform `(fig, ax)` plot return),
> **#134 closed** (PR #220 — Wrapper/Tool ABCs), **#132 closed** (PR #243 — resolved, *no rename*), the
> **`aaanalysis.pipe` API started** (`aap.predict` shipped, PR #244; ADR-0038/0040), and the
> **pyright burn-down** is live (PR #245 merged, **#249 open**). New issues filed: **#241** (golden-pipelines
> `aaanalysis.pipe`), **#242** (pyright burn-down). **Open: 57** (closed #133/#134; +new #241/#242). Plan:
> **`~/.claude/plans/foamy-churning-shore.md`** — see **PR ↔ Issue activity** + **Optimal order** below.
>
> _Regenerate: `python3 .claude/skills/github-issues/scripts/fetch_issues.py`, then re-run the skill._

Audits every open issue against package scope (interpretable, CPP-centered, sequence-based protein
prediction; `pro` extra for heavy deps; semver-strict v1) and the standards in `CLAUDE.md` +
`.claude/rules/sharp-edges.md`. Verdicts: ✅ Ready · 🔄 Revisit (decision/under-spec/oversized) ·
⏸️ Defer-v2 · ❌ Reject (rule conflict) · ☑️ Done/Partial.

## Snapshot (56 open)
- **Just merged — the agentic-readiness wave** (2026-06-21→23): **#231** front-doors → **#69 closed**;
  **#232** type track (ADR-0035/0036 + `py.typed` + advisory pyright); **#233** error enrichment;
  **#234** `df_feat` data dictionary (→ **#26 partial**); **#235** smoke gate; **#236** utils.py barrel
  split (`_constants.py`); **#237** pyright slice; **#238** rich `ut.DICT_DF_SCHEMAS` (all key frames, →
  **#26 more**); **#239** honest required-arg signatures (+ `plot_get_clist` kind-driven + method-spacing
  test). The 2026-06-14 lanes also landed: **#127/#214** (PR #222 → `get_seq_kws`), **#211/#24**
  (PR #223, sklearn `Pipeline` proof), **#130/#88/#77** closed.
- **Headline next** (per `foamy-churning-shore.md`): most of the agentic-readiness spine is **done** —
  ADR-0038/0040 ✅, #134 ✅, #133 ✅, #132 ✅ (spine complete). **Remaining = finish `#241` `aaanalysis.pipe`**
  (the `cpp_feature_map`/`explain` pipelines + the sklearn `SequenceFeatureTransformer`) and the
  **`#242` pyright burn-down** tail (#249 open). See Track 1.
- **Scope rule (this session):** *agentic readiness* (legible/typed/contracted/improvable primitives) is
  the active program; **science/product is a SEPARATE track** — structure-XAI (#119/#120), the **v2.X XAI
  suite (#47–#55, + #44)**, design (#57/#59/#60), uncertainty (#16, core), viz layer (#56, v1.4), ShapModel
  estimator (#229, v1.1) are real but **not** agentic readiness.
  The boundary: **MCP / machine-readable tool contract → ProtXplain; usability + improvability →
  AAanalysis.**
- **New since last refresh:** **#226/#227** (Usage-Principles data-flow infogram — #227 artwork *blocks*
  #226 page swap; both Lane E docs, dcos/prio:3, unmilestoned) · **#229** (ShapModel unbiased
  probability-interpolation estimator for fuzzy labels — v1.1, XAI/prio:3, science track).
- **Agentic-spine status:** **#134 ✅ closed** (PR #220) · **#133 ✅ closed** (PR #246, ADR-0039 — *full*
  `(fig, ax)` migration, beyond the plan's ADR-only intent) · **#132 ✅ closed** (PR #243 — resolved by
  documentation, *no rename*; the 4 distinct label concepts documented instead) · **#241** `aaanalysis.pipe` in
  progress (`aap.predict` shipped PR #244; remaining: `cpp_feature_map`/`explain`/`SequenceFeatureTransformer`)
  · **#242** pyright burn-down in progress (PR #245 merged, **#249 open**) · **#26** partial (df_feat +
  family contracts via #234/#238; remaining = per-residue-score shape + a stable ProtXplain anchor).

## Issue ↔ PR activity
> **Issues are the spine** (left); their PR(s) are on the right — **—** when an issue has no PR yet.
> `✅` = a PR closed the issue · `↗` = PR(s) advance it (no closing keyword, or still in progress).

**Issues with PR activity** (newest first)
| issue | state | PR(s) | what landed / in flight |
|---|---|---|---|
| #133 | ✅ closed | #246 | one uniform `(fig, ax)` `*Plot` return + ADR-0039 |
| #134 | ✅ closed | #220 | inherit Wrapper/Tool ABCs + meta-test |
| #132 | ✅ closed | #243 | label names documented; resolved as *no rename* (distinct concepts) |
| #241 | 🔄 in progress | #244 (`aap.predict`), #247/#250 (ADRs) ↗ | `aaanalysis.pipe` golden pipelines |
| #242 | 🔄 in progress | #245, #249 ↗ | pyright burn-down (ongoing) |
| #26 | 🔄 partial | #234, #238 ↗ | df_feat + family data dictionary |
| #18 | ✅ closed | #234 | `df_feat` output schema |
| #69 | ✅ closed | #231 | subpackage front-doors |
| #77 | ✅ closed | #230 | check_cat audit |
| #88 | ✅ closed | #224 | subprocess/network/file hardening |
| #211 | ✅ closed | #223 | sklearn `Pipeline` compat proof |
| #24 | ✅ closed | #223 | sklearn wrapper (superseded by #241) |
| #127 | ✅ closed | #222 | `get_seq_kws` (`df_seq=`/`sample=`) |
| #214 | ✅ closed | #222 | SHAP-plot `sample=`/`df_seq=` sugar |
| #130 | ✅ closed | #221 | `single_logo` from `df_parts` |

**Issues with no PR yet ( — ):** every other open issue — #16, #22, #23, #25, #27, #28, #29, #33, #35,
#36, #37, #40, #42, #44–#56, #57, #59, #60, #62, #64, #65, #75, #76, #79, #80, #87, #89, #91, #93,
#106, #107, #108, #109, #119, #120, #126, #131, #210, #219, #226, #227, #229. See the per-issue audit.

**Program / tooling PRs with no backing issue** (the agentic-readiness infra): #232 (`py.typed`+pyright),
#233 (errors), #235 (smoke), #236 (utils split), #237 (pyright slice), #238 (`DICT_DF_SCHEMAS`), #239
(honest signatures), #240 (ADR-0038), #247 (ADR-0040), #248 (tooling), #250 open (ADR-0041, advances #241).

---

## ▶ Optimal order

### Track 1 — Agentic-readiness program (ONE driver session; mostly serialized; per `foamy-churning-shore.md`)
> Each item = its own worktree+branch+PR. Several touch CONFIRM-FIRST surfaces — ask before those.

1. ✅ **Decision layer** — **ADR-0038** (boundary refines ADR-0035) + **ADR-0040** (golden pipelines) +
   `CONTEXT.md` term shipped (PRs #240/#247).
2. ✅ **Filed 2 issues:** **#241** golden-pipelines `aaanalysis.pipe` (supersedes closed #24, child #126),
   **#242** pyright burn-down.
3. ✅ **#134** Wrapper/Tool — **closed** (PR #220). `SeqMut` deliberately not a `Tool`; documented in the meta-test.
4. ✅ **#132** — **closed** (PR #243 + closed as resolved): *no rename* (SHAP target-class ≠ positive marker;
   `list_labels` 2D ≠ `labels` 1D); the 4 label concepts documented instead.
5. 🔄 **#241 `aaanalysis.pipe`** — `aap.predict` shipped (PR #244). **Remaining:** `cpp_feature_map`
   (args `subcategories` / `dpulearn=True/False` / graded `optimization`), `explain` (pro), and
   `SequenceFeatureTransformer` (BaseEstimator+TransformerMixin, leak-free). **Largest; CONFIRM-FIRST**
   (top-level `aa.pipe` exposure + `__all__`).
6. ✅ **#133** — **closed** (PR #246, **ADR-0039**): one uniform `(fig, ax)` `*Plot` return — the *full*
   breaking migration was taken now (beyond the plan's ADR-only intent).
7. 🔄 **#242 pyright burn-down** — PR #245 merged (plotting + budget ratchet); **#249 open** (validators +
   metrics). Per-subpackage, ongoing, advisory/non-blocking.

**Track-1 remaining = just (5) the rest of `aaanalysis.pipe` + (7) the pyright burn-down tail.**

### Track 2 — Docs onramp (Lane E; parallel-safe, doc files only; ONE doc owner)
**#106** epic → **#107** (nav + landing routing + **API stable/experimental** — the agent-relevant slice) ·
**#108** · **#109** (use_cases/) · **#227** (infogram artwork) → **#226** (page swap) · **#80** gallery ·
**#35** protocols tail. Coordinate with Track 1 where the pipe API changes the cheat sheet / tutorials.

### Track 3 — Science / product (separate; parallel lanes; NOT agentic readiness)
- **Lane C — Protein design (serialize):** **#57 → #59 → #60** (#37 gate landed; built on AAMut/SeqMut+ΔCPP).
- **Lane G — Evaluation:** **#91 → #93** (reuse `aaanalysis.metrics` + `comp_bootstrap_ci`).
- **Lane I — Pro structure plots (serialize, CONFIRM-FIRST new `feature_engineering_pro/`):** **#119 → #120.**
- **Lane A — Schema consumers:** **#29 → #33 → #26-finish** (build on frozen `LIST_COLS_FEAT`/`sort_cols_feat`).
- **Lane D — Data/sampling:** **#25, #28** (on `AAWindowSampler`).
- **Lane F — Structure/conservation (pro, serialize):** **#65 → #40/#42**; **#64 deferred (ADR-0012).**
- **Lane B — Perf:** **#62** (GPU/parallel; opt-in, CPU default — the only open perf issue).
- **XAI long-tail:** **#229** (v1.1, ShapModel estimator) · **#56** (v1.4, viz layer) · **#16/#53**
  (uncertainty — merge intent) · the **v2.X XAI suite #47–#55** (mostly ProtXplain-scoped; #55 eval-metrics
  slice stays in-core).
- **Plot polish:** **#219** (dense-plot row-label overlap gate — `_cpp_plot.py`; serialize with any pipe-API
  edits to the same file), **#131** (session-persistent plot style — CONFIRM-FIRST `config.py`), **#75**
  (route output through logging).

---

## Overlap clusters — DO NOT develop in parallel
| Cluster | Issues | Primary | Why they collide |
|---|---|---|---|
| **`_cpp_plot.py` surface** | pipe `cpp_feature_map`, #219 | pipe API | `#133` migration ✅ landed (PR #246) — `*Plot` now returns `(fig, ax)`; the pending `cpp_feature_map` (wraps `CPPPlot.feature_map`) + #219 (edits the same methods) build on it → one owner |
| **API-consistency spine** | ✅ complete | — | #134 ✅ #133 ✅ #132 ✅ all closed — spine done |
| **`aaanalysis.pipe` ↔ #126/#24** | pipe issue, #126, #24(closed) | pipe issue | pipe API is the #126-ergonomics realization; supersedes the closed #24 wrapper |
| Output schema | #29, #33, #26 | #29 | serialize behind frozen `LIST_COLS_FEAT`/`sort_cols_feat` |
| Protein design | #57 → #59 → #60 | #57 | shared mutation API in `protein_design/` |
| Pro structure plots | #119, #120 | #119 | #120 blocked-by #119; share `feature_engineering_pro/` + StructureView (ADR-0028) |
| Structure/conservation/MSA | #40, #64, #65, #42 | #65 | shared `data_handling_pro` + pending move |
| Embedding | #22, #23, #47 | #22 | shared embedding surface (`[embed]` input path #145/ADR-0029 landed) |
| Docs infogram | #226, #227 | #227 | #226 page swap blocked-by #227 artwork |
| Uncertainty XAI | #16, #53 | #16 | overlapping bootstrap/variance intent |

---

## Per-issue audit

### Recently closed (resolved on master — context only)
- **Agentic-readiness wave (2026-06-21→23):** **#69** (front-doors, #231) · **#26 partial** (#234 df_feat
  dictionary + #238 `DICT_DF_SCHEMAS`) · type track #232 (ADR-0035/0036, `py.typed`, advisory pyright) ·
  error enrichment #233 · smoke #235 · utils split #236 · pyright slice #237 · honest signatures #239.
- **2026-06-14 lanes:** **#127/#214** (PR #222 → `get_seq_kws`) · **#211/#24** (PR #223, sklearn Pipeline
  proof) · **#130** (single_logo from df_parts) · **#88** (security hardening) · **#77** (check_cat audit).
- **Earlier:** #74 (`deprecated()`+CHANGELOG) · #135 (docstring contracts) · #157/#158/#129 (dPULearn/AAclust/
  ShapModel sugar) · perf sweep #19/#180/#186/#187/#188(ADR-0032)/#199/#200/#201 · #37 (AAMut/SeqMut — merged,
  **kept OPEN for review**; gate for #57–60 landed).

### API-consistency spine + ergonomics (#126 epic) — `aaanalysis/`
| # | prio | verdict | note (complements the issue) |
|---|---|---|---|
| 126 | 2 | 🔄 epic | Don't "implement #126" — drive children. The **`aaanalysis.pipe` API is its convenience realization**; #131/#132/#133/#134 are its consistency children. |
| 134 | 2 | ✅ **closed** (PR #220) | Wrapper/Tool ABCs bound + meta-tests. `SeqMut` deliberately NOT a `Tool` (has `mutate`/`scan`/`suggest`/`eval`, no `run`); `AAWindowSampler` excluded — both documented in the meta-test. |
| 132 | 2 | ✅ **closed** (PR #243) | **No rename.** SHAP's `label_target_class` is a *general* target-class selector (any class), NOT the positive/test marker; `list_labels` (2D multi-dataset) is genuinely distinct from `labels` (1D). The 4 label concepts documented instead (docstring_guide + CONTEXT.md). Closed as resolved. |
| 133 | 3 | ✅ **closed** (PR #246) | One uniform `(fig, ax)` `*Plot` return shipped (**ADR-0039** + `test_plot_return_contract.py`) — the full breaking migration, beyond the plan's ADR-only intent. |
| 131 | 2 | 🔄 (CONFIRM-FIRST `config.py`) | Session-persistent plot style; unset → byte-identical default. Plot-polish lane. |
| 219 | 2 | ✅ | Dense-plot row-label overlap gate + shrink-to-floor-then-grow on `feature_map`/`heatmap`/`ranking`/`profile`. **Serialize with the pipe `cpp_feature_map` + #133** (same `_cpp_plot.py`). |
| 241 | 2 | 🔄 in progress (PR #244) | **`aaanalysis.pipe` golden-pipelines** + `SequenceFeatureTransformer`. `aap.predict` shipped (PR #244, ADR-0040). Remaining slices: `cpp_feature_map` / `explain` / the transformer. Supersedes closed #24. |
| 242 | 3 | 🔄 in progress (PR #245✓, #249 open) | **pyright burn-down** (baseline ~1203). PR #245 cleared plotting + added a budget ratchet; **#249 open** (core validators + metrics). Advisory/non-blocking, per-subpackage. |

### Docs (Lane E)
| # | prio | verdict | note |
|---|---|---|---|
| 106 | 1 | 🔄 epic | Docs-architecture parent; split into #107/#108/#109; coordinate #35/#80, one owner. |
| 107 | 2 | 🔄 | Nav + landing routing + **API stable/experimental** split (the agent-relevant slice). Cheapest/highest-leverage child. |
| 108 | 3 | ✅ | 'You will learn' box + wire comparison harness. Low-risk additive. |
| 109 | 2 | 🔄 | End-to-end biological `use_cases/`; confirm boundary vs protocols (#35)/tutorials. |
| 226 | 3 | 🔄 | Swap data-flow infogram into `usage_principles.rst`, drop mini-ecosystem fig. **Blocked-by #227.** |
| 227 | 3 | 🔄 | **Design the data-flow infogram** (df_seq→df_parts→df_feat→model→explain) — could mirror the `aaanalysis.pipe` flow. Blocks #226. |
| 80 | 2 | ✅ | nbsphinx thumbnail gallery; any new docs dep = pyproject CONFIRM-FIRST. |
| 35 | 2 | ✅ | Protocols epic (10/10 shipped); tail = standalone *Scale selection (AAclust)* protocol. |

### Protein design (Lane C, serialize) · Evaluation (Lane G)
| # | prio | verdict | note |
|---|---|---|---|
| 57 | 1 | 🔄 | Top-k residue selection from feature importance; **unblocked (#37)**; first in Lane C. |
| 59 | 1 | 🔄 | Weighted/Pareto mutation ranking; after #57; keep interpretable. |
| 60 | 2 | 🔄 | Uncertainty/diversity candidate selection; after #59. |
| 16 | 1 | 🔄 | Bootstrap/variance over CPP features; coordinate with #53 (uncertainty XAI). |
| 91 | 1 | 🔄 | Model eval & comparison (repeated-CV + bootstrap CIs + paired ΔMCC); helpers vs Tool-class+Plot — settle first. #93 builds on it. |
| 93 | 3 | 🔄 | Learning-curve "sampling-limited?" utility; on #91. |

### Pro structure plots (Lane I) · Schema / data
| # | prio | verdict | note |
|---|---|---|---|
| 119 | 1 | 🔄 pro | `CPPStructurePlot.map_structure` → StructureView; reuse `get_positions_`/`get_df_pos`/`load_structure`/`fetch_alphafold`; new `feature_engineering_pro/` (CONFIRM-FIRST). Blocks #120. |
| 120 | 2 | 🔄 pro | `CPPStructurePlot.interactive` (ipywidgets); after #119. |
| 29 | 2 | ✅ | z-score/min-max normalization appended via `sort_cols_feat` tail; serialize ahead of #33/#26. |
| 33 | 2 | ✅ | CSV/parquet/metadata export of `df_feat`; canonical `LIST_COLS_FEAT` order. |
| 26 | 3 | ☑️ partial | df_feat + family contracts documented & tested (#234/#238). Remaining: per-residue-score shape + a stable importable anchor for ProtXplain. |
| 28 | 2 | 🔄 | Sliding-window CPP on `AAWindowSampler`; overlaps #27. |
| 27 | 3 | 🔄 | Custom-region abstraction; coordinate with output schema. |
| 25 | 3 | 🔄 | Curate benchmark dataset (`_data/` CONFIRM-FIRST). |
| 76 | 3 | 🔄 | Explicit entry removal + opt-out flag (today silent). |
| 22 | 2 | 🔄 partial | Embedding input path landed (`[embed]`); gap = per-sequence pooled-vector fusion with CPP. |
| 23 | 3 | 🔄 | CPP↔embedding correlation/decorrelation; pairs with #22. |
| 210 | 3 | 🔄 epic | Ecosystem integration — drive children (sklearn proof shipped via #211/#24); no heavy core deps. |

### Structure/conservation · Perf · XAI long-tail · misc
| # | prio | verdict | note |
|---|---|---|---|
| 65 | 3 | 🔄 pro | `get_msa` misnamed; add real homolog+align backends (ADR-0017). |
| 40 | 2 | 🔄 pro partial | DSSP→scales slice; coordinate pending `data_handling_pro` move. |
| 42 | 3 | 🔄 | PTM/structure scales touch `_data/` (CONFIRM-FIRST); split. |
| 64 | 3 | ⏸️ | Conservation feature **deferred by ADR-0012**; blocked on #65 + pro move. |
| 62 | 3 | 🔄 | GPU/parallel; opt-in, CPU default; only open perf issue; v1.5. |
| 229 | 3 | 🔄 **v1.1** | ShapModel unbiased probability-interpolation estimator (replace threshold-sweep); `_backend/shap_model/`. **Highest-priority XAI item — NOT in the v2.X tail.** |
| 56 | 2 | 🔄 **v1.4** | Visualization layer (multi-scale explanation): split static parts in-core; interactive (dash/NGLview/PyMOL) → ProtXplain. **NOT in the v2.X tail.** |
| 47 | 3 | 🔄 v2.X pro | *Feature attribution* → CPP scale space; on #63 shap bars; coord #22. |
| 48 | 3 | 🔄 v2.X | *Concept-based*: cluster CPP scales → concepts (AAclust). |
| 55 | 3 | 🔄 v2.X | *XAI evaluation*: fidelity/stability **metrics slice → `aaanalysis/metrics/`** (in-core); cross-method DL-XAI bench → ProtXplain. |
| 49,50,51,52,53,54 | 3 | 🔄 v2.X → ProtXplain | *example-based / rule / neural / surrogate / uncertainty-aware / causal* XAI — heavy deps (FAISS/alibi/dice/imodels/captum/PyMC/DoWhy); keep out of core (`pro-core-boundary.md`). #53 merges intent with #16. |
| 44 | 3 | 🔄 v2.X | Temporal CPP (ΔCPP/Δt); needs a time-point data contract. *(topic:core, not XAI.)* |
| 45 | 3 | 🔄 v1.3 | **Core, not XAI:** Motif integration into CPP; FIMO p/q overlay → `df_feat` (ADR-0021). |
| 46 | 3 | 🔄 v1.5 | **Core, not XAI:** Functional alignment; CPP-distance in-scope, rest → ProtXplain. |
| 36 | 3 | 🔄 | Scale-selection method needs a concrete spec first. |
| 87 | 3 | 🔄 | Named CPP strategy preset; doc-first, sugar only if it earns surface. |
| 89 | 3 | 🔄 | First-class between-residues/bond-centered features; on AAWindowSampler. |
| 79 | 3 | 🔄 | PSSM = `(L,20)`→`dict_num` glue into `run_num`, not a new pipeline. |
| 75 | 3 | ✅ | Route output through logging, keep `print_out` shim. |

## Defer / Reject appendix (with cited rule)
- **#64 ⏸️** — deferred by **ADR-0012**; blocked on #65 + the pro move.
- **integration/e2e tiers** — shipped (#173); the old v2-defer note is obsolete.
- **v2.X XAI suite (#47–#55) / #56-interactive / #62 🔄→scope** — not rule-violations, but balloon the
  dependency surface or belong to **ProtXplain** (`pro-core-boundary.md`); keep the boundary. (#55's
  fidelity/stability *metrics* slice stays in-core under `aaanalysis/metrics/`.)
- **#119 / #120 — pro-only**: py3Dmol/ipywidgets in the `pro` extra (CONFIRM-FIRST), never core.
- No issue currently triggers a hard-rule **Reject** (no `AAanalysisError`/SECURITY.md/ruff/mypy asks).
