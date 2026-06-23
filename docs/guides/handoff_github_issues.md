# GitHub issues — implementation handoff

> **Refresh — 2026-06-23.** The **agentic-readiness wave merged** (#231–#239 + the 2026-06-14 lane
> PRs #220–#224). Closed since last refresh (8): **#24, #69, #77, #88, #127, #130, #211, #214**. New (3):
> **#226/#227** (data-flow infogram docs), **#229** (ShapModel fuzzy-label estimator). **Open 60 → 57.**
> The next program is set by the approved plan **`~/.claude/plans/foamy-churning-shore.md`** (the
> `aaanalysis.pipe` convenience API + the API-consistency spine) — see **Optimal order** below.
>
> _Generated/refreshed by the `github-issues` skill. Regenerate:_
> _`python3 .claude/skills/github-issues/scripts/fetch_issues.py`, then re-run the skill._

Audits every open issue against package scope (interpretable, CPP-centered, sequence-based protein
prediction; `pro` extra for heavy deps; semver-strict v1) and the standards in `CLAUDE.md` +
`.claude/rules/sharp-edges.md`. Verdicts: ✅ Ready · 🔄 Revisit (decision/under-spec/oversized) ·
⏸️ Defer-v2 · ❌ Reject (rule conflict) · ☑️ Done/Partial.

## Snapshot (57 open)
- **Just merged — the agentic-readiness wave** (2026-06-21→23): **#231** front-doors → **#69 closed**;
  **#232** type track (ADR-0035/0036 + `py.typed` + advisory pyright); **#233** error enrichment;
  **#234** `df_feat` data dictionary (→ **#26 partial**); **#235** smoke gate; **#236** utils.py barrel
  split (`_constants.py`); **#237** pyright slice; **#238** rich `ut.DICT_DF_SCHEMAS` (all key frames, →
  **#26 more**); **#239** honest required-arg signatures (+ `plot_get_clist` kind-driven + method-spacing
  test). The 2026-06-14 lanes also landed: **#127/#214** (PR #222 → `get_seq_kws`), **#211/#24**
  (PR #223, sklearn `Pipeline` proof), **#130/#88/#77** closed.
- **Headline next = the approved plan** (`foamy-churning-shore.md`): a **second convenience API
  `import aaanalysis.pipe as aap`** (stateless golden pipelines + a sklearn-compliant
  `SequenceFeatureTransformer`) + the **API-consistency spine** (#134 → #132 → #133-ADR) + **2 new issues
  to file** (golden-pipelines; pyright burn-down) + an **ADR-0035 boundary clarification**.
- **Scope rule (this session):** *agentic readiness* (legible/typed/contracted/improvable primitives) is
  the active program; **science/product is a SEPARATE track** — structure-XAI (#119/#120), XAI families
  (#47–#56), design (#16/#57/#59/#60), ShapModel estimator (#229) are real but **not** agentic readiness.
  The boundary: **MCP / machine-readable tool contract → ProtXplain; usability + improvability →
  AAanalysis.**
- **New since last refresh:** **#226/#227** (Usage-Principles data-flow infogram — #227 artwork *blocks*
  #226 page swap; both Lane E docs, dcos/prio:3, unmilestoned) · **#229** (ShapModel unbiased
  probability-interpolation estimator for fuzzy labels — v1.1, XAI/prio:3, science track).
- **Still open & partial:** **#134** (Wrapper/Tool ABCs bound on TreeModel/ShapModel/dPULearn/AAMut +
  meta-test; the `SeqMut.run` decision remains) · **#26** (df_feat + family contracts now documented &
  tested via #234/#238; remaining = per-residue-score shape + a stable importable anchor for ProtXplain).
- `deprecated()` machinery exists (`_utils/decorators.py` via `ut`, from #74) — the breaking children
  **#132/#133** wire to it directly, no machinery to build.

---

## ▶ Optimal order

### Track 1 — Agentic-readiness program (ONE driver session; mostly serialized; per `foamy-churning-shore.md`)
> Each item = its own worktree+branch+PR. Several touch CONFIRM-FIRST surfaces — ask before those.

1. **Decision layer (docs PR, low conflict — do first):** clarify **ADR-0035** ("Boundary refined"
   section — don't renumber a merged ADR) + add a `CONTEXT.md` glossary term "Agentic readiness".
2. **File 2 issues:** (a) **golden-pipelines `aaanalysis.pipe`** (supersedes closed **#24**, child of
   **#126**, relates #35/#210/#25); (b) **pyright burn-down** (currently untracked — the #232 follow-up).
3. **#134 finish** (Wrapper/Tool) — settle the `SeqMut.run` decision; ensure all model/tool classes bound;
   keep the `isinstance` meta-test green. *Mechanical, non-breaking — start here.*
4. **#132** label/marker unification — canonical `label_test`/`label_ref` + `labels`; old names
   (`label_target_class`, `list_labels`) as **deprecated aliases** (wire `ut.deprecated`). Non-breaking.
5. **`aaanalysis.pipe` (the new issue)** — 3 golden pipelines `cpp_feature_map` (args: `subcategories`,
   `dpulearn=True/False`, graded `optimization`) / `predict` / `explain` + `SequenceFeatureTransformer`
   (BaseEstimator+TransformerMixin, leak-free). **Largest; CONFIRM-FIRST** (new namespace + `__all__`).
6. **#133** plot-return-contract **ADR only** (decide `(fig, ax)` vs `ax`+`.figure`); breaking migration
   deferred to a versioned change.
7. **pyright burn-down** — per-subpackage, ongoing, advisory/non-blocking.

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
- **XAI long-tail:** **#229** (ShapModel estimator), **#55** (metrics slice in-core; rest → ProtXplain),
  **#16/#53** (uncertainty — merge intent), **#47–#54/#56** (ProtXplain-scoped).
- **Plot polish:** **#219** (dense-plot row-label overlap gate — `_cpp_plot.py`; serialize with any pipe-API
  edits to the same file), **#131** (session-persistent plot style — CONFIRM-FIRST `config.py`), **#75**
  (route output through logging).

---

## Overlap clusters — DO NOT develop in parallel
| Cluster | Issues | Primary | Why they collide |
|---|---|---|---|
| **`_cpp_plot.py` surface** | pipe `cpp_feature_map`, #219, #133 | pipe API | golden `cpp_feature_map` wraps `CPPPlot.feature_map`; #219 edits the same methods; #133 changes their return type → one owner |
| **API-consistency spine** | #134, #132, #133 | #134 | template/ABC + param-name + return-contract all touch the model/plot class signatures → serialize |
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
| 134 | 2 | ☑️ partial (OPEN) | ABCs bound (TreeModel/ShapModel/dPULearn→`Wrapper`, AAMut→`Tool`) + meta-test. **Open:** does `SeqMut` get a `.run` → inherit `Tool`? Settle in a review session. *Track 1, step 3.* |
| 132 | 2 | 🔄 (decision) | Canonical `label_test`/`label_ref` + `labels`; `label_target_class`/`list_labels` → deprecated aliases via `ut.deprecated`. Non-breaking. *Track 1, step 4.* |
| 133 | 3 | 🔄 (ADR) | One `*Plot` return contract (`(fig,ax)` vs `ax`+`.figure`). **Breaking → ADR now, migrate later.** *Track 1, step 6.* |
| 131 | 2 | 🔄 (CONFIRM-FIRST `config.py`) | Session-persistent plot style; unset → byte-identical default. Plot-polish lane. |
| 219 | 2 | ✅ | Dense-plot row-label overlap gate + shrink-to-floor-then-grow on `feature_map`/`heatmap`/`ranking`/`profile`. **Serialize with the pipe `cpp_feature_map` + #133** (same `_cpp_plot.py`). |
| *(new)* | — | 🆕 | **`aaanalysis.pipe` golden-pipelines** — file the issue (Track 1, step 2a); spec in `foamy-churning-shore.md` Workstream D. Supersedes closed #24. |
| *(new)* | — | 🆕 | **pyright burn-down** — file the issue (Track 1, step 2b). |

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
| 229 | 3 | 🔄 | ShapModel unbiased probability-interpolation estimator (replace threshold-sweep). Science/XAI; touches `_backend/shap_model/`. |
| 55 | 3 | 🔄 | Fidelity/stability **metrics slice → `aaanalysis/metrics/`**; cross-method DL-XAI bench → ProtXplain. |
| 47 | 3 | 🔄 pro | Attributions → CPP scale space; on #63 shap bars; coord #22. |
| 48 | 3 | 🔄 | Cluster CPP scales → concepts (AAclust). |
| 49,50,51,52,54,56 | 3/2 | 🔄 → ProtXplain | New heavy deps (FAISS/alibi/dice/imodels/captum/DoWhy/dash) — keep out of core (`pro-core-boundary.md`). |
| 53 | 3 | 🔄 | Uncertainty XAI (PyMC) → ProtXplain; merge intent with #16. |
| 45 | 3 | 🔄 | Motif hits → df_feat overlay; real FIMO p/q (ADR-0021). |
| 46 | 3 | 🔄 | Split: CPP-distance alignment in-scope; rest → ProtXplain. |
| 44 | 3 | 🔄 | Temporal CPP (ΔCPP/Δt); needs a time-point data contract; v2.X. |
| 36 | 3 | 🔄 | Scale-selection method needs a concrete spec first. |
| 87 | 3 | 🔄 | Named CPP strategy preset; doc-first, sugar only if it earns surface. |
| 89 | 3 | 🔄 | First-class between-residues/bond-centered features; on AAWindowSampler. |
| 79 | 3 | 🔄 | PSSM = `(L,20)`→`dict_num` glue into `run_num`, not a new pipeline. |
| 75 | 3 | ✅ | Route output through logging, keep `print_out` shim. |

## Defer / Reject appendix (with cited rule)
- **#64 ⏸️** — deferred by **ADR-0012**; blocked on #65 + the pro move.
- **integration/e2e tiers** — shipped (#173); the old v2-defer note is obsolete.
- **#49–#54 / #56 / #62 🔄→scope** — not rule-violations, but balloon the dependency surface or belong to
  **ProtXplain** (`pro-core-boundary.md`); keep the boundary.
- **#119 / #120 — pro-only**: py3Dmol/ipywidgets in the `pro` extra (CONFIRM-FIRST), never core.
- No issue currently triggers a hard-rule **Reject** (no `AAanalysisError`/SECURITY.md/ruff/mypy asks).
