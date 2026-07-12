# ADR-0057 — v2 API naming system and the future XAI `*Explainer` layer

Status: Accepted — 2026-07-12

## Context

AAanalysis assigns every public class one canonical abbreviation (the instance
variable and example-notebook filename stem), enforced by
`tests/unit/api_tests/test_class_abbreviation_registry.py` and documented in the
*Class abbreviations* section of `docs/source/index/docstring_guide.rst`.

Two naming clean-ups preceded this ADR and are now on `master`/working tree:

- The `aaanalysis.pipe` golden-pipeline facade alias moved `aap` → **`ap`**
  (`import aaanalysis.pipe as ap`).
- The preprocessor and tool-class abbreviations were regularised:
  `SequencePreprocessor` `sp`→`seqp`, `EmbeddingPreprocessor` `ep`→`embp`,
  `StructurePreprocessor` `stp`→`strp`, `AnnotationPreprocessor` `ap`→`annp`;
  `AAMut` `aamut`→`aam`, `SeqMut` `seqmut`→`seqm`, `SeqOpt` `seqopt`→`seqo`,
  `AAPred` `aapred`→`aap`; `CPPStructurePlot` `csp`→`cpps_plot`.

The Explainability (XAI) roadmap (issues #16, #47–#56, #60, #91, #93, #265,
#276) proposes a large expansion. Without a naming/architecture decision each
issue would invent its own class shape and abbreviation, and the abbreviation
space (especially two-letter forms such as `sm`) would collide. This ADR fixes
the **naming system**, the **XAI layer architecture**, and the **reserved
future abbreviations**, so downstream issues have a single target to build
against.

## Decision

### 1. Type-suffix naming system

Four class families encode their *kind* in the final letter of the
abbreviation; all other tool/engine classes use the prefix + concept-initial
rule already in the style guide.

| Suffix | Family | Members |
|---|---|---|
| `*p` | preprocessors | `seqp`, `embp`, `strp`, `annp` |
| `*f` | feature containers | `sf`, `nf` |
| `*m` | models (fit / produce a model or estimate) | `tm`, `rm`, `sm` |
| `*x` | XAI explainers (post-hoc) | `fx`, `cx`, `ex`, `rx`, `nx`, `sx` |
| `*_plot` | plot companions | `cpp_plot`, `aap_plot`, `seqo_plot`, `fx_plot`, … |

`aa` = package alias; `ap` = `aaanalysis.pipe` module alias (reserved, never a
class).

### 2. XAI layer architecture: one `*Explainer` per method category

The XAI taxonomy has one class **per method category**, not per method. Each
class is a thin wrapper exposing **one method per external approach** (e.g.
`FeatureExplainer` exposes `pdp`, `ale`, `lime`, `pfi`), projecting results into
CPP feature space where possible. Post-hoc explainers (take an already-fitted
model + data → produce an explanation) take the `*Explainer`/`*x` form. The
complementary trust layers that *produce* a model or estimate keep the `*Model`
form.

Shipped `*Model` wrappers stay unchanged: `TreeModel` (`tm`, predictor +
importance), `ReliabilityModel` (`rm`, uncertainty/conformal), `ShapModel`
(`sm`, kept as the legacy SHAP wrapper). `TreeModel` is **not** renamed — it is a
predictor that is *explained*, not a post-hoc explainer.

Planned XAI `*Explainer` classes (reserved abbreviations):

| Class | Abbr | Home subpackage | Scope / new deps | Wraps (method per approach) | Issue |
|---|---|---|---|---|---|
| `FeatureExplainer` | `fx` | `explainable_ai_pro` | pro (sklearn-inspection, `pyALE`, `lime`) | `pfi`, `pdp`, `ice`, `ale`, `lime` (SHAP stays in `ShapModel`) | #47 |
| `ExampleExplainer` | `ex` | `explainable_ai` (+pro) | core prototypes (AAclust medoids); pro counterfactuals (`dice-ml`/`alibi`) | `prototypes` (MMD-CRITIC), `counterfactuals` (Wachter, DiCE, CEM) | #49 |
| `RuleExplainer` | `rx` | `explainable_ai_pro` | pro (`imodels`/`alibi`) | `anchor`, `rulefit`, `lore`, `trepan` | #50 |
| `NeuralExplainer` | `nx` | `explainable_ai_pro` | pro (`captum`, DL stack) | `integrated_gradients`, `lrp`, `grad_cam`, `tcav` (concept, #48), `gnn` | #51, #48 |
| `SurrogateExplainer` | `sx` | `explainable_ai` (+pro) | core tree/linear; pro symbolic (`gplearn`/`pysr`, #265) | `tree`, `linear`, `symbolic` | #52, #265 |
| `CausalExplainer` | `cx` | `explainable_ai_pro` | pro (`dowhy`, `econml`) | `dowhy`, `econml` | #54 |

Each gets a `*_plot` companion (`fx_plot`, `ex_plot`, …), covering the
visualization layer (#56). **XAI evaluation (#55)** is *not* a class — its
in-core slice is `aaanalysis/metrics/` functions (`comp_fidelity`,
`comp_stability`). **Uncertainty (#16) and uncertainty-aware XAI (#53) overlap
and must be deduped into `ReliabilityModel`** before either is scheduled.

### 3. Future non-XAI classes

| Class | Abbr | Home subpackage | Scope | Purpose | Issue |
|---|---|---|---|---|---|
| `ModelTrainer` | `mt` | `prediction` | core | Paper-fidelity nested-CV Monte-Carlo + ensemble engine behind `predict_samples` | #276 |
| `LearningCurve` (+Plot) | `lc` / `lc_plot` | `prediction` | core | "Is this task sampling-limited?" subsample-vs-metric curve | #93 |
| `ModelComparison` | `mc` | `prediction` | core | Repeated-CV + bootstrap CIs + paired ΔMCC (may instead be `metrics` funcs) | #91 |
| `ActiveLearner` | `al` | `protein_engineering` | core | Uncertainty/diversity mutation-candidate selection | #60 |
| `MSAGenerator` | `msa` | `data_handling_pro` | pro (biopython) | Homolog search + alignment → true MSA | #65 |
| `FunctionalAligner` | `fa` | `seq_analysis` | core | CPP-distance ("functional") alignment (may be a `comp_*` func) | #46 |

Likely **not** classes (methods/modes on existing classes): temporal CPP (#44)
and structural CPP (#365) → `CPP` modes; conservation (#64) / residue-pair (#89)
→ `SequenceFeature` methods; #335/#397/#398 → `AAPred`; #391/#392/#393 →
`AAPredPlot`.

### 4. PascalCase pass (implemented 2026-07-12)

`AAlogo` → `AALogo`, `AAlogoPlot` → `AALogoPlot` (abbreviations `aal` /
`aal_plot` unchanged). Because 1.0.3 shipped the lowercase names, the old
`AAlogo` / `AAlogoPlot` remain importable as **deprecated back-compatible
aliases**, resolved lazily via a module-level `__getattr__` in both
`aaanalysis/__init__.py` and `aaanalysis/seq_analysis/__init__.py` (emitting a
`DeprecationWarning`), so they stay out of `__all__` and the abbreviation
registry. Module and test filenames keep their lowercase `aalogo` stem
(module names are lowercase regardless of class casing).

### 5. Dependencies and extras

New heavy deps (`captum`, `dowhy`, `econml`, `dice-ml`/`alibi`, `imodels`,
`pyALE`, `lime`, `gplearn`/`pysr`) are pro-only and gated behind the `pro`
extra via the `missing_feature_stub` mechanism; a genuinely new install extra
(e.g. a dedicated `xai` extra) requires user approval per the CONFIRM-FIRST
list. Prefer reusing existing deps for any core slice (AAclust, sklearn) and
routing the heavy backends to `pro`.

## Consequences

- The style guide (`docstring_guide.rst`) carries the type-suffix map and the
  reserved `*Explainer` table; the reserved abbreviations are locked so no
  future PR reuses `fx`/`cx`/`ex`/`rx`/`nx`/`sx`.
- When a future class lands it must: register in `REGISTRY` **and** the guide
  table, add its example notebook `<abbr>_<method>.ipynb`, and (for pro) gate in
  `__init__.py` with `missing_feature_stub`. A new dedicated backend subpackage
  extends `DEDICATED_OWNERS` in `test_backend_import_hygiene.py`.
- An optional safeguard is a `RESERVED_V2_SHORTCUTS` set checked when a new
  class registers, rejecting an out-of-scheme abbreviation in review.
- Each XAI issue (#47–#56) is now a *method-adding* task against a known target
  class rather than an open-ended design; the "resolve scope before coding"
  banners on those issues resolve to "add method(s) to the reserved class, core
  slice in `explainable_ai`, heavy backends in `explainable_ai_pro`."
