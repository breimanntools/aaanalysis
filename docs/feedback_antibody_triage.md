# Triage: antibody non-specificity feedback → AAanalysis

Triage of `feedback_aaanalysis_antibody/FEEDBACK.md` — an external drop from a
project that reproduced *Sakhnini et al.* (antibody/nanobody non-specificity) and
tried to beat its PLM predictor with AAanalysis CPP.

**Headline finding.** The feedback was written against **stale GitHub `main`**
(editable install, aaanalysis 1.0.3 / commit `acab636`, 2026-06-06). Local
`master` is far ahead, so a large fraction of the P0/P1 asks **already shipped**
and the author never saw them. Two more items were **withdrawn or downgraded**
once examined against current code:

- **Prefix-sum `feature_matrix`** — withdrawn by the author. The "~21 s/feature,
  5 % CPU, 12–16 GB RSS" was measured on a stale/partial build; the 5 % duty +
  climbing RSS is env/memory-bandwidth starvation, not the summation. And a float
  `cumsum`-then-subtract reorders additions, so it cannot be bit-identical to
  per-slice `np.mean` → it would break the ADR-0015 regression anchor. **Dropped.**
- **Standalone `select_features(X, y, …)`** — the *capability already exists*
  (`comp_auc_adjusted` + `NumericalFeature.filter_correlation` +
  `TreeModel.select_features`); the real gap is a documented composition recipe
  plus one truncation warning. **Downgraded** from "build a function" to docs +
  warning + a Protocol.

---

## Verified implementation state (current `master`)

| Capability | State | Evidence |
|---|---|---|
| macOS-safe `n_jobs` / spawn | ✅ shipped | `_get_feature_matrix_fast.py`, `_filters_c/_get_feature_matrix_c.py::_pick_n_jobs_cython`, spawn-safe `_progress.py` |
| Feature-matrix Cython kernel + fallback notice | ✅ shipped | `_filters_c/_inner.pyx` (`compute_segment_mean`…); `_backend/cpp_run.py::_pick_feature_matrix_builder` emits a one-time `ut.print_out` fallback notice |
| Matrix caching | ✅ shipped | content-addressed `lru_cache` on scales + per-instance `AALookupCache` (`_get_feature_matrix_fast.py`) |
| Batched / memory-bounded build | ✅ shipped | `cpp_run_batch` (scale-axis) + `cpp_run_sample_batched` (sample-axis) |
| Embedding ingestion + fusion (per-residue `(L,D)`) | ✅ shipped | `EmbeddingPreprocessor` + `combine_dict_nums` + `CPP.run_num` (struct / embedding / fused arms) |
| Bootstrap CI / MCC / per-protein AP | ✅ shipped | `comp_bootstrap_ci`, `comp_detection_metrics`, `comp_per_protein_ap` |
| Structure parsing (pro) | ✅ shipped | `StructurePreprocessor` — DSSP / PDB / AlphaFold / PAE / contacts |
| Model-based feature selection | ✅ shipped | `TreeModel.select_features` (`top_k`/`threshold`/`frequency`, ADR-0023) |
| Redundancy primitive | ✅ shipped | `NumericalFeature.filter_correlation(X, max_cor)` (order-dependent) |
| Redundancy knob in CPP | ✅ shipped | `max_cor=0.5` / `max_overlap=0.5` / `check_cat` in `CPP.run` |
| Prefix-sum vectorization | ❌ not done (intentionally) | direct slice-mean + Cython; cumsum breaks ADR-0015 parity |
| Model-free `select_features(X,y)` standalone | ❌ not present | composable from existing primitives; **#32** deliberately excludes it |
| Generic `aa.evaluate` / `aa.compare` | ❌ not present | only `comp_bootstrap_ci`; `.eval` is per-class |
| Charge-patch / spatial-charge feature | ❌ not present | `StructurePreprocessor` has contacts, not charge patches |
| TreeModel sklearn compatibility | ❌ not present | custom class; `feat_importance` (no `feature_importances_`, no `BaseEstimator`) |
| Redundancy "< n_filter returned" warning | ❌ not present | `_redundancy_filter.py::filtering` truncates silently |
| Learning-curve / sampling-limited utility | ❌ not present | no issue |

---

## Per-item verdict

| Feedback item (FEEDBACK.md) | Verdict | Action |
|---|---|---|
| **P0** perf: prefix-sum + macOS `n_jobs` + caching + batching | **Mostly shipped; prefix-sum dropped** | resolve; `warnings.warn`-vs-print nit + version note → **#74** |
| **P0** standalone `select_features(X,y)` + expose redundancy cap | **Capability exists; docs gap** | new **Protocol** issue + new **truncation-warning** issue |
| **P1** fusion / external-feature pathway | **Partial** (per-residue done; per-sequence pooled-vector + joint scoring not first-class) | scope-note on **#22** |
| **P1** built-in evaluation (repeated-CV, CIs, paired ΔMCC) | **Valid gap** | new **Model evaluation & comparison** issue (3 design options) |
| **P1** structure / charge-patch feature type | **Dropped** (empirically weak: charge-patch-alone MCC ≈ 0.491) | none |
| **P2** TreeModel sklearn-compat / `feature_importances_` | **Valid gap** | new **TreeModel sklearn-compat** issue |
| **P2** AAclust determinism / `n_clusters` semantics / clamp | **Partial** (`random_state` shipped; raises, no clamp) | scope-note on **#43** |
| **P2** dPULearn defaults / errors (`n_unl_to_neg`, labels) | **Minor papercut** | new small **dPULearn papercut** issue (prio:3) |
| **P2** native CPP⊕PLM stacking + error-correlation diagnostic | **Covered** | open **#22** (stacking) + **#23** (correlation) |
| **P2** decorrelation-aware view selection | **Covered** | open **#23** |
| **P2** learning-curve / "is this sampling-limited?" utility | **Valid gap** | new **Learning-curve diagnostic** issue |

---

## Issue-action list

**New issues (6):**
1. *Protocol: feature selection & redundancy reduction* (`type:dcos`, `prio:2`, `topic:core`) — relates #35, #32; refs ADR-0023.
2. *Model evaluation & comparison: repeated-CV + bootstrap CIs + paired ΔMCC* (`type:feature`, `prio:1`, `topic:core`) — presents 3 design options (helpers / `Tool`-class+Plot / hybrid; class recommended); relates #24, #25, #16.
3. *TreeModel: expose `feature_importances_` + sklearn estimator interface* (`type:feature`, `prio:3`, `topic:XAI`) — relates #24.
4. *Learning-curve utility: "is this task sampling-limited?"* (`type:feature`, `prio:3`, `topic:core`).
5. *Warn when CPP redundancy filter returns fewer than `n_filter` features* (`type:feature`, `prio:3`, `topic:core`) — relates #78, Protocol.
6. *dPULearn: clearer `n_unl_to_neg` / labels errors and defaults* (`type:bug`, `prio:3`, `topic:core`).

**Comments / scope-notes on existing issues (4):**
- **#22** — per-*sequence* pooled-vector fusion + joint CPP⊕PLM scoring is the still-missing slice (per-residue `(L,D)` fusion already shipped).
- **#43** — add graceful clamp + document `n_clusters` semantics; note `random_state` already shipped.
- **#23** — confirm it already covers the CPP↔embedding correlation + decorrelation-view ask.
- **#74** — record which release carries the perf speedups; consider promoting the Cython-fallback `ut.print_out` notice to `warnings.warn` so it survives suppressed output.

**Dropped:** prefix-sum vectorization (ADR-0015 parity + not the real bottleneck); charge-patch feature type (empirically weak).
