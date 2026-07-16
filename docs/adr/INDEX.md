# ADR index

Auto-generated overview of every Architecture Decision Record. The conventions
(when to write one, the template, status rules) live in [README.md](README.md).

Regenerate this table with:
`python .claude/skills/agent-readiness-audit/scripts/check_adrs.py --write-index`

<!-- ADR-INDEX:START — auto-generated; regenerate with check_adrs.py --write-index -->
| ADR | Title | Status | Date | Notes |
|----:|-------|--------|------|-------|
| [0001](0001-cpp-backend-architecture.md) | CPP backend architecture: unified numerical pipeline + Cython acceleration | Accepted | 2026-05-25 |  |
| [0002](0002-structure-preprocessor-alphafold-features.md) | StructurePreprocessor: AlphaFold per-residue features (feature-set rev 1.1) | Accepted | 2026-05-25 |  |
| [0003](0003-residue-annotation-layer.md) | Residue-level annotation layer (AnnotationPreprocessor) | Accepted | 2026-05-25 |  |
| [0004](0004-domain-segmentation-wrapper.md) | Domain segmentation: wrap ChainSaw + AFragmenter; Merizo stays BYO | Accepted | 2026-05-25 |  |
| [0005](0005-feature-preprocessor-family.md) | Feature-preprocessor family: consolidate into `data_handling_pro`, unify the protocol | Accepted | 2026-05-29 | partially superseded by ADR-0011, 2026-06-01 |
| [0006](0006-standalone-rank-plot.md) | Per-protein rank plot ships as a standalone `aa.plot_rank`, not a `*Plot` method | Superseded |  |  |
| [0007](0007-cppgrid-across-configs.md) | `CPPGrid`: a class that sweeps across configurations on a threads-default backend | Accepted | 2026-05-31 |  |
| [0008](0008-n-jobs-semantics.md) | Unified `n_jobs` contract: `None` means optimized | Accepted | 2026-05-31 |  |
| [0009](0009-discard-cppstate-for-lru.md) | Reuse scale tensors via an internal content-hash LRU, not a public `CPPState` | Accepted | 2026-05-31 | D2 superseded by ADR-0014, 2026-06-03 |
| [0010](0010-package-version-1.1.0.md) | Next package release is v1.1.0, not v1.0.4 | Accepted | 2026-06-01 |  |
| [0011](0011-embedding-encode-and-builder-naming.md) | EmbeddingPreprocessor gains `encode`; unify builder names to `build_scales` / `build_cat` | Accepted | 2026-06-01 |  |
| [0012](0012-defer-comp-seq-cons.md) | Defer `comp_seq_cons`: neither test nor integrate the orphaned conservation module | Accepted | 2026-06-02 |  |
| [0013](0013-wire-sample-batched-cpp-run.md) | Wire `cpp_run_sample_batched` to a public `CPP.run(n_sample_batches=)` | Accepted | 2026-06-03 |  |
| [0014](0014-clear-cache-internal-not-public.md) | Demote cache eviction to an internal utility; drop public `CPP.clear_cache()` | Accepted | 2026-06-03 |  |
| [0015](0015-cpp-regression-anchor.md) | Exact-value CPP regression anchor, pinned to one canonical CI cell | Accepted | 2026-06-03 |  |
| [0016](0016-coverage-measurement-and-gates.md) | Measure coverage on the package only; ratchet module gates | Accepted | 2026-06-03 |  |
| [0017](0017-alphafold-fetch-and-acquisition-verbs.md) | `StructurePreprocessor.fetch_alphafold` + the preprocessor verb taxonomy | Accepted | 2026-06-04 |  |
| [0018](0018-options-override-only-scales-cache.md) | `options['df_scales'|'df_cat']` are override-only; default memoization is internal | Accepted | 2026-06-05 |  |
| [0019](0019-fimo-source-build-in-ci.md) | Build FIMO from source on the Linux CI matrix to gate `scan_motif` end-to-end | Accepted | 2026-06-05 |  |
| [0020](0020-negative-sampler-subsumed-by-aawindowsampler.md) | Issue #66 `NegativeSampler` is subsumed by `AAWindowSampler` | Accepted | 2026-06-05 |  |
| [0021](0021-scan-motif-fimo-significance.md) | `scan_motif` is a true FIMO significance scanner, not a parity twin | Accepted | 2026-06-05 |  |
| [0022](0022-prediction-task-level-taxonomy.md) | Prediction-task taxonomy: residue / domain / protein, by unit-of-comparison | Accepted | 2026-06-06 | amended 2026-07-14 — reconciled the level vocabulary with the shipped `AAPred.predict(level=)` API |
| [0023](0023-tree-model-select-features.md) | `TreeModel.select_features` is a post-fit method, not a new selector class | Accepted | 2026-06-06 |  |
| [0024](0024-feature-map-shap-via-shap-plot.md) | `feature_map` gains SHAP support via the `shap_plot` toggle, not issue #63's `stack_by` / `df_imp` | Accepted | 2026-06-06 |  |
| [0025](0025-interpretability-tiered-explainable-scale-sets.md) | Interpretability-tiered "explainable" scale sets in `load_scales` | Accepted | 2026-06-06 |  |
| [0026](0026-feature-pruning-empirical-not-scale-correlation.md) | Feature pruning is empirical (sample-level), df_feat-in/out methods on `SequenceFeature` | Accepted | 2026-06-11 |  |
| [0027](0027-protein-design-mutation-deltacpp-scope.md) | Protein design (AAMut/SeqMut): scope boundary and model-free ΔCPP | Accepted | 2026-06-11 | amended 2026-06-24 — see *Amendment* below and ADR-0042 |
| [0028](0028-cppstructureplot-structureview-return-wrapper.md) | `CPPStructurePlot` returns a `StructureView` wrapper, not a matplotlib `Axes` | Accepted | 2026-06-11 | amended 2026-06-28: matplotlib backend removed |
| [0029](0029-fetch-embeddings.md) | EmbeddingPreprocessor.fetch_embeddings + fetch_* covers model-weight acquisition | Accepted | 2026-06-12 |  |
| [0030](0030-changelog-and-deprecation-policy.md) | Strict-semver deprecation policy, `deprecated` decorator, and a two-file changelog | Accepted | 2026-06-13 |  |
| [0031](0031-integration-e2e-test-policy.md) | Integration & e2e test tiers: scope, taxonomy, and merge-gating | Accepted | 2026-06-13 |  |
| [0032](0032-numerical-equivalence-tolerance-policy.md) | Numerical-equivalence tolerance policy for output-affecting optimizations | Accepted | 2026-06-13 |  |
| [0033](0033-performance-audit-outcomes.md) | Whole-library performance audit: accepted wins and rejected optimizations | Accepted | 2026-06-13 |  |
| [0035](0035-validation-ut-check-not-pydantic.md) | Input validation stays `ut.check_*`; no Pydantic; agent-typed contracts live in ProtXplain | Accepted | 2026-06-22 |  |
| [0036](0036-type-hints-contract-checker-deferred-pyright.md) | Ship `py.typed` and adopt pyright (non-blocking) now; type hints are the contract | Accepted | 2026-06-22 |  |
| [0037](0037-perf-ab-gate.md) | Same-runner A/B vs the latest PyPI release makes wall-clock a merge gate | Accepted | 2026-06-22 |  |
| [0038](0038-agentic-readiness-boundary.md) | Agentic-readiness boundary: usability/improvability is ours, agent integration is ProtXplain's | Accepted | 2026-06-23 |  |
| [0039](0039-plot-return-contract.md) | One uniform `(fig, ax)` return contract for every `*Plot` method | Accepted | 2026-06-24 |  |
| [0040](0040-golden-pipelines-convenience-api.md) | Golden pipelines: the `aaanalysis.pipe` (ap) convenience API | Accepted | 2026-06-24 | amended 2026-07-14 — verb names aligned to the shipped `aaanalysis.pipe` API |
| [0041](0041-pipe-pipeline-conventions.md) | `aaanalysis.pipe` pipeline conventions and the core golden pipelines | Accepted | 2026-06-24 |  |
| [0042](0042-seqmut-model-guided-prediction-shift.md) | SeqMut model-guided prediction shift (ML-guided directed evolution) | Accepted | 2026-06-24 |  |
| [0043](0043-seqopt-optimization-layer.md) | SeqOpt optimization layer (SHAP-guided, fuzzy-labeled multi-objective directed evolution) | Accepted | 2026-06-24 |  |
| [0044](0044-find-features-search-protocol.md) | `find_features` staged sensitivity search and multi-objective selection | Accepted | 2026-06-24 |  |
| [0045](0045-seqopt-deap-parity-and-pure-python-operators.md) | SeqOpt: full pure-Python EA operator set + DEAP parity (ship ours) | Accepted | 2026-06-25 |  |
| [0046](0046-predict-samples-multi-model-harness.md) | `predict_samples` multi-model comparison harness; paper-fidelity training engine deferred | Accepted | 2026-06-25 |  |
| [0047](0047-cpp-structure-explore-integrated-predictor.md) | `CPPStructurePlot.explore`: integrated per-site predictor with `output=` dispatch | Accepted | 2026-06-28 |  |
| [0048](0048-select-scales-curation-surface.md) | `AAclust.pre_select_scales` for metadata-only scale exclusion | Accepted | 2026-06-29 |  |
| [0049](0049-cpp-profile-nonnegative-shap-signed.md) | CPP profile y-axis is non-negative; three plot-rendering "fixes" are rejected | Accepted | 2026-07-04 |  |
| [0050](0050-cpp-redundancy-legacy-exact.md) | CPP redundancy criterion: `legacy` default, `exact` opt-in | Accepted | 2026-07-05 |  |
| [0051](0051-treemodel-per-round-seeding.md) | TreeModel per-round seeding (fixed seed → independent rounds) | Accepted | 2026-07-06 |  |
| [0052](0052-bh-pvalue-monotonicity.md) | BH-adjusted p-value monotonicity (canonical Benjamini–Hochberg) | Accepted | 2026-07-06 |  |
| [0053](0053-dpulearn-project-out-of-sample.md) | dPULearn.project: out-of-sample projection into the fitted PC space | Accepted | 2026-07-06 |  |
| [0054](0054-reject-distributed-signal-diagnostic.md) | Reject the distributed-signal (joint-vs-marginal lift) diagnostic | Accepted | 2026-07-08 |  |
| [0055](0055-cpp-bootstrap-stability-selection.md) | CPP bootstrap / stability annotation | Accepted | 2026-07-08 |  |
| [0056](0056-rank-plot-into-aapredplot.md) | Per-protein rank scatter moves into `AAPredPlot.predict_group(kind="rank_scatter")` | Accepted | 2026-07-09 |  |
| [0057](0057-v2-api-naming-and-future-xai-layer.md) | v2 API naming system and the future XAI `*Explainer` layer | Accepted | 2026-07-12 |  |
| [0058](0058-aapred-eval-pooled-cv-splitter.md) | `AAPred.eval` accepts a custom CV splitter and scores it by pooled out-of-fold prediction | Accepted | 2026-07-12 |  |
| [0059](0059-aapred-predict-oof-scores.md) | AAPred.predict_oof: cross-validated out-of-fold scores for the training set | Accepted | 2026-07-12 |  |
| [0060](0060-packaging-install-from-wheel-gate.md) | Packaging gate: ship the sdist sources, build wheel + sdist, install from both, import the public API | Accepted | 2026-07-14 |  |
| [0062](0062-find-features-nested-selection-scope.md) | find_features nested feature selection (selection_scope="global"|"fold") | Accepted | 2026-07-16 |  |
<!-- ADR-INDEX:END -->
