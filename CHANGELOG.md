# Changelog

All notable changes to **AAanalysis** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
from v1.x onward, any rename or removal of a public symbol (one re-exported by
`aaanalysis/__init__.py`) ships at least one **minor** release carrying a
`DeprecationWarning` (via `aaanalysis.utils.deprecated`) before the symbol is
removed. See the *Versioning and Deprecation Policy* in `CONTRIBUTING.rst`.

This is the terse, developer-facing index. The narrative, RTD-rendered release
notes — with cross-references and examples — live in
[`docs/source/index/release_notes.rst`](docs/source/index/release_notes.rst).

## [Unreleased]

### Added
- `ReliabilityModel.eval(use_calibrated=..., add_metrics=...)`: score the calibrated column
  (`score_calibrated`) instead of the raw `score`, and append Brier-score / expected
  calibration error (ECE) rows (`bin='brier'` / `bin='ece'`, value in `mean_score`); defaults
  keep the previous output byte-identical. `ReliabilityModelPlot.reliability_diagram(label=...)`
  annotates Brier / ECE in the legend when present and supports raw-vs-calibrated overlays on
  one `ax` (addresses #480).
- `ReliabilityModel.fit(calibrate=True)` now warns (`UserWarning`) when no calibrator can be
  fitted (for example, too few samples in a class for internal cross-validation or a model that
  cannot be cloned), instead of leaving the failure silent.
  fitted (too few samples in a class for the internal cross-validation, or a model that cannot
  be cloned), instead of leaving the failure silent.
- `ReliabilityModel` applicability domain is banded and inspectable: `predict` appends
  `ad_status` (`inside` / `borderline` / `outside` / `unknown`, never null; `in_domain` equals
  `ad_status == "inside"`) and `ad_nearest_train` (0-based row index of the closest training
  sample); `fit` gains `ad_borderline=0.1` (band width above the boundary) and exposes the
  fitted `ad_threshold_` (raw training k-NN distance threshold; when it is positive,
  `ood_score == ad_knn / ad_threshold_`) and `ad_method_` (`"knn"`). Apart from the
  `ad_knn_dist` → `ad_knn` rename, existing `predict` columns keep their order and values
  (partially addresses #473). **Not included:** the mutation-candidate entry
  point that would score `SeqMut` / `SeqOpt` output in one call. `predict` still takes a
  prebuilt feature matrix `X`, so a candidate set is scored by building that matrix with
  `SequenceFeature.feature_matrix` first.
- Output contract for the prediction tier, advancing the per-sample and per-residue half of
  the documented boundary contract (addresses #26; the `df_feat` half is already covered by
  `DICT_DF_FEAT` / the CPP output schema): `DICT_DF_SCHEMAS` now documents `df_pred`
  (`AAPred.predict`, all three levels), `df_rel` (`ReliabilityModel.predict`) and
  `df_eval_reliability` (`ReliabilityModel.eval`), rendered on the Data Schemas page and guarded
  by contract tests that pin the column names, order and dtypes as literals, so a renamed,
  dropped, retyped or undocumented column fails even if the schema is edited to match. The
  `score` column is documented per scale (`proba` -> `[0, 1]`, `percent` -> `[0, 100]`), so
  both `AAPred.predict(score_range=...)` outputs are checked against the same contract. The
  domain level's `is_best` column is now routed through `COL_IS_BEST` (no output change).

### Changed
- Prediction/design-tier consistency pass (these classes are still experimental, so no
  deprecation cycle; addresses #510):
  - `ReliabilityModel.fit(ci=...)` is now a fraction in (0, 1), default `0.90` (was a percent,
    `90.0`), matching `ModelEvaluator.run(ci=0.95)` and `comp_bootstrap_ci`. A percent value
    raises a `ValueError` with a hint.
  - `ReliabilityModel.predict` column `ad_knn_dist` renamed to `ad_knn` (matches
    `ad_mahalanobis` / `ad_leverage`).
  - `ReliabilityModel.eval` column `n` renamed to `n_samples`; its columns are registered as
    `COLS_EVAL_RELIABILITY` (summary-row shape unchanged).
  - `SeqOpt` default `mode` is now `"importance"` (core-only), so bare `SeqOpt()` constructs in
    a base install; `"impact"` stays the documented headline mode.
  - `AAPred.eval` (and `list_metrics`) accepts `"mcc"`, the same metric vocabulary as
    `ModelEvaluator`.
  - Documented that `score_std` in `ReliabilityModel` is the spread across ensemble members,
    whereas in `AAPred` / `ModelEvaluator` it is the spread across CV folds.
  - `AAPred` and `SeqOpt` now validate `df_scales`, and `AAPred.eval` validates `list_parts`, in
    their frontend checks: an invalid value raises a `ValueError` naming the parameter instead of
    failing later (or being silently ignored).
### Added
- `NumericalFeature.from_pssm` converts PSI-BLAST ASCII PSSM files and precomputed `(L, 20)`
  arrays into a canonical-amino-acid-order `dict_num`. File columns are reordered; arrays must
  already use that order. Values are normalized to `[0, 1]` by default (or kept raw with
  `normalize=False`), and `return_scales=True` supplies matching 20-column `df_scales` and
  `df_cat` for `get_parts` -> `CPP.run_num` (#79).
- `SequenceFeature.get_split_kws(strategy=...)`: `"compositional"` / `"positional"` CPP
  strategy presets, equal to the explicit `split_types` / `n_split_min` / `n_split_max`
  calls (`Segment(1,1)`; `Segment(2..15)` + `Pattern` + `PeriodicPattern`).
  `strategy=None` keeps the output unchanged; combining a preset with non-default
  `split_types`, `n_split_min`, or `n_split_max` raises `ValueError` (#87).
- `ModelEvaluator.learning_curve` / `ModelEvaluatorPlot.learning_curve`: cross-validated
  metric-vs-training-size curve with bootstrap CIs (stratified, nested subsets of each training
  fold; test folds untouched) to tell a sampling-limited task from a saturated one. The default
  grid has five fraction candidates (which resolve to five distinct sizes on sufficiently large
  data), each with a bootstrap CI. A fractional size is resolved within each training fold, so
  `1.0` is every fold's complete training set and reproduces
  `ModelEvaluator.run` when both calls use the same `random_state`, `n_cv`, `n_rounds`, and
  metrics, also when the training folds differ in size (#93).
- `AAPredPlot.group_cluster`: `kind="dendrogram"` with `layout="rectangular"|"circular"` draws
  the sample relation tree alone, leaves colored by `labels` / `labels_row` (one strip or ring
  each, titled legends). It reuses the clustermap's linkage (now computed explicitly with SciPy
  and passed to seaborn), so both kinds show the same topology; the clustermap figure is
  unchanged when `fastcluster` is not installed (with `fastcluster`, seaborn used to compute the
  linkage itself, so exact ties between equidistant merges may be broken differently).
  (Addresses #391)

### Fixed
- `ReliabilityModel.fit`: non-finite numbers (`NaN`, `inf`, `-inf`) are rejected for `ci`,
  `ad_percentile` and `conformal_alpha`. `NaN` passed both range comparisons silently
  (`nan < 0` and `nan > 1` are each `False`), so `fit(ci=float("nan"))` was accepted and
  `predict` then returned `NaN` `ci_low` / `ci_high` columns.
- `ReliabilityModel.eval(use_calibrated=True)` no longer claims the model was fitted with
  `calibrate=False` when calibration was requested but failed; the error names the real reason.
- `ReliabilityModel.eval`: passing `X` without `labels` raises instead of silently ignoring the
  given features. Passing `labels` without `X` scores the training features against that
  labelling, which must match them in length and may only use labels observed during `fit`.
- `ReliabilityModelPlot.reliability_diagram` validates `figsize`, `color`, `title`, and `ax`.
- `StructurePreprocessor.encode_pae` / `encode`: read the AlphaFold DB PAE JSON layout
  (a one-element list wrapping the `predicted_aligned_error` dict), so files from
  `fetch_alphafold` load without a manual unwrap.
- `CPPPlot.ranking`: the `Σ` total / SHAP sign key no longer overprints the short bars'
  percentage labels.
- `SeqOptPlot`: `convergence` y-label fits its panel; `mutation_map` default
  `figsize=(8, 6)`; `parallel_coordinates` colors by the first objective (with colorbar)
  when only one front is drawn.
- Tutorials: typo / wrong-name fixes, four notebooks repaired to pass `nbformat`
  validation, all tutorials re-executed against the current code.
- `StructurePreprocessor.get_domains`: AFragmenter adapter reads the current
  `ClusteringResult` API (0-based `cluster_intervals`); choppings were silently empty.
- `AALogoPlot`: upright, size-fitted P-site labels for long windows (bottom panel only
  in `multi_logo`); TMD / JMD part labels shrink on very short parts.
- `CPPPlot.eval`: bar annotations capped to the bar height (no overprinting).
- `SeqOptPlot.pareto_front`: solid color for a single front.
- `AAPredPlot.eval` / comparison charts: long condition names auto-rotated.
- `aaanalysis.pipe.plot_eval`: axis-impact panel wraps long axis names.
- `CPPPlot.feature_map` / `heatmap`: explicit `cbar_xywh` (with `y`) is honored by the
  bottom-row layout; vertical colorbars get right-side ticks.
- Quiet by default: DSSP no longer re-emits mkdssp's mmCIF probe as a warning;
  `fetch_embeddings` keeps Hub status text / weight-loading bars off stderr; the
  `build_scales` dataset-dependence note is verbose-gated (was a `UserWarning`);
  `ShapModel` runs `KernelExplainer` silently.
- Examples: `display_df` everywhere; real executed `fetch_alphafold`, `get_dssp`,
  `encode_dssp`, `get_domains`, `encode_domains`, `fetch_uniprot` examples;
  `compare_sets_negatives` example no longer depends on `upsetplot` (0.9 breaks on
  pandas 3); nbformat repairs and spelling fixes.

### Documentation
- New tutorial `tutorial3e_cpp_embeddings_structure`: embeddings and AlphaFold
  structures into `CPP.run_num`, fusion, and `CPPStructurePlot` painting.
- New usage-principles page *Golden Pipelines*: the `aaanalysis.pipe` spine, the
  `(result, plot, df_eval)` return shape (and `plot_eval` returning a list of figures),
  the parity anchors, and the experimental-API note.
- `aaanalysis.pipe.find_features`: the docstring now states that only `search="fast"` has a
  parity-anchored `df_feat`; `"balanced"` / `"exhaustive"` are reproducible but have no explicit
  chain.

### Tests
- `aaanalysis.pipe` contract suite: `predict_samples` defaults pinned byte-identical to the
  explicit `feature_matrix` -> `cross_validate` chain, comparing the full learned state of every
  fitted predictor (nested ensembles included), not only its predictions.
- The documented golden path is executed and its statement budget is parsed from the code block on
  the *Golden Pipelines* page, so page and test cannot drift apart.
- Example notebooks of `aaanalysis.pipe` are held to zero parameter-coverage gaps.
- Protocols P2 (exploratory sequence analysis) and P3 (sampling) brought to the
  protocol quality rubric: key mental model, concept-contrast figures (bits vs
  probability, shuffled baseline, label-split logos; distance band, reference
  composition, motif-matched lookalikes, similarity filters), each demonstrated
  method's public parameters covered by name across its calls, and demonstrated
  mistakes. This pass covers P2 and P3 only; the other protocols are unchanged.
- `CPP.run`: corrected the `n_sample_batches` description, which promised peak memory
  bounded by the batch size rather than by the sample count `n`. Measurements show it bounds
  the dominant per-batch scale-value tensor, while the `(n_samples, n_pre_filter)` survivor
  matrix and its test statistics stay resident, so peak memory still grows linearly with `n`
  at a constant batch size, on a roughly 13x flatter slope than the single-pass run. Behaviour
  is unchanged; only the documentation was wrong.
- `CPP.run`: clarified the chunking contract. `n_sample_batches` now creates exactly the requested
  number of balanced, non-empty sample batches and documents that it bounds the per-batch
  scale-value tensor, not total peak memory: the `(n_samples, n_survivors)` matrix and its test
  statistics remain resident. The `n_batches` documentation now notes its per-batch FDR semantics.
- `CPP.run` and `CPP.run_num`: `n_sample_batches` now creates exactly the requested number of
  balanced, non-empty sample batches. It bounds the dominant per-batch working set, but not total
  peak memory: the pre-filtered candidate matrix and its test statistics remain resident. The
  documentation now distinguishes this from `n_batches`: `CPP.run` applies FDR correction per
  selected-feature batch, while `CPP.run_num` batches only pass-1 statistics.

## [1.1.0] - 2026-09-10

This release substantially expands the feature-engineering surface: a unified
feature-preprocessor family (embedding / structure / annotation sources), a
numerical mode for CPP, a configuration-sweep wrapper, sequence-window sampling,
and a suite of site-localization metrics and plotting helpers. It also introduces
a prediction tier (`AAPred`, `ModelEvaluator`, `ReliabilityModel`), a
multi-objective design tier (`SeqOpt`), a scikit-learn transformer, and the
`aaanalysis.pipe` convenience API. The public surface grows from 31 to 54
re-exported symbols.

### Added
- **Prediction tier**: `AAPred` / `AAPredPlot` (evaluate and deploy sequence-based
  prediction models), `ModelEvaluator` / `ModelEvaluatorPlot` (cross-validated
  evaluation and paired model comparison with bootstrap confidence intervals), and
  `ReliabilityModel` / `ReliabilityModelPlot` (per-prediction trust: probability
  calibration, applicability-domain / out-of-distribution scoring, conformal sets).
- **Design tier**: `SeqOpt` / `SeqOptPlot` — multi-objective, ML-guided directed
  evolution over sequence variants (NSGA-II), with Pareto-front, hypervolume,
  convergence and genealogy plots.
- `SequenceFeatureTransformer`: leak-free CPP feature selection exposed as a
  scikit-learn transformer (`fit` / `transform` / `get_feature_names_out`), so CPP
  features can be used inside a `Pipeline` without leaking across folds.
- `aaanalysis.pipe` (`ap`): a second, stateless convenience API of high-level
  "golden pipelines" — `obtain_samples`, `find_features`, `predict_samples` and
  `plot_eval` (plus `explain_features` under `[pro]`).
- Named sample-color constants `COLOR_SAMPLES_POS` / `COLOR_SAMPLES_NEG` /
  `COLOR_SAMPLES_UNL` / `COLOR_SAMPLES_REL_NEG`, so the positive / negative /
  unlabeled / reliable-negative palette can be referenced by name rather than
  duplicated as literals.
- `EmbeddingPreprocessor`, `StructurePreprocessor` (`[pro]`),
  `AnnotationPreprocessor` (`[pro]`), and `combine_dict_nums` for building
  per-residue numerical tensors as `CPP.run_num` input. New `[embed]` extra
  isolates the heavy `torch` / `transformers` dependencies.
- `CPPGrid` configuration-sweep wrapper and `CPP.run_num` numerical mode.
- `SequenceFeature` label helpers (`get_labels_ovr` / `get_labels_ovo` /
  `get_labels_quantile` / `get_labels_tiered`), `get_df_parts_from_windows`,
  and `get_feature_descriptions`.
- `AAclust.select_scales` and `AAclust.select_proteins`.
- `AAWindowSampler` sequence-window sampler and `scan_motif` (`[pro]`, MEME/FIMO).
- `aa.metrics` site-localization helpers: `comp_per_protein_ap`,
  `comp_detection_metrics`, `comp_bootstrap_ci`, `comp_smooth_scores`.
- Per-protein max-score-vs-rank scatter via `AAPredPlot.predict_group(kind="rank_scatter")`.
- `aa.__version__` top-level attribute.
- `aaanalysis.utils.deprecated(reason, version_removed)` decorator helper for
  marking public symbols deprecated under the strict-semver policy (internal
  helper; not part of the public API).
- This `CHANGELOG.md`.
- `Version Guard` CI workflow (`.github/scripts/check_version_ahead.py`): fails a
  build unless `[project] version` is strictly ahead of the latest release published
  on PyPI (git-tag fallback offline), so `master` never reports a published version.
- `get_provenance(random_state, data)`: opt-in, JSON-serializable plain-`dict`
  provenance record carrying the **effective resolved seed** (after the
  `options['random_state']` override), a deterministic-vs-stochastic flag, the
  package / Python / key-dependency versions, the git commit when resolvable, and
  an optional `sha256` input hash. Nothing attaches it to outputs; no return type
  changes.

### Changed
- **Version bumped to `1.1.0`** (from the published `1.0.3`), so a development
  checkout is distinguishable from a released install. The version stays manually
  maintained and always names the next unreleased number; see *Version truth* in
  `CONTRIBUTING.rst`.
- **Uniform plot return contract: every `*Plot` method now returns a `(fig, ax)`
  pair.** Previously the methods returned three inconsistent shapes (`(fig, ax)`,
  a bare `Axes`, or `(ax, df)`). The returned object is a thin tuple subclass that
  unpacks as `fig, ax = plot(...)` and also forwards attribute access to `ax`, so
  legacy `ax = plot(...); ax.set_title(...)` keeps working — this part is
  backward-compatible. **Breaking (scheduled for the next major):**
  `AAclustPlot.centers` / `medoids` now return `(fig, ax)` and expose the
  PCA-component DataFrame on the `df_components_` attribute instead of as the
  second return value, so `ax, df = centers(...)` no longer unpacks correctly —
  use `fig, ax = centers(...)` then read `aac_plot.df_components_`.
- **CPP performance work lands in this release.** The Cython feature-matrix
  kernel, macOS-safe threaded `n_jobs`, scale / AA-index caching, and scale /
  sample batching together replace the hour-long, low-CPU runs seen on `1.0.3`
  and earlier. **Users on `≤1.0.3` should upgrade** rather than debug a
  performance pathology that is already fixed on `master`.
- The Cython-fallback notice (shown when the compiled extension is missing and
  CPP falls back to the ~2× slower pure-Python kernel) is now a one-time
  `UserWarning` instead of an easily-missed INFO print, so it surfaces even with
  `aa.options['verbose'] = False`.
- `SequenceFeature.feature_matrix` accepts a `batch=` list of `df_parts` for a
  single Cython pass.
- `SequenceFeature.get_df_parts` / `NumericalFeature.get_parts` gain a
  `pos`-anchor input mode (`tmd_len=`).
- Unified `n_jobs` parallelism convention across `CPP` / `CPPGrid`, with an
  `options['n_jobs']` global override.
- `CPPPlot.feature` titles the plot with the feature's human-readable
  description, controlled by new `show_title` / `title_wrap_width` parameters.
- Same-output speedups for internal hotspots (no API/output change):
  `AAWindowSampler` redundancy/similarity filtering (vectorized, ~30x at scale),
  `AAclust` sample-to-medoid correlation distances (single pass), and per-feature
  Kullback-Leibler divergence used by `dPULearn.eval(comp_kld=True)`
  (parallelized, honors `options['n_jobs']`). Plus `AAWindowSampler`
  candidate-center band filtering (~40x) and `sample_motif_matched` PWM scoring
  (~12x), and `SequencePreprocessor.encode_one_hot` (~3x), vectorized with
  identical output. Plus `StructurePreprocessor.encode_pdb` CA-CA contact counts
  (`contact_count_8A`/`12A`) vectorized (~50x, identical counts) and its
  per-(target, atom) sequence alignment cached across the ~26 redundant
  re-alignments each entry triggers (~12x off the alignment overhead, byte-identical encoder output).
- `StructurePreprocessor.fetch_alphafold` and `AnnotationPreprocessor.fetch_uniprot`
  reuse a pooled HTTP session (one `requests.Session` per worker thread) instead of
  opening a fresh connection per request, and gain an opt-in `max_workers` parameter
  for threaded bulk fetching. Concurrency is **off by default** (`max_workers=None`/`1`
  is the unchanged sequential path) because parallel requests to AlphaFold DB / UniProt
  risk HTTP-429 throttling; results are reassembled in input order, so the status
  table / `df_annot` and on-disk files are byte-identical regardless of worker count.
- `dPULearn.fit` gains flexible, package-consistent label handling via
  `label_pos` / `label_unl` / `label_neg` markers: pass standard `{0, 1}` labels
  directly with `label_unl=0`, or an arbitrary positive/unlabeled/negative
  encoding. Pre-labeled negatives (`label_neg`) are kept and never re-selected —
  only unlabeled samples are candidates. The negative count is now specified one
  of two ways (exactly one): the new `n_neg` (the **total** number of negatives
  wanted; dPULearn identifies `n_neg` minus the pre-labeled negatives), or the
  existing `n_unl_to_neg` (the number identified **directly from the unlabeled
  pool**). Output labels always use the package convention (1 = positive,
  0 = negative, 2 = unlabeled); the recommended input encoding is unchanged.
- Numerical-equivalence tolerance policy for performance optimizations
  (developer-facing; checklist in `CONTRIBUTING.rst`): three tiers —
  **T1** byte-identical (default), **T2** numerically-equivalent
  (`allclose(atol=1e-10, rtol=0)` + identical discrete decisions), **T3**
  statistically-equivalent (quality metric within a documented band) — with the
  evidence + pinned regression anchor each tier requires. Unblocks
  previously-excluded algorithmic optimizations (e.g. AAclust binary-search `k`),
  each as its own tier-declared PR. No user-facing behavior change.
- Build: the from-source build now requires `setuptools>=83`
  (`[build-system].requires`), which patches CVE-2026-59890 (an sdist
  `MANIFEST.in` exclusion bypass on macOS APFS/HFS+ filesystems). Installing the
  published wheel is unaffected; this only tightens the build backend used when
  building from the sdist.
- Build: this is the first release to ship binary wheels (25: CPython
  3.10-3.14 across Linux x86_64/aarch64, macOS Intel/ARM, and Windows), so
  installing no longer compiles the Cython kernel from the sdist. Linux wheels
  are built on `manylinux_2_28` images and tagged with every compatibility
  level the compiled extension satisfies (down to `manylinux2014`/glibc 2.17);
  note that `Pillow>=12.3` (a hard dependency) publishes no wheels below
  manylinux_2_28, so in practice the dependency stack needs glibc >= 2.28.
  musllinux (Alpine) wheels are not provided: `scikit-learn` ships no musl
  wheels, so the dependency stack is not pip-installable on musl systems
  either way.

### Deprecated
- `AAlogo` / `AAlogoPlot` are deprecated in favour of the PascalCase `AALogo` /
  `AALogoPlot`. The old names remain importable from both `aaanalysis` and
  `aaanalysis.seq_analysis` and now emit a `DeprecationWarning` on attribute
  access; they are scheduled for removal in the next major release. Existing code
  keeps working unchanged — update the import at your convenience.
- The strict-semver deprecation policy and the `deprecated` decorator are now in
  force for all further public-API renames and removals.

## [1.0.3] - 2026-04-28
### Added
- `AALogo` and `AALogoPlot` for amino acid logo visualization.

### Changed
- Dropped end-of-life Python 3.9; added 3.13 and 3.14 (now 3.10–3.14).
- Migrated dependency management to a single `pyproject.toml` with `[pro]` /
  `[docs]` / `[dev]` extras; added full `uv` support.

> Note: `1.0.3` and earlier **predate the CPP performance work** (see the
> Unreleased *Changed* section). Installs pinned to these versions can see
> hour-long, low-CPU CPP runs; upgrading resolves it.

## 1.0.2 - 2025-06-17
### Changed
- Faster CPP pipeline: 3–5× faster `CPP.run()` via optimized part-split-scale
  generation and filtering.
- `CPP.feature_map()` adds a cumulative per-residue importance bar plot.

### Fixed
- Minor dependency-resolution and edge-case fixes.

## 1.0.1 - 2025-01-29
### Changed
- Better `[pro]` IDE integration (jump to implementation, not `__init__.py`).
- Preserve original import-error messages for missing `[pro]` dependencies.

### Fixed
- Consistent subcategory ordering between heatmap and bar plot in the feature map.
- `numpy>=2.0.0` compatibility.

## [1.0.0] - 2024-07-01
### Added
- `SequencePreprocessor`, `comp_seq_sim`, `filter_seq`, and global `jmd_n/c_len`
  options.

### Changed
- Renamed `ShapExplainer` → `ShapModel`; renamed the *Perturbation* module to
  *Protein Design*. Biopython is now `[pro]`-only.

### Fixed
- Script-level multiprocessing support outside functions/classes.

## 0.1.5 - 2024-04-18
### Changed
- Relicensed MIT → BSD-3-Clause; added a Code of Conduct.

### Fixed
- Replaced native `multiprocessing` with `joblib` for CPP / feature-matrix
  construction.

## 0.1.4 - 2024-04-09
### Added
- Split core vs `[pro]` install profiles.

### Changed
- General API consistency improvements; Python support extended to 3.12.

## 0.1.3 - 2024-02-09
### Added
- `TreeModel`, `ShapExplainer`, `NumericalFeature`, and `load_features`.

## 0.1.2 - 2023-11-06
### Added
- `CPPPlot`, `dPULearnPlot`, `AAclustPlot`, and the `options` interface.

## [0.1.1] - 2023-09-11
- Test release of the first beta version (`CPP`, `dPULearn`, `AAclust`,
  `SequenceFeature`, `load_dataset`, `load_scales`).

<!-- Real tags: 0.1.1, v1.0.0, v1.0.3, and v1.1.0 (this release); intermediate
     versions (1.0.1/1.0.2, 0.1.2–0.1.5) were never tagged, so only the real tags
     are linked here. -->
[Unreleased]: https://github.com/breimanntools/aaanalysis/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/breimanntools/aaanalysis/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/breimanntools/aaanalysis/compare/v1.0.0...v1.0.3
[1.0.0]: https://github.com/breimanntools/aaanalysis/compare/0.1.1...v1.0.0
[0.1.1]: https://github.com/breimanntools/aaanalysis/releases/tag/0.1.1
