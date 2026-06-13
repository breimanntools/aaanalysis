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

This release substantially expands the feature-engineering surface: a unified
feature-preprocessor family (embedding / structure / annotation sources), a
numerical mode for CPP, a configuration-sweep wrapper, sequence-window sampling,
and a suite of site-localization metrics and plotting helpers.

### Added
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
- `plot_rank` per-protein max-score-vs-rank scatter.
- `aa.__version__` top-level attribute.
- `aaanalysis.utils.deprecated(reason, version_removed)` decorator helper for
  marking public symbols deprecated under the strict-semver policy (internal
  helper; not part of the public API).
- This `CHANGELOG.md`.

### Changed
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
  (~12x), vectorized with identical output. Plus `StructurePreprocessor.encode_pdb`
  CA-CA contact counts (`contact_count_8A`/`12A`) vectorized (~50x, identical counts).

### Deprecated
- None. The strict-semver deprecation policy and the `deprecated` decorator are
  now in force for all future public-API renames and removals.

## [1.0.3] - 2026-04-06
### Added
- `AAlogo` and `AAlogoPlot` for amino acid logo visualization.

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

<!-- Only tags 0.1.1, v1.0.0, v1.0.3 exist; intermediate versions (1.0.1/1.0.2,
     0.1.2–0.1.5) were never tagged, so only the real tags are linked here. -->
[Unreleased]: https://github.com/breimanntools/aaanalysis/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/breimanntools/aaanalysis/compare/v1.0.0...v1.0.3
[1.0.0]: https://github.com/breimanntools/aaanalysis/compare/0.1.1...v1.0.0
[0.1.1]: https://github.com/breimanntools/aaanalysis/releases/tag/0.1.1
