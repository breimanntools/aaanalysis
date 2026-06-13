Release notes
=============

Version 1.1
--------------------------------

v1.1.0 (Unreleased)
--------------------------------

This release substantially expands the feature-engineering surface: a unified
**feature-preprocessor family** (embedding / structure / annotation sources),
a **numerical mode** for CPP, a configuration-sweep wrapper, sequence-window
sampling, and a suite of site-localization metrics and plotting helpers.

Added
~~~~~

**Data Handling**

- **EmbeddingPreprocessor**: Instance-based class for per-residue protein
  language model (PLM) embeddings. The primary ``encode`` method normalizes raw
  embeddings into a ``[0, 1]`` per-residue ``dict_num`` (``method='minmax' |
  'quantile' | 'sigmoid'``) ready for ``CPP.run_num``; the secondary
  ``build_scales`` / ``build_cat`` pair collapses them into pseudo-scales /
  pseudo-categories for ``CPP.run``. The ``fetch_embeddings`` method
  (``aaanalysis[embed]``) downloads a curated PLM (ESM-2, ESM-1b, ProtT5,
  ProstT5) from the Hugging Face Hub and computes per-protein
  (``mode='protein'``, mean/max/cls pooling) or per-residue
  (``mode='residue'``) embeddings, with a hardware-aware size guard; the
  ``pool_embeddings`` helper reduces per-residue arrays to per-protein vectors.
  A new ``[embed]`` install extra isolates the heavy ``torch`` / ``transformers``
  dependencies (see ADR-0029).
- **StructurePreprocessor** (``aaanalysis[pro]``): Converts PDB / CIF / AlphaFold
  files (and AlphaFold PAE sidecars) into ``[0, 1]``-normalized per-residue
  numerical tensors. Methods: ``get_dssp``, ``encode_dssp``, ``encode_pdb``,
  ``encode_pae``, ``get_domains``, ``encode_domains``, ``build_scales``,
  ``build_cat``.
- **AnnotationPreprocessor** (``aaanalysis[pro]``): Fetches from UniProt (or
  ingests user / predictor labels) per-residue PTM and functional-site
  annotations and encodes them into per-residue tensors. Methods:
  ``fetch_uniprot``, ``ingest``, ``register_feature``, ``encode``,
  ``build_scales``, ``build_cat``, ``to_df_seq``.
- **combine_dict_nums**: Concatenates multiple per-residue tensors
  (embeddings / structure / annotation) along the feature axis to build a
  combined ``CPP.run_num`` input.

**Feature Engineering**

- **CPPGrid**: ``Tool``-style wrapper (``run`` + ``eval``) that runs a grid sweep
  of ``CPP`` configurations in one call, parallelized across configurations.
  Configurations that differ only in ``n_filter`` are collapsed into a single CPP
  run, with the remaining configurations served as exact ``head(n)`` slices.
  ``run`` also stores ``list_df_feat_`` / ``df_params_``; ``eval(sort_by=...)``
  scores the configurations (by ``avg_ABS_AUC`` by default) and returns them
  best-first.
- **CPP.run_num**: New numerical-mode method whose per-residue value source is a
  pre-sliced numerical tensor (``dict_num_parts``) rather than an amino-acid →
  scale lookup, enabling embedding / structure / annotation features through the
  same pipeline and output schema as ``CPP.run``.
- **SequenceFeature.get_labels_ovr / get_labels_ovo**: Convert multi-class
  ``labels`` into binary label sets for ``CPP`` — one-vs-rest (K full-length
  arrays, all samples kept) or one-vs-one (per class-pair). The row-dropping
  ``get_labels_ovo`` takes the value source (``df_parts`` and/or ``dict_num_parts``)
  and returns each pair's row-matched copy ready for ``CPP.run`` / ``CPP.run_num``.
- **SequenceFeature.get_labels_quantile / get_labels_tiered**: Discretize a
  continuous target into binary ``labels`` for regression-style ``CPP`` — a single
  quantile cut (all samples kept), or a fixed positive set swept against
  stepwise-lowered negative cuts, ``get_labels_tiered`` returning each tier's
  row-matched ``df_parts`` / ``dict_num_parts`` subset.
- **SequenceFeature.get_df_parts_from_windows**: Assemble a reference ``df_parts``
  from per-part window sets (e.g. ``AAWindowSampler.sample_synthetic`` output), so
  each sequence part can be generated with its own recipe.
- **SequenceFeature.get_feature_descriptions**: Build one standardized,
  human-readable sentence per ``PART-SPLIT-SCALE`` feature id, combining the
  sequence region, the split (e.g. ``"segment 2 of 4"``), and the AAontology scale
  name, category, and subcategory. Complements the compact ``get_feature_names``
  label; the description is additive (the ``'feature'`` id is unchanged) and can be
  assigned to an optional ``'feature_description'`` ``df_feat`` column.
- **AAclust.select_scales**: Convenience wrapper around ``AAclust.fit`` that takes
  an amino acid scales DataFrame (rows = amino acids, columns = scale IDs) and
  returns the redundancy-reduced subset of its columns (one medoid scale per
  cluster) directly — collapsing the manual transpose, ``names`` bookkeeping, and
  medoid-name indexing into a single call ready for ``CPP``.
- **AAclust.select_proteins**: Protein-level redundancy reduction over a pre-pooled
  per-protein feature matrix (``X``: CPP features, pooled embeddings, or structural /
  DSSP-derived features). Clusters the proteins and selects one representative (medoid)
  per cluster, annotating ``df_seq`` with ``cluster`` / ``is_representative`` /
  ``dist_to_rep`` (``return_data='annotated' | 'filtered' | 'both'``) — the numerical
  counterpart to sequence-identity reduction via ``filter_seq``.

**Sequence Analysis**

- **AAWindowSampler**: Samples fixed-length sequence windows for PU-learning and
  hard-negative-mining workflows (``sample_same_protein``,
  ``sample_different_protein``, ``sample_motif_matched``, ``sample_synthetic``).
- **scan_motif** (``aaanalysis[pro]``): scans candidate proteins for
  statistically significant PWM occurrences via MEME/FIMO (selection by match
  p-value against a background model), complementing the pure-Python
  ``AAWindowSampler.sample_motif_matched`` PWM-sum sampler.

**Metrics**

- **comp_per_protein_ap**: Per-protein average precision for site-localization
  ranking, with an optional ``tolerance=±k`` variant for positional jitter.
- **comp_detection_metrics**: Recall / precision / F1 / MCC at a fixed score
  threshold, pooled across per-residue predictions.
- **comp_bootstrap_ci**: Seeded percentile confidence interval over a
  per-protein metric vector for small-N uncertainty reporting. Returns a dict
  ``{'mean', 'ci_low', 'ci_high'}``.
- **comp_smooth_scores**: Peak-preserving (``max(smoothed, raw)``), NaN-aware
  smoothing of per-residue score tracks.

**Plotting**

- **plot_rank**: Standalone per-protein max-score-vs-rank scatter with group
  coloring and optional threshold lines (pairs with the new ``aa.metrics``
  functions).

**Package**

- **aa.__version__**: The installed package version is now exposed as a
  top-level attribute via ``importlib.metadata``.
- **CHANGELOG.md + deprecation policy**: A root ``CHANGELOG.md``
  (`Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_ format) now gives a
  terse, developer-facing index alongside these narrative notes. The project
  adopts strict semantic versioning: from v1.x onward, any rename or removal of a
  public symbol ships at least one minor release carrying a ``DeprecationWarning``
  first. A ``deprecated(reason, version_removed)`` decorator helper (internal,
  ``aaanalysis.utils``) marks such symbols and prepends a deprecation note
  to their docstring. See the *Versioning and Deprecation Policy* in
  ``CONTRIBUTING.rst``.

**Documentation**

- **Prediction tasks** concept-overview page (*Usage Principles*): maps a
  biological question to the right AAanalysis workflow via a task table keyed on
  *unit of comparison* and *reference construction* (not biological scale alone),
  covering the residue / domain / protein levels plus the determinant-discovery,
  design/engineering, and relational-boundary rows. The front door to the
  Protocols catalog; taxonomy recorded in ADR-0022.
- **A minimal CPP analysis** tutorial (``tutorial0_minimal``): the shortest
  end-to-end loop — load a dataset, run CPP, read out the signature — paired with
  the new concept page.

Changed
~~~~~~~

- **CPP performance work**: The Cython feature-matrix kernel, macOS-safe threaded
  ``n_jobs``, scale / AA-index caching, and scale / sample batching land in this
  release, replacing the hour-long, low-CPU CPP runs seen on ``1.0.3`` and
  earlier. **Users on** ``≤1.0.3`` **should upgrade** rather than debug a
  performance pathology that is already fixed.
- **CPP Cython-fallback notice**: When the compiled extension is missing and CPP
  falls back to the ~2× slower pure-Python kernel, the one-time notice is now a
  ``UserWarning`` instead of an easily-missed INFO print, so it surfaces even with
  ``aa.options['verbose'] = False``.
- **SequenceFeature.feature_matrix**: New ``batch=`` parameter accepts a list of
  ``df_parts`` and builds them in a single Cython pass, returning a list of
  feature matrices — faster than per-call construction for many small part tables.
- **SequenceFeature.get_df_parts / NumericalFeature.get_parts**: New ``pos``-anchor
  input mode (``tmd_len=``) explodes each 1-based anchor in the ``pos`` column
  into one three-part (``jmd_n`` / ``tmd`` / ``jmd_c``) row, identified by
  ``entry_win``.
- **SequenceFeature.get_df_parts**: Several-fold faster on large inputs — the
  per-row ``DataFrame.apply`` driver was replaced with a vectorized iteration over
  the raw column arrays. The output (parts, column order, index, values) is
  unchanged.
- **CPP / feature-engineering same-output speedups**: Three byte-identical
  optimizations. ``SequenceFeature.prune_by_correlation`` /
  ``NumericalFeature.filter_correlation`` vectorize the inner correlation-triangle
  comparison while preserving the greedy, order-dependent skip (the selected mask
  is unchanged). ``CPP.simplify``'s redundancy reduction replaces a per-pair double
  pandas lookup into the scale-correlation table with a numpy view built once,
  keeping the sequential greedy tie-break (kept set and order unchanged). The greedy
  swap loop drops a per-candidate full-matrix copy in favor of a single mutated-column
  save/restore (memory only; scored matrix and selected set unchanged).
- **n_jobs**: Unified parallelism convention across ``CPP`` / ``CPPGrid``
  (``1`` serial, ``-1`` all cores, ``N>1`` exactly N, ``None`` optimized), with an
  ``options['n_jobs']`` global override.
- **CPPPlot.feature**: Now titles the plot with the feature's human-readable
  description (from ``SequenceFeature.get_feature_descriptions``), line-wrapped via
  the new ``show_title`` (default ``True``) and ``title_wrap_width`` (default ``45``)
  parameters. A subsequent ``plt.title(...)`` still overrides it; ``feature_map`` and
  ``ranking`` are unchanged.
- **Docstring discoverability**: Surfaced previously implicit API contracts at the
  docstrings users actually read (no behavior change). ``CPP.run_num`` /
  ``NumericalFeature.get_parts`` now state the ``get_parts`` → ``run_num`` call order
  and the ``[0, 1]`` normalization contract (and what breaks if unnormalized); the
  ``[pro]`` classes / functions (``ShapModel``, ``StructurePreprocessor``,
  ``AnnotationPreprocessor``, ``comp_seq_sim``, ``filter_seq``, ``scan_motif``) carry
  a ``[pro]`` install marker in their summary; and ``SeqMut`` cross-links the canonical
  ``df_seq`` format spec (``SequenceFeature.get_df_parts``).
- **Performance (same output)**: Several internal hotspots were vectorized or
  parallelized without changing results. ``AAWindowSampler`` redundancy /
  similarity filtering now compares amino-acid windows with vectorized NumPy
  operations (identical keep/drop decisions; ~30x faster at scale), ``AAclust``
  sample-to-medoid correlation distances are computed in one pass, and the
  per-feature Kullback-Leibler divergence (used by ``dPULearn.eval`` with
  ``comp_kld=True``) is parallelized over features and honors
  ``options['n_jobs']``. Public APIs and outputs are unchanged.
  A further pass vectorizes ``AAWindowSampler`` window-sampling internals:
  candidate-center band filtering (~40x faster at scale) and per-window PWM
  scoring in ``sample_motif_matched`` (~12x), again with identical results.
  ``SequencePreprocessor.encode_one_hot`` is also vectorized (~3x), with a
  byte-identical feature matrix. And ``StructurePreprocessor.encode_pdb`` CA-CA
  contact counts (``contact_count_8A`` / ``contact_count_12A``) use a vectorized
  per-residue distance computation (~50x at scale) with byte-identical counts.
  ``encode_pdb`` additionally caches the per-(target, atom) global sequence
  alignment that its encoders otherwise re-run ~26 times per entry (chain pick
  plus each per-feature value mapping); the first optimal alignment is
  deterministic, so cached and recomputed encoder output are byte-identical
  (~12x off the repeated-alignment overhead).
- **Pooled, optionally concurrent web fetches**:
  ``StructurePreprocessor.fetch_alphafold`` and
  ``AnnotationPreprocessor.fetch_uniprot`` now route every request through a
  pooled ``requests.Session`` (one per worker thread) rather than opening a
  fresh connection per request, and accept a new ``max_workers`` parameter for
  threaded bulk fetching. Concurrency is **off by default** (``max_workers=None``
  or ``1`` keeps the unchanged sequential path) because parallel requests to
  AlphaFold DB / UniProt risk HTTP-429 throttling; when enabled, results are
  reassembled in input order, so the returned status table / ``df_annot`` and
  the on-disk files are byte-identical regardless of worker count.
- **dPULearn.fit**: Flexible, package-consistent label handling via ``label_pos`` /
  ``label_unl`` / ``label_neg`` markers. Pass standard ``{0, 1}`` labels directly with
  ``label_unl=0`` (``0`` = unlabeled, ``1`` = positive), or any positive / unlabeled /
  negative encoding. **Only unlabeled samples are candidates** — pre-labeled negatives
  (``label_neg``) are kept and never re-selected. The negative count is specified one of
  two ways (exactly one): the new ``n_neg`` (the **total** number of negatives wanted, so
  dPULearn identifies ``n_neg`` minus the pre-labeled negatives), or the existing
  ``n_unl_to_neg`` (the number identified **directly from the unlabeled pool**, for direct
  control). Output labels always use the package convention (``1`` = positive, ``0`` =
  negative, ``2`` = unlabeled); the recommended input encoding is unchanged.
- **Numerical-equivalence tolerance policy** (developer-facing): A new policy
  (ADR-0032, summarized in ``CONTRIBUTING.rst``) defines three tiers of acceptable
  output change for performance optimizations — **T1 byte-identical** (default),
  **T2 numerically-equivalent** (``allclose(atol=1e-10, rtol=0)`` plus identical
  discrete decisions), and **T3 statistically-equivalent** (documented quality
  metric within an agreed band) — and the evidence + pinned regression anchor each
  tier requires. It unblocks previously-excluded algorithmic optimizations (e.g.
  AAclust binary-search ``k``), each landing as its own tier-declared PR. No user-
  facing behavior changes in this release.


Version 1.0 (Stable Version)
--------------------------------

v1.0.3 (2026-04-06)
--------------------------------

Added
~~~~~
- **AAlogo**: New class for amino acid logo visualization.
- **AAlogoPlot**: New plotting class for AAlogo visualizations.

Changed
~~~~~~~
- **Python Support**: Dropped Python 3.9 (end-of-life) and added Python 3.13 and 3.14 support.
  Supported versions are now 3.10, 3.11, 3.12, 3.13, and 3.14.
- **Dependency Management**: Migrated from ``requirements.txt`` files to a single
  ``pyproject.toml`` as the source of truth for all dependencies. Introduced structured
  dependency extras: ``aaanalysis[pro]``, ``aaanalysis[docs]``, and ``aaanalysis[dev]``.
- **Package Manager**: Added full ``uv`` support alongside existing ``pip`` and ``Poetry``
  compatibility.
- **CI/CD**: Updated all GitHub Actions workflows to reflect new Python version matrix
  and consolidated dependency installation via extras.

Other
~~~~~
- **Documentation**: Updated ``ReadTheDocs`` configuration to install dependencies
  directly from ``pyproject.toml`` via ``aaanalysis[docs]`` extra.
- **Cleanup**: Removed legacy ``requirements.txt``, ``docs/requirements_dev.txt``,
  and ``docs/requirements_wo_pro.txt`` files.


v1.0.2 (2025-06-17)
--------------------------------

Improved
~~~~~~~~
- **Faster CPP Pipeline**: Major performance boost in ``CPP.run()`` through optimized generation and filtering of
  part-split-scale combinations. Depending on the number of scales, runtime is now **3–5× faster** on standard hardware.
- **Feature Map Enhancement**: ``CPP.feature_map()`` now includes a **top bar plot** showing cumulative feature importance
  per residue, improving interpretability. This visualization is also included in the CPP profile output.

Fixed
~~~~~
- **StructurePreprocessor.fetch_alphafold**: Resolve download URLs through the
  AlphaFold API instead of a hardcoded file version. AlphaFold DB renamed its
  files ``v4`` → ``v6``, which had silently broken every fetch (all entries
  returned ``alphafold_ok=False``); the fetch now tracks the current version
  automatically. Added a ``network``-marked live test (``tests/integration/``)
  so an upstream API/version change is caught instead of slipping past the
  mocked unit tests.
- **General Bug Fixes**: Minor fixes related to dependency resolution and edge-case behavior.
- **Documentation**: Removed inconsistencies in documentation for selected functions and plotting options.

Other
~~~~~
- **Branding**: Introduced updated logo and favicon (legacy version preserved under `docs/source/_artwork/logos/legacy/`).
- **Landing Page Visual**: Added a main conceptual sketch to the documentation landing page illustrating the core CPP idea
  — comparing two sequence sets to derive their critical difference, the **physicochemical signature**.


v1.0.1 (2025-01-29)
--------------------------------

Improved
~~~~~~~~
- **Pro Feature Accessibility**: Improved integration of **aaanalysis[pro]** features in IDEs. Clicking on a pro
  feature now directs users to its exact class implementation instead of the main ``__init__.py`` file.

- **Import Error Handling**: Improved error handling for missing dependencies in the **aaanalysis[pro]** version.
  If dependencies are installed but errors occur during import, users now receive the original import error messages.

Fixed
~~~~~
- **Feature Map Plot**: Resolved a potential mismatch in subcategory ordering between heatmap and bar plot
  in ``aa.cpp_plot().featuremap()``. Previously, subcategories with nearly identical names (e.g., "α-helix (C-term)"
  and "α-helix (C-term, out)") could appear in an inconsistent order.
- **General Bug Fixes**: Minor bug fixes to improve overall stability and functionality.

Other
~~~~~
- **Dependencies**: All dependencies have been updated to ensure compatibility with the latest versions, including
  full support for ``numpy>=2.0.0``.


v1.0.0 (2024-07-01)
--------------------------------

Added
~~~~~
- **SequencePreprocessor**: A utility data preprocessing class (data handling module).
- **comp_seq_sim**: A function for computing pairwise sequence similarity (data handling module).
- **filter_seq**: A function for redundancy-reduction of sequences (data handling module).
- **options**: Juxta Middle Domain (JMD) length can now be globally adjusted using the **jmd_n/c_len** options.

Changed
~~~~~~~
- **ShapModel**: The **ShapExplainer** class has been renamed to **ShapModel** for consistency with the **TreeModel**
  class and to avoid confusion with the ShapExplainer models from the
  `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ package.
- **Dependencies**: Biopython is now a required dependency only for the **aaanalysis[pro]** version.
- **Module Renaming**: The **Perturbation** module has been renamed to **Protein Design** module
  to better reflect its broad functionality.

Fixed
~~~~~
- **Multiprocessing**: Now supported directly at the script level, outside of any functions or classes,
  in the top-level of the script (global namespace).

Version 0.1 (Beta Version)
--------------------------

v0.1.5 (2024-04-18)
-------------------

Added
~~~~~
- **Code of Conduct**: Introduced a Code of Conduct to foster a welcoming and inclusive community environment.
  We encourage all contributors to review the `Code of Conduct <https://github.com/breimanntools/aaanalysis/blob/master/CODE_OF_CONDUCT.md>`_
  to understand the expectations and responsibilities when participating in the project.

Changed
~~~~~~~
- **License Update**: Transitioned the project license from MIT to `BSD-3-Clause <https://github.com/breimanntools/aaanalysis/blob/master/LICENSE>`_
  to better align with our project's community engagement and protection goals. This change affects how the software
  can be used and redistributed.

Fixed
~~~~~
- **Multiprocessing**: Replaced native ``multiprocessing`` with the ``joblib`` module for **CPP** and
  **internal feature matrix** creation. This change prevents a ``RuntimeError`` that occurred when the main function
  is not explicitly used.

Other
~~~~~
- **Dependencies**: Update the ``seaborn`` dependency to version 0.13.2 or higher to resolve the legend argument
  error present in versions earlier than 0.13

v0.1.4 (2024-04-09)
-------------------

Added
~~~~~
- **Installation Options**: Introduced separate installation profiles for the core and professional versions.
  The **core version** has reduced dependencies to enhance installation robustness, installable using ``pip install aaanalysis``.
  The **professional version**, designed for advanced usage, includes packages required for our explainable AI module
  such as SHAP, installable using ``pip install aaanalysis[pro]``.

Changed
~~~~~~~
- **API Improvements**: General improvement of API for consistency and higher user-friendliness.

Fixed
~~~~~
- **General Issues**: Fix of different check function related API issues.

Other
~~~~~
- **Python Dependency**: Updated the Python version compatibility from <= 3.10 to <= 3.12.

v0.1.3 (2024-02-09)
-------------------

Added
~~~~~
- **TreeModel**: Wrapper class of tree-based models for Monte Carlo estimates of predictions and feature importance.
  `See TreeModel <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.TreeModel.html>`_.
- **ShapExplainer**: A wrapper for SHAP (SHapley Additive exPlanations) explainers to obtain Monte Carlo estimates for
  feature impact. `See ShapExplainer <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.ShapExplainer.html>`_.
- **NumericalFeature**: Utility feature engineering class to process and filter numerical data structures.
  `See NumericalFeature <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.NumericalFeature.html>`_.
- **Load_feature**: Utility function to load feature sets for protein benchmarking datasets.
  `See load_features <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_features.html>`_.


Changed
~~~~~~~
- **API Improvements**: General improvement of API for consistency and higher user-friendliness.

Fixed
~~~~~
- **Interface**: Change of internal documentation decorator to hard-coded documentation for better IDE responsiveness.
- **General Issues**: Fix of different check function related API issues.

v0.1.2 (2023-11-06)
-------------------

Added
~~~~~
- **CPPPlot**: Plotting class for CPP features.
  `See CPPPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPPPlot.html>`_.
- **dPULearnPlot**: Plotting class for results of negative identifications by dPULearn.
  `See dPULearnPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearnPlot.html>`_.
- **AAclustPlot**: Plotting class for AAclust clustering results.
  `See AAclustPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.AAclustPlot.html>`_.
- **Options**: Set system-level settings by a dictionary-like interface (similar to pandas).
  `See options <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.options.html>`_.
- **Plotting functions**: Extension of plotting utility functions.

Changed
~~~~~~~
- **API Improvements**: General improvement of API.

Fixed
~~~~~
- **API Improvements**: General improvement of API (Application Programming Interface).

Other
~~~~~
- **Python Dependency**: Supports Python versions 3.9 and 3.10.

v0.1.1 (2023-09-11)
-------------------
Test release of the first beta version.

v0.1.0 (2023-09-11)
-------------------
First release of the beta version including
`CPP <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPP.html>`_,
`dPULearn <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearn.html>`_,
and `AAclust <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.AAclust.html>`_ algorithms
as well as the
`SequenceFeature <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.SequenceFeature.html>`_
utility class and data loading functions
`load_dataset <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_dataset.html>`_
and `load_scales <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_scales.html>`_.
