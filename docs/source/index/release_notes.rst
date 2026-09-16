.. _release_notes:

Release Notes
=============

Version 1.2
--------------------------------

v1.2.0 (Unreleased)
--------------------------------

In development.

Tools that are still under active development are marked beta: their API may change
between minor releases without the usual deprecation cycle. The
:ref:`Beta Features <beta_features>` page lists every beta tool in one place.

Added
~~~~~
- :class:`~aaanalysis.DesignConstraints`: one validated container for the design limits that
  :class:`~aaanalysis.AAMut`, :class:`~aaanalysis.SeqMut` and :class:`~aaanalysis.SeqOpt` all
  express — immutable and mutable positions, permitted and forbidden substitutions, a mutation
  budget, identity bounds to the parent, and required or forbidden motifs. Positions are 1-based
  over the parent sequence.
- :meth:`~aaanalysis.DesignConstraints.check` returns ``(ok, reasons)`` so a rejected candidate
  explains itself, :meth:`~aaanalysis.DesignConstraints.as_predicate` adapts the same limits to the
  feasibility callable :meth:`~aaanalysis.SeqOpt.run` consumes, and ``to_dict`` / ``from_dict``
  round-trip a constraint set through JSON.
- All three design classes accept the object as ``constraints``. The ``region``, ``to_aa`` and
  ``n_mut_max`` parameters keep their meaning and are now shorthand that builds one internally, so
  a limit has a single definition; combining a shorthand with an object that sets the same limit
  differently raises. Results are unchanged when no object is passed.
- :meth:`~aaanalysis.AAPred.eval_selective`: the risk-coverage trade-off as a table. Samples are
  ranked by a per-sample confidence signal and every metric is scored again on the most-confident
  fraction, at each level of a coverage grid, so a refusal threshold can be chosen from evidence.
  The confidence source is the caller's (``confidence=``); the default is the out-of-fold score
  margin. The ``coverage=1.0`` row is the ordinary out-of-fold score. Classification only, and a
  measurement only: nothing abstains and :meth:`~aaanalysis.AAPred.eval` is untouched.
- **CPP feature intervals**: ``CPP(bootstrap=True, bootstrap_kws=dict(ci=0.95))`` retains the
  statistics each resampling round already computes and summarises them into a percentile interval
  per feature, adding ``abs_auc_ci_low`` / ``_high`` and ``mean_dif_ci_low`` / ``_high`` to
  ``df_feat`` at no extra runs. The interval is conditional on selection, so it is read together
  with ``selection_frequency``. Leaving ``ci`` unset keeps the output unchanged.
- :meth:`~aaanalysis.ReliabilityModel.eval` gains ``use_calibrated`` and ``add_metrics``, which
  score the calibrated column and append Brier score and expected calibration error, so comparing
  the raw and calibrated tables shows whether calibration helped. Defaults leave the table
  unchanged. :meth:`~aaanalysis.ReliabilityModelPlot.reliability_diagram` gains ``label``,
  annotates both metrics in the legend, and lets a raw and a calibrated curve share one axis.
- :meth:`~aaanalysis.ReliabilityModel.fit` now warns when ``calibrate=True`` cannot be honoured,
  for example too few members in a class, instead of failing silently.
- **Banded applicability domain**: :meth:`~aaanalysis.ReliabilityModel.predict` appends
  ``ad_status`` (``inside`` / ``borderline`` / ``outside`` / ``unknown``) and ``ad_nearest_train``,
  and :meth:`~aaanalysis.ReliabilityModel.fit` gains ``ad_borderline`` and exposes the fitted
  ``ad_threshold_`` and ``ad_method_``. Apart from the ``ad_knn_dist`` to ``ad_knn`` rename,
  existing columns are unchanged.
- :meth:`~aaanalysis.ReliabilityModel.predict_candidates`: scores a designed candidate set in one
  call. It takes the table the design methods emit, rebuilds the feature matrix from the wild-type
  coordinates in ``df_seq``, and delegates to :meth:`~aaanalysis.ReliabilityModel.predict`. The
  result is row-aligned with the candidates, so the two join directly.
- :meth:`~aaanalysis.NumericalFeature.from_pssm`: position-specific scoring matrices as a CPP value
  source. It reads PSI-BLAST ASCII files and precomputed arrays, reorders file columns into
  canonical amino acid order, and with ``return_scales=True`` also returns the matching
  ``df_scales`` and ``df_cat``, so a matrix runs through :meth:`~aaanalysis.CPP.run_num` unchanged.
- :meth:`~aaanalysis.SequenceFeature.get_split_kws` gains a ``strategy`` preset:
  ``"compositional"`` returns the whole-part segment split and ``"positional"`` the sub-segments
  plus patterns. Together they cover the default split set; ``strategy=None`` is unchanged, and a
  preset cannot be combined with non-default split arguments.
- :meth:`~aaanalysis.ModelEvaluator.learning_curve` answers whether a task is sampling-limited: it
  repeats the cross-validation on nested subsets of each training fold, scores on the untouched
  test fold, and returns one row per model, size and metric with a bootstrap interval. At the full
  fraction it reproduces :meth:`~aaanalysis.ModelEvaluator.run` exactly for the same settings.
  :meth:`~aaanalysis.ModelEvaluatorPlot.learning_curve` draws it with the band.
- :meth:`~aaanalysis.AAPredPlot.group_cluster` gains ``kind='dendrogram'`` with a rectangular or
  circular ``layout``, coloured by the existing annotations. It shares the clustermap's linkage, so
  both kinds show the same topology.
- The :ref:`Data Schemas <df_schemas>` page now documents the prediction outputs downstream tools
  read: ``df_pred``, ``df_rel`` and ``df_eval_reliability``. Contract tests pin their column names,
  order and dtypes, so renaming, dropping, retyping or undocumenting one fails the suite.

Changed
~~~~~~~
- **Consistency pass on the prediction and design tier.** These classes are marked beta, so the
  changes land without a deprecation cycle: :meth:`~aaanalysis.ReliabilityModel.fit` takes ``ci``
  as a fraction in ``(0, 1)`` rather than a percent, and a percent now raises; ``ad_knn_dist`` is
  renamed ``ad_knn`` and the ``n`` column of
  :meth:`~aaanalysis.ReliabilityModel.eval` is renamed ``n_samples``;
  :class:`~aaanalysis.SeqOpt` defaults to ``mode="importance"``, so it constructs in a base
  install; :meth:`~aaanalysis.AAPred.eval` accepts ``"mcc"``, giving
  :class:`~aaanalysis.AAPred` and :class:`~aaanalysis.ModelEvaluator` one metric vocabulary; and
  both classes now validate ``df_scales`` and ``list_parts`` instead of ignoring an invalid value.
- :meth:`~aaanalysis.CPP.run`: ``n_sample_batches`` creates exactly the requested number of
  balanced, non-empty batches, and the multiple-testing correction is now pooled across scale
  batches, so a batched run matches the single-pass result.

Fixed
~~~~~
- :meth:`~aaanalysis.ReliabilityModel.fit` rejects non-finite values for ``ci``,
  ``ad_percentile`` and ``conformal_alpha``. A ``NaN`` passed both range comparisons silently and
  produced ``NaN`` interval columns.
- :meth:`~aaanalysis.ReliabilityModel.eval`: passing ``X`` without ``labels`` now raises, where the
  features were silently ignored. Passing ``labels`` alone scores the training features against
  that labelling.
- :meth:`~aaanalysis.ReliabilityModelPlot.reliability_diagram` validates its arguments in the
  frontend, so an invalid value names the parameter instead of raising from matplotlib.
- :meth:`~aaanalysis.StructurePreprocessor.encode_pae` reads the PAE JSON exactly as the AlphaFold
  Database serves it, so downloaded files load without a manual unwrap, and
  :meth:`~aaanalysis.StructurePreprocessor.get_domains` reads the ``ClusteringResult`` that current
  AFragmenter releases return.
- **Plot legibility**: the total and key of :meth:`~aaanalysis.CPPPlot.ranking` no longer overprint
  short bars; :meth:`~aaanalysis.CPPPlot.eval` caps its annotations to the bar height;
  :class:`~aaanalysis.SeqOptPlot` fixes the convergence y-label, a taller mutation map and
  single-front colouring; long condition names rotate or wrap in the comparison charts; an explicit
  ``cbar_xywh`` is honoured by :meth:`~aaanalysis.CPPPlot.feature_map` and
  :meth:`~aaanalysis.CPPPlot.heatmap`; and :class:`~aaanalysis.AALogoPlot` keeps P-site and part
  labels from overprinting.
- **Quiet by default**: the structure and embedding preprocessors no longer echo third-party probe
  messages, progress bars or dataset-dependency notes to stderr, and
  :class:`~aaanalysis.ShapModel` runs its explainer silently.
- **Notebooks**: every table is shown with ``display_df``, the structure and UniProt examples are
  real executed walkthroughs instead of commented-out stubs, and several notebooks were repaired to
  pass ``nbformat`` validation.

Documentation
~~~~~~~~~~~~~
- New tutorial *CPP with protein language model embeddings and AlphaFold structures*: the
  recommended embedding and structure paths into :meth:`~aaanalysis.CPP.run_num`, and painting the
  signature onto the 3D model.
- New usage-principles page :ref:`Golden Pipelines <golden_pipelines>`: what the
  :mod:`aaanalysis.pipe` layer is for, its shared return shape, and the parity anchors that keep it
  honest against the explicit path. :func:`~aaanalysis.pipe.find_features` documents that only its
  fast search is parity-anchored.
- Protocols *P2: Exploratory sequence analysis* and *P3: Sampling* brought to the protocol quality
  rubric: each opens with a mental model and shows every concept it names as a contrast figure.
- :meth:`~aaanalysis.CPP.run`: the ``n_sample_batches`` documentation no longer claims that peak
  memory is bounded by the batch size. Measurements show it bounds the dominant per-batch tensor
  while the survivor matrix stays resident, so peak memory still grows with the sample count, on a
  far flatter slope.

Version 1.1
--------------------------------

v1.1.0 (2026-09-10)
--------------------------------

Added
~~~~~
- **Per-residue input preprocessors**: :class:`~aaanalysis.EmbeddingPreprocessor` (protein language
  model embeddings), :class:`~aaanalysis.StructurePreprocessor` (``[pro]``; PDB / CIF / AlphaFold and
  PAE files) and :class:`~aaanalysis.AnnotationPreprocessor` (``[pro]``; UniProt records) turn each
  source into the per-residue tensors :meth:`~aaanalysis.CPP.run_num` consumes.
- **Input helpers**: :func:`~aaanalysis.combine_dict_nums` concatenates per-residue tensors from
  several sources, :func:`~aaanalysis.get_labels` derives a binary label vector from a sequence
  DataFrame, :meth:`~aaanalysis.SequencePreprocessor.pad_parts` pads part columns to equal length,
  and every bundled dataset now carries a human-readable gene name.
- :func:`~aaanalysis.get_provenance`: opt-in, JSON-serializable record of how a result was
  produced (versions, parameters and the effective seed), so a figure can be traced back to the run
  behind it.
- :class:`~aaanalysis.SequenceFeatureTransformer`: scikit-learn transformer that runs CPP feature
  selection inside a pipeline, fitting the selection on the training fold only, so cross-validated
  scores stay honest instead of being inflated by selection on the full set.
- :class:`~aaanalysis.CPPGrid`: tool-style wrapper (``run`` + ``eval``) that sweeps CPP
  configurations in parallel and returns one evaluation table across the grid, so a configuration
  choice is made on evidence rather than by hand.
- :meth:`~aaanalysis.CPP.run_num`: numerical mode sourcing per-residue values from a pre-sliced
  tensor, and :meth:`~aaanalysis.CPP.run_composit`: composition mode scoring amino-acid, dipeptide
  and k-mer descriptors with CPP's discriminative statistics.
- **CPP stability and cost controls**: opt-in bootstrap / stability annotation (``bootstrap``,
  ``bootstrap_kws``) adding ``selection_frequency``, ``redundancy='legacy'|'exact'`` on
  :meth:`~aaanalysis.CPP.run` / :meth:`~aaanalysis.CPP.run_num`, and
  ``candidate_search='fast'`` on :meth:`~aaanalysis.CPP.simplify`.
- **Label helpers on** :class:`~aaanalysis.SequenceFeature`: ``get_labels_ovr`` / ``get_labels_ovo``
  convert multi-class labels to binary settings, ``get_labels_quantile`` / ``get_labels_tiered``
  discretize a continuous target.
- **Baseline featurizers on** :class:`~aaanalysis.SequenceFeature`: ``scale_composition``,
  ``aa_composition``, ``dipeptide_composition``, ``kmer_composition`` and the order-aware
  ``acc`` (scale auto-covariance), so CPP features can be compared against non-positional
  baselines. :meth:`~aaanalysis.NumericalFeature.feature_matrix` builds the matrix for
  :meth:`~aaanalysis.CPP.run_num`-selected features.
- **Part and description helpers**: :meth:`~aaanalysis.SequenceFeature.get_df_parts_from_windows`
  assembles a reference ``df_parts`` from windows, :meth:`~aaanalysis.SequenceFeature.get_seq_kws`
  returns one protein's part sequences, and
  :meth:`~aaanalysis.SequenceFeature.get_feature_descriptions` gives each feature a standardized,
  human-readable description.
- **sample_kws bundle**: the CPP plots take one bundle of sample-selection keywords instead of
  several flat parameters.
- **Scale and protein selection on** :class:`~aaanalysis.AAclust`: ``pre_select_scales`` filters by
  AAontology metadata before clustering, ``select_scales`` returns the redundancy-reduced set
  directly, and ``select_proteins`` reduces redundancy over a per-protein matrix.
  :class:`~aaanalysis.AAclustPlot` ``centers`` / ``medoids`` accept ``df_scales``.
- :class:`~aaanalysis.ReliabilityModel` and :class:`~aaanalysis.ReliabilityModelPlot`: per-sample
  prediction reliability. One call per sample reports calibrated score, ensemble spread, a
  confidence interval, applicability-domain distance and a conformal set, so a prediction carries
  how much it can be trusted. The plot class renders the calibration curve and the trust axes.
- :class:`~aaanalysis.ModelEvaluator` and :class:`~aaanalysis.ModelEvaluatorPlot`: model-agnostic
  evaluation harness. Repeated stratified cross-validation over several seeds, bootstrap intervals
  per metric, and a paired comparison on identical folds, so two models are separated by evidence
  rather than by a single split.
- **New** :class:`~aaanalysis.AAPred` **capabilities**: ``eval(baseline=...)`` compares CPP features
  against composition baselines, ``eval(cv=...)`` accepts an arbitrary scikit-learn splitter,
  ``predict_oof`` returns cross-validated out-of-fold per-sample scores, and ``score_to_group``
  maps scores to named confidence bands. :class:`~aaanalysis.AAPredPlot` gains
  ``eval(kind='heatmap')`` and ``predict_group(kind='rank_scatter')``.
- :class:`~aaanalysis.ShapModel` (``[pro]``): accession-based ``fit`` interface, and an unbiased
  fuzzy estimator that is now the default.
- :class:`~aaanalysis.CPPStructurePlot` (``[pro]``): paints per-residue CPP and CPP-SHAP impact onto
  a 3D structure.
- **PU learning**: :meth:`~aaanalysis.dPULearn.fit` accepts a positives / unlabeled split directly,
  :meth:`~aaanalysis.dPULearn.project` projects held-out samples into the same space, and
  :class:`~aaanalysis.AAWindowSampler` samples fixed-length windows for PU learning.
- :func:`~aaanalysis.scan_motif` (``[pro]``): scans candidate proteins for statistically
  significant position-weight-matrix motif hits, reporting each hit with its match p-value.
- **Protein engineering**: :class:`~aaanalysis.SeqOpt` and :class:`~aaanalysis.SeqOptPlot` perform
  multi-objective, machine-learning-guided directed evolution over one wild-type (core; only
  ``mode="impact"`` needs ``[pro]``). :class:`~aaanalysis.SeqMut` gains a model-guided mode and
  ``combine`` for multi-mutation variants, and :class:`~aaanalysis.SeqMutPlot` renders the
  prediction-shift landscape.
- **Metrics**: :func:`~aaanalysis.comp_per_protein_ap`, :func:`~aaanalysis.comp_detection_metrics`,
  :func:`~aaanalysis.comp_bootstrap_ci` and :func:`~aaanalysis.comp_smooth_scores` for
  site-localization ranking and thresholded detection.
- **Plot sizing**: ``cell_size`` holds every grid cell at an exact physical size, ``seq_size``
  defaults to ``"auto"`` and fits residue letters to the cell, and ``fontsize_labels`` gains
  ``"auto"``, which tracks the font scale and shrinks on overlap.
- **Named sample colors**: ``COLOR_SAMPLES_POS`` / ``_NEG`` / ``_UNL`` / ``_REL_NEG`` expose the
  canonical group colors as public constants.
- ``options['plot_settings']``: opt-in, session-persistent :func:`~aaanalysis.plot_settings`, so
  every subsequent figure adopts the publication style automatically.
- :mod:`aaanalysis.pipe` (``ap``): an opt-in convenience API of stateless, one-call pipelines —
  ``find_features`` (staged CPP AutoML, with ``selection_scope="global"|"fold"`` for honest
  evaluation), ``predict_samples``, ``explain_features`` and ``plot_eval``.
- ``aa.__version__`` exposes the installed version, and a root ``CHANGELOG.md`` gives a terse,
  developer-facing index alongside these notes.

Changed
~~~~~~~
- **Module rename**: ``protein_design`` is now ``protein_engineering``.
- :class:`~aaanalysis.AAPred`: capability-based estimator validation, so a model is rejected at
  construction rather than mid-run.
- :class:`~aaanalysis.TreeModel`: per-round seeding fix, so a fixed ``random_state`` reproduces
  exactly.
- **Consistent plot sizing**: ``auto_font`` applies across the CPP plots, constant-cell sizing now
  shrinks as well as grows, and every public plot method returns a single, uniform object.
- **CPP performance**: a Cython feature-matrix kernel, macOS-safe threaded ``n_jobs``, and a unified
  parallelism convention across :class:`~aaanalysis.CPP` and :class:`~aaanalysis.CPPGrid`. Many
  internal hotspots were vectorized with unchanged output.
- **Feature-matrix and part building**: :meth:`~aaanalysis.SequenceFeature.feature_matrix` gains
  ``batch``, ``df_seq`` and ``list_parts``; ``get_df_parts`` and
  :meth:`~aaanalysis.NumericalFeature.get_parts` gain a position-anchor mode (``tmd_len``).
- :meth:`~aaanalysis.CPPPlot.feature`: titles the plot with the feature's human-readable description.
- :func:`~aaanalysis.load_dataset`: new ``verbose`` parameter (default ``False``).
- :meth:`~aaanalysis.dPULearn.fit`: flexible label handling via ``label_pos`` / ``label_unl``.
- **Docstring discoverability**: previously implicit API contracts are stated in the docstrings,
  so the expected inputs, outputs and defaults are readable from the reference itself.
- **Web fetches**: ``fetch_alphafold`` and ``fetch_uniprot`` are pooled and optionally concurrent.
- **Library output**: all messages flow through a named logger, so they can be captured or silenced.
- **Release engineering**: a packaging workflow builds and checks the distributions, a committed
  benchmark suite guards the hot paths, and ``aaanalysis.__version__`` on master no longer collides
  with the released version.

Fixed
~~~~~
- **BH-adjusted p-values**: ``p_val_fdr_bh`` in ``df_feat`` now follows the canonical
  Benjamini-Hochberg procedure.
- :meth:`~aaanalysis.CPP.run` with ``n_jobs > 1`` no longer crashes in non-interactive contexts, and
  CPP splits no longer fail on free peptides or very short parts.
- **Source install**: the published sdist omitted the Cython sources and could not build; it now does.
- **Plot layout**: composite-plot furniture no longer lands on the heatmap, dense grids keep a
  consistent layout at any figure size, and the sequence bar in CPP-SHAP plots renders correctly
  with ``seq_char_fill=True``.
- **Golden pipelines**: an invalid call names the offending argument instead of failing obscurely.

Deprecated
~~~~~~~~~~
- ``AAlogo`` / ``AAlogoPlot`` are deprecated in favour of :class:`~aaanalysis.AALogo` /
  :class:`~aaanalysis.AALogoPlot`. The strict-semver deprecation policy and the ``deprecated``
  decorator are now in force.

Documentation
~~~~~~~~~~~~~
- The rendered docs state which version they document, derived from ``aaanalysis.__version__``.
- New **Prediction tasks** concept page maps a biological question to the right workflow, and a new
  **A minimal CPP analysis** tutorial gives the shortest end-to-end loop.
- **Navigation**: the sidebar is grouped into *Overview*, *Guides*, *Reference* and *Project*, the
  landing page gains a routing table, and the API reference is split into building blocks and
  golden pipelines.
- **Guides**: a new *Use Cases* subchapter walks a published study end to end, every tool tutorial
  opens with a uniform *You will learn* box, and the tutorials landing page opens with a gallery of
  headline figures.


Version 1.0 (Stable Version)
--------------------------------

v1.0.3 (2026-04-28)
--------------------------------

Added
~~~~~
- :class:`~aaanalysis.AALogo`: New class for amino acid logo visualization.
- :class:`~aaanalysis.AALogoPlot`: New plotting class for AALogo visualizations.

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
- :meth:`~aaanalysis.StructurePreprocessor.fetch_alphafold`: Resolve download URLs through the
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
- :class:`~aaanalysis.SequencePreprocessor`: A utility data preprocessing class (data handling module).
- :func:`~aaanalysis.comp_seq_sim`: A function for computing pairwise sequence similarity (data handling module).
- :func:`~aaanalysis.filter_seq`: A function for redundancy-reduction of sequences (data handling module).
- **options**: Juxta Middle Domain (JMD) length can now be globally adjusted using the **jmd_n/c_len** options.

Changed
~~~~~~~
- :class:`~aaanalysis.ShapModel`: The **ShapExplainer** class has been renamed to :class:`~aaanalysis.ShapModel` for consistency with the :class:`~aaanalysis.TreeModel`
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
- **Multiprocessing**: Replaced native ``multiprocessing`` with the ``joblib`` module for :class:`~aaanalysis.CPP` and
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
- :class:`~aaanalysis.TreeModel`: Wrapper class of tree-based models for Monte Carlo estimates of predictions and feature importance.
  `See TreeModel <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.TreeModel.html>`_.
- **ShapExplainer**: A wrapper for SHAP (SHapley Additive exPlanations) explainers to obtain Monte Carlo estimates for
  feature impact. `See ShapExplainer <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.ShapExplainer.html>`_.
- :class:`~aaanalysis.NumericalFeature`: Utility feature engineering class to process and filter numerical data structures.
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
- :class:`~aaanalysis.CPPPlot`: Plotting class for CPP features.
  `See CPPPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPPPlot.html>`_.
- :class:`~aaanalysis.dPULearnPlot`: Plotting class for results of negative identifications by dPULearn.
  `See dPULearnPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearnPlot.html>`_.
- :class:`~aaanalysis.AAclustPlot`: Plotting class for AAclust clustering results.
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
