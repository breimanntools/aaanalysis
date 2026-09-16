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

- Design limits are now one shared, validated object. :class:`~aaanalysis.DesignConstraints`
  collects everything a design campaign has to say about which variants are admissible: the
  positions that must keep their wild-type residue (``immutable_positions``), the span a
  substitution may fall in (``mutable_positions``), the target residues that are allowed or
  banned, globally or per position (``permitted_substitutions`` / ``forbidden_substitutions``),
  the mutation budget (``n_mut_max``), how close a variant must stay to its parent
  (``min_identity`` / ``max_identity``), and the motifs it must avoid or keep
  (``forbidden_motifs`` / ``required_motifs``). Every position is a **1-based position in the
  parent sequence**, the convention the ``region`` parameter and the ``pos`` column already use.
- The primary contract is :meth:`~aaanalysis.DesignConstraints.check`, which returns
  ``(ok, reasons)`` for a candidate sequence and names each violated limit in a fixed field
  order, so a rejected candidate explains itself instead of disappearing.
  :meth:`~aaanalysis.DesignConstraints.as_predicate` adapts the same limits to the
  ``genome -> bool`` callable :meth:`~aaanalysis.SeqOpt.run` already consumes, and
  :meth:`~aaanalysis.DesignConstraints.to_dict` / :meth:`~aaanalysis.DesignConstraints.from_dict`
  round-trip a constraint set through JSON.
- :class:`~aaanalysis.AAMut`, :class:`~aaanalysis.SeqMut` and :class:`~aaanalysis.SeqOpt` accept
  the same object as ``constraints``: :meth:`~aaanalysis.AAMut.run` applies its residue-level
  substitution rules, :meth:`~aaanalysis.SeqMut.scan` and :meth:`~aaanalysis.SeqMut.suggest` drop
  the excluded mutations from the scan, :meth:`~aaanalysis.SeqMut.combine` appends an
  ``is_feasible`` column and a ``reasons`` column rather than dropping variants, and
  :meth:`~aaanalysis.SeqOpt.run` restricts its search space and penalizes the sequence-level
  limits through its existing feasibility path. The ``region``, ``to_aa`` and ``n_mut_max``
  parameters keep their meaning and are now shorthand that builds a
  :class:`~aaanalysis.DesignConstraints` internally, so no limit is expressed by two independent
  mechanisms; combining a shorthand with an object that sets the same limit to a different value
  raises a ``ValueError``. Results are unchanged when no object is passed, and
  ``SeqOpt.run(constraints=[...])`` still accepts its published list of feasibility callables.
- :class:`~aaanalysis.DesignConstraints` is part of the public API: it is re-exported at the top
  level (``aa.DesignConstraints``), listed in the :ref:`API reference <protein_engineering_api>`
  under *Protein Engineering*, registered with the canonical abbreviation ``dc``, and each of its
  four methods (:meth:`~aaanalysis.DesignConstraints.check`,
  :meth:`~aaanalysis.DesignConstraints.as_predicate`,
  :meth:`~aaanalysis.DesignConstraints.to_dict`,
  :meth:`~aaanalysis.DesignConstraints.from_dict`) ships an example notebook.
- The risk-coverage trade-off is now measurable.
  :meth:`~aaanalysis.AAPred.eval_selective` ranks the samples by a per-sample confidence signal
  and scores every metric again on the most-confident fraction of them, at each level of a
  coverage grid, so "at 60% coverage the balanced accuracy is 0.93" can be read off a table
  instead of guessed. The confidence source is the caller's choice (``confidence=...`` takes the
  score margin, an uncertainty measure, or a negated applicability-domain distance); the default
  ranks by the out-of-fold score margin. The returned ``df_eval_selective`` carries ``metric``,
  ``coverage``, ``n_retained``, ``score`` and ``score_aurc`` (the area under that metric's
  coverage-performance curve, divided by the coverage span, so a flat curve at ``0.8`` has an
  area of ``0.8``). The ``coverage=1.0`` row reproduces the ordinary out-of-fold score.
  This is a measurement, not an abstaining predictor: nothing is refused, and choosing a refusal
  threshold from the curve stays with the caller. It covers classification;
  :meth:`~aaanalysis.AAPred.eval` is untouched.

- CPP features can now carry an uncertainty estimate. :class:`~aaanalysis.CPP` accepts a
  confidence level in its bootstrap configuration
  (``CPP(bootstrap=True, bootstrap_kws=dict(ci=0.95))``), and the statistics that each
  resampling round already computes are retained and summarised into a central percentile
  interval per feature. ``df_feat`` then gains ``abs_auc_ci_low`` / ``abs_auc_ci_high`` and
  ``mean_dif_ci_low`` / ``mean_dif_ci_high`` after ``selection_frequency``, for
  :meth:`~aaanalysis.CPP.run`, :meth:`~aaanalysis.CPP.run_num` and
  :meth:`~aaanalysis.CPP.run_composit` alike, without any additional runs. The interval is
  conditional on selection: a feature contributes a value only in the rounds in which it was
  selected, so it is read together with ``selection_frequency``, and a feature selected in
  fewer than two rounds gets ``NaN`` bounds. Leaving ``ci`` unset keeps the output unchanged.
- Calibration quality is now measurable. :meth:`~aaanalysis.ReliabilityModel.eval` gains two
  keyword-only parameters: ``use_calibrated=True`` bins the calibrated probability
  (``score_calibrated``) instead of the raw ``score``, and ``add_metrics=True`` appends the Brier
  score (``bin='brier'``) and the expected calibration error (``bin='ece'``, equal-width bins
  weighted by bin size), each stored in ``mean_score``. Comparing the raw and the calibrated
  table on held-out data shows whether ``calibrate=True`` helped. With both left at their
  defaults the returned table is unchanged; ``use_calibrated=True`` on a model fitted with
  ``calibrate=False`` raises a ``ValueError``.
- :meth:`~aaanalysis.ReliabilityModelPlot.reliability_diagram` gains ``label``, annotates the
  Brier score and ECE in the curve's legend entry when the table carries them, and draws the
  diagonal only once, so the raw and the calibrated curve can share one ``ax``.
- :meth:`~aaanalysis.ReliabilityModel.fit` now warns when ``calibrate=True`` cannot be honoured
  (for example, because a class holds fewer members than the internal cross-validation needs or
  the model cannot be cloned). ``score_calibrated`` is ``NaN`` in that case, and
  ``eval(use_calibrated=True)`` raises a ``ValueError`` naming that reason instead of reporting a
  ``calibrate=False`` that was never passed.
- The applicability domain of :class:`~aaanalysis.ReliabilityModel` is now banded and
  inspectable. :meth:`~aaanalysis.ReliabilityModel.predict` appends two columns at the end of its
  table: ``ad_status`` (``inside`` if ``ood_score <= 1``, ``borderline`` up to
  ``1 + ad_borderline``, ``outside`` above, and ``unknown`` when the training reference has no
  usable spread) and ``ad_nearest_train``, the 0-based row index of the closest training sample.
  ``in_domain`` stays the bool shorthand for ``ad_status == "inside"``.
  :meth:`~aaanalysis.ReliabilityModel.fit` gains ``ad_borderline`` (default ``0.1``) and exposes
  the fitted boundary as ``ad_threshold_`` (when it is positive,
  ``ood_score == ad_knn / ad_threshold_``) and the decision rule as ``ad_method_`` (``"knn"``).
  Apart from the ``ad_knn_dist`` → ``ad_knn`` rename, existing columns keep their order and
  values.
- A mutation-candidate set is now scored in one call.
  :meth:`~aaanalysis.ReliabilityModel.predict_candidates` takes the candidate table that
  :meth:`~aaanalysis.SeqMut.mutate`, :meth:`~aaanalysis.SeqMut.combine` and
  :meth:`~aaanalysis.SeqOpt.run` emit (an ``entry`` column plus the candidate sequence in
  ``sequence_mut``; ``col_seq`` selects a different column, e.g. ``sequence`` to score
  wild-types), rebuilds its feature matrix with
  :meth:`~aaanalysis.SequenceFeature.feature_matrix` from the wild-type TMD coordinates in
  ``df_seq``, and delegates to :meth:`~aaanalysis.ReliabilityModel.predict`. The returned table
  carries the ``df_rel`` columns, is row-aligned with the candidates (it keeps their index, so
  ``df_cand.join(df_rel)`` attaches it), and matches a manual
  :meth:`~aaanalysis.SequenceFeature.feature_matrix` + ``predict`` round-trip exactly. Designed
  candidates are pushed away from the training data by construction, so ``ood_score`` /
  ``ad_status`` / ``ad_nearest_train`` are what say which of them the model can still be
  trusted on.
- The :ref:`Data Schemas <df_schemas>` page now documents the prediction outputs that downstream
  tools read, advancing the per-sample and per-residue half of the documented output contract:
  ``df_pred`` from :meth:`~aaanalysis.AAPred.predict` (sequence, domain and window levels),
  ``df_rel`` from :meth:`~aaanalysis.ReliabilityModel.predict`, and ``df_eval_reliability`` from
  :meth:`~aaanalysis.ReliabilityModel.eval`. Contract tests pin the column names, order and
  dtypes as literals, so one of these columns being renamed, dropped, retyped or left
  undocumented fails the suite. The ``score`` column carries one documented range per
  scale, ``[0, 1]`` for ``score_range='proba'`` and ``[0, 100]`` for ``'percent'``, so both
  outputs of :meth:`~aaanalysis.AAPred.predict` are checked against the same contract.


- :meth:`~aaanalysis.NumericalFeature.from_pssm` makes position-specific scoring matrices (PSSMs) a
  CPP value source. It reads PSI-BLAST ASCII ``.pssm`` files (a folder, single file, or an
  ``entry`` to file/array dict) and precomputed ``(L, 20)`` arrays. File columns are reordered
  from PSI-BLAST order (``ARNDCQEGHILKMFPSTWYV``) into canonical amino acid order; arrays must
  already use that order. By default it maps log-odds with a sigmoid or percentages by dividing
  by 100, and can instead return the selected raw values. It can optionally check each matrix
  against the sequences in ``df_seq``. With ``return_scales=True`` it also returns the matching
  20-column ``df_scales`` and ``df_cat``, so a PSSM runs through
  :meth:`~aaanalysis.NumericalFeature.get_parts` and :meth:`~aaanalysis.CPP.run_num` unchanged.
- :meth:`~aaanalysis.SequenceFeature.get_split_kws` gained a ``strategy`` preset for the CPP
  strategy: ``strategy="compositional"`` returns the single whole-part ``Segment`` split (equal to
  ``split_types="Segment", n_split_min=1, n_split_max=1``) and ``strategy="positional"`` returns
  sub-segments plus ``Pattern`` and ``PeriodicPattern`` (equal to ``n_split_min=2, n_split_max=15``
  over all three split types). Both presets together cover the default split set; the default
  ``strategy=None`` leaves the output unchanged. A preset cannot be combined with non-default
  ``split_types``, ``n_split_min``, or ``n_split_max`` values.
- :meth:`~aaanalysis.ModelEvaluator.learning_curve` answers "is this task sampling-limited?": it
  repeats the stratified cross-validation of :meth:`~aaanalysis.ModelEvaluator.run` on stratified,
  nested subsets of increasing size of every training fold, scores each model on the full,
  unchanged test fold, and returns one row per (model, training size, metric) with the mean, std,
  and a bootstrap confidence interval. The default grid has five fraction candidates (which
  resolve to five distinct sizes on sufficiently large data), each with a bootstrap CI. A
  fractional size is resolved within each training fold, so the fraction ``1.0`` uses every
  fold's complete training set and reproduces :meth:`~aaanalysis.ModelEvaluator.run` exactly when
  both calls use the same ``random_state``, ``n_cv``, ``n_rounds``, and metrics (also for unequal
  training folds). :meth:`~aaanalysis.ModelEvaluatorPlot.learning_curve` draws the metric versus
  training size per model with the CI band. A still-rising curve suggests collecting more data; a
  flat one suggests changing the representation or model.
- :meth:`~aaanalysis.AAPredPlot.group_cluster`: ``kind='dendrogram'`` draws the sample relation
  tree without the heatmap, with ``layout='rectangular'`` or a radial ``layout='circular'`` tree.
  The leaves are colored by the existing ``labels`` / ``labels_row`` annotations (one strip or
  ring each, with titled legends). The tree comes from the same linkage as
  ``kind='clustermap'`` (now computed once with ``scipy`` and handed to seaborn), so both kinds
  show the same topology and leaf order. The clustermap figure itself is unchanged when the
  optional ``fastcluster`` package is not installed; with ``fastcluster``, seaborn used to
  compute the linkage internally, so exact ties between equidistant merges may now be broken
  differently.

Changed
~~~~~~~

- Consistency pass on the prediction and design tier. These classes are still marked
  experimental, so the changes land without a deprecation cycle:

  - :meth:`~aaanalysis.ReliabilityModel.fit`: ``ci`` is now a fraction in ``(0, 1)`` with default
    ``0.90`` (it was a percent, ``90.0``), the same unit as :meth:`~aaanalysis.ModelEvaluator.run`
    and :func:`~aaanalysis.comp_bootstrap_ci`. Passing a percent raises a ``ValueError`` that says
    so.
  - :meth:`~aaanalysis.ReliabilityModel.predict`: the ``ad_knn_dist`` column is renamed to
    ``ad_knn``, following the ``ad_<method>`` pattern of ``ad_mahalanobis`` and ``ad_leverage``.
  - :meth:`~aaanalysis.ReliabilityModel.eval`: the ``n`` column is renamed to ``n_samples``. The
    shape of the table, including its summary row, is unchanged.
  - :class:`~aaanalysis.SeqOpt`: the default ``mode`` is now ``"importance"``, which needs no
    fitted model, reference set, or ``[pro]`` extra, so ``aa.SeqOpt()`` constructs in a base
    install. ``mode="impact"`` remains the recommended SHAP-guided search.
  - :meth:`~aaanalysis.AAPred.eval`: ``"mcc"`` (Matthews correlation coefficient) is accepted, so
    ``AAPred`` and :class:`~aaanalysis.ModelEvaluator` share one metric vocabulary.
  - The :class:`~aaanalysis.ReliabilityModel` docstring now states that its ``score_std`` is the
    spread of a sample's score across ensemble members, whereas the ``score_std`` of
    :meth:`~aaanalysis.AAPred.eval` and :meth:`~aaanalysis.ModelEvaluator.run` is the spread of a
    metric across cross-validation folds.
  - :class:`~aaanalysis.AAPred` and :class:`~aaanalysis.SeqOpt` validate ``df_scales``, and
    :meth:`~aaanalysis.AAPred.eval` validates ``list_parts``, so an invalid value raises a
    ``ValueError`` naming the parameter instead of failing later or being ignored.

Fixed
~~~~~

- :meth:`~aaanalysis.ReliabilityModel.fit` rejects non-finite numbers (``NaN``, ``inf``, ``-inf``)
  for ``ci``, ``ad_percentile`` and ``conformal_alpha``. A ``NaN`` passed both range comparisons
  silently (``nan < 0`` and ``nan > 1`` are each ``False``), so ``fit(ci=float("nan"))`` was
  accepted and :meth:`~aaanalysis.ReliabilityModel.predict` then returned ``NaN`` ``ci_low`` /
  ``ci_high`` columns.
- :meth:`~aaanalysis.ReliabilityModel.eval`: passing ``X`` without ``labels`` now raises, where
  the given features were silently ignored before. Passing ``labels`` without ``X`` scores the
  training features against that labelling; it must match them in length and may only use labels
  observed during :meth:`~aaanalysis.ReliabilityModel.fit`.
- :meth:`~aaanalysis.ReliabilityModelPlot.reliability_diagram` validates ``figsize``, ``color``,
  ``title``, and ``ax`` in its frontend, so an invalid value raises a ``ValueError`` naming the
  parameter instead of a matplotlib traceback.
- :meth:`~aaanalysis.StructurePreprocessor.encode_pae` (and the ``encode`` router) now reads the
  PAE JSON exactly as the AlphaFold Database serves it, a one-element list wrapping the
  ``predicted_aligned_error`` dict, so files downloaded by
  :meth:`~aaanalysis.StructurePreprocessor.fetch_alphafold` load without a manual unwrap.
- :meth:`~aaanalysis.CPPPlot.ranking`: the ``Σ`` total and the SHAP positive/negative key no longer
  overprint the percentage labels of the shortest bars; the anchor now skips the label width.
- :class:`~aaanalysis.SeqOptPlot`: ``convergence`` uses a two-line y-label that fits its third
  panel, ``mutation_map`` defaults to a taller figure (``figsize=(8, 6)``) so the 20 amino-acid rows
  stay readable, and ``parallel_coordinates`` colors the lines by the first objective (with a
  colorbar) when a single front is drawn, where rank coloring made every line the same color.
- Tutorials: corrected typos and wrong names (e.g. ``load_dataest``, ``plot_setting``,
  ``ShapExplainer``, ``Part-Slit``, a broken link to the ShapModel tutorial), repaired four
  notebooks that failed ``nbformat`` validation, and re-executed every tutorial so the stored
  figures match the current plotting code.
- :meth:`~aaanalysis.StructurePreprocessor.get_domains`: the AFragmenter adapter now reads the
  ``ClusteringResult`` that current AFragmenter releases return (0-based ``cluster_intervals``),
  so ``tool='afragmenter'`` yields real chopping strings instead of silently empty ones.
- :class:`~aaanalysis.AALogoPlot`: with ``target_p1_site`` the P-site labels of long windows are
  drawn upright at a size that fits one position, and only under the bottom panel of
  ``multi_logo``; the TMD / JMD part labels shrink on very short parts instead of overprinting
  the boundary position numbers (shared by every TMD-JMD plot).
- :meth:`~aaanalysis.CPPPlot.eval`: the pos / neg mean-difference and cluster-count annotations
  are capped to the bar height, so they no longer overprint each other in small figures.
- :meth:`~aaanalysis.SeqOptPlot.pareto_front`: a single front is drawn in one solid color
  instead of the pale end of the rank colormap.
- :meth:`~aaanalysis.AAPredPlot.eval` (and every comparison bar chart): long condition names such
  as ``balanced_accuracy`` are rotated automatically so they and the legend stay readable.
- :func:`~aaanalysis.pipe.plot_eval`: the axis-impact panel wraps long axis names instead of
  rotating them into each other.
- :meth:`~aaanalysis.CPPPlot.feature_map` / :meth:`~aaanalysis.CPPPlot.heatmap`: an explicit
  ``cbar_xywh`` with a ``y`` value is now honored (the automatic bottom-row layout used to move the
  colorbar back under the grid), and a vertical ``cbar_kws`` orientation puts the ticks on the
  right of the bar.
- Quiet by default: :meth:`~aaanalysis.StructurePreprocessor.get_dssp` / ``encode_dssp`` no
  longer re-emit mkdssp's harmless "does not seem to be an mmCIF file" probe as a warning;
  :meth:`~aaanalysis.EmbeddingPreprocessor.fetch_embeddings` keeps the Hub client's status
  text and the weight-loading progress bar out of stderr; the ``build_scales`` "pseudo-scales are
  dataset-dependent" note of the three preprocessors is a verbose-gated message instead of a
  ``UserWarning``; :class:`~aaanalysis.ShapModel` runs ``shap.KernelExplainer`` silently.
- Example notebooks: every table is shown with ``display_df``; the ``fetch_alphafold``,
  ``get_dssp``, ``encode_dssp``, ``get_domains``, ``encode_domains`` and ``fetch_uniprot``
  examples are real, executed walkthroughs on AlphaFold models and UniProt records instead of
  commented-out stubs; the ``compare_sets_negatives`` example draws the negative-set overlap
  with matplotlib because ``upsetplot`` 0.9 does not run on pandas 3; three notebooks were
  repaired to pass ``nbformat`` validation and spelling mistakes were corrected.

Documentation
~~~~~~~~~~~~~

- New tutorial *CPP with protein language model embeddings and AlphaFold structures*
  (``tutorial3e``): the recommended embedding and AlphaFold paths into
  :meth:`~aaanalysis.CPP.run_num`, source fusion, and painting the signature onto the 3D model
  with :class:`~aaanalysis.CPPStructurePlot`.
- New usage-principles page :ref:`Golden Pipelines <golden_pipelines>`: what the
  :mod:`aaanalysis.pipe` layer is for, the four-pipeline spine, the shared
  ``(result, plot, df_eval)`` return shape (with :func:`~aaanalysis.pipe.plot_eval` as the
  documented exception, returning a list of matplotlib figures), and the parity anchors that keep
  the convenience layer honest against the explicit primitive path.
- :func:`~aaanalysis.pipe.find_features` documents its parity scope explicitly: only its
  ``search="fast"`` ``df_feat`` is anchored byte-identical to an explicit chain, while
  ``"balanced"`` and ``"exhaustive"`` search many configurations and are reproducible for a fixed
  ``random_state`` without being parity-anchored.
- The parity of :func:`~aaanalysis.pipe.predict_samples` with the explicit
  :meth:`~aaanalysis.SequenceFeature.feature_matrix` / ``cross_validate`` chain is now pinned over
  the complete learned state of every fitted predictor, and the example notebooks of
  :mod:`aaanalysis.pipe` are held to zero parameter-coverage gaps.
- The documented load → find-features → prediction path is executed in the standard unit suite,
  with its at-most-ten-statement budget read from the same code block displayed on the
  :ref:`Golden Pipelines <golden_pipelines>` page.
- Protocols *P2: Exploratory sequence analysis* and *P3: Sampling* brought to the
  protocol quality rubric: each opens with a key mental model and shows every concept it
  names as a contrast figure (probability versus information logo, real versus shuffled
  baseline, substrate versus non-substrate logos; distance band, reference composition,
  motif-matched lookalikes versus the pool they are drawn from, anti-leakage and
  redundancy filters). The public parameters of each demonstrated
  :class:`~aaanalysis.AALogo`, :class:`~aaanalysis.AALogoPlot`,
  :class:`~aaanalysis.SequenceFeature` and :class:`~aaanalysis.AAWindowSampler` method
  are covered by name across its calls, including
  :meth:`~aaanalysis.AAWindowSampler.sample_motif_matched`, and the common mistakes are
  demonstrated in code. The remaining protocols are unchanged by this pass.
- :meth:`~aaanalysis.CPP.run`: the ``n_sample_batches`` description promised that peak memory is
  bounded by the batch size rather than by the full sample count ``n``. Peak-RSS measurements at a
  constant batch size over 100, 200 and 400 samples show that it bounds the dominant term, the
  per-batch scale-value tensor, while the ``(n_samples, n_pre_filter)`` pre-filter survivor matrix
  and the test statistics computed on it stay resident: peak memory still grows linearly with
  ``n``, on a roughly 13x flatter slope than the single-pass run. The parameter documentation now
  states what is bounded and what is not; the behaviour of ``run`` is unchanged.
- :meth:`~aaanalysis.CPP.run`: ``n_sample_batches`` now creates exactly the requested number of
  balanced, non-empty sample batches. It bounds the per-batch scale-value tensor, but not total
  peak memory: the ``(n_samples, n_survivors)`` survivor matrix and its test statistics remain
  resident. The ``n_batches`` documentation now also states its per-batch FDR-correction semantics.
- :meth:`~aaanalysis.CPP.run` and :meth:`~aaanalysis.CPP.run_num`: ``n_sample_batches`` now
  creates exactly the requested number of balanced, non-empty sample batches. It bounds the
  dominant per-batch working set, but not total peak memory: the pre-filtered candidate matrix and
  its test statistics remain resident. The documentation now distinguishes this from
  ``n_batches``: :meth:`~aaanalysis.CPP.run` applies FDR correction per selected-feature batch,
  while :meth:`~aaanalysis.CPP.run_num` batches only pass-1 statistics.

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
- :func:`~aaanalysis.get_provenance`: opt-in, JSON-serializable record of how a result was produced.
- :class:`~aaanalysis.SequenceFeatureTransformer`: scikit-learn transformer that performs CPP feature
  selection inside a pipeline without leaking the test fold.
- :class:`~aaanalysis.CPPGrid`: tool-style wrapper (``run`` + ``eval``) for a parallel grid over CPP
  configurations.
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
  prediction reliability (calibration, uncertainty, applicability domain, conformal sets) and its
  figures.
- :class:`~aaanalysis.ModelEvaluator` and :class:`~aaanalysis.ModelEvaluatorPlot`: model-agnostic
  evaluation harness with repeated cross-validation, bootstrap intervals and paired comparison.
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
- :func:`~aaanalysis.scan_motif` (``[pro]``): scans candidate proteins for significant PWM motif hits.
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
