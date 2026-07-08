.. _release_notes:

Release Notes
=============

Version 1.1
--------------------------------

v1.1.0 (Unreleased)
--------------------------------

This release substantially expands the feature-engineering surface: a unified
**feature-preprocessor family** (embedding / structure / annotation sources), a
**numerical CPP mode**, a configuration-sweep wrapper, sequence-window sampling,
site-localization metrics, and an opt-in golden-pipeline API.

Added
~~~~~

**Data Handling**

- :class:`~aaanalysis.EmbeddingPreprocessor`: Per-residue protein language model (PLM) embeddings.
  ``encode`` normalizes raw embeddings into a ``[0, 1]`` per-residue ``dict_num``
  (``minmax`` / ``quantile`` / ``sigmoid``) for :meth:`~aaanalysis.CPP.run_num`; ``build_scales`` /
  ``build_cat`` collapse them into pseudo-scales for :meth:`~aaanalysis.CPP.run`. ``fetch_embeddings``
  (``[embed]`` extra) downloads a curated PLM (ESM-2, ESM-1b, ProtT5, ProstT5) from the
  Hugging Face Hub and computes per-protein (mean/max/cls pooling) or per-residue
  embeddings; ``pool_embeddings`` reduces per-residue arrays to per-protein vectors. The
  new ``[embed]`` extra isolates the heavy ``torch`` / ``transformers`` dependencies.
- :class:`~aaanalysis.StructurePreprocessor` (``[pro]``): Converts PDB / CIF / AlphaFold files (and PAE
  sidecars) into ``[0, 1]``-normalized per-residue tensors (``get_dssp``, ``encode_dssp``,
  ``encode_pdb``, ``encode_pae``, ``get_domains``, ``encode_domains``, ``build_scales``,
  ``build_cat``).
- :class:`~aaanalysis.AnnotationPreprocessor` (``[pro]``): Fetches UniProt (or ingests user / predictor)
  per-residue PTM and functional-site annotations and encodes them into tensors
  (``fetch_uniprot``, ``ingest``, ``register_feature``, ``encode``, ``build_scales``,
  ``build_cat``, ``to_df_seq``).
- **combine_dict_nums**: Concatenates per-residue tensors (embedding / structure /
  annotation) along the feature axis into one combined ``CPP.run_num`` input.
- **get_labels**: Derives a binary ``int`` label vector from a sequence DataFrame's
  label column (``positive_label`` mapped to ``1``, everything else to ``0``) — the
  single-call form of the recurring ``(df[col] == x).astype(int).to_numpy()`` expression.
- :func:`~aaanalysis.combine_dict_nums`: Concatenates per-residue tensors (embedding / structure /
  annotation) along the feature axis into one combined :meth:`~aaanalysis.CPP.run_num` input.
- :meth:`~aaanalysis.SequencePreprocessor.pad_parts`: Pads the sequence-part columns of a
  ``df_parts`` DataFrame to a uniform length with a gap symbol (``length`` target or each column's
  per-part max; N-terminal, C-terminal, or symmetric ``both``). The selected ``list_parts`` columns
  are padded (default all) and a padded copy is returned (non-selected columns and the index
  unchanged; input never mutated). Enables analyzing short, variable-length parts at a finer,
  uniform resolution via *pad* → :class:`~aaanalysis.CPP` (``accept_gaps=True``) → larger uniform
  ``n_split_max`` than the shortest real part allows.

**Feature Engineering**

- :class:`~aaanalysis.CPPGrid`: ``Tool``-style wrapper (``run`` + ``eval``) that runs a parallel grid
  sweep of :class:`~aaanalysis.CPP` configurations in one call; configurations differing only in
  ``n_filter`` collapse into a single run. ``eval(sort_by=...)`` scores the
  configurations (``avg_ABS_AUC`` by default) best-first.
- :meth:`~aaanalysis.CPP.run_num`: Numerical mode sourcing per-residue values from a pre-sliced tensor
  (``dict_num_parts``) instead of an amino-acid → scale lookup — embedding / structure /
  annotation features through the same pipeline and output schema as :meth:`~aaanalysis.CPP.run`.
- :meth:`~aaanalysis.CPP.run_composit`:
  Composition mode — build a ``df_feat`` of **composition** features (iFeature-style descriptors
  [Chen18]_) scored with CPP's discriminative statistics. ``composition="aac"`` (amino-acid
  composition) *is* positional CPP — a one-hot identity scale set with the whole-part ``Segment(1,1)``
  split, so it yields ``<PART>-Segment(1,1)-<AA>`` features **with positions** drawn by the feature
  map. ``composition="dpc"`` (dipeptide) / ``"kmer"`` (general ``k``) are **non-positional** (a k-mer
  is an adjacent-tuple property, not a per-residue scale): the ``20 ** k`` k-mers are scored and
  filtered by adjusted AUC (top ``n_filter``), a min-occurrence guard (``min_count``), and optional
  correlation dedup (``max_cor``). The composition matrices themselves come from
  :meth:`~aaanalysis.SequenceFeature.kmer_composition` (which documents the compositional approaches).
- **CPP bootstrap / stability annotation** (:class:`~aaanalysis.CPP` constructor: ``bootstrap``,
  ``n_bootstrap``, ``resample``, ``bootstrap_frac``): Opt-in resampling-based **stability annotation**,
  a thin wrapper applied uniformly by :meth:`~aaanalysis.CPP.run`, :meth:`~aaanalysis.CPP.run_num`, and
  :meth:`~aaanalysis.CPP.run_composit`. With ``bootstrap=True`` the data is resampled ``n_bootstrap``
  times (``resample='reference'`` fixes the test group and resamples only the reference group; also
  ``'both'`` / ``'test'``; per-group draw size ``bootstrap_frac``, with replacement) and re-selected
  each round to score how often each feature is selected, then the **ordinary full-data run is returned
  with a ``selection_frequency`` column** (0 to 1) added. The selected features are exactly those of a
  normal run (``n_filter`` stays the selection criterion); ``selection_frequency`` flags which are
  reproducible under resampling — a trust / interpretability aid, not a change to the list or accuracy.
  ``bootstrap=True`` applies tuned defaults (``n_bootstrap=20``, ``bootstrap_frac=0.8``,
  ``resample='reference'``); the default ``bootstrap=False`` is byte-identical to previous versions.
  Reuses the constructor ``random_state``.
- **CPP.run ``redundancy='legacy'|'exact'``** (also :meth:`~aaanalysis.CPP.run_num`): Opt-in
  position-overlap criterion for the redundancy-reduction step. The default ``'legacy'`` is
  byte-identical to previous versions (published signatures stay reproducible); ``'exact'``
  compares the actual residue positions — an interpretability enhancement (a more concentrated
  signature, fewer redundant subcategories) that does not change predictive performance. For a
  stronger, more efficient reduction, see :meth:`~aaanalysis.CPP.simplify`.
- **CPP.simplify ``candidate_search='fast'``**: Opt-in heuristic capping the candidate
  scales evaluated per feature, for a large speed-up on big scale pools (mainly
  ``greedy``). The default ``'exact'`` reproduces the previous result; ``'fast'`` is
  statistically equivalent (kept-feature Jaccard ≥ 0.95, ΔavgABS_AUC ≤ 0.005 on the
  canonical data).
- **SequenceFeature.get_labels_ovr / get_labels_ovo**: Convert multi-class ``labels``
  into binary sets for :class:`~aaanalysis.CPP` — one-vs-rest (all samples kept) or one-vs-one (per
  class-pair, each pair's value source row-matched).
- **SequenceFeature.get_labels_quantile / get_labels_tiered**: Discretize a continuous
  target into binary ``labels`` — a single quantile cut, or a fixed positive set swept
  against stepwise-lowered negative cuts (each tier row-matched).
- **SequenceFeature.scale_composition**: Scale-composition baseline featurizer that turns
  sequences + scales into a ``(n_seq, n_scales)`` matrix by averaging each scale over a
  sequence span (``list_parts=None`` → whole ``jmd_n`` + ``tmd`` + ``jmd_c``), dropping
  missing / non-canonical residues — the sequence's mean profile in scale-space (the
  scale-based analogue of amino-acid composition). The no-positional-split baseline to
  compare against ``feature_matrix`` / CPP; optional ``return_df=True`` for a labeled frame.
- :meth:`~aaanalysis.NumericalFeature.feature_matrix`: Turns :meth:`~aaanalysis.CPP.run_num`-selected
  features back into a model matrix ``X`` — the numerical analog of
  :meth:`~aaanalysis.SequenceFeature.feature_matrix`. Reconstructs each ``PART-SPLIT-SCALE`` value
  from the per-residue tensors in ``dict_num_parts``, with per-part lengths taken from ``df_parts``
  (the same length source :meth:`~aaanalysis.CPP.run_num` uses), re-applying the split to the part's
  residue axis rather than the JMD-offset ``positions`` display numbering. ``X`` is therefore
  byte-identical to the values :meth:`~aaanalysis.CPP.run_num` computed and preserves the per-residue
  context that per-AA-averaged sequence features discard.
- **SequenceFeature.aa_composition**: Amino-acid-composition (AAC) baseline featurizer that
  turns sequences into a ``(n_seq, 20)`` matrix — the fraction of each of the 20 canonical
  amino acids (``ut.LIST_CANONICAL_AA`` column order) over a sequence span
  (``list_parts=None`` → whole ``jmd_n`` + ``tmd`` + ``jmd_c``), dropping gaps /
  non-canonical residues so each row sums to 1. Fully vectorized; the no-positional-split
  residue-frequency baseline to compare against ``feature_matrix`` / CPP; optional
  ``return_df=True`` for a labeled frame.
- **SequenceFeature.dipeptide_composition**: Dipeptide-composition (DPC) baseline featurizer
  that turns sequences into a ``(n_seq, 400)`` matrix — the fraction of each of the 400
  ordered adjacent canonical amino-acid pairs (``AA, AC, ..., YY``) over a sequence span,
  dropping gaps / non-canonical residues before pairing (adjacencies span dropped residues
  and cross concatenated part boundaries); each row with at least two canonical residues sums
  to 1. Captures local sequential order that plain composition discards; fully vectorized;
  optional ``return_df=True`` for a labeled frame.
- **SequenceFeature.kmer_composition**: General k-mer-composition baseline featurizer — the
  fraction of each of the ``20 ** k`` ordered overlapping k-mers of adjacent canonical residues
  over a sequence span, a ``(n_seq, 20 ** k)`` matrix (columns in
  ``itertools.product(ut.LIST_CANONICAL_AA, repeat=k)`` order). ``k`` selects the composition:
  ``k=1`` is amino-acid composition (identical to ``aa_composition``), ``k=2`` dipeptide
  composition (identical to ``dipeptide_composition``), and higher ``k`` (up to 4) captures
  longer local sequential order. Same non-canonical-dropping, gap-free-span, each-row-sums-to-1
  semantics as the ``k=1`` / ``k=2`` special cases; fully vectorized (one ``bincount`` over
  base-20 k-mer codes); optional ``return_df=True`` for a labeled frame. ``return_scales=True`` also
  returns the CPP-ready ``(df_scales, df_cat)``: for ``k=1`` the ``(20, 20)`` one-hot identity scale set
  + amino-acid-class ``df_cat`` (feed to :meth:`~aaanalysis.CPP.run` with a whole-part ``Segment(1,1)``
  split to get amino-acid composition as a real ``df_feat`` / feature map); for ``k>=2`` ``df_scales`` is
  ``None`` (a k-mer is not a per-residue scale) and ``df_cat`` categorizes the k-mers by residue class.
- **SequenceFeature.get_df_parts_from_windows**: Assemble a reference ``df_parts`` from
  per-part window sets (e.g. ``AAWindowSampler.sample_synthetic`` output).
- **SequenceFeature.get_seq_kws**: Return one protein's ``{jmd_n_seq, tmd_seq, jmd_c_seq}``
- :meth:`~aaanalysis.SequenceFeature.get_df_parts_from_windows`: Assemble a reference ``df_parts`` from
  per-part window sets (e.g. :meth:`~aaanalysis.AAWindowSampler.sample_synthetic` output).
- :meth:`~aaanalysis.SequenceFeature.get_seq_kws`: Return one protein's ``{jmd_n_seq, tmd_seq, jmd_c_seq}``
  as a ready-to-splat ``seq_kws`` dict (by entry or position), parts taken from
  ``df_parts`` so the residues stay bound to the feature geometry — removing the manual
  slicing glue when feeding :meth:`~aaanalysis.CPPPlot.profile` / ``feature_map`` (e.g. sample-level SHAP
  plots).
- :meth:`~aaanalysis.SequenceFeature.get_feature_descriptions`: One standardized, human-readable
  sentence per ``PART-SPLIT-SCALE`` feature id (region + split + AAontology scale name /
  category). Additive (the ``'feature'`` id is unchanged); fills an optional
  ``'feature_description'`` column.
- :meth:`~aaanalysis.AAclust.pre_select_scales`: Metadata-only pre-filter that drops scales by AAontology
  ``category`` (``cat_out``) / ``subcategory`` (``subcat_out``) via ``df_cat`` — the
  preparation step before ``select_scales`` or ``filter_coverage`` (no clustering).
- :meth:`~aaanalysis.AAclust.select_scales`: Wrapper around :meth:`~aaanalysis.AAclust.fit` that returns the
  redundancy-reduced scale subset (one medoid per cluster) directly, ready for :class:`~aaanalysis.CPP`.
- :meth:`~aaanalysis.AAclust.select_proteins`: Protein-level redundancy reduction over a per-protein
  feature matrix ``X`` — clusters proteins, selects one medoid per cluster, annotates
  ``df_seq`` with ``cluster`` / ``is_representative`` / ``dist_to_rep`` — the numerical
  counterpart to :func:`~aaanalysis.filter_seq`.
- **AAclustPlot.centers / medoids accept ``df_scales``**: Pass scales via ``df_scales``
  (transposed internally) instead of ``centers(np.array(df_scales).T, ...)``; pass
  proteins / embeddings / CPP features via ``X`` (used as-is). The explicit ``X``
  signature is unchanged.

**Prediction**

- :class:`~aaanalysis.ReliabilityModel`: Per-sample **prediction reliability** — quantifies how much
  to trust a prediction, separately from the score itself (a model can be confident about a ``0.55``
  and worthless about a ``1.0`` out-of-distribution score). Wraps a fitted predictor (an
  :class:`~aaanalysis.AAPred`, a :class:`~aaanalysis.TreeModel`, or any scikit-learn classifier) plus
  its training data and reports, per sample: stability (ensemble/bootstrap ``score_std`` and a
  confidence interval), an **applicability-domain** out-of-distribution signal (k-NN distance,
  Mahalanobis, leverage → ``ood_score`` / ``in_domain``), calibrated sharpness (``margin`` /
  ``entropy``), a marginal split-conformal prediction set that can abstain, and a headline
  ``reliable`` flag (in-domain, stable, and a confident conformal singleton). ``fit`` / ``predict`` /
  ``eval``; core scikit-learn only, no new required dependency.
- :class:`~aaanalysis.ReliabilityModelPlot`: Visualizes :class:`~aaanalysis.ReliabilityModel` output —
  a per-sample ``ranking`` (each prediction's score with its uncertainty interval, colored by trust
  status), a calibration curve (``reliability_diagram``), the out-of-distribution score distribution
  (``ood_hist``), and a score-vs-OOD ``trust_map`` colored by the ``reliable`` flag.

**Explainable AI**

- **ShapModel — accession-based interface** (``[pro]``): ``fit`` accepts entry-keyed
  soft labels (``fuzzy_labels={'P05067': 0.6}``) together with ``df_seq``;
  ``add_feat_impact`` / ``add_sample_mean_dif`` accept ``df_seq`` and a ``samples``
  parameter taking row positions or entry names. The array-``labels`` path is unchanged;
  ``sample_positions`` is a deprecated alias for ``samples`` (removed in 1.2.0).
- **ShapModel — unbiased fuzzy estimator, now the default** (``[pro]``): ``fit`` gains
  ``fuzzy_aggregation``, defaulting to the new ``'interpolate'`` estimator. It weights a
  soft label by *exactly* ``p`` — fitting at 0 (``S0``) and at 1 (``S1``) and blending
  ``p * S1 + (1 - p) * S0`` — the unbiased alternative to the biased threshold sweep, which
  stays available as a first-class option via ``fuzzy_aggregation='threshold'``. For
  ``interpolate``, ``n_rounds`` (default ``5``) is a speed/stability dial: ``1`` is the fast
  exact two-fit estimate (~2x faster than the threshold default on the same cell), ``5`` adds
  light Monte-Carlo averaging, and the mean converges (run-to-run spread below ~5%) around
  ``n_rounds ≈ 15–20``; a fixed ``random_state`` keeps every run reproducible.
- :class:`~aaanalysis.CPPStructurePlot` (``[pro]``): Paints per-residue CPP / CPP-SHAP feature impact onto an
  interactive 3D protein structure, rendered with `py3Dmol <https://pypi.org/project/py3Dmol/>`_
  (no matplotlib structure fallback — the cartoon is always a real 3D view). ``map_structure(
  df_feat, pdb=...)`` (or ``uniprot=...`` to auto-fetch the AlphaFold model) returns a
  ``StructureView`` (``show`` / ``write_html`` / ``_repr_html_``). Supports an ``'impact'``
  red-white-blue ramp and a ``'plddt'`` AlphaFold-confidence mode, with ``whole`` / ``fade`` /
  ``zoom`` focus. By default each feature's full impact is painted on every residue it spans
  (app-fidelity colouring); ``normalize_by_span=True`` switches to the span-normalized sum used
  by :meth:`~aaanalysis.CPPPlot.profile` and the :meth:`~aaanalysis.CPPPlot.feature_map` top per-position bar. ``plot_combined``
  returns a ``CombinedView`` showing the py3Dmol cartoon next to the :meth:`~aaanalysis.CPPPlot.feature_map`
  image (``write_html`` exports the pair; ``savefig(path)`` saves the feature-map panel as a static
  PNG / PDF for papers — the 3D cartoon is interactive and has no headless image), reproducing the
  deployed cleavage app's signature layout. ``interactive`` returns a live `ipywidgets <https://ipywidgets.readthedocs.io>`_
  explorer (added to the ``[pro]`` extra) where a site slider drives a user ``predictor`` and
  repaints the linked 3D structure and :meth:`~aaanalysis.CPPPlot.feature_map` together (debounced), the
  notebook-native version of the app's per-site explore loop. A **highlight (position) slider**
  links the two panels live: picking a residue lights it up in the 3D cartoon and marks its
  feature-map column without re-predicting, and with the ``ipympl`` (``%matplotlib widget``)
  backend the feature map becomes **clickable** for the same highlight (``ipympl`` is optional —
  the slider is the always-present link, no extra dependency). ``plot_linked`` returns a
  ``LinkedView`` — a self-contained HTML where **hovering a feature-map column highlights the
  corresponding residue** in the 3Dmol cartoon (the app's signature interaction); ``write_html``
  exports it as a standalone, shareable page. ``explore(df_feat, sequence, df_seq=..., labels=...,
  model=...)`` is the integrated one call: it builds a **built-in per-site predictor** (compute the
  query window's values for the fixed feature set, predict the probability, attach the per-site SHAP
  impact via a default :class:`~aaanalysis.ShapModel` refit — no :meth:`~aaanalysis.CPP.run` rediscovery) and dispatches to a
  selectable ``output`` (``'widget'`` / ``'html'`` / ``'static'``); ``model`` takes a name
  (``'rf'`` / ``'svm'`` / ``'log_reg'``), an estimator, or a list, and a custom
  ``predictor=(sequence, p1) -> df_feat`` remains the escape hatch. With ``output='html'``,
  passing ``sites=[...]`` bakes a **multi-site live** standalone page: a client-side JS slider
  switches the pre-computed prediction per P1 (feature map + structure restyle) with no kernel,
  keeping the column-residue linking (warned past 40 sites, hard-capped at 200).

**PU Learning**

- **dPULearn.fit — positives/unlabeled split input**: for the common positive / unlabeled
  setup, ``fit`` now accepts ``X_pos`` and ``X_unlabeled`` separately (an alternative to
  ``X`` + ``labels``) instead of stacking them by hand and building a ``1`` / ``2`` label
  vector. After fitting, the new ``dPULearn.mask_neg_`` attribute holds the **boolean mask
  of reliable negatives** — over the rows of ``X_unlabeled`` in the split mode, over ``X``
  otherwise (equal to the manual ``labels_[len(X_pos):] == 0`` result exactly). ``fit`` still
  returns ``self`` and the existing ``fit(X, labels=...)`` path is unchanged.
- :meth:`~aaanalysis.dPULearn.project`: Projects held-out samples from the same feature space into
  the **fitted PC space** (the ``PCi`` columns of ``df_pu_``) after PCA-based identification, so new
  proteins can be placed alongside the identified negatives. ``method`` selects the reconstructed
  linear map — ``'lstsq'`` (default, affine least-squares) or ``'components'`` (exact PCA-geometry
  map); both are exact on the fitted samples and interpolate for new ones.
  :meth:`~aaanalysis.dPULearnPlot.pca` gains ``df_pu_add`` / ``names_add`` / ``colors_add`` to
  overlay one or more projected groups (a one-call four-group PCA); the default (``df_pu_add=None``)
  output is unchanged.

**Sequence Analysis**

- :class:`~aaanalysis.AAWindowSampler`: Samples fixed-length sequence windows for PU-learning and
  hard-negative-mining workflows (``sample_same_protein``, ``sample_different_protein``,
  ``sample_motif_matched``, ``sample_synthetic``).
- :func:`~aaanalysis.scan_motif` (``[pro]``): Scans candidate proteins for statistically significant PWM
  occurrences via MEME/FIMO, complementing the pure-Python
  :meth:`~aaanalysis.AAWindowSampler.sample_motif_matched` sampler.

**Protein Engineering**

- **SeqOpt — multi-objective protein engineering** (**core**; only ``mode="impact"`` needs
  ``aaanalysis[pro]``): A new :class:`~aaanalysis.SeqOpt` optimizer
  (with :class:`~aaanalysis.SeqOptPlot`) performs **machine-learning-guided directed evolution** of one
  wild-type — searching the Pareto front across several objectives at once, with a
  model-bound :class:`~aaanalysis.SeqMut` as the fitness engine and a re-implementation of NSGA-II for
  selection (this is protein *engineering*, not *de novo design*). Two guidance modes:
  ``mode="impact"`` refits :class:`~aaanalysis.ShapModel` each generation under fuzzy labeling to target the
  strongest-``feat_impact`` residues; ``mode="importance"`` walks positions by static
  ``feat_importance``. The evolutionary toolbox is a complete pure-Python re-implementation
  (DEAP is a dev/test-only parity oracle; runtime stays DEAP-free): ``crossover`` (uniform /
  one- / two-point), ``mutation`` (substitution / shift), ``variation`` (varAnd / varOr),
  ``survival`` ((mu+lambda) / (mu,lambda) / eaSimple), ``constraints`` (delta / closest-valid
  penalty), a single-objective Hall of Fame (``hall_of_fame_``), and a memory-bounded
  (chunked) vectorized non-dominated sort. Objectives accept any
  ``callable(sequence) -> float`` (an external scikit / torch model or sequence-level
  tool / web API), cached per variant. ``run`` returns ``df_pareto`` (objective columns +
  ``rank`` + ``crowding``) backed by a cumulative Pareto archive; ``eval`` reports
  hypervolume / front size / spread / convergence. **Visualization**: :class:`~aaanalysis.SeqOptPlot` covers
  ``pareto_front`` (2-D / 3-D), ``parallel_coordinates``, ``convergence`` (hypervolume +
  spread + per-objective best/mean/worst band), ``hypervolume``, ``mutation_map`` (front
  substitution-enrichment heatmap) and ``genealogy`` (mutational-lineage tree). Reproducible
  via ``random_state`` / ``seed``.
- **SeqMut model-guided mode (ML-guided directed evolution)**: :class:`~aaanalysis.SeqMut` is optionally
  model-aware — binding a fitted classifier (``SeqMut(model=..., target_class=...)``, any
  object with ``predict_proba``) makes ``scan`` / ``suggest`` / ``mutate`` report
  ``delta_pred`` (the prediction-score shift in percentage points) and ``suggest`` rank
  by it. Without a model, :class:`~aaanalysis.SeqMut` stays the deterministic, model-free ΔCPP tool.
- :meth:`~aaanalysis.SeqMut.combine`: Scores combined multi-mutation variants — several point mutations
  applied to one sequence and evaluated as a single design.
- :class:`~aaanalysis.SeqMutPlot`: ``mutation_landscape`` renders the ``delta_pred`` prediction-shift
  mutation-scan heatmap; new ``variant_impact`` (ranked-variant bar) and ``epistasis``
  (pairwise non-additivity) plots.

**Metrics**

- :func:`~aaanalysis.comp_per_protein_ap`: Per-protein average precision for site-localization ranking,
  with an optional ``tolerance=±k`` variant for positional jitter.
- :func:`~aaanalysis.comp_detection_metrics`: Recall / precision / F1 / MCC at a fixed score threshold,
  pooled across per-residue predictions.
- :func:`~aaanalysis.comp_bootstrap_ci`: Seeded percentile confidence interval over a per-protein metric
  vector (returns ``{'mean', 'ci_low', 'ci_high'}``).
- :func:`~aaanalysis.comp_smooth_scores`: Peak-preserving (``max(smoothed, raw)``), NaN-aware smoothing
  of per-residue score tracks.

**Plotting**

- :func:`~aaanalysis.plot_rank`: Standalone per-protein max-score-vs-rank scatter with group coloring and
  optional threshold lines (pairs with the new ``aa.metrics`` functions).
- **COLOR_SAMPLES_POS / COLOR_SAMPLES_NEG / COLOR_SAMPLES_UNL / COLOR_SAMPLES_REL_NEG**:
  Public, named constants for the canonical sample-group colors (positive / negative /
  unlabeled / reliable-negative). They equal the ``plot_get_cdict("DICT_COLOR")["SAMPLES_*"]``
  values exactly, so a named constant replaces indexing the color dict by string key.

**Golden Pipelines**

- **aaanalysis.pipe** (``aap``): A second, opt-in convenience API of stateless, one-call
  *golden pipelines* over the AAanalysis primitives (``import aaanalysis.pipe as aap``).
- **aap.find_features**: Staged, interpretable CPP AutoML search. Stage 1
  cross-validates the full Cartesian Part × Split × Scale grid and ranks each axis by
  its marginal-mean impact; Stage 2 refines the single highest-impact axis against
  ``n_filter``; Stage 3 refines the winning feature set (:meth:`~aaanalysis.CPP.simplify` + recursive
  feature elimination, each kept only if it is not Pareto-dominated). Selection is
  multi-objective: within each stage the Pareto-optimal-then-simplest configuration
  across all ``metric`` wins, scored by the averaged cross-validated performance of one
  or more ``model`` s. The winner is ranked by tree-based importance and drawn as the
  feature map. The ``search`` grade scopes the effort (``"fast"`` is byte-identical to
  the explicit single-CPP path); it returns ``(df_feat, ax, df_eval)`` where ``ax`` also
  carries the publication eval figures (``ax.eval``) and ``df_eval`` has one
  ``<metric>_mean``/``_std`` column per metric plus ``stage`` / ``is_pareto`` / ``rank``
  / ``is_selected``.
- **aap.plot_eval**: Publication-ready evaluation figures of a ``find_features`` sweep —
  the high-dimensional Part × Split × Scale grid is **decomposed** into a series of clean
  2D ``viridis`` heatmaps (the two most-informative axes on each panel, the least on the
  slice), with a shared colorbar, the selected configuration starred, plus marginal-impact
  and ``n_filter`` panels. Returns the list of figures so each drops straight into a paper;
  also usable standalone on a ``find_features`` eval table.

**Package**

- **aa.__version__**: The installed package version is exposed at top level via
  ``importlib.metadata``.
- **CHANGELOG.md + deprecation policy**: A root ``CHANGELOG.md``
  (`Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_ format) gives a terse,
  developer-facing index alongside these narrative notes. From v1.x onward, any rename or
  removal of a public symbol ships at least one minor release carrying a
  ``DeprecationWarning`` first; an internal ``deprecated(reason, version_removed)``
  decorator marks such symbols. See *Versioning and Deprecation Policy* in
  ``CONTRIBUTING.rst``.

**Documentation**

- **Prediction tasks** concept page (*Usage Principles*): maps a biological question to
  the right workflow via a task table keyed on unit of comparison and reference
  construction, across the residue / domain / protein levels — the front door to the
  Protocols catalog.
- **A minimal CPP analysis** tutorial (``tutorial0_minimal``): the shortest end-to-end
  loop — load a dataset, run CPP, read out the signature.
- **Documentation navigation**: the sidebar is grouped into four sections — *Overview*,
  *Guides* (Tutorials · Protocols · Use Cases), *Reference*, and *Project* — and the landing
  page gains a "You want to… / Go to" routing table; the previously unwired **Comparison
  Harness** tutorial (``tutorial6_comparison_harness``) is now reachable.
- **Use Cases** guide (third *Guides* subchapter): each use case showcases a published
  study end to end from bundled data. The first, *Charting γ-secretase substrates by
  explainable AI* (``use_case1_gamma_secretase``), walks the full AAanalysis pipeline of
  Breimann and Kamp *et al.*, Nat. Commun. 2025 on the bundled ``DOM_GSEC`` /
  ``DOM_GSEC_PU`` sets: AAlogo sequence logos of the three protein groups, AAclust
  redundancy-reduced scale sets, the CPP + TreeModel signature and feature map, dPULearn
  reliable-negative mining (with PCA and logo), a prediction benchmark (feature
  engineering × data expansion) plus a CPP/dPULearn optimization heatmap, and SHAP
  single-residue explanations for individual substrates (APP, N-cadherin).
- **Standardized tutorial header box**: every tool tutorial now opens with a uniform
  green *You will learn* box (Tool · Input · Output · Best used for · Related protocol ·
  Related API), giving a one-glance answer to *what tool, what goes in, what comes out,
  and where to go next* and cross-linking the matching protocol and API reference.
- **Split API reference**: the reference is now two pages, each listing its members
  directly at the top level. *API* documents the explicit **building blocks**
  (``import aaanalysis as aa``) grouped by category; the new *API (Pipelines)* page documents
  the **golden pipelines** (``import aaanalysis.pipe as aap``), one function per pipeline.
  Golden pipelines are no longer mixed into the building-block page or the Tutorials
  section; Getting Started links both references.

Changed
~~~~~~~

- :class:`~aaanalysis.TreeModel`: **per-round seeding fix** — with a fixed ``random_state`` (or the
  global ``options["random_state"]``), :meth:`~aaanalysis.TreeModel.fit` now reseeds each round to
  ``random_state + i`` so the rounds are independent. Previously every round fit identical estimators,
  so ``feat_importance_std`` (and :meth:`~aaanalysis.TreeModel.predict_proba`'s ``pred_std``) collapsed
  to exactly ``0`` and rounds 2..N were wasted. Fixed-seed importances change once (degenerate → real
  Monte-Carlo mean with non-zero std); the ``random_state=None`` default is unchanged.
- **Uniform plot return contract**: Every public ``*Plot`` method now returns a single
  ``(fig, ax)`` pair (forwarding attribute access to ``ax``, so existing
  ``ax = plot(...); ax.set_title(...)`` code keeps working), replacing the previous mix
  of shapes. **Breaking change, scheduled for the next major release:**
  :meth:`~aaanalysis.AAclustPlot.centers` / ``medoids`` return ``(fig, ax)`` and expose the PCA-component
  DataFrame on the ``df_components_`` attribute instead of as the second return value.
- **CPP performance**: The Cython feature-matrix kernel, macOS-safe threaded ``n_jobs``,
  scale / AA-index caching, and scale / sample batching land in this release, replacing
  the hour-long, low-CPU CPP runs of ``≤1.0.3`` — users on those versions should upgrade.
  When the compiled extension is missing and CPP falls back to the pure-Python kernel,
  the one-time notice is now a ``UserWarning`` (visible even with ``verbose=False``).
- :meth:`~aaanalysis.SequenceFeature.feature_matrix`: New ``batch=`` parameter accepts a list of
  ``df_parts`` built in a single Cython pass (faster for many small part tables).
- **SequenceFeature.feature_matrix**: New ``df_seq=`` / ``list_parts=`` parameters build
  ``df_parts`` internally (via ``get_df_parts``), collapsing the ``get_df_parts`` →
  ``feature_matrix`` two-step into one call. Exactly one of ``df_parts`` / ``df_seq`` is
  required; the existing ``df_parts=`` path is unchanged (byte-identical).
- **get_df_parts / NumericalFeature.get_parts**: New ``pos``-anchor mode (``tmd_len=``)
  explodes each 1-based anchor into one ``jmd_n`` / ``tmd`` / ``jmd_c`` row
  (``entry_win``). ``get_df_parts`` is also several-fold faster (vectorized; output
  unchanged).
- **n_jobs**: Unified parallelism convention across :class:`~aaanalysis.CPP` / :class:`~aaanalysis.CPPGrid` (``1`` serial,
  ``-1`` all cores, ``N>1`` exactly N, ``None`` optimized), with an ``options['n_jobs']``
  global override.
- :meth:`~aaanalysis.CPPPlot.feature`: Titles the plot with the feature's human-readable description,
  line-wrapped via ``show_title`` (default ``True``) and ``title_wrap_width`` (default
  ``45``).
- **load_dataset verbose reporting**: New ``verbose`` parameter (default ``False``)
  reports how many entries each removal step (``min_len``, ``max_len``, and
  ``non_canonical_aa='remove'``) drops, making the previously silent filtering
  observable. The returned data is unchanged; to retain every entry use
  ``non_canonical_aa='keep'``.
- **Docstring discoverability**: Surfaced previously implicit API contracts at the
  docstrings users read (no behavior change) — the ``get_parts`` → ``run_num`` call order
  and ``[0, 1]`` normalization contract, and a ``[pro]`` install marker on the pro
  classes / functions.
- :meth:`~aaanalysis.dPULearn.fit`: Flexible label handling via ``label_pos`` / ``label_unl`` /
  ``label_neg`` markers (only unlabeled samples are candidates; pre-labeled negatives are
  kept and never re-selected). The negative count is set by exactly one of ``n_neg`` (the
  total wanted) or ``n_unl_to_neg`` (drawn directly from the unlabeled pool); output uses
  the package convention (``1`` positive, ``0`` negative, ``2`` unlabeled).
- **Pooled, optionally concurrent web fetches**: ``fetch_alphafold`` / ``fetch_uniprot``
  route every request through a pooled ``requests.Session`` and accept a ``max_workers``
  parameter. Concurrency is off by default (parallel requests risk HTTP-429 throttling);
  when enabled, results reassemble in input order, so output is byte-identical.
- **Performance (same output)**: Many internal hotspots were vectorized or parallelized
  with byte-identical results — :class:`~aaanalysis.AAWindowSampler` filtering / sampling, :class:`~aaanalysis.AAclust`
  medoid distances, the per-feature KLD path in :meth:`~aaanalysis.dPULearn.eval`, ``encode_one_hot``,
  :meth:`~aaanalysis.AAMut.comp_substitution_impact`, ``get_sliding_aa_window``, and several
  :class:`~aaanalysis.StructurePreprocessor` encoders (``encode_pdb`` contact / disulfide / pLDDT, a shared
  per-entry chain-pick and alignment cache, ``get_dssp``). Public APIs and outputs are
  unchanged.
- **Developer tooling**: A committed ``pytest-benchmark`` suite (``tests/benchmarks/``,
  ``[bench]`` extra) micro-benchmarks the hot entry points as a non-gating nightly; a
  numerical-equivalence tolerance policy defines three tiers (T1 byte-identical, T2
  ``allclose`` plus identical discrete decisions, T3 statistically-equivalent within an
  agreed band) for output-affecting optimizations; and an advisory pyright ratchet
  (``.github/pyright_baseline.txt``) drives the type-contract count down per subpackage
  (now 887, every public-API signature pyright-clean). None gate a merge or change the
  public API.

Changed
~~~~~~~

- **Module rename**: the ``protein_design`` subpackage is now ``protein_engineering``,
  matching its user-facing name (``AAMut`` / ``SeqMut`` / ``SeqOpt`` are amino-acid
  mutation and directed-evolution tools). The public classes are unchanged and imported
  the same way (``import aaanalysis as aa`` → :class:`~aaanalysis.AAMut`,
  :class:`~aaanalysis.SeqMut`, :class:`~aaanalysis.SeqOpt` and their plot classes); only a
  full-path import such as ``from aaanalysis.protein_design import SeqMut`` must become
  ``from aaanalysis.protein_engineering import SeqMut``.

Fixed
~~~~~

- **BH-adjusted p-values (#343)**: ``p_val_fdr_bh`` in ``df_feat`` now follows canonical
  Benjamini–Hochberg — the reverse cumulative-minimum (monotonicity) step was missing, so the
  reported values could be non-monotone / slightly conservative in non-monotone regions. Only the
  reported column changes; feature selection and ranking (``abs_auc`` / ``abs_mean_dif``) are
  unaffected.
- :meth:`~aaanalysis.CPP.run` with ``n_jobs > 1`` no longer crashes in non-interactive
  contexts (e.g. ``python -c``, heredocs, some subprocess shells) where starting a
  ``multiprocessing.Manager`` for the cross-process progress bar raised ``EOFError`` /
  ``OSError``. The Manager is now created best-effort: on failure CPP degrades to the
  thread-safe, single-process progress path and the run completes normally instead of
  aborting (previously the only workaround was ``n_jobs=1``). When the Manager is
  available, behavior and output are unchanged.
- **CPP splits on free peptides / short parts (#338)**: ``aap.find_features`` and the
  ``Pattern`` / ``PeriodicPattern`` splits were unusable on free peptides with no flanking
  context (the linear-epitope case). ``find_features(search="fast")`` and its Stage-3
  simplify step ignored the requested / winning split configuration and always used the
  default (``len_max=15``, ``n_split_max=15``), so any target region shorter than ~15
  residues raised. Now:

  - **CPP auto-caps splits to the shortest part** instead of raising. When a ``df_parts``
    sequence part is too short for the requested ``split_kws``, :class:`~aaanalysis.CPP`
    caps the ``Segment`` ``n_split_max`` and drops the ``Pattern`` / ``PeriodicPattern``
    split types that cannot fit (``Segment`` is always kept), emits one ``UserWarning``, and
    stores the capped ``split_kws`` as ``self.split_kws`` (used by both :meth:`~aaanalysis.CPP.run`
    and :meth:`~aaanalysis.CPP.run_num`). For parts long enough for the requested splits this
    is a no-op, so results for flanked inputs are byte-identical.
  - **``find_features`` handles free peptides end to end.** The bounded ``kws`` dict now accepts
    ``len_max`` (and actually honors ``n_split_max``). Passing ``kws={"n_jmd": 0}`` (no flanking
    context) switches the part sweep to **TMD-only** (the whole peptide is one part, instead of
    half-TMD composite fragments) and **caps the swept ``n_split_max`` range** to the shortest
    part length, deduped, with a ``UserWarning``. The staged ``balanced`` / ``exhaustive`` searches
    no longer hard-error on short parts. On normal (long-part) inputs the range cap is a no-op.


Version 1.0 (Stable Version)
--------------------------------

v1.0.3 (2026-04-06)
--------------------------------

Added
~~~~~
- :class:`~aaanalysis.AAlogo`: New class for amino acid logo visualization.
- :class:`~aaanalysis.AAlogoPlot`: New plotting class for AAlogo visualizations.

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
