.. _beta_features:

Beta Features
=============

Some AAanalysis tools are still under active development and are marked beta. For a beta tool, the API (signatures, defaults, return objects) may change between minor releases without the usual deprecation cycle, so pin a version (e.g. ``pip install aaanalysis==1.1.0``) if you depend on the current behaviour. Everything else in the public API keeps the ordinary deprecation cycle.

Each tool below repeats this warning in its own documentation. The list is generated from the code: a symbol appears here exactly when its docstring carries the ``**Experimental.**`` warning, and a drift test keeps this page in sync, so it cannot go stale.

.. list-table::
   :header-rows: 1
   :widths: 26 56 18

   * - Tool
     - Purpose
     - Added in
   * - :class:`~aaanalysis.SequenceFeatureTransformer`
     - Leak-free CPP feature selection as a scikit-learn transformer.
     - 1.1.0
   * - :class:`~aaanalysis.SeqOpt`
     - Multi-objective directed evolution over sequence variants.
     - 1.1.0
   * - :class:`~aaanalysis.SeqOptPlot`
     - Pareto-front and convergence plots for ``SeqOpt`` results.
     - 1.1.0
   * - :class:`~aaanalysis.AAPred`
     - Evaluation and deployment of sequence-based prediction models.
     - 1.1.0
   * - :class:`~aaanalysis.AAPredPlot`
     - Evaluation and prediction figures for ``AAPred`` results.
     - 1.1.0
   * - :class:`~aaanalysis.ReliabilityModel`
     - Per-prediction trust: calibration, uncertainty, and applicability domain.
     - 1.1.0
   * - :class:`~aaanalysis.ReliabilityModelPlot`
     - Calibration and trust-axis plots for ``ReliabilityModel`` outputs.
     - 1.1.0
   * - :class:`~aaanalysis.ModelEvaluator`
     - Cross-validated evaluation and paired comparison of models.
     - 1.1.0
   * - :class:`~aaanalysis.CPPStructurePlot`
     - CPP feature impact painted onto a 3D protein structure.
     - 1.1.0
