.. _overview:

.. image:: _artwork/logos/model_AAanalysis.png
   :alt: AAanalysis Model Overview
   :align: center
   :class: aa-model
   :width: 80%

**AAanalysis** (Amino Acid analysis) is a Python framework for interpretable sequence-based protein prediction.
Its foundation are the following algorithms:

- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein
  sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on
  unbalanced and small datasets.
- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales
  (e.g., amino acid scales).

In addition, AAanalysis provide functions for loading various protein benchmark datasets, amino acid scales,
and their two-level classification (**AAontology**). We combined **CPP** with the explainable
AI  `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ framework to explain sample level predictions with
single-residue resolution.

If you are looking to make publication-ready plots with a view lines of code, see our
:doc:`Plotting Prelude </generated/plotting_prelude>`.

You can find the source code of AAanalysis at `GitHub <https://github.com/breimanntools/aaanalysis>`_.
