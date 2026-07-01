..
   Developer Notes:
   The paths to tutorials are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation.
..


.. _tutorials:

Tutorials
=========
Tutorials teach the AAanalysis **tools** — what each one does, its parameters, and
the outputs it returns. They cover the *mechanics*; for how to combine tools into a
valid end-to-end analysis, see the :ref:`Protocols <protocols>`, which link back
here for the mechanics instead of repeating them — so the two stay distinct with no
overlap. New to AAanalysis? Begin with :ref:`Getting Started <getting_started>` for
your first result, then return here to go deeper on each tool.

Data Handling
-------------
Learn how to load protein benchmarking datasets and amino acid scale sets in the **Data Loader** and **Scale Loader**  tutorials.

.. toctree::
   :maxdepth: 1

   generated/tutorial2a_data_loader
   generated/tutorial2b_scales_loader

Feature Engineering
-------------------
Explore interpretable feature engineering, the core of AAanalysis, with the :class:`~aaanalysis.AAclust`, :class:`~aaanalysis.SequenceFeature`,
and :class:`~aaanalysis.CPP` tutorials, then see how CPP turns different data representations (scales, embeddings, structure)
into features. Because :meth:`~aaanalysis.SequenceFeature.feature_matrix` returns a plain numeric matrix, these features
drop directly into a stock ``scikit-learn`` ``Pipeline`` — the prediction protocol demonstrates this end to end.

.. toctree::
   :maxdepth: 1

   generated/tutorial3a_aaclust
   generated/tutorial3b_sequence_feature
   generated/tutorial3c_cpp
   generated/tutorial3d_data_representations

PU Learning
-----------
Start positive-Unlabeled (PU) learning to tackle unbalanced and small data through our :class:`~aaanalysis.dPULearn` tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial4a_dpulearn

Explainable AI
--------------
Explaining sample level predictions at single-residue resolution is introduced in our :class:`~aaanalysis.ShapModel` tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial5a_shap_model

Evaluation & Comparison
-----------------------
Learn the evaluation tools — :class:`~aaanalysis.CPPGrid` configuration sweeps, per-protein
site-localization metrics, and fair ranking under cross-validation — in the
:class:`~aaanalysis.CPPGrid` tutorial. These are the mechanics that the *P10: Validation*
protocol puts to work end to end.

.. toctree::
   :maxdepth: 1

   generated/tutorial6_comparison_harness

Protein Engineering
-------------------
Optimize an existing sequence with :class:`~aaanalysis.SeqOpt` — machine-learning-guided directed
evolution — and read the results with :class:`~aaanalysis.SeqOptPlot`. This is **protein engineering**
(mutating a known protein), distinct from **de novo protein design** (generating new
proteins, e.g. RFdiffusion → ProteinMPNN → AlphaFold). The :class:`~aaanalysis.SeqOpt`
tutorial walks a complete case study: training a substrate classifier, engineering a
"super substrate" for gamma-secretase, and visualizing the Pareto front, convergence,
mutation map and lineage.

.. toctree::
   :maxdepth: 1

   generated/tutorial7_protein_engineering