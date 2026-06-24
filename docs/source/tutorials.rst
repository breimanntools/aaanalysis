..
   Developer Notes:
   The paths to tutorials are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation.
..


.. _tutorials:

Tutorials
=========
Tutorials teach **one tool at a time** — what a function does, its parameters,
and the outputs it returns. New to AAanalysis? Begin with
:ref:`Getting Started <getting_started>` for your first result, then return here
to go deeper on each tool.

Data Handling
-------------
Learn how to load protein benchmarking datasets and amino acid scale sets in the **Data Loader** and **Scale Loader**  tutorials.

.. toctree::
   :maxdepth: 1

   generated/tutorial2a_data_loader
   generated/tutorial2b_scales_loader

Feature Engineering
-------------------
Explore interpretable feature engineering, the core of AAanalysis, with the **AAclust**, **SequenceFeature**,
and **CPP** tutorials, then see how CPP turns different data representations (scales, embeddings, structure)
into features. Because **SequenceFeature.feature_matrix** returns a plain numeric matrix, these features
drop directly into a stock ``scikit-learn`` ``Pipeline`` — the prediction protocol demonstrates this end to end.

.. toctree::
   :maxdepth: 1

   generated/tutorial3a_aaclust
   generated/tutorial3b_sequence_feature
   generated/tutorial3c_cpp
   generated/tutorial3d_data_representations

PU Learning
-----------
Start positive-Unlabeled (PU) learning to tackle unbalanced and small data through our **dPULearn** tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial4a_dpulearn

Explainable AI
--------------
Explaining sample level predictions at single-residue resolution is introduced in our **ShapModel** tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial5a_shap_model

Evaluation & Comparison
-----------------------
Assemble the building blocks into one honest small-N evaluation in the **Comparison Harness**
tutorial: sweep CPP configurations with **CPPGrid**, score per-protein site-localization
metrics, and rank configurations fairly under cross-validation.

.. toctree::
   :maxdepth: 1

   generated/tutorial6_comparison_harness