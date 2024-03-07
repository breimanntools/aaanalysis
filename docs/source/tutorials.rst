..
   Developer Notes:
   The paths to tutorials are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation.
..


.. _tutorials:

Tutorials
=========

Getting Started
---------------
For an introduction into AAanalysis, explore our **Quick start** and **Slow start** tutorials, both offering the same examples
with the latter explaining the conceptual background. The **Plotting Prelude** tutorial can help you to create publication-ready plots.


.. toctree::
   :maxdepth: 1

   generated/tutorial1_quick_start
   generated/tutorial1_slow_start
   generated/plotting_prelude

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
and **CPP** tutorials.

.. toctree::
   :maxdepth: 1

   generated/tutorial3a_aaclust
   generated/tutorial3b_sequence_feature
   generated/tutorial3c_cpp

PU Learning
-----------
Start positive-Unlabeled (PU) learning to tackle unbalanced and small data through our **dPULearn** tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial4a_dpulearn

Explainable AI
--------------
Explaining sample level predictions at single-residue resolution is introduced in our **ShapExplainer** tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial5a_shap_explainer