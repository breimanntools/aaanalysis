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
Explaining sample level predictions at single-residue resolution is introduced in our **ShapModel** tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial5a_shap_model

Protocols
---------
Task-oriented, pipeline-ordered recipes that answer a single biological question
end-to-end — *when to use it, what goes in, the minimal code, what comes out, how
to interpret it, common mistakes,* and *what to do next*. Unlike the tutorials
(which teach one function at a time), protocols teach **workflows**. This is a
living catalog that grows along the AAanalysis pipeline; see the
`Protocols epic <https://github.com/breimanntools/aaanalysis/issues/35>`_.

.. toctree::
   :maxdepth: 1

   generated/protocol1_cpp_signature
   generated/protocol2_prediction_tasks
   generated/protocol3_sampling
   generated/protocol4_engineer_features
   generated/protocol5_compositional_positional
   generated/protocol6_feature_selection
   generated/protocol7_classifier
   generated/protocol8_interpretability
   generated/protocol9_validation