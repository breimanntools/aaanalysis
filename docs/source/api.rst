.. module:: aaanalysis.py
.. currentmodule:: aaanalysis

.. _api:

===
API
===

This Application Programming Interface (API) is the public interface for the objects and functions of our AAanalysis
Python toolkit, which can be imported by:

.. code-block:: python

    import aaanalysis as aa

You can then access all methods and objects via the `aa` alias, such as `aa.load_dataset`.

.. _data_api:

Data Handling
-------------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    load_dataset
    load_scales
    load_features
    read_fasta
    to_fasta
    comp_seq_sim
    filter_seq
    SequencePreprocessor

.. _feature_engineering_api:

Feature Engineering
-------------------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    AAclust
    AAclustPlot
    SequenceFeature
    NumericalFeature
    CPP
    CPPPlot

.. _pu_learning_api:

PU Learning
-----------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    dPULearn
    dPULearnPlot

.. _explainable_ai_api:

Explainable AI
--------------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    TreeModel
    ShapExplainer

.. _protein_design_api:

Protein Design
--------------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    AAMut
    AAMutPlot
    SeqMut
    SeqMutPlot

.. _utility_functions_api:

Utility Functions
-----------------
.. autosummary::
    :toctree: generated/

    comp_auc_adjusted
    comp_bic_score
    comp_kld
    display_df
    options
    plot_gcfs
    plot_get_cdict
    plot_get_clist
    plot_get_cmap
    plot_legend
    plot_settings
