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

AAanalysis exposes two interfaces. The **golden pipelines** (``aap``) are the
high-level, one-call entry point; the **building blocks** below are the explicit
objects and functions they compose, for full control.

.. _pipe_api:

Golden Pipelines (``aap``)
==========================

The golden pipelines chain the standard ``load → CPP → model → explain → plot``
workflow into a single call — the implicit counterpart to the explicit building
blocks below (much as ``pyplot`` sits over Matplotlib's ``Axes`` / ``Figure``).
They are stateless wrappers whose defaults match the explicit path, imported
under their own alias:

.. code-block:: python

    import aaanalysis.pipe as aap

.. currentmodule:: aaanalysis.pipe

.. autosummary::
    :toctree: generated/

    obtain_samples
    find_features
    predict_samples
    plot_eval
    explain_features

.. currentmodule:: aaanalysis

Building Blocks
===============

The explicit objects and functions the pipelines compose — use them directly
when you need to customise a step.

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
    SequencePreprocessor
    StructurePreprocessor
    EmbeddingPreprocessor
    AnnotationPreprocessor
    combine_dict_nums

.. _sequence_analysis_api:

Sequence Analysis
-----------------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    AAlogo
    AAlogoPlot
    AAWindowSampler
    comp_seq_sim
    filter_seq
    scan_motif

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
    CPPGrid
    CPPPlot
    CPPStructurePlot

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
    ShapModel

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
    comp_bootstrap_ci
    comp_detection_metrics
    comp_kld
    comp_per_protein_ap
    comp_smooth_scores
    display_df
    options
    plot_gcfs
    plot_get_cdict
    plot_get_clist
    plot_get_cmap
    plot_legend
    plot_rank
    plot_settings
