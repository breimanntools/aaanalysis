.. module:: aaanalysis.py
.. currentmodule:: aaanalysis

.. _api:

===
API
===

This Application Programming Interface (API) is the public interface for the building
blocks of our AAanalysis Python toolkit: the explicit objects and functions, imported
by:

.. code-block:: python

    import aaanalysis as aa

You can then access all methods and objects via the `aa` alias, such as `aa.load_dataset`.

For the high-level, one-call **golden pipelines** (``aap``) that chain these building
blocks into complete workflows, see the :ref:`API (Pipelines) <api_pipe>` reference.

.. _data_api:

Data Handling
=============
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
=================
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
===================
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
===========
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    dPULearn
    dPULearnPlot

.. _explainable_ai_api:

Explainable AI
==============
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    TreeModel
    ShapModel

.. _protein_engineering_api:

Protein Engineering
===================
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst

    AAMut
    AAMutPlot
    SeqMut
    SeqMutPlot
    SeqOpt
    SeqOptPlot

.. _utility_functions_api:

Utility Functions
=================
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
