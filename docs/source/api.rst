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

    load_dataset
    load_scales
    load_features

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
    ShapModel

.. _perturbation_api:

Perturbation
------------
.. autosummary::
    :toctree: generated/
    :template: autosummary/class_template.rst


    AAMut
    AAMutPlot
    SeqMut
    SeqMutPlot

.. _plot_api:

Plot Utilities
--------------
.. autosummary::
    :toctree: generated/

    plot_get_clist
    plot_get_cmap
    plot_get_cdict
    plot_settings
    plot_gcfs
    plot_legend
    display_df

.. _metric_api:

Metric Utilities
----------------
.. autosummary::
    :toctree: generated/

    comp_auc_adjusted
    comp_bic_score
    comp_kld
