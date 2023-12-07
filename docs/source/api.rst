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

    aaanalysis.load_dataset
    aaanalysis.load_scales
    aaanalysis.load_features

.. _feature_engineering_api:

Feature Engineering
-------------------
.. autosummary::
    :toctree: generated/

    aaanalysis.AAclust
    aaanalysis.AAclustPlot
    aaanalysis.SequenceFeature
    aaanalysis.CPP
    aaanalysis.CPPPlot

.. _pu_learning_api:

PU Learning
-----------
.. autosummary::
    :toctree: generated/

    aaanalysis.dPULearn

.. _explainable_ai_api:

Explainable AI
--------------
.. autosummary::
    :toctree: generated/

    aaanalysis.TreeModel
    aaanalysis.ShapModel

.. _perturbation_api:

Perturbation
------------
.. autosummary::
    :toctree: generated/

    aaanalysis.AAMut
    aaanalysis.AAMutPlot
    aaanalysis.SeqMut
    aaanalysis.SeqMutPlot

.. _plot_api:

Plot Utilities
--------------
.. autosummary::
    :toctree: generated/

    aaanalysis.plot_get_clist
    aaanalysis.plot_get_cmap
    aaanalysis.plot_get_cdict
    aaanalysis.plot_settings
    aaanalysis.plot_gcfs
    aaanalysis.plot_legend


