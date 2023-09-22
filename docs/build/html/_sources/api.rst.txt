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

Data
----
.. autosummary::
    :toctree: generated/

    aaanalysis.load_dataset
    aaanalysis.load_scales

.. _feature_engineering_api:

Feature Engineering
-------------------
.. autosummary::
    :toctree: generated/

    aaanalysis.AAclust
    aaanalysis.SequenceFeature
    aaanalysis.CPP
    aaanalysis.CPPPlot

.. _pu_learning_api:

PU Learning
-----------
.. autosummary::
    :toctree: generated/

    aaanalysis.dPULearn

.. _plot_api:

Explainable AI
--------------


Perturbation
------------



Plot Utilities
--------------
.. autosummary::
    :toctree: generated/

    aaanalysis.plot_settings
    aaanalysis.plot_set_legend
    aaanalysis.plot_gcfs
    aaanalysis.plot_get_cmap
    aaanalysis.plot_get_cdict

