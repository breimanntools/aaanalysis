.. currentmodule:: aaanalysis

===
API
===

This page contains the API reference for public objects and functions in AAanalysis. For more examples
and practical usage, refer to our :ref:`example notebooks <api_examples>`.
For convenience, it is common to import the module as follows:

.. code-block:: python

    import aaanalysis as aa

Then you can access all methods and objects via the `aa` alias, such as `aa.load_dataset`.

.. _data_api:

Data Loading
------------
.. autosummary::
    :toctree: generated/

    aaanalysis.load_dataset
    aaanalysis.load_scales

.. _aaclust_api:

AAclust
-------
.. autosummary::
    :toctree: generated/

    aaanalysis.AAclust

.. _cpp_api:

CPP Module
----------
.. autosummary::
    :toctree: generated/

    aaanalysis.CPP
    aaanalysis.SequenceFeature
    aaanalysis.SplitRange

.. _dpulearn_api:

dPUlearn
--------
.. autosummary::
    :toctree: generated/

    aaanalysis.dPULearn

.. _plot_api:

Plot Utilities
--------------
.. autosummary::
    :toctree: generated/

    aaanalysis.plot_settings
    aaanalysis.plot_set_legend
    aaanalysis.plot_gcfs

