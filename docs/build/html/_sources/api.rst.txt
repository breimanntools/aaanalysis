API
===

Import AAanalysis as:

.. code-block:: python

   import aaanalysis as aa

Load data
---------
The `data_loader` module provides utilities for loading protein benchmark datasets, amino acid scale sets, and more.

.. automodule:: aaanalysis.data_loader
   :members:

AAclust
-------
The `AAclust` is a k-optimized clustering wrapper framework designed for redundancy reduction of numerical scales.

.. automodule:: aaanalysis.aaclust
   :members:

CPP Module
----------
The `CPP` (Comparative Physicochemical Profiling) module focuses on feature engineering by comparing two sets
of protein sequences to identify distinct features.

.. automodule:: aaanalysis.cpp
   :members:

dPUlearn
--------
The `dPUlearn` offers a deterministic Positive-Unlabeled (PU) Learning algorithm, especially beneficial for training on unbalanced and small datasets.

.. automodule:: aaanalysis.dpulearn
   :members:
