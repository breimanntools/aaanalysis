.. AAanalysis documentation master file, created by
   sphinx-quickstart on Fri Aug 11 17:25:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the AAanalysis documentation
=======================================

**AAanalysis** (Amino Acid analysis) is a Python framework to enable interpretable protein prediction,
providing the following algorithms:

- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales (e.g., amino acid scales)
- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on unbalanced and small datasets.

Moreover, AAanalysis provides functions for loading protein benchmark datasets (**load_data**),
amino acid scale sets (**load_scales**), and their in-depth two-level classification (**AAontology**).

Install
=======
**AAanalysis** can be installed from AAanalysis can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install -u aaanalysis

Contents
========

.. toctree::
   :maxdepth: 3

   Tutorial <tutorials>
   API <api>
   References <references>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`