Welcome to the AAanalysis documentation
=======================================

**AAanalysis** (Amino Acid analysis) is a platform to enable interpretable protein prediction, providing the following algorithms:

- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales (e.g., amino acid scales)
- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on unbalanced and small datasets.

Moreover, AAanalysis provides functions for loading protein benchmark datasets, amino acid scale sets, and their in-depth two-level classification (**AAontology**).

Install
-------

AAanalysis can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install aaanalysis

or via GitHub repository:

.. code-block:: bash

   git clone https://github.com/breimanntools/aaanalysis

Set up a virtual environment and install dependencies using the `requirements.txt` file:

.. code-block:: bash

   pip install -r requirements.txt

Contents
--------
