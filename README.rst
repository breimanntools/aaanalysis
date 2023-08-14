Welcome to the AAanalysis documentation!
========================================

**AAanalysis** (Amino Acid analysis) is a Python framework for interpretable sequence-based protein prediction,
providing the following algorithms:

- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales (e.g., amino acid scales)
- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on unbalanced and small datasets.

Moreover, AAanalysis provides functions for loading protein benchmark datasets (**load_data**),
amino acid scale sets (**load_scales**), and their in-depth two-level classification (**AAontology**).

Install
-------

AAanalysis can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install -u aaanalysis

or via GitHub repository:

.. code-block:: bash

   git clone https://github.com/breimanntools/aaanalysis
   cd aaanalysis

Set up a virtual environment and install dependencies using poetry:

.. code-block:: bash

   poetry install

Citation
--------

If you use 'AAanalysis' in your research, please cite the appropriate publication:

**AAontology**:
   ´[Breimann23b]_´ Breimann et al. (2023),
   *AAontology: An ontology of amino acid scales for interpretable machine learning*,
   `bioRxiv <https://www.biorxiv.org/content/10.1101/2023.08.03.551768v1>`__.

**AAclust**:
   [Citation details and link if available]

**CPP**:
   [Citation details and link if available]

**dPULearn**:
   [Citation details and link if available]
