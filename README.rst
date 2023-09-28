Welcome to the AAanalysis documentation
=======================================
.. Developer Notes:
    Please update badges in README.rst and vice versa
.. image:: https://github.com/breimanntools/aaanalysis/workflows/Build/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions
   :alt: Build Status

.. image:: https://github.com/breimanntools/aaanalysis/workflows/Python-check/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions
   :alt: Python-check

.. image:: https://img.shields.io/pypi/status/aaanalysis.svg
   :target: https://pypi.org/project/aaanalysis/
   :alt: PyPI - Status

.. image:: https://img.shields.io/pypi/pyversions/aaanalysis.svg
   :target: https://pypi.python.org/pypi/aaanalysis
   :alt: Supported Python Versions

.. image:: https://img.shields.io/pypi/v/aaanalysis.svg
   :target: https://pypi.python.org/pypi/aaanalysis
   :alt: PyPI - Package Version

.. image:: https://anaconda.org/conda-forge/aaanalysis/badges/version.svg
   :target: https://anaconda.org/conda-forge/aaanalysis
   :alt: Conda - Package Version

.. image:: https://readthedocs.org/projects/aaanalysis/badge/?version=latest
   :target: https://aaanalysis.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/breimanntools/aaanalysis.svg
   :target: https://github.com/breimanntools/aaanalysis/blob/master/LICENSE
   :alt: License

.. image:: https://pepy.tech/badge/aaanalysis
   :target: https://pepy.tech/project/aaanalysis
   :alt: Downloads

**AAanalysis** (Amino Acid analysis) is a Python framework for interpretable sequence-based protein prediction,
providing the following algorithms:

- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales (e.g., amino acid scales)
- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on unbalanced and small datasets.

Moreover, AAanalysis provides functions for loading protein benchmark datasets (**load_data**),
amino acid scale sets (**load_scales**), and their in-depth two-level classification (**AAontology**).

If you are looking to make publication-ready plots with a view lines of code, see our
`Plotting Prelude <https://https://aaanalysis.readthedocs.io/en/latest/generated/plotting_prelude.htmll>`_.

Install
=======
**AAanalysis** can be installed either from `PyPi <https://pypi.org/project/aaanalysis>`_ or
`conda-forge <https://anaconda.org/conda-forge/aaanalysis>`_:

.. code-block:: bash

   pip install -u aaanalysis
   or
   conda install -c conda-forge aaanalysis

Contributing
============
We appreciate bug reports, feature requests, or updates on documentation and code. For details, please refer to
`Contributing Guidelines <CONTRIBUTING.rst>`_. These include specifics about AAanalysis and also notes on Test
Guided Development (TGD) using ChatGPT. For further questions or suggestions, please email stephanbreimann@gmail.com.

Citations
=========
If you use AAanalysis in your work, please cite the respective publication as follows:

**AAclust**:
   [Citation details and link if available]

**AAontology**:
   Breimann, Kamp, Steiner, Frishman (2023),
   *AAontology: An ontology of amino acid scales for interpretable machine learning*,
   `bioRxiv <https://www.biorxiv.org/content/10.1101/2023.08.03.551768v1>`__.

**CPP**:
   [Citation details and link if available]

**dPULearn**:
   [Citation details and link if available]
