Welcome to the AAanalysis documentation!
========================================
..
    Developer Notes:
    Please update badges in README.rst and vice versa

.. |Build Status| image:: https://github.com/breimanntools/aaanalysis/workflows/Build/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions
   :alt: Build

.. |Python Check| image:: https://github.com/breimanntools/aaanalysis/workflows/Python-check/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions
   :alt: Python-check

.. |PyPI Status| image:: https://img.shields.io/pypi/status/aaanalysis.svg
   :target: https://pypi.org/project/aaanalysis/
   :alt: PyPI - Status

.. |Supported Python Versions| image:: https://img.shields.io/pypi/pyversions/aaanalysis.svg
   :target: https://pypi.python.org/pypi/aaanalysis
   :alt: Supported Python Versions

.. |PyPI Version| image:: https://img.shields.io/pypi/v/aaanalysis.svg
   :target: https://pypi.python.org/pypi/aaanalysis
   :alt: PyPI - Package Version

.. |Conda Version| image:: https://anaconda.org/conda-forge/aaanalysis/badges/version.svg
   :target: https://anaconda.org/conda-forge/aaanalysis
   :alt: Conda - Package Version

.. |Documentation Status| image:: https://readthedocs.org/projects/aaanalysis/badge/?version=latest
   :target: https://aaanalysis.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |License| image:: https://img.shields.io/github/license/breimanntools/aaanalysis.svg
   :target: https://github.com/breimanntools/aaanalysis/blob/master/LICENSE
   :alt: License

.. |Downloads| image:: https://pepy.tech/badge/aaanalysis
   :target: https://pepy.tech/project/aaanalysis
   :alt: Downloads

..
    Missing badges
    |Build Status| |Python Check| |Conda Version|

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **Package**
     - |PyPI Status| |PyPI Version| |Supported Python Versions| |Downloads| |License|
   * - **Testing**
     - |Documentation Status|

**AAanalysis** (Amino Acid analysis) is a Python framework for interpretable sequence-based protein prediction.
Its foundation are the following algorithms:

- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein
  sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on
  unbalanced and small datasets.
- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales
  (e.g., amino acid scales).

In addition, AAanalysis provide functions for loading various protein benchmark datasets, amino acid scales,
and their two-level classification (**AAontology**). We combined **CPP** with the explainable
AI  `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ framework to explain sample level predictions with
single-residue resolution.

If you are looking to make publication-ready plots with a view lines of code, see our
`Plotting Prelude <https://aaanalysis.readthedocs.io/en/latest/generated/plotting_prelude.html>`_.


You can find the official documentation at `Read the Docs <https://aaanalysis.readthedocs.io/en/latest/>`_.

Install
=======
**AAanalysis** can be installed either from `PyPi <https://pypi.org/project/aaanalysis>`_ or
`conda-forge <https://anaconda.org/conda-forge/aaanalysis>`_:

.. code-block:: bash

   pip install -u aaanalysis
   or
   conda install -c conda-forge aaanalysis

**Note**: Please use Python 3.9 and pip to avoid any dependency issues. Support for Python 3.10 to 3.12 is
planned for the next release.

Contributing
============
We appreciate bug reports, feature requests, or updates on documentation and code. For details, please refer to
`Contributing Guidelines <CONTRIBUTING.rst>`_. These include specifics about AAanalysis and also notes on Test
Guided Development (TGD) using ChatGPT. For further questions or suggestions, please email stephanbreimann@gmail.com.

Citations
=========
If you use AAanalysis in your work, please cite the respective publication as follows:

**AAclust**:
   Breimann and Frishman (2024a),
   *AAclust: k-optimized clustering for selecting redundancy-reduced sets of amino acid scales*,
   `bioRxiv <https://www.biorxiv.org/content/10.1101/2024.02.04.578800v1>`__.

**AAontology**:
   Breimann *et al.* (2024b),
   *AAontology: An ontology of amino acid scales for interpretable machine learning*,
   `bioRxiv <https://www.biorxiv.org/content/10.1101/2023.08.03.551768v1>`__.

**CPP**:
   Breimann and Kamp *et al.* (2024c),
   *Interpretable feature engineering by CPP reveals the physicochemical signature of γ-secretase substrates*,
   .. # Link if available

**dPULearn**:
   Breimann and Kamp *et al.* (2024c),
   *Interpretable feature engineering by CPP reveals the physicochemical signature of γ-secretase substrates*,
   .. # Link if available
