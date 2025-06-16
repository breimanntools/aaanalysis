Welcome to the AAanalysis documentation!
========================================
..
    Developer Notes:
    Please make sure that badges in badges.rst (Read The Docs)
    and README.rst (GitHub) are the same.

.. Group 1: Package badges
.. |PyPI Status| image:: https://img.shields.io/pypi/status/aaanalysis.svg
   :target: https://pypi.org/project/aaanalysis/
   :alt: PyPI - Status

.. |PyPI Version| image:: https://img.shields.io/pypi/v/aaanalysis.svg
   :target: https://pypi.python.org/pypi/aaanalysis
   :alt: PyPI - Package Version

.. |Supported Python Versions| image:: https://img.shields.io/pypi/pyversions/aaanalysis.svg
   :target: https://pypi.python.org/pypi/aaanalysis
   :alt: Supported Python Versions

.. |Downloads| image:: https://pepy.tech/badge/aaanalysis
   :target: https://pepy.tech/project/aaanalysis
   :alt: Downloads

.. |License| image:: https://img.shields.io/github/license/breimanntools/aaanalysis.svg
   :target: https://github.com/breimanntools/aaanalysis/blob/master/LICENSE
   :alt: License

.. Group 2: Testing badges
.. |Unit Tests| image:: https://github.com/breimanntools/aaanalysis/actions/workflows/main.yml/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions/workflows/main.yml
   :alt: CI/CD Pipeline

.. |CodeQL| image:: https://github.com/breimanntools/aaanalysis/actions/workflows/codeql_analysis.yml/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions/workflows/codeql_analysis.yml
   :alt: CodeQL

.. |Codecov| image:: https://codecov.io/gh/breimanntools/aaanalysis/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/breimanntools/aaanalysis
   :alt: Codecov

.. |Documentation Status| image:: https://readthedocs.org/projects/aaanalysis/badge/?version=latest
   :target: https://aaanalysis.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. Group 3: Potential badges for future
.. |Conda Version| image:: https://anaconda.org/conda-forge/aaanalysis/badges/version.svg
   :target: https://anaconda.org/conda-forge/aaanalysis
   :alt: Conda - Package Version


..
    Missing badges
    |Conda Version|

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - **Package**
     - |PyPI Status| |PyPI Version| |Supported Python Versions| |Downloads| |License|
   * - **Testing**
     - |Unit Tests| |CodeQL| |Codecov| |Documentation Status|

.. image:: docs/source/_artwork/logos/model_AAanalysis.png
   :alt: Overview of AAanalysis components
   :align: center
   :width: 100%

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
**AAanalysis** can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install aaanalysis

For extended features, including our explainable AI module, please use the 'professional' version:

.. code-block:: bash

   pip install aaanalysis[pro]

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
   `Bioinformatics Advances <https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae165/7852846>`__.

**AAontology**:
   Breimann *et al.* (2024b),
   *AAontology: An ontology of amino acid scales for interpretable machine learning*,
   `Journal of Molecular Biology <https://www.sciencedirect.com/science/article/pii/S0022283624003267>`__.

**CPP**:
   Breimann and Kamp *et al.* (2025),
   *Charting γ-secretase substrates by explainable AI*,
   **Nature Communications** (link if available)

**dPULearn**:
   Breimann and Kamp *et al.* (2025),
   *Charting γ-secretase substrates by explainable AI*,
   **Nature Communications** (link if available)
