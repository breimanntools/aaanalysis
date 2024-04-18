Release notes
=============

Version 0.1 (Beta Version)
--------------------------

v0.1.5 (Released 2024-04-18)
----------------------------

Added
~~~~~
- **Code of Conduct**: Introduced a Code of Conduct to foster a welcoming and inclusive community environment.
  We encourage all contributors to review the `Code of Conduct <https://github.com/breimanntools/aaanalysis/blob/master/CODE_OF_CONDUCT.md>`_
  to understand the expectations and responsibilities when participating in the project.

Changed
~~~~~~~
- **License Update**: Transitioned the project license from MIT to `BSD-3-Clause <https://github.com/breimanntools/aaanalysis/blob/master/LICENSE>`_
  to better align with our project's community engagement and protection goals. This change affects how the software
  can be used and redistributed.

Fixed
~~~~~
- **Multiprocessing**: Replaced native ``multiprocessing`` with the ``joblib`` module for CPP and internal feature matrix
  creation. This change prevents a ``RuntimeError`` that occurred when the main function is not explicitly used.

v0.1.4 (Released 2024-04-09)
----------------------------

Added
~~~~~
- **Installation Options**: Introduced separate installation profiles for the core and professional versions.
  The **core version** has reduced dependencies to enhance installation robustness, installable using ``pip install aaanalysis``.
  The professional version, designed for advanced usage, includes packages required for our explainable AI module
  such as SHAP, installable using ``pip install aaanalysis[pro]``.

Changed
~~~~~~~
- **API Improvements**: General improvement of API for consistency and higher user-friendliness.

Fixed
~~~~~
- **General Issues**: Fix of different check function related API issues.

Other
~~~~~
- **Python Dependency**: Updated the Python version compatibility from <= 3.10 to <= 3.12.

v0.1.3 (Released 2024-02-09)
----------------------------

Added
~~~~~
- **TreeModel**: Wrapper class of tree-based models for Monte Carlo estimates of predictions and feature importance.
  `See TreeModel <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.TreeModel.html>`_.
- **ShapExplainer**: A wrapper for SHAP (SHapley Additive exPlanations) explainers to obtain Monte Carlo estimates for
  feature impact. `See ShapExplainer <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.ShapExplainer.html>`_.
- **NumericalFeature**: Utility feature engineering class to process and filter numerical data structures.
  `See NumericalFeature <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.NumericalFeature.html>`_.
- **Load_feature**: Utility function to load feature sets for protein benchmarking datasets.
  `See load_features <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_features.html>`_.


Changed
~~~~~~~
- **API Improvements**: General improvement of API for consistency and higher user-friendliness.

Fixed
~~~~~
- **Interface**: Change of internal documentation decorator to hard-coded documentation for better IDE responsiveness.
- **General Issues**: Fix of different check function related API issues.

v0.1.2 (Released 2023-11-06)
----------------------------

Added
~~~~~
- **CPPPlot**: Plotting class for CPP features.
  `See CPPPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPPPlot.html>`_.
- **dPULearnPlot**: Plotting class for results of negative identifications by dPULearn.
  `See dPULearnPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearnPlot.html>`_.
- **AAclustPlot**: Plotting class for AAclust clustering results.
  `See AAclustPlot <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.AAclustPlot.html>`_.
- **Options**: Set system-level settings by a dictionary-like interface (similar to pandas).
  `See options <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.options.html>`_.
- **Plotting functions**: Extension of plotting utility functions.

Changed
~~~~~~~
- **API Improvements**: General improvement of API.

Fixed
~~~~~
- **API Improvements**: General improvement of API (Application Programming Interface).

Other
~~~~~
- **Python Dependency**: Supports Python versions 3.9 and 3.10.

v0.1.1 (Released 2023-09-11)
----------------------------
Test release of the first beta version.

v0.1.0 (Released 2023-09-11)
----------------------------
First release of the beta version including
`CPP <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPP.html>`_,
`dPULearn <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearn.html>`_,
and `AAclust <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.AAclust.html>`_ algorithms
as well as the
`SequenceFeature <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.SequenceFeature.html>`_
utility class and data loading functions
`load_dataset <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_dataset.html>`_
and `load_scales <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_scales.html>`_.
