Release notes
=============

Version 1.0 (Stable Version)
--------------------------------

v1.0.2 (2025-06-17)
--------------------------------

Improved
~~~~~~~~
- **Faster CPP Pipeline**: Major performance boost in ``CPP.run()`` through optimized generation and filtering of
  part-split-scale combinations. Depending on the number of scales, runtime is now **3–5× faster** on standard hardware.
- **Feature Map Enhancement**: ``CPP.feature_map()`` now includes a **top bar plot** showing cumulative feature importance
  per residue, improving interpretability. This visualization is also included in the CPP profile output.

Fixed
~~~~~
- **General Bug Fixes**: Minor fixes related to dependency resolution and edge-case behavior.
- **Documentation**: Removed inconsistencies in documentation for selected functions and plotting options.

Other
~~~~~
- **Branding**: Introduced updated logo and favicon (legacy version preserved under `docs/source/_artwork/logos/legacy/`).
- **Landing Page Visual**: Added a main conceptual sketch to the documentation landing page illustrating the core CPP idea
  — comparing two sequence sets to derive their critical difference, the **physicochemical signature**.


v1.0.1 (2025-01-29)
--------------------------------

Improved
~~~~~~~~
- **Pro Feature Accessibility**: Improved integration of **aaanalysis[pro]** features in IDEs. Clicking on a pro
  feature now directs users to its exact class implementation instead of the main ``__init__.py`` file.

- **Import Error Handling**: Improved error handling for missing dependencies in the **aaanalysis[pro]** version.
  If dependencies are installed but errors occur during import, users now receive the original import error messages.

Fixed
~~~~~
- **Feature Map Plot**: Resolved a potential mismatch in subcategory ordering between heatmap and bar plot
  in ``aa.cpp_plot().featuremap()``. Previously, subcategories with nearly identical names (e.g., "α-helix (C-term)"
  and "α-helix (C-term, out)") could appear in an inconsistent order.
- **General Bug Fixes**: Minor bug fixes to improve overall stability and functionality.

Other
~~~~~
- **Dependencies**: All dependencies have been updated to ensure compatibility with the latest versions, including
  full support for ``numpy>=2.0.0``.


v1.0.0 (2024-07-01)
--------------------------------

Added
~~~~~
- **SequencePreprocessor**: A utility data preprocessing class (data handling module).
- **comp_seq_sim**: A function for computing pairwise sequence similarity (data handling module).
- **filter_seq**: A function for redundancy-reduction of sequences (data handling module).
- **options**: Juxta Middle Domain (JMD) length can now be globally adjusted using the **jmd_n/c_len** options.

Changed
~~~~~~~
- **ShapModel**: The **ShapExplainer** class has been renamed to **ShapModel** for consistency with the **TreeModel**
  class and to avoid confusion with the ShapExplainer models from the
  `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ package.
- **Dependencies**: Biopython is now a required dependency only for the **aaanalysis[pro]** version.
- **Module Renaming**: The **Perturbation** module has been renamed to **Protein Design** module
  to better reflect its broad functionality.

Fixed
~~~~~
- **Multiprocessing**: Now supported directly at the script level, outside of any functions or classes,
  in the top-level of the script (global namespace).

Version 0.1 (Beta Version)
--------------------------

v0.1.5 (2024-04-18)
-------------------

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
- **Multiprocessing**: Replaced native ``multiprocessing`` with the ``joblib`` module for **CPP** and
  **internal feature matrix** creation. This change prevents a ``RuntimeError`` that occurred when the main function
  is not explicitly used.

Other
~~~~~
- **Dependencies**: Update the ``seaborn`` dependency to version 0.13.2 or higher to resolve the legend argument
  error present in versions earlier than 0.13

v0.1.4 (2024-04-09)
-------------------

Added
~~~~~
- **Installation Options**: Introduced separate installation profiles for the core and professional versions.
  The **core version** has reduced dependencies to enhance installation robustness, installable using ``pip install aaanalysis``.
  The **professional version**, designed for advanced usage, includes packages required for our explainable AI module
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

v0.1.3 (2024-02-09)
-------------------

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

v0.1.2 (2023-11-06)
-------------------

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

v0.1.1 (2023-09-11)
-------------------
Test release of the first beta version.

v0.1.0 (2023-09-11)
-------------------
First release of the beta version including
`CPP <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPP.html>`_,
`dPULearn <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearn.html>`_,
and `AAclust <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.AAclust.html>`_ algorithms
as well as the
`SequenceFeature <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.SequenceFeature.html>`_
utility class and data loading functions
`load_dataset <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_dataset.html>`_
and `load_scales <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.load_scales.html>`_.
