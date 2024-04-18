# Release notes

## Version 0.1 (beta version)

## v0.1.5 (Released 2024-04-18)

### Added
- **Code of Conduct**: Introduced a Code of Conduct to foster a welcoming and inclusive community environment.
    We encourage all contributors to review the Code of Conduct to understand the expectations and
    responsibilities when participating in the project.

### Changed
- **License update**: Transitioned the project license from MIT to BSD-3-Clause to better align with our project's
    community engagement and protection goals. This change affects how the software can be used and redistributed.

### Fixed
- **Multiprocessing**: Replaced native multiprocessing with the `joblib` module for CPP and internal feature matrix
    creation. This change prevents a `RuntimeError` that occurred when the main function is not explicitly used.


## v0.1.4 (Released 2024-04-09)

### Added
- **Installation options**: Introduced separate installation profiles for the core and professional versions.
    The core version now has reduced dependencies, enhancing installation robustness. For advanced usage, the
    professional version includes necessary packages such as SHAP for the AAanalysis explainable AI module
    (`TreeModel` and `ShapExplainer`). This version can be installed using `pip install aaanalysis[pro]`.

### Changed
- **API improvements**: General improvement of API for consistency and higher user-friendliness.

### Fixed
- **General issues**: Fix of different check function related API issues.

### Other
- **Python dependency**: Updated the Python version compatibility from <= 3.10 to <= 3.12.


## v0.1.3 (Released 2024-02-09)

### Added
- **TreeModel**: Wrapper class of tree-based models for Monte Carlo estimates of predictions and feature importance.
- **ShapExplainer**: A wrapper for SHAP (SHapley Additive exPlanations) explainers to obtain Monte Carlo estimates
    for feature impact.

### Changed
- **API improvements**: General improvement of API for consistency and higher user-friendliness.

### Fixed
- **Interface**: Change of internal documentation decorator to hard coded documentation for better IDE responsiveness.
- **General issues**: Fix of different check function related API issues.

### Other


## v0.1.2 (Released 2023-11-06)

### Added
- **CPPPlot**: Plotting class for CPP features.
- **dPULearnPlot**: Plotting class for results of negative identifications by dPULearn.
- **AAclustPlot**: Plotting class for AAclust clustering results.
- **options**: Set system-level settings by a dictionary-like interface (similar to pandas)

### Changed
- **API improvements**: General improvement of API (Application Programming Interface).

### Fixed
- **API improvements**: General improvement of API (Application Programming Interface).

### Other
- **Python dependency**: Requires Python <= 3.10 (>3.8).


## v0.1.1 (Released 2023-09-11)
Test release of first beta version.


## v0.1.0 (Released 2023-09-11)
First release of beta version including
`CPP <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.CPP.html>`_,
`dPULearn <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.dPULearn.html>`_,
and `AAclust <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.AAclust.html>`_ algorithms
as well as
`SequenceFeature <https://aaanalysis.readthedocs.io/en/latest/generated/aaanalysis.SequenceFeature.html>`_
utility class.
