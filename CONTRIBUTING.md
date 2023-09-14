# Contributing to AAanalysis

- [Introduction](#introduction)
- [Writing helpful bug reports](#writing-helpful-bug-reports)
- [Installing the latest version](#installing-the-latest-version)
- [Setting up a local development environment](#setting-up-a-local-development-environment)
- [Pull Requests (PRs)](#pull-requests-prs)
- [Documentation](#documentation)

## Introduction

Thank you for considering contributing to AAanalysis. AAanalysis is an open-source effort dedicated to providing a 
framework for interpretable protein prediction, and your involvement is invaluable.

You can contribute by:

- Reporting bugs and suggesting features in our GitHub issue tracker.
- Proposing and implementing improvements via Pull Requests.
- Engaging in discussions on the relevant topics.

If you're new to the project, consider looking for issues tagged with `good first issue`.

## Writing helpful bug reports

It's invaluable for maintainers when bug reports include a Minimal Reproducible Example (MRE). An MRE should be:

- Minimal: Use as few lines of code as possible that still showcase the problem.
- Self-contained: Ensure it includes all necessary data and imports.
- Reproducible: Verify that it consistently reproduces the problem.

For a guide on crafting effective bug reports, see [How To Craft Minimal Bug Reports](https://matthewrocklin.com/minimal-bug-reports).

## Installing the latest version

To obtain the latest version of aaanalysis, you can install directly from the master branch:

```bash
pip install git+https://github.com/breimanntools/aaanalysis.git@master
```

This allows you to test the latest developments before they're officially released.
Alternatively, if you plan to modify the source code, follow the steps below 
to set up a local development environment.

## Setting up a local development environment

### Fork and Clone

1. Fork the GitHub repository (https://github.com/breimanntools/aaanalysis)
2. Clone your fork:  
```bash
git clone https://github.com/YOUR_USERNAME/aaanalysis.git
```

### Install Dependencies
After you've cloned the repository, you can set up a Python environment and install all the required dependencies.

1. Navigate to project folder
```bash
cd aaanalysis
```
2. Crete a new isolated Python environment using `conda`:
```bash
conda create -n aanalysis python=3.9
conda activate aanalysis
```
3. Install dependencies using  `poetry`:
```bash
poetry install
```

### Unit Tests with Pytest

We use pytest for running unit tests. To run the tests locally, use:

```bash
pytest
```

This will execute all the test cases in the tests/ directory.

## Pull Requests (PRs)

For significant changes, please start by opening an Issue to discuss the proposed changes.
For minor adjustments, such as typo fixes, you can directly submit a PR.

Ensure your PR is:

- Concise: Keep the scope limited to make the review process smoother.
- Descriptive: Use clear branch names like `fix/data-loading-issue` or `doc/update-readme`.

When your PR is ready for review, ensure:

- It's updated with the master branch.
- All tests pass.

### Previewing changes on Pull Requests

The documentation is automatically built for each Pull Request, allowing for a convenient preview:

- Look for "All checks have passed", then click "Show all checks".
- Navigate to the check titled "docs/readthedocs.org".
- Click on "Details" to see the preview.

## Documentation

Documentation is a crucial part of the project. If you make any modifications to the documentation,
please ensure they render correctly.

### Coding Principles

We strive for a modular, robust, and easily extendable codebase. We achieve this by adhering to flat class 
hierarchies and functional programming principles, as outlined in [this paper](https://dl.acm.org/doi/10.5555/3288797). 
We also prioritize user-friendly interfaces, complete with descriptive error messages 
and [Python type hints](https://docs.python.org/3/library/typing.html).

### Naming Conventions

We aim to maintain interface consistency with established libraries like scikit-learn, pandas, matplotlib, and seaborn. 

For the sake of consistency, we employ two types of template classes:

- **Wrapper**: These classes are designed to extend scikit-learn models (or similar interfaces). They feature `.fit` 
and `.eval` methods for model training and evaluation, respectively. Examples include extending KMeans 
for specific functionalities like redundancy-reduction via AAclust.

- **Tool**: These standalone classes focus on specialized tasks, such as feature engineering for protein prediction. 
Each class offers `.run` and `.eval` methods to execute its complete processing pipeline and generate various evaluation metrics.

Both `Wrapper` and `Tool` classes come with accompanying plotting classes to visualize their analysis or evaluation results.

### Documentation style

- **Docstring Style**: We use Numpy-style docstrings. Learn more in the
[Numpy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html).
  
- **Markup Language**: Documentation is in reStructuredText (.rst). For an introduction, 
see this [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

- **Autodoc**: We use `sphinx.ext.autodoc` for automatic inclusion of docstrings in the documentation.

- **Further Details**: See `docs/source/conf.py` for more.


### Building the docs locally

To generate the documentation locally:

- Go to the `docs` directory.
- Execute `make html`.
- Open `_build/html/index.html` in a browser.
