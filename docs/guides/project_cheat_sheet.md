# Project Cheat Sheet

This is an overview of different bash commands and step-by-step guides used in this project.

## Table of Contents

1. [Version Control](#version-control)
2. [Development Environment](#development-environment)
3. [Package Management](#package-management)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Continuous Integration/Continuous Deployment (CI/CD)](#cicd)
7. [Code Quality & Analysis](#code-quality)

---

## Version Control

Version control systems help you manage changes to source code over time.

- **Git**: Distributed version control system
- **GitHub**: Web-based platform for hosting Git repositories

### Git

1. `git init`: Initialize a new Git repository.
2. `git add .`: Stage all changes.
3. `git commit -m "message"`: Commit staged changes.
4. `git pull origin <branch>`: Fetch remote changes.
5. `git push origin <branch>`: Push local commits.

### GitHub

(Actions on GitHub are usually performed via GUI, so bash commands are limited.)

---

## Development Environment

Development environments manage dependencies and settings for your project.

- **uv**: Fast Python package and environment manager (recommended)
- **Virtualenv**: Virtual environment management for Python (alternative)

### uv

1. `uv venv`: Create a new virtual environment.
2. `uv venv --python 3.12`: Create environment with specific Python version.
3. `source .venv/bin/activate`: Activate environment (Mac/Linux).
4. `.venv\Scripts\activate`: Activate environment (Windows).

### Virtualenv (alternative)

1. `virtualenv <env_name>`: Create new environment.
2. `source <env_name>/bin/activate`: Activate environment (Mac/Linux).
3. `deactivate`: Deactivate environment.
4. `pip list`: List installed packages.

---

## Package Management

Package management tools manage libraries and dependencies.
All dependencies are defined in `pyproject.toml` — this is the single source of truth.

Install extras (works with uv, pip, and poetry):

    pip install aaanalysis              # core dependencies only
    pip install aaanalysis[pro]         # core + pro (biopython, shap, etc.)
    pip install aaanalysis[docs]        # core + docs (sphinx, nbsphinx, etc.)
    pip install aaanalysis[dev]         # core + pro + docs + dev tools

    uv pip install aaanalysis           # core dependencies only
    uv pip install aaanalysis[pro]      # core + pro (biopython, shap, etc.)
    uv pip install aaanalysis[docs]     # core + docs (sphinx, nbsphinx, etc.)
    uv pip install aaanalysis[dev]      # core + pro + docs + dev tools

### uv (recommended)

1. `uv add <package>`: Add a dependency to pyproject.toml.
2. `uv remove <package>`: Remove a dependency.
3. `uv sync`: Install all dependencies from lockfile.
4. `uv sync --extra dev`: Install with dev extras.
5. `uv pip install -e .`: Install package in editable mode.
6. `uv pip install -e .[dev]`: Install in editable mode with dev extras.
7. `uv lock`: Update the lockfile.
8. `uv build`: Build the package.
9. `uv publish`: Publish to PyPI.

### pip (alternative)

1. `pip install <package>`: Install a package.
2. `pip uninstall <package>`: Uninstall a package.
3. `pip install -e .[dev]`: Install in editable mode with dev extras.
4. `pip show <package>`: Show package info.

### Poetry (alternative)

1. `poetry install`: Install all dependencies.
2. `poetry install --extras pro`: Install with pro extras.
3. `poetry install --extras dev`: Install with dev extras.
4. `poetry add <package>`: Add a dependency.
5. `poetry remove <package>`: Remove a dependency.
6. `poetry build`: Build the package.
7. `poetry publish`: Publish to PyPI.

---

## Testing

Testing frameworks validate that your code works as intended.

- **pytest**: Testing framework for Python

### pytest

1. `pytest`: Run all tests.
2. `pytest <file>`: Run specific test file.
3. `pytest -k <test_name>`: Run specific test by name.
4. `pytest --durations=0`: Show all tests sorted by duration.
5. `pytest -x`: Stop after the first failure.
6. `pytest --maxfail=<num>`: Stop after the specified number of failures.
7. `pytest --lf`: Run only the tests that failed at the last run.
8. `pytest --ff`: Run the last failures first, then all tests.
9. `pytest --cov`: Generate coverage report for the entire codebase.
10. `pytest --cov=<module>`: Generate coverage for a specific module.
11. `pytest --cov-report=<type>`: Specify coverage report type (html, xml, term).
12. `pytest -v`: Increase verbosity of the output.
13. `pytest -q`: Decrease verbosity of the output.
14. `pytest --tb=short`: Display a shorter traceback format.
15. `pytest --disable-warnings`: Disable warnings display during test run.

---

## Documentation

Documentation tools help you maintain project documentation.

- **Sphinx**: Documentation generator
- **Read the Docs**: Online documentation hosting (uses `aaanalysis[docs]` via `.readthedocs.yaml`)
- **Docstrings**: Inline code documentation

### Sphinx

1. `sphinx-quickstart`: Initialize documentation.
2. `make html`: Build HTML docs (run from `docs/`).
3. `make clean`: Remove build directory.
4. `make latexpdf`: Build PDF docs.
5. `sphinx-build -b linkcheck ./docs ./docs/_build`: Check links.

### Read the Docs

(Managed via `.readthedocs.yaml` in the repo root and the RTD web GUI.)

### Sphinx directives

    :mod: - Reference a Python module.
    Example: :mod:`os` would link to the documentation for the os module.

    :func: - Reference a Python function.
    Example: :func:`print` would link to the documentation for the print function.

    :data: - Reference a module-level variable.
    Example: :data:`sys.path` would link to the documentation for the sys.path variable.

    :const: - Reference a "constant". This can be any immutable primitive data type (like a string or number).
    Example: :const:`True` would link to the documentation where the Python constant True is described.

    :class: - Reference a Python class.
    Example: :class:`matplotlib.axes.Axes` would link to the documentation for the Axes class in the matplotlib.axes module.

    :meth: - Reference a method of a Python class.
    Example: :meth:`str.split` would link to the documentation for the split method of the str class.

    :attr: - Reference a class attribute.
    Example: :attr:`Exception.args` would link to the documentation for the args attribute of the Exception class.

    :exc: - Reference a Python exception.
    Example: :exc:`ValueError` would link to the documentation for the ValueError exception.

---

## Continuous Integration/Continuous Deployment (CI/CD)

CI/CD tools automate testing and deployment.

- **GitHub Actions**: CI/CD service built into GitHub (workflows in `.github/workflows/`)

### GitHub Actions workflows

- **Test Coverage** (`test_coverage.yml`): Runs on push/PR to master. Installs `.[pro]`, runs pytest with coverage, uploads to Codecov.
- **Build Wheels** (`build_wheels.yml`): Runs on release. Builds wheels via cibuildwheel and publishes to PyPI.

(Workflows are managed via YAML files in `.github/workflows/`; no bash commands needed.)

---

## Code Quality & Analysis

Code quality tools enforce best practices and standards.

- **Flake8**: Linting tool
- **Black**: Code formatter
- **Coverage.py**: Code coverage tool

### Flake8

1. `flake8`: Run on all files.
2. `flake8 <file>`: Run on specific file.
3. `flake8 --ignore=E1,E2`: Ignore specific errors.
4. `flake8 --max-line-length=120`: Set line length.
5. `flake8 --statistics`: Show stats.

### Black

1. `black .`: Format all files.
2. `black <file>`: Format specific file.
3. `black --check .`: Check formatting without making changes.
4. `black --diff`: Show what changes would be made.
5. `black --line-length 80`: Set line length.

### Coverage.py

1. `coverage run -m pytest`: Run tests and collect coverage data.
2. `coverage report`: Terminal report.
3. `coverage html`: HTML report.
4. `coverage xml`: XML report.
5. `coverage erase`: Erase collected data.