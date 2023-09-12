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

- **Anaconda/Miniconda**: Package manager and environment manager
- **Virtualenv**: Virtual environment management for Python

### Anaconda/Miniconda

1. `conda create --name <env_name> python=<version>`: Create new environment.
2. `conda activate <env_name>`: Activate environment.
3. `conda deactivate`: Deactivate environment.
4. `conda list`: List installed packages.
5. `conda install <package>`: Install a package.

### Virtualenv

1. `virtualenv <env_name>`: Create new environment.
2. `source <env_name>/bin/activate`: Activate environment.
3. `deactivate`: Deactivate environment.
4. `pip list`: List installed packages.
5. `pip install <package>`: Install a package.

---

## Package Management

Package management tools manage libraries and dependencies.

- **pip**: The Python Package Installer
- **Poetry**: Dependency management and packaging
- **Setuptools**: Library for package management

### pip

1. `pip install <package>`: Install a package.
2. `pip uninstall <package>`: Uninstall a package.
3. `pip freeze > requirements.txt`: Generate requirements.
4. `pip install -r requirements.txt`: Install from requirements.
5. `pip show <package>`: Show package info.

### Poetry

1. `poetry init`: Initialize a new project.
2. `poetry add <package>`: Add a dependency.
3. `poetry build`: Build package.
4. `poetry publish`: Publish to PyPI.
5. `poetry install`: Install dependencies.

### Setuptools

1. `python setup.py install`: Install package.
2. `python setup.py sdist`: Create source distribution.
3. `python setup.py bdist_wheel`: Create built distribution.
4. `python setup.py test`: Run tests.
5. `python setup.py --help`: Show help.

---

## Testing

Testing frameworks validate that your code works as intended.

- **pytest**: Testing framework for Python
- **unittest**: Built-in Python library for unit testing

### pytest

1. `pytest`: Run all tests.
2. `pytest <file>`: Run specific tests.
3. `pytest -k <test_name>`: Run specific test.
4. `pytest --cov=<module>`: Generate coverage.
5. `pytest --fixtures`: Show fixtures.

### unittest

1. `python -m unittest discover`: Discover and run tests.
2. `python -m unittest <test_file>.<TestClass>`: Run specific class.
3. `python -m unittest <test_file>.<TestClass>.<test_method>`: Run specific test.
4. `python -m unittest -k <test_name>`: Run tests matching name.
5. `python -m unittest -v`: Verbose output.

---

## Documentation

Documentation tools help you maintain project documentation.

- **Sphinx**: Documentation generator
- **Read the Docs**: Online documentation hosting
- **Docstrings**: Inline code documentation

### Sphinx

1. `sphinx-quickstart`: Initialize documentation.
2. `make html`: Build HTML docs.
3. `make clean`: Remove build directory.
4. `make latexpdf`: Build PDF docs.
5. `sphinx-build -b linkcheck ./docs ./docs/_build`: Check links.

### Read the Docs

(Managed via a web GUI; bash commands are limited.)

### Docstrings

(Not applicable as Docstrings are inline comments; no bash commands.)

---

## Continuous Integration/Continuous Deployment (CI/CD)

CI/CD tools automate testing and deployment.

- **Travis CI**: Hosted CI/CD service
- **GitHub Actions**: CI/CD service built into GitHub
- **Jenkins**: Self-hosted CI/CD service

### Travis CI

(Managed via `.travis.yml`; no specific bash commands.)

### GitHub Actions

(Managed via `.github/workflows` YAML files; no bash commands needed.)

### Jenkins

(Managed via a web GUI; bash commands are limited.)

---

## Code Quality & Analysis

Code quality tools enforce best practices and standards.

- **Flake8**: Linting tool
- **Black**: Code formatter
- **Coverage.py**: Code coverage tool

### Flake8

1. `flake8`: Run on all files.
2. `flake8 <file>`: Run on specific file.
3. `flake8 --ignore=E1,E2`: Ignore errors.
4. `flake8 --max-line-length=120`: Set line length.
5. `flake8 --statistics`: Show stats.

### Black

1. `black .`: Format all files.
2. `black <file>`: Format specific file.
3. `black --check .`: Check formatting.
4. `black --diff`: Show changes.
5. `black --line-length 80`: Set line length.

### Coverage.py

1. `coverage run -m pytest`: Run tests and collect data.
2. `coverage report`: Terminal report.
3. `coverage html`: HTML report.
4. `coverage xml`: XML report.
5. `coverage erase`: Erase data.
