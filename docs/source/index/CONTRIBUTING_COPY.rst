.. Developer Notes:
    - This file summarizes Python dev conventions for this project.
    - Refer to 'Vision' for project aims and 'Documentation' for naming conventions.
    - This file mirrors CONTRIBUTING.rst: modify CONTRIBUTING.rst first, then update this copy.
    - Remove '/docs/source' from image paths for CONTRIBUTING_COPY.
    - EXCEPTION — the 'Agentic Engineering' section is RTD-only and has NO counterpart in
      CONTRIBUTING.rst (the root file drops it). When re-syncing the two files, do not delete
      this section. It is an inline RST port of /docs/guides/agentic_engineering.md (canonical).
    Some minor doc tools
    - You can use Traffic analytics (https://docs.readthedocs.io/en/stable/analytics.html) for doc traffic.
    - Check URLs with LinkChecker (bash: linkchecker ./docs/_build/html/index.html).

.. _contributing:

============
Contributing
============

.. contents::
  :local:
  :depth: 1

Introduction
============

Welcome and thank you for considering a contribution to AAanalysis! We are an open-source project focusing on
interpretable protein prediction. Your involvement is invaluable to us. Contributions can be made in the following ways:

- Filing bug reports or feature suggestions on our `GitHub issue tracker <https://github.com/breimanntools/aaanalysis/issues>`_.
- Submitting improvements via Pull Requests.
- Participating in project discussions.

Newcomers can start by tackling issues labeled `good first issue <https://github.com/breimanntools/aaanalysis/issues>`_.
Please email stephanbreimann@gmail.com for further questions or suggestions?

Project Vision
==============

Objectives
----------

- Establish a comprehensive toolkit for interpretable, sequence-based protein prediction.
- Enable robust learning from small and unbalanced datasets, which are common in life sciences.
- Integrate seamlessly with machine learning and explainable AI libraries such as `scikit-learn <https://scikit-learn.org/stable/>`_
  and `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_.
- Offer flexible interoperability with other Python packages like `biopython <https://biopython.org/>`_.

Non-goals
---------

- Reimplementation of existing solutions.
- Ignoring the biological context.
- Reliance on opaque, black-box models.

Principles
----------

- Algorithms should be biologically inspired and combine empirical insights with cutting-edge computational methods.
- We emphasize fair, accountable, and transparent machine learning, as detailed
  in `Interpretable Machine Learning with Python <https://www.packtpub.com/product/interpretable-machine-learning-with-python/9781800203907>`_.
- We're committed to offering diverse evaluation metrics and interpretable visualizations, aiming to extend to other aspects of
  explainable AI such as causal inference.


Bug Reports
===========

For effective bug reports, please include a Minimal Reproducible Example (MRE):

- **Minimal**: Include the least amount of code to demonstrate the issue.
- **Self-contained**: Ensure all necessary data and imports are included.
- **Reproducible**: Confirm the example reliably replicates the issue.

Further guidelines can be found `here <https://matthewrocklin.com/minimal-bug-reports>`_.


Development Installation
========================

Latest Version
--------------

To install the latest development version using pip, execute the following:

.. code-block:: bash

  pip install git+https://github.com/breimanntools/aaanalysis.git@master

Local Development Environment
-----------------------------

Fork and Clone the Repository
"""""""""""""""""""""""""""""

1. Fork the `repository <https://github.com/breimanntools/aaanalysis>`_
2. Clone your fork and enter the project directory:

.. code-block:: bash

  git clone https://github.com/YOUR_USERNAME/aaanalysis.git
  cd aaanalysis

Install Dependencies
""""""""""""""""""""

Navigate to the project folder and set up a Python environment.

**1. Navigate to project folder**

.. code-block:: bash

    cd aaanalysis

**2. Create and activate a virtual environment using ``venv``**

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # Use ``venv\Scripts\activate`` on Windows

**3a. Install using uv (recommended)**

This is the recommended installation method for contributors. It installs the package in editable mode
together with all development dependencies:

.. code-block:: bash

    uv pip install -e ".[dev]"

**3b. Install using pip**

.. code-block:: bash

    pip install -e ".[dev]"

**General Notes**

- ``pyproject.toml`` is the single source of truth for all dependencies.
- ``uv.lock`` ensures reproducible installs for contributors using uv.
- **Additional Requirement**: Some non-Python utilities might need to be installed separately, such as Pandoc.
- **Manage Dependencies**: Ensure dependencies are updated as specified in ``pyproject.toml``
  after pulling updates from the repository.

Run Unit Tests
""""""""""""""

We utilize `pytest <https://docs.pytest.org/en/7.4.x/>`_ and `hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_.

Run the full test suite with:

.. code-block:: bash

  pytest tests

Run a specific file or directory with:

.. code-block:: bash

  pytest path/to/test_file.py

This will execute all the test cases in the tests/ directory. Check out our
`README on testing <https://github.com/breimanntools/aaanalysis/blob/master/tests/README_TESTING>`_. See further
useful commands in our `Project Cheat Sheet <https://github.com/breimanntools/aaanalysis/blob/master/docs/guides/project_cheat_sheet.md>`_.


Pull Requests
=============

For substantial changes, start by opening an issue for discussion. For minor changes like typos, submit a pull request directly.

Ensure your pull request:

- Is focused and concise.
- Has a descriptive and clear branch name like ``fix/data-loading-issue`` or ``doc/update-readme``.
- Is up-to-date with the ``master`` branch.
- Passes all tests.

Preview Changes
---------------

To preview documentation changes in pull requests, follow the "docs/readthedocs.org" check link under "All checks have passed".

GitHub Push
-----------

Before pushing code changes to GitHub, test your changes and update any relevant documentation.
It's recommended to work on a separate branch for your changes. Follow these steps for pushing to GitHub:

1. **Create a Branch**: If not already done, create a new branch:

   .. code-block:: bash

       git checkout -b your-branch-name

2. **Stage, Commit, and Push**: Stage your changes, commit with a clear message, and push to the branch:

   .. code-block:: bash

       git add .
       git commit -m "Describe your changes"
       git push origin your-branch-name

3. **Open a Pull Request**: Visit the GitHub repository to create a pull request for your branch.


For more detailed instructions, see the official `GitHub documentation <https://docs.github.com/en>`_.

Documentation Standards
=======================

Documentation is a crucial part of the project. If you make any modifications to the documentation,
please ensure they render correctly.

Naming Conventions
------------------

We strive for consistency of our public interfaces with well-established libraries like
`scikit-learn <https://scikit-learn.org/stable/>`_, `pandas <https://pandas.pydata.org/>`_,
`matplotlib <https://matplotlib.org/>`_, and `seaborn <https://seaborn.pydata.org/>`_.

Class Templates
"""""""""""""""

We primarily use two class templates for organizing our codebase:

- **Wrapper**: Designed to extend models from libraries like scikit-learn. These classes contain `.fit` and `.eval` methods
  for model training and evaluation, respectively.

- **Tool**: Standalone classes that focus on specialized tasks, such as feature engineering for protein prediction.
  They feature `.run` and `.eval` methods to carry out the complete processing pipeline and generate various evaluation metrics.

The remaining classes should fulfill two further purposes, without being directly implemented using class inheritance.

- **Data visualization**: Supplementary plotting classes for `Wrapper` and `Tool` classes, named accordingly using a
  `Plot` suffix (e.g., 'CPPPlot'). These classes implement an `.eval` method to visualize the key evaluation measures.
- **Analysis support**: Supportive pre-processing classes  for `Wrapper` and `Tool` classes.

Function and Method Naming
""""""""""""""""""""""""""

We semi-strictly adhere to the naming conventions established by the aforementioned libraries. Functions/Methods
processing data values should correspond with the names specified in our primary `pd.DataFrame` columns, as defined in
`aaanalysis/_utils/_utils_constants.py`.

Code Philosophy
---------------

We aim for a modular, robust, and easily extendable codebase. Therefore, we adhere to flat class hierarchies
(i.e., only inheriting from `Wrapper` or `Tool` is recommended) and functional programming principles, as outlined in
`A Philosophy of Software Design <https://dl.acm.org/doi/10.5555/3288797>`_.
Our goal is to provide a user-friendly public interface using concise description and
`Python type hints <https://docs.python.org/3/library/typing.html>`_ (see also this Python Enhancement Proposal
`PEP 484 <https://peps.python.org/pep-0484/>`_
or the `Robust Python <https://www.oreilly.com/library/view/robust-python/9781098100650/>`_ book).
For the validation of user inputs, we use comprehensive checking functions with descriptive error messages.

Documentation Style
-------------------

- **Docstring Style**: We use the `Numpy Docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and
  adhere to the `PEP 257 <https://peps.python.org/pep-0257/>`_ docstring conventions. The authoritative,
  worked-out conventions (class/method templates, recurring-parameter baselines, citation rules, versioning,
  examples, and the rule → checker-code table) live in the :ref:`Docstring Style Guide <docstring_guide>`;
  it is the single source of truth and is enforced by an internal checker.

- **Code Style**: Please follow the `PEP 8 <https://peps.python.org/pep-0008/>`_ and
  `PEP 20 <https://peps.python.org/pep-0020/>`_ style guides for Python code.

- **Markup Language**: Documentation is in reStructuredText (.rst). See for an introduction (
  `reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_) and for
  cheat sheets (`reStructureText Cheatsheet <https://docs.generic-mapping-tools.org/6.2/rst-cheatsheet.html>`_ or
  `Sphinx Tutorial <https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_).

- **Autodoc**: We use `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_
  for automatic inclusion of docstrings in the documentation, including its
  `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_,
  `napoleon <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/#>`_, and
  `sphinx-design <https://sphinx-design.readthedocs.io/en/rtd-theme/>`_ extensions.

- **Further Details**: See our `conf.py <https://github.com/breimanntools/aaanalysis/blob/master/docs/source/conf.py>`_
  for more.

Documentation Layers
---------------------
This project's documentation is organized across four distinct layers, each with a specific focus and level of detail:

- **Docstrings**: Concise code description, with minimal usage examples and references to other layers (in 'See also').

- **Usage Principles**: Bird's-eye view with background and key principles, reflecting by selected code examples.

- **Tutorial**: Close-up on public interface, as step-by-step guide on essential usage with medium detail.

- **Tables**:  Close-up on data or other tabular overviews, with detailed explanation of columns and critical values.

See our reference order here (exceptions confirm the rules):

.. image :: /_artwork/diagrams/ref_order.png

The `API <https://aaanalysis.readthedocs.io/en/latest/api.html>`_ showcases **Docstrings** for our public objects
and functions. Within these docstrings, scientific
`References <https://aaanalysis.readthedocs.io/en/latest/index/references.html>`_
may be mentioned in their extended sections.

For the order in which docstring *See Also* entries reference these layers, see the
:ref:`Docstring Style Guide <docstring_guide>`.

Note that the Usage Principles documentation is open for direct linking to References,
Tutorials, and Tables, which can as well include links to References.

Building the Docs
-----------------

To generate the documentation locally:

- Go to the `docs` directory.
- Run `make html`.

.. code-block:: bash

  cd docs
  make html

- Open `_build/html/index.html` in a browser.

Building New PyPi package version
---------------------------------

AAanalysis is packaged and released using Poetry. To create and publish a new version on PyPi, follow these steps:

1. **Ensure Poetry is installed**

   If Poetry is not installed, install it with:

   .. code-block:: bash

      pip install poetry

2. **Update the version**

   Update the version number (**MAJOR.MINOR.PATCH**) in ``pyproject.toml``.

   Versioning follows semantic versioning:

   - **MAJOR**: incompatible API changes
   - **MINOR**: backward-compatible new functionality
   - **PATCH**: backward-compatible bug fixes

   Alternatively, Poetry can update the version automatically:

   .. code-block:: bash

      poetry version patch
      # or
      poetry version minor
      # or
      poetry version major

3. **Build the package**

   From the project root directory, build the distribution files:

   .. code-block:: bash

      poetry build

   This creates the source distribution and wheel files in the ``dist`` directory.

4. **Publish to PyPI**

   Upload the package to PyPI:

   .. code-block:: bash

      poetry publish

   You will need valid PyPI credentials or an API token configured for Poetry.

5. **Verify the upload**

   After publishing, verify that the package appears correctly on PyPI and that the metadata and files are accurate.

Agentic Engineering
===================
AAanalysis is developed with AI-assisted ("agentic") workflows. Just as important as the
tooling are the **automated gates every change must pass before a human merges it**. The
durable artifacts of this process (ADRs, ``CONTEXT.md``, ``CLAUDE.md`` / ``.claude/rules``,
and the code) are the single sources of truth; per-issue planning notes are ephemeral and not
committed. The canonical version of this protocol lives in
`docs/guides/agentic_engineering.md <https://github.com/breimanntools/aaanalysis/blob/master/docs/guides/agentic_engineering.md>`_.

Workflow (step by step)
-----------------------
1. **Pick the issue.** No skill required. Optionally clean up or generate the issue wording
   first with ``/triage`` or ``/to-issues`` (house style: ``docs/guides/issue_style_guide.rst``).
2. **``/grill-with-docs`` — the highest-leverage step.** A custom command that sharpens the
   spec against the *live* codebase and refreshes ``CONTEXT.md`` / ADRs **before any code is
   written**. The closest built-in, ``/init``, only (re)generates codebase docs — it does not
   do the adversarial spec-vs-reality pass. Keep grill for the real work; use ``/init`` only as
   a one-time bootstrap when ``CONTEXT.md`` does not exist yet.
3. **Branch + isolated worktree.** ``git switch -c <type>/<slug>`` off ``master`` (plain git).
   **Always pair it with** ``git worktree add`` **so each task gets its own checkout** —
   concurrent streams then cannot contaminate each other.
4. **Implement.** Plan mode. Use a structured plan for multi-file or architectural changes; drop
   to plain edits for trivial diffs. Plan mode is preferable when you want to approve the
   approach before commits land.
5. **Push → open PR.** ``gh`` / ``git``. A PR needs ≥1 commit; push a scaffolding commit and open
   a **draft PR early** so CI + the Read the Docs (RTD) preview run while you build.
6. **Automated review gate (machine-enforced, not eyeballed).** Run ``/review`` (reviews the PR
   diff) and ``/security-review`` (scans the pending diff for vulnerabilities). The quality gates
   below must be green first. **Never merge red** — automated checks *gate* the human review,
   they do not replace it.
7. **Refine on the same branch.** Back to plan mode; push more commits (the PR and the RTD
   preview update automatically).
8. **Keep current.** Periodically merge ``master`` → branch (plain git). A good fit for the
   ``/schedule`` skill: auto-**sync** each morning so you wake to a synced branch or an early
   conflict warning. **A scheduled job syncs only — it must never resolve conflicts or merge a
   branch to** ``master`` **unattended; it just flags you.**
9. **Arm auto-merge.** Once the step-6 review is green and you've read the RTD preview + PR diff,
   enable GitHub-native auto-merge: ``gh pr merge --auto --squash``. GitHub then merges the moment
   every required check passes and the branch is conflict-free, so **"never merge red" still
   holds** — a red check blocks the merge instead of completing it.
10. **Auto-fix red CI.** If GitHub Actions reports a failure, pull the failing logs
    (``gh run view --log-failed``, or ``gh run watch`` to follow live), reproduce locally, fix
    **forward on the same branch**, and push. Armed auto-merge re-arms itself and completes once
    the re-run is green. Disarm with ``gh pr merge --disable-auto`` if you need to hold the PR.
11. **Delete the branch.** Plain git, with permission.

Quality gates
-------------
.. note::

   **Never merge red.** These automated checks gate the merge. With ``gh pr merge --auto``
   GitHub enforces it for you — it completes the merge only once every required check is green
   and the branch is conflict-free.

.. list-table::
   :header-rows: 1
   :widths: 16 24 60

   * - Gate
     - "Green" means
     - AAanalysis mechanism
   * - **Tests**
     - full matrix passes
     - ``.github/workflows/main.yml`` ("Unit Tests"): ``pytest tests -m "not regression" -x -n auto`` on **Ubuntu py3.10–3.14** + **Windows py3.10 & 3.14** (Windows brackets min+max; the full Windows range and the ``-m regression`` exact-value CPP anchor run in the nightly).
   * - **Coverage**
     - **≥ 88 %** line coverage, package-only
     - ``.github/workflows/test_coverage.yml``: ``pytest … --cov=aaanalysis --cov-fail-under=88`` (+ Codecov ``patch`` / ``project``). Measured on the package only (``--cov=aaanalysis``, never ``--cov=./``).
   * - **Docs**
     - RTD builds; API + examples render
     - ``readthedocs.org`` check: Sphinx + nbsphinx. ``docs/source/conf.py`` runs ``export_example_notebooks_to_rst`` with ``nbsphinx_execute='never'`` — it renders committed notebook outputs, it does not execute them.
   * - **Docstrings**
     - numpydoc shape, named ``Returns``, per-method ``Examples`` include, no doc-vs-signature drift
     - the ``/docstrings`` skill: ``check_docstrings.py``, ``doc_signature_drift.py``, ``check_example_notebooks.py``. **Local gate** (not yet a CI job).
   * - **Notebooks execute**
     - every ``examples/`` + ``tutorials/`` notebook runs clean with embedded outputs
     - ``pytest --nbmake --nbmake-timeout=120 examples/ tutorials/``. **Local gate only — NOT in blocking CI.** Re-run and re-commit outputs before every push.
   * - **Architecture**
     - matches ``CONTEXT.md`` / ADRs; no cross-class backend imports or layering violations
     - machine: ``tests/unit/api_tests/test_backend_import_hygiene.py``. Spec / ADR conformance is human + ``/grill-with-docs``.
   * - **Parameter coverage**
     - every public parameter is exercised by name in tests
     - ``tests/unit/api_tests/test_param_coverage.py``.
   * - **Lint (errors)**
     - no syntax errors / undefined names
     - ``.github/workflows/codeql_analysis.yml`` ("code-quality" job): ``flake8 . --select=E9,F63,F7,F82``.
   * - **Style / types (full)**
     - black (88) / isort / flake8 (88) / mypy clean
     - **manual at review** — no pre-commit, ruff, or type-checker in CI (v2 target).
   * - **Security**
     - CodeQL clean
     - ``.github/workflows/codeql_analysis.yml`` ("Analyze" job).
   * - **Issue linkage**
     - the PR's lifecycle keyword is set per policy
     - ``Closes #NN`` in the **PR body** (see Process notes → *Issue lifecycle*).

Process notes (hard-won)
------------------------
- **Isolated worktrees per task.** Create a ``git worktree`` per branch so two concurrent
  streams never share one working tree / ``HEAD``. Sharing a single checkout caused commits to
  land on the wrong branch and uncommitted work to bleed across tasks. Do the edits in the
  worktree, commit, push, then ``git worktree remove``.
- **Issue lifecycle —** ``Closes #NN``. GitHub auto-closes a referenced issue on merge to the
  default branch when a closing keyword (``Closes`` / ``Fixes`` / ``Resolves #NN``) appears in
  **either the PR body or the merge (squash) commit message**. To **keep an issue open through a
  merge, remove the keyword from the PR body before merging** — fixing only the commit-message
  text is not enough.
- **Auto-merge + auto-fix loop.** ``gh pr merge --auto --squash`` is the default finish: it is
  safe because GitHub merges only on all-green + conflict-free, preserving *never merge red*. When
  a check goes red, **fix forward on the same branch** — the armed auto-merge needs no re-issuing
  and completes on the green re-run. Use ``gh pr merge --disable-auto`` to hold a PR.
- **Notebooks are a local-only gate.** Because nbmake is not in blocking CI, a broken example
  surfaces only on RTD (as wrong/un-rendered output) or in a local run. Always run
  ``pytest --nbmake examples/ tutorials/`` and commit fresh outputs before pushing.
