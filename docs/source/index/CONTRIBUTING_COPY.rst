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

Optimizations that Change Output
""""""""""""""""""""""""""""""""

Most optimizations must be **byte-identical** to the code they replace. An
optimization that changes output at all — even only at the floating-point
last-bit (ULP) level or in tie-breaks — is acceptable only under the
**numerical-equivalence tolerance policy**, which defines three tiers
of acceptable change and the evidence each requires:

- **T1 — Byte-identical** (default): output is bit-for-bit identical. Covered by
  the change's own unit / parity tests; no extra evidence.
- **T2 — Numerically-equivalent**: ``np.allclose(atol=1e-10, rtol=0)`` on all
  numerical outputs **and** identical discrete decisions (labels, selected
  features / medoids, kept / dropped / ranked sets). Examples: ULP-level
  reductions, ``allclose`` distance/correlation reformulations.
- **T3 — Statistically-equivalent**: results differ but documented quality
  metrics (clustering quality, downstream AUC, kept-feature overlap) stay within
  an agreed, numerically stated band on named canonical datasets. Reserved for
  genuinely algorithmic changes (e.g. AAclust binary-search ``k``) — never a
  fallback for a change that could meet T2.

**Reviewer acceptance checklist** for a T2 / T3 optimization PR:

1. The PR **names its tier** and lands at the *strictest* tier the change can
   satisfy.
2. It links a validation harness (gitignored ``dev_scripts/perf_*_validate.py``
   pattern) that asserts the equivalence and benchmarks old-vs-new.
3. It states the **tolerance / band numerically** (T2: ``atol=1e-10, rtol=0`` +
   the discrete-decision artifacts that stay equal; T3: e.g. ΔAUC ≤ 0.005,
   kept-feature Jaccard ≥ 0.95) and names the canonical dataset(s).
4. It commits a **regression anchor** following the established pattern
   (``@pytest.mark.regression``, pinned to the canonical Linux / floor-Python
   cell, run only in the non-gating nightly) that freezes the decision artifact /
   value (T2) or the banded metric (T3).
5. Each previously-excluded optimization lands as **its own PR**.


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

- **Data Tables**:  Close-up on data or other tabular overviews, with detailed explanation of columns and critical values.

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

Versioning and Deprecation Policy
=================================

AAanalysis follows `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_
(**MAJOR.MINOR.PATCH**) and is **semver-strict from v1.x onward**. The public API
is exactly the set of symbols re-exported by ``aaanalysis/__init__.py`` (anything
starting with ``_`` is private and may change without notice).

Version truth
-------------

``master`` must **never** report a version that is already published on PyPI. The
version in ``pyproject.toml`` is maintained by hand and always names the *next,
unreleased* number, so a development checkout and a released install are never
indistinguishable to ``importlib.metadata`` (and therefore to bug reports, cached
environments, coding agents, and reproducibility records).

CI enforces this: the ``Version Guard`` workflow runs
``.github/scripts/check_version_ahead.py``, which fails when the declared version is
not strictly greater than the latest release published on PyPI (falling back to the
latest ``vX.Y.Z`` git tag when the network is unavailable). Run it locally with:

.. code-block:: bash

   python .github/scripts/check_version_ahead.py
   python .github/scripts/check_version_ahead.py --offline   # git tags only

The version is deliberately **not** derived from git tags (e.g. setuptools-scm),
which would attach ``.devN`` / ``+g<sha>`` suffixes to every commit. Release tags are
named ``vX.Y.Z``; the pre-1.0 ``0.1.1`` tag predates that convention and the guard
accepts both spellings.

Deprecation policy
------------------

Renaming or removing a public symbol is a breaking change. To give users a
migration window, such a change is staged across releases:

1. **Deprecate, don't remove.** Keep the symbol working and decorate it with
   ``aaanalysis.utils.deprecated(reason=..., version_removed=...)``. Calling it
   (or, for a class, instantiating it) then emits a ``DeprecationWarning`` naming
   the replacement and the planned removal version, and the docstring gains a
   deprecation note rendered in the API docs.

   .. code-block:: python

       import aaanalysis.utils as ut

       @ut.deprecated(reason="Use 'new_name' instead.", version_removed="1.2.0")
       def old_name(...):
           ...

2. **Ship at least one minor release** carrying the ``DeprecationWarning`` before
   the symbol is removed.
3. **Remove** the symbol only in a subsequent **minor** (or major) release, and
   record it under ``Removed`` in the changelog.

PATCH releases never rename or remove public symbols; MINOR releases add
backward-compatible functionality (and may *introduce* deprecations); MAJOR
releases may complete removals.

Changelog
---------

Every user-visible change (new public symbol, signature change, behavior change,
deprecation, or important bug fix) is recorded **in the same pull request** in two
places:

- ``CHANGELOG.md`` (repo root, `Keep a Changelog
  <https://keepachangelog.com/en/1.1.0/>`_ format) — a terse, one-line-per-change
  index under the top ``Unreleased`` section
  (``Added`` / ``Changed`` / ``Deprecated`` / ``Removed`` / ``Fixed``).
- ``docs/source/index/release_notes.rst`` — the narrative, RTD-rendered notes
  under the current ``Unreleased`` version, with cross-references and examples.

At release time, the ``Unreleased`` heading in both files is renamed to the new
version with its date.

Building New PyPi package version
---------------------------------

AAanalysis compiles a Cython kernel, so every release ships **platform wheels and a
source distribution**. Publishing is automated and token-free: **cutting a GitHub
Release is the publish action.** Publishing a Release triggers the canonical workflow
``.github/workflows/release.yml``, which builds the full wheel matrix (Linux, macOS
and Windows across every supported Python version) plus the sdist with
``cibuildwheel``, then uploads them to PyPI over **OIDC trusted publishing** -- no
long-lived API tokens and no manual ``uv publish`` step. Because every wheel is built
in CI, the published release installs cleanly with ``pip install aaanalysis``,
``uv add aaanalysis``, or any other installer, on every supported platform.

**One-time setup** (must already exist before the first release; needed again only if
the project is re-hosted or the environment is removed):

- a **PyPI trusted publisher** registered for this repository at
  https://pypi.org/manage/project/aaanalysis/settings/publishing/ , pointing at
  workflow ``release.yml`` and environment ``pypi``; and
- a GitHub deployment **environment named** ``pypi``.

Until both exist, PyPI rejects the OIDC exchange and the publish step fails closed.

To cut a release:

1. **Confirm the version to release.**

   ``master`` already carries the next unreleased version (see `Version truth`_), so
   the number in ``pyproject.toml`` is normally the one you are about to release --
   confirm it still matches the change set instead of bumping it a second time.
   Versioning follows semantic versioning:

   - **MAJOR**: incompatible API changes
   - **MINOR**: backward-compatible new functionality
   - **PATCH**: backward-compatible bug fixes

   If the number must change, edit ``[project] version`` in ``pyproject.toml``
   (``uv version --bump patch`` / ``minor`` / ``major``).

2. **Stamp the release notes and merge to master.**

   Rename the ``Unreleased`` heading in ``CHANGELOG.md`` and
   ``docs/source/index/release_notes.rst`` to this version with its release date, and
   merge that to ``master``. You release *from* this commit.

3. **(Optional) Local sanity build.**

   .. code-block:: bash

      uv build

   This writes an sdist and a *single* platform wheel to ``dist/`` -- handy for
   eyeballing the built metadata, but it is **not** what gets published (the release
   rebuilds every wheel in CI). Do not upload it by hand.

4. **Cut the GitHub Release.**

   Draft a new Release, target the stamped commit, and give it a **new tag**
   ``vX.Y.Z`` -- the Release creates the tag. Publishing the Release triggers
   ``release.yml``; watch the run to green. (The build jobs also run on a manual
   ``workflow_dispatch`` as a dry run, but the publish step runs **only** for a real
   Release.)

   .. code-block:: text

      GitHub -> Releases -> Draft a new release
        Choose a tag:  v1.1.0   ("Create new tag: v1.1.0 on publish"); target = release commit
        Release title: v1.1.0
        Description:   the release_notes / CHANGELOG entry for this version
      -> Publish release

5. **Verify the upload.**

   Confirm the version, files and metadata on https://pypi.org/project/aaanalysis/ ,
   then smoke-test a clean install from PyPI -- e.g. ``pip install aaanalysis`` in a
   fresh virtual environment (and/or ``uv add aaanalysis`` in a scratch project) --
   and check ``import aaanalysis as aa; aa.__version__``.

6. **Bump master ahead again.**

   **Immediately** bump ``[project] version`` in ``pyproject.toml`` to the next
   unreleased number (e.g. ``1.2.0``) and open a fresh ``Unreleased`` section in
   ``CHANGELOG.md`` and ``docs/source/index/release_notes.rst``. This keeps
   `Version truth`_ intact: until it lands, ``master`` reports a version that is
   already on PyPI and the ``Version Guard`` workflow fails -- by design, as the
   reminder that the bump is still owed.

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
Eight steps in three phases. Phases group the work; within the Build phase you iterate, so the
order there is natural rather than rigid.

**Prepare**

1. **Pick & sharpen the issue.** No skill required to start. Optionally clean up or generate the
   wording with ``/triage`` or ``/to-issues`` (house style: ``docs/guides/issue_style_guide.md``);
   for "what next?", ``/github-issues`` produces a prioritized, parallelization-aware plan.
2. **``/grill-with-docs`` — the highest-leverage step.** Sharpen the spec against the *live*
   codebase and refresh ``CONTEXT.md`` / ADRs **before any code is written**. The closest built-in,
   ``/init``, only (re)generates codebase docs — it does not do the adversarial spec-vs-reality
   pass; use ``/init`` only as a one-time bootstrap when ``CONTEXT.md`` does not exist yet.
3. **Branch + isolated worktree.** ``git switch -c <type>/<slug>`` off ``master`` (``fix/``,
   ``feat/``, ``doc/``, ``refactor/``), **always paired with** ``git worktree add`` so concurrent
   streams never share a checkout (see Process notes → *Isolated worktrees*).

**Build**

4. **Implement + open a draft PR early.** Use plan mode for multi-file or architectural changes
   (approve the approach before commits land); drop to plain edits for trivial diffs. Honor the
   path-scoped rules in ``.claude/rules/`` that auto-load for the files you touch. A PR needs ≥1
   commit, so push a scaffolding commit and open a **draft PR** early — CI and the Read the Docs
   (RTD) preview then run while you keep pushing to refine. **Push and open the PR *before* the
   human-review gate (step 6), not after** — opening the PR is what starts CI/RTD, so the checks run
   *while* the user reviews or decides to skip. **Before a substantive push, run the fast local unit
   gate** (``pytest tests -m "not regression" -x -n auto -c tests/pytest.ini``) — a local run catches
   the obvious break far faster than a red-CI round-trip. **No change is done until you have walked
   the ripple checklist below** — a code edit almost always lands alongside its docstring, example,
   test, and other mirrors *in the same PR*.
5. **Automated review gate (machine-enforced, not eyeballed).** Run ``/review`` (reviews the PR
   diff) and ``/security-review`` (scans the pending diff for vulnerabilities); for a substantial
   diff reach for ``/code-review high`` (or ``ultra`` for the deep cloud review) and ``/simplify``,
   and when public API or docstrings change run ``/docstrings``. The quality gates below must be
   green. **Never merge red** — automated checks *gate* the human review, they do not replace it.
   *Meanwhile, as a background concern:* periodically merge ``master`` → branch to
   stay current (a good fit for the ``/schedule`` skill: auto-**sync** each morning so you wake to
   a synced branch or an early conflict warning). **A scheduled job syncs only — it must never
   resolve conflicts or merge a branch to** ``master`` **unattended; it just flags you.**

**Review, merge & clean up**

6. **Human review gate — the PR is already up; the user picks how to review.** Because the PR was
   pushed and opened in step 4, the GitHub Actions + RTD preview are **already running while the user
   decides** (confirm with ``gh pr checks <n>`` / ``gh run list --branch <branch>``). This is a
   deliberate checkpoint for human judgement on the *content* of the change — **not** a decision about
   whether to push (that already happened) — distinct from and on top of the automated gates in step 5
   and CI. Do **not** advance to merge on your own; surface the fork and **wait for the user's
   decision**:

   - **(a) Manual PR review — iterate.** The user reviews the PR diff on GitHub and leaves comments.
     Address **each** comment by refactoring **forward on the same branch** (honor the ripple
     checklist and the auto-loaded ``.claude/rules/``), re-run the fast local gate, push, and report
     back per comment. Then **loop**: wait for the next round and repeat the *review → refactor →
     push* cycle until the user signals the review is complete (e.g. "merge it" / "looks good") —
     only then proceed to step 7. Each re-push re-triggers CI, and any armed auto-merge waits for the
     new green.
   - **(b) Skip review — approve + auto-merge.** The user opts out of a manual pass: post a short
     **approving review comment** (e.g. "Skipping manual review — automated gates green, all good") so
     the skip is recorded on the PR, then proceed to step 7. The PR merges and closes itself once CI
     is green.

   Recommend (a) for substantial or architectural diffs and (b) for trivial ones, but **never assume
   the answer — the user picks.**
7. **Arm auto-merge; fix-forward on red.** Once the user has cleared the review gate (step 6) — on the
   skip path, after the approving review comment — and you've read the RTD preview + PR diff, enable
   GitHub-native auto-merge: ``gh pr merge --auto --merge`` — a **merge commit, never** ``--squash``
   (squash rewrites the branch into a new SHA, which loses the individual commits *and* blinds the
   step-8 cleanup detection; merge commits keep ``git branch --merged`` / ``-d`` honest). The merge
   **method** is its own explicit decision — never fold ``--squash`` / ``--merge`` into the step-6
   skip-review option. GitHub
   merges the moment every required check passes and the branch is conflict-free, so **"never
   merge red" still holds** — a red check blocks the merge instead of completing it. If CI goes
   red, pull the failing logs (``gh run view --log-failed``, or ``gh run watch`` to follow live),
   reproduce locally, and fix **forward on the same branch**; armed auto-merge re-arms itself and
   completes on the green re-run. Disarm with ``gh pr merge --disable-auto`` to hold the PR, or
   skip ``--auto`` for a hard human gate.
8. **Clean up — gated on merge + a green** ``master``. Key cleanup off the **merge state, never a
   single CI run**: once ``gh pr view <n> --json state,mergedAt`` shows ``MERGED``, let the
   push-triggered workflows on ``master`` run and **wait for them to pass** — that confirms the
   merge didn't break anything the branch CI couldn't see (master may have moved under it). An
   intervening push, from this session or another, just re-runs checks and armed auto-merge waits
   for the *new* green, so the trigger survives the race. Then, **with permission** and after
   ``git fetch origin --prune`` + confirming no work is lost — because PRs land as **merge commits**,
   ``git branch --merged master`` lists the branch and ``git status --porcelain`` in the worktree is
   empty — ``git switch master`` → ``git worktree remove <path>`` (a tree with uncommitted work needs
   ``--force``, which also needs permission) → ``git worktree prune`` → ``git branch -d <branch>``.
   The remote branch is auto-deleted by GitHub on merge when the repo's "automatically delete head
   branches" setting is on; otherwise ``git push origin --delete <branch>``. A scheduled/unattended
   job may *detect and flag* "ready to clean up" but must never delete on its own — the same sync-only
   discipline as the step-5 background sync.

Propagate every change — the ripple checklist
----------------------------------------------
**No code change is complete until every surface that mirrors it is updated in the same PR**
(tutorials may trail in a separate PR, but never a different release). The package is heavily
cross-referenced — docstrings include example notebooks, the cheat sheet and tables are generated
from the public API, meta-tests assert consistency across files — so "I only touched code" is
almost never true. When you change code, walk this list and update what applies:

- **Docstrings** — numpydoc on every changed class / method / function; new ``[Key]_`` citations
  go in ``docs/source/index/references.rst``, new conventions in ``docstring_guide.rst``.
  *Enforced:* the ``/docstrings`` checkers (now blocking CI).
- **Public API** — ``aaanalysis/__init__.py`` ``__all__`` for any symbol added / removed / renamed
  (**CONFIRM-FIRST**); the API reference (``docs/source/api.rst`` + autosummary ``generated/``)
  flows from ``__all__`` + docstrings.
- **Examples** — ``examples/<abbr>_<method>.ipynb`` (one per public method, included into the
  docstring via ``.. include:: examples/<name>.rst``): cover every public parameter and re-run with
  executed outputs (``aa.display_df(...)``, ``plt.show()``).
- **Tutorials** — ``tutorials/*.ipynb`` when the change alters a taught workflow.
- **Protocols** — ``protocols/protocol<N>_*.ipynb`` when an end-to-end workflow changes
  (``docs/guides/protocol_style_guide.md``); they render under *Examples : Protocols* on RTD.
- **Tests** — unit tests for the change, plus the cross-file meta-tests that catch drift: parameter
  coverage (``tests/unit/api_tests/test_param_coverage.py``), class-abbreviation registry
  (``test_class_abbreviation_registry.py``), backend import hygiene, and the extras / missing-stub
  parity tests.
- **Cheat sheet** — ``docs/_cheatsheet/content.py`` (single source of truth → regenerate the html /
  pdf); every snippet must use only public ``__all__`` symbols with real signatures.
- **Data Tables** — ``docs/source/index/tables*.rst`` (generated by ``docs/source/create_tables_doc.py``)
  when scales / datasets / overview rows change.
- **Release Notes** — ``docs/source/index/release_notes.rst`` (the changelog): add an entry under
  the current *Unreleased* section.
- **Contributing** — ``CONTRIBUTING.rst`` **and** its RST port
  ``docs/source/index/CONTRIBUTING_COPY.rst`` when the dev process changes.
- **Glossary / ADRs** — ``CONTEXT.md`` when terminology shifts; a new ``docs/adr/NNNN-*.md`` for an
  architectural decision (ideally settled in step 2 via ``/grill-with-docs``).
- **Conventions** — ``CLAUDE.md`` / ``.claude/rules/*`` when the change establishes or alters a rule.
- **Build / deps** — ``pyproject.toml`` (**CONFIRM-FIRST**) for dependency, extras, or version
  bumps; ``aaanalysis/config.py`` (**CONFIRM-FIRST**) for a new global option.

Most of these surface late — a stale cheat sheet, a red meta-test, a wrong RTD render — not in the
fast unit job. Check them at implement time (step 4), not after CI goes red.

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
     - ``.github/workflows/main.yml`` ("Unit Tests"): ``pytest tests -m "not regression and not integration and not e2e" -x -n auto`` on **Ubuntu py3.10–3.14** + **Windows py3.10 & 3.14** (Windows brackets min+max; the full Windows range and the ``-m regression`` exact-value CPP anchor run in the nightly).
   * - **Integration & E2E**
     - cross-component seams + protocol workflows pass
     - ``.github/workflows/integration_e2e.yml`` ("Integration & E2E Tests"): ``pytest tests/integration tests/e2e -m "not regression" -n auto`` on **Ubuntu py3.10 + 3.14** (core-only, offline). The fourth master-gating workflow (push + PR to master); excluded from the Unit Tests matrix so it runs once.
   * - **Coverage**
     - **≥ 88 %** line coverage, package-only
     - ``.github/workflows/test_coverage.yml``: ``pytest … --cov=aaanalysis --cov-fail-under=88`` (+ Codecov ``patch`` / ``project``). Measured on the package only (``--cov=aaanalysis``, never ``--cov=./``).
   * - **Docs**
     - RTD builds; API + examples render
     - ``readthedocs.org`` check: Sphinx + nbsphinx. ``docs/source/conf.py`` runs ``export_example_notebooks_to_rst`` with ``nbsphinx_execute='never'`` — it renders committed notebook outputs, it does not execute them.
   * - **Docstrings**
     - numpydoc shape, named ``Returns``, per-method ``Examples`` include, no doc-vs-signature drift
     - the ``/docstrings`` skill: ``check_docstrings.py``, ``doc_signature_drift.py``, ``check_example_notebooks.py``. The first two are **blocking CI** in the ``codeql_analysis.yml`` "code-quality" job; ``check_example_notebooks`` runs there **advisory (non-blocking)** until the remaining notebook param-coverage gaps are cleared. All three also run locally via the skill.
   * - **Notebooks execute**
     - every ``examples/`` + ``tutorials/`` notebook runs clean with embedded outputs
     - ``pytest --nbmake --nbmake-timeout=120 examples/ tutorials/``. **Local gate only — NOT in blocking CI.** Re-run and re-commit outputs before every push.
   * - **Architecture**
     - matches ``CONTEXT.md`` / ADRs; no cross-class backend imports or layering violations
     - machine: ``tests/unit/api_tests/test_backend_import_hygiene.py``. Spec / ADR conformance is human + ``/grill-with-docs``.
   * - **Parameter coverage**
     - every public parameter is exercised by name in tests
     - ``tests/unit/api_tests/test_param_coverage.py`` — runs in the "Unit Tests" job (an ordinary test under ``tests/``, picked up by ``pytest tests``).
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
  worktree, commit, push, then ``git worktree remove``. **If you run parallel agents the
  per-task worktree is mandatory:** in a shared checkout a concurrent agent's commit/push can
  land *between* your ``git status`` and your commit. Mitigate even in a shared tree — re-check
  ``git status`` immediately before staging, commit **explicit pathspecs only** (never a blind
  ``git add -A`` / ``git commit -a``), and never commit, revert, or discard changes you did not
  make; stop and surface unexpected edits instead.
- **Issue lifecycle —** ``Closes #NN``. GitHub auto-closes a referenced issue on merge to the
  default branch when a closing keyword (``Closes`` / ``Fixes`` / ``Resolves #NN``) appears in
  **either the PR body or the merge commit message**. To **keep an issue open through a
  merge, remove the keyword from the PR body before merging** — fixing only the commit-message
  text is not enough.
- **Auto-merge + auto-fix loop.** ``gh pr merge --auto --merge`` (a **merge commit, never**
  ``--squash``) is the default finish: it is
  safe because GitHub merges only on all-green + conflict-free, preserving *never merge red*. When
  a check goes red, **fix forward on the same branch** — the armed auto-merge needs no re-issuing
  and completes on the green re-run. Use ``gh pr merge --disable-auto`` to hold a PR.
- **Notebooks are a local-only gate.** Because nbmake is not in blocking CI, a broken example
  surfaces only on RTD (as wrong/un-rendered output) or in a local run. Always run
  ``pytest --nbmake examples/ tutorials/`` and commit fresh outputs before pushing.
