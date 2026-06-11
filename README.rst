Welcome to the AAanalysis documentation!
========================================
..
    Developer Notes:
    Please make sure that badges in badges.rst (Read The Docs)
    and README.rst (GitHub) are the same.


.. =========================
.. Distribution
.. =========================

.. |License| image:: https://img.shields.io/github/license/breimanntools/aaanalysis.svg
   :target: https://github.com/breimanntools/aaanalysis/blob/master/LICENSE
   :alt: License

.. |PyPI Version| image:: https://img.shields.io/pypi/v/aaanalysis.svg
   :target: https://pypi.org/project/aaanalysis/
   :alt: PyPI - Package Version

.. |Supported Python Versions| image:: https://img.shields.io/pypi/pyversions/aaanalysis.svg
   :target: https://pypi.org/project/aaanalysis/
   :alt: Supported Python Versions

.. |Downloads| image:: https://pepy.tech/badge/aaanalysis
   :target: https://pepy.tech/project/aaanalysis
   :alt: Downloads

.. |GitHub Stars| image:: https://img.shields.io/github/stars/breimanntools/aaanalysis.svg?style=social
   :target: https://github.com/breimanntools/aaanalysis
   :alt: GitHub Stars


.. =========================
.. Status
.. =========================

.. |PyPI Status| image:: https://img.shields.io/pypi/status/aaanalysis.svg
   :target: https://pypi.org/project/aaanalysis/
   :alt: PyPI - Status

.. |Unit Tests| image:: https://github.com/breimanntools/aaanalysis/actions/workflows/main.yml/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions/workflows/main.yml
   :alt: CI/CD Pipeline

.. |Codecov| image:: https://codecov.io/gh/breimanntools/aaanalysis/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/breimanntools/aaanalysis
   :alt: Codecov

.. |CodeQL| image:: https://github.com/breimanntools/aaanalysis/actions/workflows/codeql_analysis.yml/badge.svg
   :target: https://github.com/breimanntools/aaanalysis/actions/workflows/codeql_analysis.yml
   :alt: CodeQL

.. |Documentation Status| image:: https://readthedocs.org/projects/aaanalysis/badge/?version=latest
   :target: https://aaanalysis.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. =========================
.. Table
.. =========================

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - **Distribution**
     - |License| |PyPI Version| |Supported Python Versions| |Downloads| |GitHub Stars|
   * - **Status**
     - |PyPI Status| |Unit Tests| |Codecov| |CodeQL| |Documentation Status|

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

Cheat Sheet
===========
A one-page visual reference to the AAanalysis workflow, the main classes grouped by
capability, the prediction levels (residue / domain / protein), and the
*Part × Split × Scale* feature ontology — every snippet uses the public API.

.. image:: docs/source/_artwork/cheat_sheet_preview.png
   :alt: AAanalysis cheat sheet (page 1 of 3)
   :target: https://aaanalysis.readthedocs.io/en/latest/index/cheat_sheet.html
   :width: 75%
   :align: center

`Download the cheat sheet (PDF, 3 pages) <docs/source/_static/cheat_sheet.pdf>`_ |
`open it interactively <https://aaanalysis.readthedocs.io/en/latest/index/cheat_sheet.html>`_

Install
=======
**AAanalysis** can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install aaanalysis

For extended features, including the explainable AI module:

.. code-block:: bash

    pip install "aaanalysis[pro]"

If you use uv, the equivalent commands are:

.. code-block:: bash

    uv pip install aaanalysis
    uv pip install "aaanalysis[pro]"

Contributing
============
We appreciate bug reports, feature requests, or updates on documentation and code. For details, please refer to
`Contributing Guidelines <CONTRIBUTING.rst>`_. These include specifics about AAanalysis and also notes on Test
Guided Development (TGD) using ChatGPT. For further questions or suggestions, please email stephanbreimann@gmail.com.

Agentic engineering
===================
AAanalysis is developed with AI-assisted ("agentic") workflows. Their durable artifacts — not ephemeral
planning notes — are the single sources of truth:

- **Issue triage & planning.** ``/github-issue-handoff`` audits every open issue against the coding
  standards and writes one prioritized plan to ``docs/guides/handoff_github_issues.md`` (refreshed on
  demand). Per-issue scratch files are **not** committed.
- **Spec & design.** ``/grill-with-docs`` interviews the contributor to resolve each design decision and
  sharpen terminology against the project glossary. Decisions that are hard to reverse are recorded as
  **Architecture Decision Records** (``docs/adr/``); the domain vocabulary lives in **CONTEXT.md**; the
  transient implementation plan stays outside the repo.
- **Standards & guardrails.** Implementation rules live in ``CLAUDE.md`` and the path-scoped files under
  ``.claude/rules/`` (frontend/backend split, docstrings, testing, reproducibility, …) and are backed by
  meta-tests — e.g. per-parameter test coverage and backend-import hygiene
  (``tests/unit/api_tests/``).

So the authoritative record of *why* the code looks the way it does is the **ADRs + CONTEXT.md +
CLAUDE.md/.claude/rules**, kept in sync as decisions are made.

The loop for a single change is: pick the issue → ``/grill-with-docs`` (spec vs. reality) →
branch in an isolated ``git worktree`` → implement → open a draft PR early → pass the automated
quality gates → human review → merge → delete the branch. The full step-by-step protocol and the
exact gate each change must clear (test matrix, ≥88 % coverage, RTD docs, architecture and
parameter-coverage meta-tests, lint/security) are documented in
`docs/guides/agentic_engineering.md <docs/guides/agentic_engineering.md>`_.

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
   `Nature Communications <https://www.nature.com/articles/s41467-025-60638-z>`__.

**dPULearn**:
   Breimann and Kamp *et al.* (2025),
   *Charting γ-secretase substrates by explainable AI*,
   `Nature Communications <https://www.nature.com/articles/s41467-025-60638-z>`__.
