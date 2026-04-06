.. Developer Notes:
    - This file summarizes Python dev conventions for this project.
    - Refer to 'Vision' for project aims and 'Documentation' for naming conventions.
    - Only modify  CONTRIBUTING.rst and then update the /docs/source/index/CONTRIBUTING_COPY.rst.
    - Remove '/docs/source' from image paths for CONTRIBUTING_COPY.
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
useful commands in our `Project Cheat Sheet <https://github.com/breimanntools/aaanalysis/blob/master/docs/project_cheat_sheet.md>`_.


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
  adhere to the `PEP 257 <https://peps.python.org/pep-0257/>`_ docstring conventions.

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

.. image :: /docs/source/_artwork/diagrams/ref_order.png

The `API <https://aaanalysis.readthedocs.io/en/latest/api.html>`_ showcases **Docstrings** for our public objects
and functions. Within these docstrings, scientific
`References <https://aaanalysis.readthedocs.io/en/latest/index/references.html>`_
may be mentioned in their extended sections. 

For additional links in docstrings, use the *See Also* section in this order:

- `Usage Principles <https://aaanalysis.readthedocs.io/en/latest/index/usage_principles.html>`_,
- `Tables <https://aaanalysis.readthedocs.io/en/latest/index/tables.html>`_,
- `Tutorials <https://aaanalysis.readthedocs.io/en/latest/tutorials.html>`_. 

Only include **External library** references when absolutely necessary. 
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

Advanced: Test with ChatGPT
===========================
To optimize testing, use ChatGPT with the template below and fill in the blank spaces between ``START OF CODE``
and ``END OF CODE``. Examples of testing templates can be found here:
`AAanalysis unit tests <https://github.com/breimanntools/aaanalysis/blob/master/tests/unit/>`_.

.. code-block:: none

    "
    Generate test functions for a given TARGET FUNCTION using the style of the provided TESTING TEMPLATE. Please take your time to ensure thoroughness and accuracy.

    Inputs:
    TARGET FUNCTION:
    - START OF CODE
    -----------------------------------------
    [your code here]
    -----------------------------------------
    - END OF CODE

    TESTING TEMPLATE:
    - START OF CODE
    -----------------------------------------
    [your code]
    -----------------------------------------
    - END OF CODE

    **Key Directive**: For the Normal Cases Test Class, EACH function MUST test ONLY ONE individual parameter of the TARGET FUNCTION using Hypothesis for property-based testing. This is crucial.

    Requirements:

    1. Normal Cases Test Class:
    - Name: 'Test[TARGET FUNCTION NAME]'.
    - Objective: Test EACH parameter *INDIVIDUALLY*.
    - Tests: Test EACH parameter, at least 10 positive and 10 negative tests for this class.

    2. Complex Cases Test Class:
    - Name: 'Test[TARGET FUNCTION NAME]Complex'.
    - Objective: Test combinations of the TARGET FUNCTION parameters.
    - Tests: At least 5 positive and 5 negative that intricately challenge the TARGET FUNCTION.

    3. General Guidelines:
    - Use Hypothesis for property-based testing, but test parameters individually for the Normal Cases Test Class .
    - Tests should be clear, concise, and non-redundant.
    - Code must be complete, without placeholders like 'TODO', 'Fill this', or 'Add ...'.
    - Explain potential issues in the TARGET FUNCTION.

    Output Expectations:
    - Two test classes: one for normal cases (individual parameters) and one for complex cases (combinations).
    - In Normal Cases, one function = one parameter tested.
    - Aim for at least 30 unique tests, totaling 150+ lines of code.

    Reminder: In Normal Cases, it's crucial to test parameters individually. Take your time and carefully create the Python code for all cases!
    "

ChatGPT has a token limit, which may truncate responses. To continue, simply ask 'continue processing' or something
similar. Repeat as necessary and compile the results.

We recommend the following workflow:

1. Repeat the prompt in new ChatGPT sessions until most of the positive test cases are covered.
2. Adjust the testing script manually such that all positive tests are passed.
3. Continue in the same session, sharing the revised script, and request the creation of negative tests.
4. Finally, provide the complete testing script, including positive and negative cases, and request the development
   of complex test cases

Test Guided Development (TGD)
-----------------------------
Leverage ChatGPT to generate testing scripts and refine your code's functionality and its interface. If ChatGPT
struggles or produces erroneous tests, it often indicates ambiguities or complexities in your function's logic,
variable naming, or documentation gaps, especially regarding edge cases. Address these insights to ensure intuitive
and robust code design through the TGD approach.

**Essential Strategies for Effective TGD**:

- **Isolated Functionality Testing**: Test one function or method at a time, adhering to unit testing principles.
  Provide an entire and well-documented function. The better the docstring, the more comprehensive our automatically
  generated tests will be.

- **Isolated Test Sessions**: Start each test scenario in a new ChatGPT session to maintain clarity and
  prevent context overlap, ensuring focused and relevant test generation.

- **Consistent Template Usage**: Align your test creation with existing templates for similar functionalities,
  utilizing them as a structured guide to maintain consistency in your test design.

- **Initial Test Baseline**: Aim for an initial set of tests where about 25% pass, providing a foundational baseline
  that identifies primary areas for iterative improvement in both tests and code.

- **Iterative Refinement and Simplification**: Use ChatGPT-generated tests to iteratively refine your code, especially
  if repeated test failures indicate areas needing clarification or simplification in your function's design.

Through an iterative TGD process, you can systematically uncover and address any subtleties or complexities in your
code, paving the way for a more robust and user-friendly application.
