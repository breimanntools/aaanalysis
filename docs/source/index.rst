..
   Developer Notes:
   This is the landing page for the AAanalysis documentation using Sphinx, containing the root `toctree` directive.
   The documentation will be hosted on Read the docs.
..

Welcome to the AAanalysis documentation!
========================================
.. include:: index/badges.rst
.. include:: index/overview.rst

Find Your Way Around
====================
The documentation is organized into pillars, each answering one question:

- **Getting Started** — *How do I install AAanalysis and get a first result?* Fast
  onboarding, no deep theory.
- **Tutorials** — *How does this tool work?* Tool-level teaching of the AAanalysis
  building blocks — each tool, its parameters, and its outputs. The *mechanics*;
  Protocols reuse these and link back rather than repeating them.
- **Protocols** — *How do I design a valid, end-to-end analysis?* Concept-level
  workflow teaching that builds the mental model for when and why to reach for each
  tool — linking to the Tutorials for the mechanics, so the two never overlap.
- **Use Cases** — *How do I adapt a full biological analysis?* End-to-end biological
  case studies *(coming soon)*.
- **API** — *What is the exact signature or parameter?* Technical reference only, no
  teaching narrative.

.. admonition:: Which section do I want?
   :class: tip

   - You are **new** and want a first result → **Getting Started**.
   - You want to learn **one specific tool** (its parameters and outputs) → **Tutorials**.
   - You want to **design a valid workflow** for a biological question → **Protocols**.
   - You want to **adapt a complete biological analysis** → **Use Cases** *(coming soon)*.
   - You want **exact technical details** of a function or class → **API**.

.. list-table:: You want to… / Go to
   :header-rows: 1
   :widths: 60 40

   * - You want to…
     - Go to
   * - Install AAanalysis and run a first CPP analysis
     - :ref:`Getting Started <getting_started>`
   * - Learn what a specific function does and how to call it
     - :ref:`Tutorials <tutorials>`
   * - Design a valid, end-to-end analysis for a biological question
     - :ref:`Protocols <protocols>`
   * - Adapt a full biological case study to your own data
     - Use Cases *(coming soon)*
   * - Look up the exact signature, parameters, or return value
     - :doc:`API <api>`

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
:doc:`Contributing Guidelines <index/CONTRIBUTING_COPY>`. These cover AAanalysis development conventions and the
automated quality gates every change must pass. For further questions or suggestions, please email
stephanbreimann@gmail.com.

Cheat Sheet
===========
The cheat sheet distills AAanalysis into a three-page summary: the golden workflow, the main
classes grouped by capability, the prediction levels (residue / domain / protein), and the
*Part × Split × Scale* feature ontology.

.. raw:: html

   <p>
     Click the image to open the interactive cheat sheet in your browser or
     <a href="_static/AAanalysis_cheat_sheet.pdf" download="AAanalysis_cheat_sheet.pdf">click here to download the PDF cheat sheet</a>.
   </p>
   <a href="_static/cheat_sheet.html" target="_blank" rel="noopener">
     <img src="https://raw.githubusercontent.com/breimanntools/aaanalysis/master/docs/source/_artwork/cheat_sheet_preview.png"
          alt="AAanalysis cheat sheet (page 1 of 3)"
          style="width: 100%; display: block; margin-left: auto; margin-right: auto;">
   </a>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: OVERVIEW

   index/introduction.rst
   index/CONTRIBUTING_COPY.rst
   index/docstring_guide.rst
   index/usage_principles.rst
   index/evaluation.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: GETTING STARTED

   getting_started.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: EXAMPLES

   tutorials.rst
   protocols.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   api.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   index/tables.rst
   index/glossary.rst
   index/references.rst
   index/release_notes.rst

The AAanalysis Ecosystem
========================
AAanalysis is the interpretable middle layer between bioinformatics I/O and the downstream machine
learning, explainable AI, and protein-design stack. It *consumes* upstream representations (sequences,
embeddings, structures) and even competitor descriptor sets, runs them through its interpretable core
(*Part × Split × Scale* · AAontology · CPP · ShapModel), and *exposes* the resulting features,
explanations, and objectives to the standard ML / XAI / optimization tools.

.. figure:: _artwork/diagrams/aaanalysis_ecosystem.svg
   :target: _static/aaanalysis_ecosystem.html
   :alt: The AAanalysis ecosystem — where AAanalysis fits in the protein-ML stack
   :width: 100%
   :align: center
   :figclass: ecosystem-figure

Explore the full `ecosystem map <_static/aaanalysis_ecosystem.html>`_ — per-category
packages, the comparison matrix, and where AAanalysis sits in the protein-ML stack. Click the diagram
to open it.

Citation
========

.. include:: index/citations.rst
