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
Pick a section by what you want to do:

.. list-table::
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
   getting_started.rst
   index/usage_principles.rst
   index/evaluation.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: GUIDES

   tutorials.rst
   protocols.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: REFERENCE

   api.rst
   index/tables.rst
   index/glossary.rst
   index/references.rst
   index/release_notes.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: PROJECT

   index/CONTRIBUTING_COPY.rst
   index/docstring_guide.rst

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
