..
   Developer Notes:
   This is the landing page for the AAanalysis documentation using Sphinx, containing the root `toctree` directive.
   The documentation will be hosted on Read the docs.
..

Welcome to the AAanalysis documentation!
========================================
.. include:: index/badges.rst
.. include:: index/overview.rst

Install
=======
**AAanalysis** can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install aaanalysis

For extended features, including our explainable AI module, please use the 'professional' version:

.. code-block:: bash

   pip install aaanalysis[pro]

Cheat Sheet
===========
The cheat sheet distills AAanalysis into a three-page summary: the golden workflow, the main
classes grouped by capability, the prediction levels (residue / domain / protein), and the
*Part × Split × Scale* feature ontology. Click the image below to download the PDF.

.. raw:: html

   <a href="_static/cheat_sheet.pdf" download="aaanalysis_cheat_sheet.pdf">
     <img src="https://raw.githubusercontent.com/breimanntools/aaanalysis/master/docs/source/_artwork/cheat_sheet_preview.png"
          alt="AAanalysis cheat sheet (page 1 of 3)"
          style="width: 100%; display: block; margin-left: auto; margin-right: auto;">
   </a>

.. toctree::
   :maxdepth: 1
   :caption: OVERVIEW

   index/introduction.rst
   index/CONTRIBUTING_COPY.rst
   index/docstring_guide.rst
   index/usage_principles.rst
   index/evaluation.rst

.. toctree::
   :maxdepth: 1
   :caption: EXAMPLES

   tutorials.rst
   protocols.rst

.. toctree::
   :maxdepth: 2
   :caption: REFERENCES

   api.rst

.. toctree::
   :maxdepth: 1

   index/tables.rst
   index/glossary.rst
   index/references.rst
   index/release_notes.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

.. include:: index/citations.rst
