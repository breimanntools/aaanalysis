..
   Developer Notes:
   This is the landing page for the AAanalysis documentation using Sphinx, containing the root `toctree` directive.
   The documentation will be hosted on Read the docs.
..

Welcome to the AAanalysis documentation!
========================================
.. include:: index/badges.rst
.. include:: index/overview.rst

.. admonition:: Cheat sheet
   :class: tip

   A one-glance reference to the AAanalysis workflow, the main classes, the
   prediction levels (residue / domain / protein), and the *Part × Split × Scale*
   feature ontology — :doc:`browse it interactively <index/cheat_sheet>` or
   :download:`download the PDF (3 pages) <_static/cheat_sheet.pdf>`.

Install
=======
**AAanalysis** can be installed from `PyPi <https://pypi.org/project/aaanalysis>`_:

.. code-block:: bash

   pip install aaanalysis

For extended features, including our explainable AI module, please use the 'professional' version:

.. code-block:: bash

   pip install aaanalysis[pro]

.. toctree::
   :maxdepth: 1
   :caption: OVERVIEW

   index/cheat_sheet.rst
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
