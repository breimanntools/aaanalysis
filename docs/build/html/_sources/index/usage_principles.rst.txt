..
   Developer Notes:
   This is the index file that outlines the usage principles for the AAanalysis package.
   Files for individual usage principles are stored in the /usage_principles directory.

   This document provides an overview of:
   - Component diagram (illustrating internal dependencies)
   - Context diagram (depicting external dependencies)

   Instead of including comprehensive tables here, refer to tables in tables.rst with concise explanations.
   Always include brief code examples that mirror the corresponding usage examples.
..

Usage Principles
================
To get started with AAanalysis, import it as follows:

.. code-block:: python

    import aaanalysis as aa

.. toctree::
   :maxdepth: 1

   usage_principles/data_flow_entry_points
   usage_principles/aaontology
   usage_principles/feature_identification
   usage_principles/pu_learning
   usage_principles/xai
