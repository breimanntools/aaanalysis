.. Developer Notes:
    This is the index file for usage principles. Files for each part are saved in the /usage_principles directory
    and the overview the AAanalysis package is given as component diagram (internal dependencies) and context diagram
    (external dependencies). Always give the concise code examples reflecting the usage examples. Instead of including
    comprehensive tables here, add them in tables.rst and refer to them with a short explanation

Usage Principles
================
Import AAanalysis as:

.. code-block:: python

    import aaanalysis as aa

.. toctree::
   :maxdepth: 1

   usage_principles/data_flow_entry_points
   usage_principles/aaontology
   usage_principles/feature_identification
   usage_principles/pu_learning
   usage_principles/xai