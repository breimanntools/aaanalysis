.. Developer Notes:
    This is the index file for all tables of the AAanalysis documentation. Each table should be saved the /tables
    directory. This file will serve as template for tables.rst, which is automatically created on the information
    provided here and in the .csv tables from the /tables directory. Add a new table as .csv in the /tables directory,
    in the overview table at the beginning of this document, and a new section with a short description of it in this
    document. Each column and important data types (e.g., categories) should be described. Each table should contain a
    'Reference' column.

Tables
======================

.. contents::
    :local:
    :depth: 1

Overview Table
------------------
All tables from the AAanalysis documentation are given here in chronological order of the project history.

.. list-table::
   :header-rows: 1
   :widths: 10 10 10

   * - Table
     - Description
     - See also
   * - Overview_benchmarks
     - Protein benchmark datasets
     - aa.load_dataset
   * - Overview_scales
     - Amino acid scale datasets
     - aa.load_scales

Protein benchmark datasets
--------------------------
