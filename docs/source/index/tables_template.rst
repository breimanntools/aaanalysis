..
   Developer Notes:
   This is the index file for all tables of the AAanalysis documentation.
   Tables should be saved in the /tables directory. This file serves as a template
   for tables.rst, which is automatically generated based on the information here and
   in the .csv tables from the /tables directory.

   To add a new table:
   1. Save it as a .csv file in the /tables directory.
   2. Add an entry for it in the "Overview Table" section below.
   3. Add a new section describing it, including each column and any important data types (e.g., categories).

   Note: Each table should include a 'Reference' column.

   Ignore the warning: 'tables_template.rst: WARNING: document isn't included in any toctree.'
..

Tables
======================

.. contents::
    :local:
    :depth: 1

Overview Table
--------------
All tables from the AAanalysis documentation are listed here, in chronological order based on the project history.

.. _0_mapper:

Protein Benchmark Datasets
--------------------------
Three types of benchmark datasets are provided:

- Residue prediction (AA): Datasets used to predict specific properties of amino acid residues.
- Domain prediction (DOM): Datasets used to predict specific properties of domains.
- Sequence prediction (SEQ): Datasets used to predict specific properties of sequences.

Datasets are named beginning with a classification (e.g., 'AA_LDR', 'DOM_GSEC', 'SEQ_AMYLO').
Some datasets have an additional version for positive-unlabeled (PU) learning containing only positive (1)
and unlabeled (2) data samples, as indicated by appending '_PU' to the dataset name (e.g., 'DOM_GSEC_PU').

.. _1_overview_benchmarks:

Amino Acid Scale Datasets
-------------------------
Various amino acid scale datasets are provided.

.. _2_overview_scales:
