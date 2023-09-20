.. Developer Notes:
    This is the index file for all tables of the AAanalysis documentation. Each table should be saved the /tables
    directory. This file will serve as template for tables.rst, which is automatically created on the information
    provided here and in the .csv tables from the /tables directory. Add a new table as .csv in the /tables directory,
    in the overview table at the beginning of this document, and a new section with a short description of it in this
    document. Each column and important data types (e.g., categories) should be described. Each table should contain a
    'Reference' column.
    Ignore 'tables_template.rst: WARNING: document isn't included in any toctree' warning

Tables
======================

.. contents::
    :local:
    :depth: 1

Overview Table
--------------
All tables from the AAanalysis documentation are given here in chronological order of the project history.

.. _0_mapper:

Protein benchmark datasets
--------------------------
Three types of benchmark datasets are provided:

- Residue prediction (AA): Datasets used to predict residue (amino acid) specific properties.
- Domain prediction (DOM): Dataset used to predict domain specific properties
- Sequence prediction (SEQ): Datasets used to predict sequence specific properties

The classification of each dataset is indicated as first part of their name followed by an abbreviation for the
specific dataset (e.g., 'AA_LDR', 'DOM_GSEC', 'SEQ_AMYLO'). For some datasets, an additional version of it is provided
for positive-unlabeled (PU) learning containing only positive (1) and unlabeled (2) data samples, as indicated by
*dataset_name*_PU (e.g., 'DOM_GSEC_PU').

.. _1_overview_benchmarks:

Amino acid scale datasets
-------------------------
Different amino acid scale datasets are provided

.. _2_overview_scales:

