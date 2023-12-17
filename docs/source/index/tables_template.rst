..
   Developer Notes:
   This is the index file for all tables of the AAanalysis documentation.
   Tables should be saved in the /tables directory. This file serves as a template
   for tables.rst, which is automatically generated based on the information here and
   in the .csv tables from the /tables directory.

   Instructions for Adding a New Table:
   1. Store the table as a .csv file in the index/tables directory. Name it using the format tX,
      where X is incremented based on the last entry's number.
   2. Update the t0_mapper.xlsx with a corresponding entry for the new table.
   3. Create a new descriptive section here that elucidates the table's columns and any
      essential data types, such as categories.

   Note: Each table should include a 'Reference' column (include exceptions in create_tables_doc.py).

   # Key Annotations for Automated Table Generation via create_tables_doc.py:
   _XXX: A string to be stripped from the references. This prevents redundancies that may result
         in broken links.
   ADD-TABLE: Placeholder indicating where tables for the corresponding section should be inserted.
..

.. _tables_XXX:

Tables
======

.. contents::
    :local:
    :depth: 1

.. _t0_mapper_XXX:

Overview Table
--------------
All tables from the AAanalysis documentation are listed here, in chronological order based on the project history.

ADD-TABLE

.. _t1_overview_benchmarks_XXX:

Protein Benchmark Datasets
--------------------------
Three types of benchmark datasets are provided:

- Residue prediction (AA): Datasets used to predict specific properties of amino acid residues.
- Domain prediction (DOM): Datasets used to predict specific properties of domains.
- Sequence prediction (SEQ): Datasets used to predict specific properties of sequences.

Datasets are named beginning with a classification (e.g., 'AA_LDR', 'DOM_GSEC', 'SEQ_AMYLO').
Some datasets have an additional version for positive-unlabeled (PU) learning containing only positive (1)
and unlabeled (2) data samples, as indicated by appending '_PU' to the dataset name (e.g., 'DOM_GSEC_PU').

ADD-TABLE

.. _t2_overview_scales_XXX:

Amino Acid Scale Datasets
-------------------------
Various amino acid scale datasets are provided.

ADD-TABLE

AAontology
----------
AAontology ([Breimann24b]_) provides a two-tiered system for amino acid classification, designed to enhance the interpretability of
sequence-based protein predictions. It encompasses 586 physicochemical scales, which are systematically arranged
into ``67 subcategories`` and further grouped into ``8 categories``. Every scale, subcategory, and main category
is clearly defined and supported by key references. The scales were grouped into their respective subcategories
using a combination of AAclust ([Breimann24a]_) clustering and assessments of biological similarity. Those scales that couldn't
be allocated to a specific subcategory are labeled as 'unclassified'.

.. _t3a_aaontology_categories_XXX:

Categories
''''''''''

ADD-TABLE

.. _t3b_aaontology_subcategories_XXX:

Subcategories
'''''''''''''

ADD-TABLE

