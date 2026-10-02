.. _data_reference:

Data
====

AAanalysis is built around a small set of tabular objects: a sequence table in, a feature
table out, and the scale and category tables that connect them. This chapter is the
reference for that data layer, in two parts.

**Data Tables** catalogues what ships with the package: the benchmark datasets of protein
sequences, the amino acid scales, and the AAontology categories that make a scale
interpretable. Start here to find out what you can load.

**Data Schemas** is the contract for the frames themselves: for each table, the columns it
carries, their dtypes, which are required, and what each one means. It is generated from
the code and kept in sync by a test, so it cannot drift from what the functions actually
return. Go here when a function hands you a DataFrame and you need to know what is in it.

.. toctree::
   :maxdepth: 1

   tables.rst
   usage_principles/df_schemas
