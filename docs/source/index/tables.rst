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

   Note: Each table should include a 'Reference' column.

   # Key Annotations for Automated Table Generation via create_tables_doc.py:
   _XXX: A string to be stripped from the references. This prevents redundancies that may result
         in broken links.
   ADD-TABLE: Placeholder indicating where tables for the corresponding section should be inserted.
..

Tables
======

.. contents::
    :local:
    :depth: 1

.. _t0_mapper:

Overview Table
--------------
All tables from the AAanalysis documentation are listed here, in chronological order based on the project history.

ADD-TABLE

.. list-table::
   :header-rows: 1
   :widths: 8 8 8

   * - Table
     - Description
     - See Also
   * - t1_overview_benchmarks
     - Protein benchmark datasets
     - aa.load_dataset
   * - t2_overview_scales
     - Amino acid scale datasets
     - aa.load_scales


.. _t1_overview_benchmarks:

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

.. list-table::
   :header-rows: 1
   :widths: 8 8 8 8 8 8 8 8 8 8

   * - Level
     - Dataset
     - # Sequences
     - # Amino acids
     - # Positives
     - # Negatives
     - Predictor
     - Description
     - Reference
     - Label
   * - Amino acid
     - AA_CASPASE3
     - 233
     - 185605
     - 705
     - 184900
     - PROSPERous
     - Prediction of caspase-3 cleavage site
     - :ref:`Song18 <Song18>`
     - 1 (adjacent to cleavage site), 0 (not adjacent to cleavage site)
   * - Amino acid
     - AA_FURIN
     - 71
     - 59003
     - 163
     - 58840
     - PROSPERous
     - Prediction of furin cleavage site
     - :ref:`Song18 <Song18>`
     - 1 (adjacent to cleavage site), 0 (not adjacent to cleavage site)
   * - Amino acid
     - AA_LDR
     - 342
     - 118248
     - 35469
     - 82779
     - IDP-Seq2Seq
     - Prediction of long intrinsically disordered regions (LDR)
     - :ref:`Tang20 <Tang20>`
     - 1 (disordered), 0 (ordered)
   * - Amino acid
     - AA_MMP2
     - 573
     - 312976
     - 2416
     - 310560
     - PROSPERous
     - Prediction of Matrix metallopeptidase-2 (MMP2) cleavage site
     - :ref:`Song18 <Song18>`
     - 1 (adjacent to cleavage site), 0 (not adjacent to cleavage site)
   * - Amino acid
     - AA_RNABIND
     - 221
     - 55001
     - 6492
     - 48509
     - GMKSVM-RU
     - Prediction of RNA-binding protein residues (RBP60 dataset)
     - :ref:`Yang21 <Yang21>`
     - 1 (binding), 0 (non-binding)
   * - Amino acid
     - AA_SA
     - 233
     - 185605
     - 101082
     - 84523
     - PROSPERous
     - Prediction of solvent accessibility (SA) of residue (AA_CASPASE3 data set)
     - :ref:`Song18 <Song18>`
     - 1 (exposed/accessible), 0 (buried/non-accessible)
   * - Sequence
     - SEQ_AMYLO
     - 1414
     - 8484
     - 511
     - 903
     - ReRF-Pred
     - Prediction of amyloidognenic regions
     - :ref:`Teng21 <Teng21>`
     - 1 (amyloidogenic), 0 (non-amyloidogenic)
   * - Sequence
     - SEQ_CAPSID
     - 7935
     - 3364680
     - 3864
     - 4071
     - VIRALpro
     - Prediction of capdsid proteins
     - :ref:`Galiez16 <Galiez16>`
     - 1 (capsid protein), 0 (non-capsid protein)
   * - Sequence
     - SEQ_DISULFIDE
     - 2547
     - 614470
     - 897
     - 1650
     - Dipro
     - Prediction of disulfide bridges in sequences
     - :ref:`Cheng06 <Cheng06>`
     - 1 (sequence with SS bond), 0 (sequence without SS bond)
   * - Sequence
     - SEQ_LOCATION
     - 1835
     - 732398
     - 1045
     - 790
     - nan
     - Prediction of subcellular location of protein (cytoplasm vs plasma membrane)
     - :ref:`Shen19 <Shen19>`
     - 1 (protein in cytoplasm), 0 (protein in plasma membrane) 
   * - Sequence
     - SEQ_SOLUBLE
     - 17408
     - 4432269
     - 8704
     - 8704
     - SOLpro
     - Prediction of soluble and insoluble proteins
     - :ref:`Magnan09 <Magnan09>`
     - 1 (soluble), 0 (insoluble)
   * - Sequence
     - SEQ_TAIL
     - 6668
     - 2671690
     - 2574
     - 4094
     - VIRALpro
     - Prediction of tail proteins
     - :ref:`Galiez16 <Galiez16>`
     - 1 (tail protein), 0 (non-tail protein)
   * - Domain
     - DOM_GSEC
     - 126
     - 92964
     - 63
     - 63
     - nan
     - Prediction of gamma-secretase substrates
     - :ref:`Breimann23c <Breimann23c>`
     - 1 (substrate), 0 (non-substrate)
   * - Domain
     - DOM_GSEC_PU
     - 694
     - 494524
     - 63
     - 0
     - nan
     - Prediction of gamma-secretase substrates (PU dataset)
     - :ref:`Breimann23c <Breimann23c>`
     - 1 (substrate), 2 (unknown substrate status)


.. _t2_overview_scales:

Amino Acid Scale Datasets
-------------------------
Various amino acid scale datasets are provided.

ADD-TABLE

.. list-table::
   :header-rows: 1
   :widths: 8 8 8 8

   * - Dataset
     - Description
     - # Scales
     - Reference
   * - scales
     - Amino acid scales (min-max normalized)
     - 586
     - :ref:`Breimann23b <Breimann23b>`
   * - scales_raw
     - Amino acid scales (raw values)
     - 586
     - :ref:`Kawashima08 <Kawashima08>`
   * - scales_classification
     - Classification of scales (Aaontology)
     - 586
     - :ref:`Breimann23b <Breimann23b>`
   * - scales_pc
     - Principal component (PC) compressed scales
     - 20
     - :ref:`Breimann23a <Breimann23a>`
   * - top60
     - Top 60 scale subsets
     - 60
     - :ref:`Breimann23a <Breimann23a>`
   * - top60_eval
     - Evaluation of top 60 scale subsets
     - 60
     - :ref:`Breimann23a <Breimann23a>`


