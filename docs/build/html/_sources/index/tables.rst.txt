..
    Developer Notes:
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
.. list-table::
   :header-rows: 1
   :widths: 8 8 8

   * - Table
     - Description
     - See also
   * - 1_overview_benchmarks
     - Protein benchmark datasets
     - aa.load_dataset
   * - 2_overview_scales
     - Amino acid scale datasets
     - aa.load_scales


Protein benchmark datasets
--------------------------
Three types of benchmark datasets are provided:

- Residue prediction (AA): Datasets used to predict residue (amino acid) specific properties.
- Domain prediction (DOM): Dataset used to predict domain specific properties.
- Sequence prediction (SEQ): Datasets used to predict sequence specific properties.

The classification of each dataset is indicated as first part of their name followed by an abbreviation for the
specific dataset (e.g., 'AA_LDR', 'DOM_GSEC', 'SEQ_AMYLO'). For some datasets, an additional version of it is provided
for positive-unlabeled (PU) learning containing only positive (1) and unlabeled (2) data samples, as indicated by
*dataset_name_PU* (e.g., 'DOM_GSEC_PU').

.. _1_overview_benchmarks:
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


Amino acid scale datasets
-------------------------
Different amino acid scale datasets are provided

.. _2_overview_scales:
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


