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

.. _tables:

Tables
======

.. contents::
    :local:
    :depth: 1

.. _t0_mapper:

Overview Table
--------------
All tables from the AAanalysis documentation are listed here, in chronological order based on the project history.


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
   * - t3a_aaontology_categories
     - AAontology scale categories
     - aa.load_scales
   * - t3b_aaontology_subcategories
     - AAontology scale subcategories
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
     - :ref:`Breimann24c <Breimann24c>`
     - 1 (substrate), 0 (non-substrate)
   * - Domain
     - DOM_GSEC_PU
     - 694
     - 494524
     - 63
     - 0
     - nan
     - Prediction of gamma-secretase substrates (PU dataset)
     - :ref:`Breimann24c <Breimann24c>`
     - 1 (substrate), 2 (unknown substrate status)


.. _t2_overview_scales:

Amino Acid Scale Datasets
-------------------------
Various amino acid scale datasets are provided.


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
     - :ref:`Breimann24b <Breimann24b>`
   * - scales_raw
     - Amino acid scales (raw values)
     - 586
     - :ref:`Kawashima08 <Kawashima08>`
   * - scales_cat
     - Classification of scales (AAontology)
     - 586
     - :ref:`Breimann24b <Breimann24b>`
   * - scales_pc
     - Principal component (PC) compressed scales
     - 20
     - :ref:`Breimann24a <Breimann24a>`
   * - top60
     - Top 60 scale subsets (AAclust) 
     - 60
     - :ref:`Breimann24a <Breimann24a>`
   * - top60_eval
     - Evaluation of top 60 scale subsets
     - 60
     - :ref:`Breimann24a <Breimann24a>`


AAontology
----------
AAontology ([Breimann24b]_) provides a two-tiered system for amino acid classification, designed to enhance the interpretability of
sequence-based protein predictions. It encompasses 586 physicochemical scales, which are systematically arranged
into ``67 subcategories`` and further grouped into ``8 categories``. Every scale, subcategory, and main category
is clearly defined and supported by key references. The scales were grouped into their respective subcategories
using a combination of AAclust ([Breimann24a]_) clustering and assessments of biological similarity. Those scales that couldn't
be allocated to a specific subcategory are labeled as 'unclassified'.

.. _t3a_aaontology_categories:

Categories
''''''''''


.. list-table::
   :header-rows: 1
   :widths: 8 8 8 8 8 8 8 8

   * - Category
     - Category Description
     - Key References
     - # Scales
     - # Unclassified Scales
     - # Subcategories
     - # Subcategories (n>=4 Scales)
     - # Subcategories (n<4 Scales)
   * - ASA/Volume
     - Subcategories regarding volume or preference of residues being accessible (e.g., solvent accessible surface area (ASA)) or buried)
     - Chothia, 1976; Lins et al., 2003
     - 64
     - 0
     - 5
     - 4
     - 1
   * - Composition
     - Frequency of occurrence in proteins (average) or in proteins with distinct cellular localization (e.g., membrane proteins or mitochondria proteins)
     - Nakashima et al., 1990; Nakashima-Nishikawa, 1992
     - 58
     - 2
     - 5
     - 5
     - 0
   * - Conformation
     - Frequency of occurrence in distinct structural conformation (e.g., α-helix, extended conformations such as β-sheet or β-strand, ranodm coil, or β-turn)
     - Tanaka-Scheraga, 1977; Chou-Fasman, 1978b; Richardson-Richardson, 1988; Qian-Sejnowski, 1988; Aurora-Rose, 1998
     - 224
     - 19
     - 24
     - 20
     - 4
   * - Energy
     - Various subcategories regarding the term of “energy” (e.g., charge, entropy, free energy, or non-bonded energy)
     - Charton-Charton, 1983; Guy, 1985; Radzicka-Wolfenden, 1988
     - 36
     - 5
     - 9
     - 3
     - 6
   * - Others
     - Subcategories that could not be assigned to the other categories (e.g., principal component analysis, mutability)
     - Sneath, 1966
     - 17
     - 5
     - 6
     - 0
     - 6
   * - Polarity
     - Subcategories regarding hydrophobicity, hydrophilicity, or amphiphilicity
     - Kyte-Doolittle, 1982; Mitaku et al., 2002; Koehler et al., 2009
     - 111
     - 6
     - 6
     - 5
     - 1
   * - Shape
     - Subcategories regarding shape and steric characteristics of residues (e.g., side chain angle, symmetry, or measures of graph-based representation of residues such as eccentricity)
     - Prabhakaran-Ponnuswamy, 1982; Karkbara-Knisley, 2016
     - 45
     - 4
     - 6
     - 4
     - 2
   * - Structure-Activity
     - Subcategories regarding flexibility, stability, or backbone dynamics
     - Vihinen et al., 1994; Bastolla et al., 2005
     - 31
     - 1
     - 6
     - 3
     - 3


.. _t3b_aaontology_subcategories:

Subcategories
'''''''''''''


.. list-table::
   :header-rows: 1
   :widths: 8 8 8 8 8

   * - Category
     - Subcategory
     - # Scales
     - Subcategory Description
     - Key References
   * - ASA/Volume
     - Accessible surface area (ASA)
     - 23
     - Solvent accessible surface area (ASA) in folded proteins (Lins), measuring a residues surface area that is accessible/exposed to solvent (typically water) at the protein surface
     - Lins et al., 2003; Chothia, 1976
   * - ASA/Volume
     - Buried
     - 12
     - Tendency of residue being buried in folded proteins (Janin), as opposed to being accessible at the protein surface (Chothia)
     - Janin et al., 1978; Chothia, 1976
   * - ASA/Volume
     - Hydrophobic ASA
     - 3
     - Hydrophobic solvent accessible surface area (ASA) in folded proteins (Lins), measuring a residues surface area that is hydrophobic and exposed to solvent (typically water) at the protein surface
     - Lins et al., 2003
   * - ASA/Volume
     - Partial specific volume
     - 9
     - Effective volume in water, accounting for physical volume and any extra water displacement caused by residue-solvent interactions (mainly hydrophobic ones)
     - Bull-Breese, 1974
   * - ASA/Volume
     - Volume
     - 17
     - Volume or size of residue
     - Bigelow, 1967
   * - Composition
     - AA composition
     - 23
     - Frequency of occurrence in proteins (Jones), denoted as amino acid (AA) composition by Dayhoff
     - Dayhoff, 1978b; Jones et al., 1992
   * - Composition
     - AA composition (surface)
     - 5
     - Frequency of occurrence at protein surface, as opposed to occurrence at protein interior (compared with “AA composition”, lower for unpolar amino acids)
     - Fukuchi-Nishikawa, 2001
   * - Composition
     - Membrane proteins (MPs)
     - 13
     - Frequency of occurrence in membrane proteins (MPs)
     - Nakashima et al., 1990; Cedano et al., 1997
   * - Composition
     - Mitochondrial proteins
     - 4
     - Frequency of occurrence in mitochondrial proteins (similar to membrane proteins, but with less Val)
     - Nakashima et al., 1990
   * - Composition
     - MPs (anchor)
     - 11
     - Frequency of occurrence in N-/C-terminal anchoring region of membrane proteins (cf. high N-terminal and C-terminal helix capping propensity (Asp, Glu resp. Lys, Arg) observed by Aurora-Rose). Characterized by a high partition energy (Guy, 1985)
     - Punta and Maritan, 2003; Aurora-Rose, 1998; Guy, 1985
   * - Composition
     - Unclassified (Composition)
     - 2
     - nan
     - nan
   * - Conformation
     - Coil
     - 13
     - Frequency of occurrence in random coil (see window positions given by Qian-Sejnowski)
     - Robson-Suzuki, 1976; Qian-Sejnowski, 1988
   * - Conformation
     - Coil (C-term)
     - 4
     - Frequency of occurrence at C-terminus in random coil (see window positions in Qian-Sejnowski)
     - Qian-Sejnowski, 1988
   * - Conformation
     - Coil (N-term)
     - 3
     - Frequency of occurrence at N-terminus in random coil (see window positions in Qian-Sejnowski)
     - Qian-Sejnowski, 1988
   * - Conformation
     - Linker (>14 AA)
     - 6
     - Frequency of occurrence in linker (length>14 residues)
     - George-Heringa, 2003
   * - Conformation
     - Linker (6-14 AA)
     - 6
     - Frequency of occurrence in medium sized linker (length between 6 to 14 residues)
     - George-Heringa, 2004
   * - Conformation
     - Unclassified (Conformation)
     - 19
     - nan
     - nan
   * - Conformation
     - α-helix
     - 36
     - Frequency of occurrence in right-handed α-helix, defined as helical structure with 3.6 residues per turn
     - Chou-Fasman, 1978b
   * - Conformation
     - α-helix (C-cap)
     - 4
     - Frequency of occurrence at C-cap position, defined as C-terminal interface residue, which is half in and half out of the helix (Richardson-Richardson); denoted as helix termination parameter (Finkelstein et al.)
     - Richardson-Richardson 1988; Aurora-Rose, 1998, Finkelstein et al., 1991
   * - Conformation
     - α-helix (C-term, out)
     - 5
     - Frequency of occurrence at C-terminus outside of right-handed α-helix, i.e., after end of helix, denoted as C-cap (Aurora-Rose) for C-terminal helix capping (Richardson-Richardson)
     - Richardson-Richardson 1988; Aurora-Rose, 1997
   * - Conformation
     - α-helix (C-term)
     - 8
     - Frequency of occurrence at C-terminus inside right-handed α-helix, i.e., before end of helix, denoted as C-cap (Aurora-Rose) for C-terminal helix capping (Richardson-Richardson)
     - Chou-Fasman, 1978b; Richardson-Richardson 1988; Aurora-Rose, 1998
   * - Conformation
     - α-helix (left-handed)
     - 11
     - Frequency of occurrence in left-handed α-helix
     - Tanaka-Scheraga, 1977
   * - Conformation
     - α-helix (N-cap)
     - 5
     - Frequency of occurrence at N-cap position, defined as N-terminal interface residue, which is half in and half out of the helix (Richardson-Richardson); denoted as helix initiation parameter (Finkelstein et al.)
     - Richardson-Richardson 1988; Aurora-Rose, 1998, Finkelstein et al., 1991
   * - Conformation
     - α-helix (N-term, out)
     - 3
     - Frequency of occurrence at N-terminus outside of right-handed α-helix, i.e., before start of helix, denoted as N-cap (Aurora-Rose) for N-terminal helix capping (Richardson-Richardson)
     - Richardson-Richardson 1988; Aurora-Rose, 1998
   * - Conformation
     - α-helix (N-term)
     - 7
     - Frequency of occurrence at N-terminus inside right-handed α-helix, i.e., after start of helix, denoted as N-cap (Aurora-Rose) for N-terminal helix capping (Richardson-Richardson)
     - Chou-Fasman, 1978b; Richardson-Richardson 1988; Aurora-Rose, 1998
   * - Conformation
     - α-helix (α-proteins)
     - 5
     - Frequency of occurrence in α-helix (based on only α-helical structures)
     - Geisow-Roberts, 1980
   * - Conformation
     - β/α-bridge
     - 2
     - Frequency of occurrence in the 'β/α-bridge' region of Ramachandran plot, reflecting conformational state between β-sheet (top-left quadrant) and right-handed α-helix (bottom-left quadrant)
     - Tanaka-Scheraga, 1977; Pauling et al., 1951
   * - Conformation
     - β-sheet
     - 21
     - Frequency of occurrence in β-sheet (see window positions in Qian-Sejnowski)
     - Chou-Fasman, 1978b; Qian-Sejnowski, 1988
   * - Conformation
     - β-sheet (C-term)
     - 5
     - Frequency of occurrence at C-terminus in β-sheet (see window positions in Qian-Sejnowski)
     - Qian-Sejnowski, 1988
   * - Conformation
     - β-sheet (N-term)
     - 5
     - Frequency of occurrence at N-terminus in β-sheet (see window positions in Qian-Sejnowski)
     - Qian-Sejnowski, 1988
   * - Conformation
     - β-strand
     - 15
     - Frequency of occurrence extended chain conformation/β-strand (typically 6-10 residues long, n>=2 parallel or antiparallel β-strands form a β-sheet)
     - Chou-Fasman, 1978; Lifson-Sander, 1979
   * - Conformation
     - β-turn
     - 21
     - Frequency of occurrence in β-turn (also called e.g. β-bend, reverse turn, tight turn), consisting of 4 consecutive residues forming a 180° back folded chain (Chou-Fasman), where two end residues (i -> i+3) form a main chain hydrogen bond
     - Chou-Fasman, 1978b, Robson-Suzuki, 1976
   * - Conformation
     - β-turn (C-term)
     - 6
     - Frequency of occurrence at 3rd or 4th position in β-turn (3rd position is denoted as chain reversal state S by Tanaka-Scheraga)
     - Chou-Fasman, 1978b; Tanaka-Scheraga, 1977
   * - Conformation
     - β-turn (N-term)
     - 6
     - Frequency of occurrence at 1st or 2nd position in β-turn (2nd position is denoted as chain reversal state R by Tanaka-Scheraga)
     - Chou-Fasman, 1978b; Tanaka-Scheraga, 1977
   * - Conformation
     - β-turn (TM helix)
     - 3
     - Frequency of occurrence in β-turn when placed in middle of transmembrane (TM) helix
     - Monné et al., 1999
   * - Conformation
     - π-helix
     - 5
     - Frequency of occurrence in right-handed π-helix, defined as helical structure with 4.4 residues per turn (compared with “α-helix”, lower for Ala, Asp, Glu, and Gln)
     - Fodje-Al-Karadaghi, 2002
   * - Energy
     - Charge
     - 2
     - Net charge and donor charge capability. Net charge is defined as 1 for positively charged residues (Arg, Lys),  0 for negatively charged residues (Asp, Glu), and 0.5 otherwise. Donor charge capability is defined as presence (1) or absence (0) of charge transfer donor capability in residue.
     - Klein et al., 1984; Charton-Charton, 1983
   * - Energy
     - Charge (negative)
     - 2
     - Negative charge and transfer charge. Negative charge is defined as presence (1) or absence (0) of negative charge in residue (Asp, Glu).Transfer charge is defined as presence (1) or absence (0) of charge transfer acceptor capability in residue.
     - Fauchere et al., 1988; Charton-Charton, 1983
   * - Energy
     - Charge (positive)
     - 1
     - Presence (1) or absence (0) of positive charge in residue (Arg, Lys, His)
     - Fauchere et al., 1988
   * - Energy
     - Electron-ion interaction pot.
     - 3
     - Electron-ion interaction potential, defined as the average energy state in a residue of all valence electrons (i.e., electrons that can participate in chemical bond formation)
     - Cosic, 1994
   * - Energy
     - Entropy
     - 3
     - Conformational entropy, associated with the number of possible conformations a residue can participate in (e.g., Ala has a low entropy due to is strong α-helix formation)
     - Hutchens, 1970
   * - Energy
     - Free energy (unfolding)
     - 8
     - Activation Gibbs free energy of unfolding in water or denaturant; measure of conformational stability
     - Yutani et al., 1987; Radzicka-Wolfenden, 1988
   * - Energy
     - Free energy (folding)
     - 5
     - Free energy of formation of α-helix or extended structure; measure of conformational instability of α-helix resp. extended structure (highest value for structure breaking Pro) 
     - Munoz-Serrano, 1994
   * - Energy
     - Isoelectric point
     - 3
     - pH at which residue is electrically neutral
     - Zimmerman et al., 1968
   * - Energy
     - Non-bonded energy
     - 4
     - Average non-bonded energy E per residue in 16 protein crystal structures, where E is computed as sum of pairwise interactions between constituent atoms (using Lennard-Jones potential)
     - Oobatake-Ooi, 1977
   * - Energy
     - Unclassified (Energy)
     - 5
     - nan
     - nan
   * - Others
     - Mutability
     - 3
     - Relative mutability of a residue, defined as the number of observed changes of a residue divided by it´s frequency of occurrence
     - Jones et al., 1992
   * - Others
     - PC 1
     - 1
     - 1. Vector of Principal Component analysis performed by Sneath, described as 'aliphaticity' (i.e., presence of linear, non-aromatic carbon chains)
     - Sneath, 1966
   * - Others
     - PC 2
     - 2
     - 2. Vector of Principal Component analysis performed by Sneath, described as 'hydrogenation' (approximately corresponds to the inverse of the number of reactive groups in a residue)
     - Sneath, 1966
   * - Others
     - PC 3
     - 2
     - 3. Vector of Principal Component analysis performed by Sneath, described as 'aromaticity' (i.e., aromatic property of residues)
     - Sneath, 1966
   * - Others
     - PC 4
     - 2
     - 4. Vector of Principal Component analysis performed by Sneath, described as 'hydroxythiolation' (might reflect hydrogen bonding potential)
     - Sneath, 1966
   * - Others
     - PC 5
     - 2
     - 1. Vector of Principal Component analysis performed by Wold
     - Wold et al., 1987
   * - Others
     - Unclassified (Others)
     - 5
     - nan
     - nan
   * - Polarity
     - Amphiphilicity
     - 6
     - Preference of residue to occur at interface of polar and non-polar solvents (esp., membrane-water interface)
     - Mitaku et al., 2002
   * - Polarity
     - Amphiphilicity (α-helix)
     - 13
     - Characteristic of residue to form amphipathic α-helices (compared to “Amphiphilicity”, higher for unpolar amino acids), highly correlating with signal sequence helical potential  (Argos et al., 1982)
     - Cornette et al., 1987; Argos et al., 1982
   * - Polarity
     - Hydrophobicity
     - 38
     - Preference of residue for non-polar/hydrophobic environment, measured as transfer free energy from non-polar solvent to water or from inside of a protein to surface/outside
     - Kyte-Doolittle, 1982; Eisenberg-McLachlan 1986
   * - Polarity
     - Hydrophobicity (interface)
     - 3
     - Preference of residue for non-polar/hydrophobic environment at membrane interfaces
     - Koehler et al., 2009
   * - Polarity
     - Hydrophobicity (surrounding)
     - 17
     - Total hydrophobicity of residues appearing within an 8 Angstrom radius volume, describing the internal packing arrangements of residues in globular proteins
     - Ponnuswamy et al., 1980; Wolfenden et al., 1981
   * - Polarity
     - Hydrophilicity
     - 28
     - Preference of residue for polar/hydrophilic environment, measured as transfer free energy from water to non-polar solvent
     - Kyte-Doolittle, 1982; Radzicka-Wolfenden, 1988
   * - Polarity
     - Unclassified (Polarity)
     - 6
     - nan
     - nan
   * - Shape
     - Graph (1. eigenvalue)
     - 5
     - Measure of graph-theoretic representation of residue, defined as eigenvalue of Laplacian matrix of undirected node-weighted graph (nodes represent atoms (weighted by mass) and edges represent molecular bonds)
     - Karkbara-Knisley, 2016
   * - Shape
     - Graph (2. eigenvalue)
     - 3
     - Measure of graph-theoretic representation of residue, defined as second smallest eigenvalue of Laplacian matrix of undirected node-weighted graph (nodes represent atoms (weighted by mass) and edges represent molecular bonds)
     - Karkbara-Knisley, 2016
   * - Shape
     - Side chain length
     - 19
     - Length of side chain and graph-based size measures like eccentricity of undirected node-weighted graph (nodes represent atoms (weighted by mass) and edges represent molecular bonds)
     - Charton-Charton, 1983; Karkbara-Knisley, 2016
   * - Shape
     - Reduced distance
     - 5
     - Reduced distance from the center of mass of a protein, defined as actual distance divided by the root-mean-square radius of gyration of the radius (e.g., reduced distance > 1 means that residue is farther away from center of mass than the average)
     - Rackovsky-Scheraga, 1977
   * - Shape
     - Shape and Surface
     - 3
     - Measure of relationships between the physical form of an amino acid and its solvent accessibility (e.g., rate at which the accessible surface increases relative to the distance from the protein center)
     - Prabhakaran-Ponnuswamy, 1982
   * - Shape
     - Steric parameter
     - 6
     - Measure of the steric factors of a residue such as branching, it´s symmetry, or side chain angles
     - Fauchere et al., 1988
   * - Shape
     - Unclassified (Shape)
     - 4
     - nan
     - nan
   * - Structure-Activity
     - Backbone-dynamics (-CH)
     - 3
     - α-CH chemical shifts of residue as measure of general backbone-dynamics/stability
     - Bundi-Wuthrich, 1979
   * - Structure-Activity
     - Backbone-dynamics (-NH)
     - 2
     - α-NH chemical shifts of residue as measure of general backbone-dynamics/stability
     - Bundi-Wuthrich, 1979
   * - Structure-Activity
     - Flexibility
     - 11
     - Flexibility parameter, measured as average B-factor (atomic temperature factor obtained from protein crystal structures)
     - Vihinen et al., 1994
   * - Structure-Activity
     - Flexibility (2 rigid neighbors)
     - 3
     - Flexibility parameter, measured as B-factor for residue being surrounded by two rigid neighbors (compared with “Flexibility”, lower for Gly and Ser)
     - Vihinen et al., 1994
   * - Structure-Activity
     - Stability
     - 7
     - Contribution of residue to protein stability (particularly depending on hydrophobic interactions)
     - Ptitsyn-Finkelstein, 1983; Zhou-Zhou, 2004;  Bastolla, 2005
   * - Structure-Activity
     - Stability (helix-coil)
     - 4
     - Contribution of residue to protein stability for helix-coil equilibrium
     - Ptitsyn-Finkelstein, 1983; Sueki et al., 1984
   * - Structure-Activity
     - Unclassified (Structure-Activity)
     - 1
     - nan
     - nan


