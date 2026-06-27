..
   Developer Note:

   This RST document lists references for the project, organized into four sections:

   1. **Algorithms**: For algorithm-based references.
   2. **Datasets and Benchmarks**: For dataset and benchmark tool references.
   3. **Use Cases**: Currently empty but reserved for application-related references.
   4. **Further Information**: For any additional, miscellaneous references.

   To add a new citation:

   1. Choose the appropriate section.
   2. Add a unique citation identifier (e.g., `[Breimann24a]`).
   3. Provide the full citation, followed by the optional link if available. Use the syntax `.. [CitationID]` for
   the citation and `` `Title <URL>`__ `` for the link.

   Make sure to update all related documents that need to reference the new citation.
..

.. _references:

Scientific References
=====================

AAanalysis Algorithms
---------------------
.. [Breimann24a] Breimann and Frishman (2024a),
   *AAclust: k-optimized clustering for selecting redundancy-reduced sets of amino acid scales*,
   `Bioinformatics Advances <https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae165/7852846>`__.

.. [Breimann24b] Breimann *et al.* (2024b),
   *AAontology: An ontology of amino acid scales for interpretable machine learning*,
   `Journal of Molecular Biology <https://www.sciencedirect.com/science/article/pii/S0022283624003267>`__.

.. [Breimann25] Breimann and Kamp *et al.* (2025),
   *Charting γ-secretase substrates by explainable AI*,
   `Nature Communications <https://www.nature.com/articles/s41467-025-60638-z>`__.

Sequence Algorithms
-------------------
.. [Li06] Li W., Godzik A. (2006),
   *Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/22/13/1658/194225>`__.

.. [Steinegger17] Steinegger M., Söding J. (2017),
   *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets*,
   `Nature Biotechnology <https://www.nature.com/articles/nbt.3988>`__.

.. [Bailey09] Bailey T.L., Boden M., Buske F.A., Frith M., Grant C.E., Clementi L., Ren J., Li W.W., Noble W.S. (2009),
   *MEME SUITE: tools for motif discovery and searching*,
   `Nucleic Acids Research <https://academic.oup.com/nar/article/37/suppl_2/W202/1135092>`__.

.. [Grant11] Grant C.E., Bailey T.L., Noble W.S. (2011),
   *FIMO: scanning for occurrences of a given motif*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/27/7/1017/231841>`__.

.. [Tareen20] Tareen A., Kinney J.B. (2020),
   *Logomaker: beautiful sequence logos in Python*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/36/7/2272/5671693>`__.

Structure Algorithms
--------------------
.. [Kabsch83] Kabsch W., Sander C. (1983),
   *Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features*,
   `Biopolymers <https://onlinelibrary.wiley.com/doi/10.1002/bip.360221211>`__.

.. [Touw15] Touw W.G., Baakman C., Black J., te Beek T.A.H., Krieger E., Joosten R.P., Vriend G. (2015),
   *A series of PDB-related databanks for everyday needs*,
   `Nucleic Acids Research <https://academic.oup.com/nar/article/43/D1/D364/2439515>`__.

.. [Sanner96] Sanner M.F., Olson A.J., Spehner J.-C. (1996),
   *Reduced surface: an efficient way to compute molecular surfaces*,
   `Biopolymers <https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0282(199603)38:3%3C305::AID-BIP4%3E3.0.CO;2-Y>`__.

.. [Jumper21] Jumper J., Evans R., Pritzel A. *et al.* (2021),
   *Highly accurate protein structure prediction with AlphaFold*,
   `Nature <https://www.nature.com/articles/s41586-021-03819-2>`__.

.. [Varadi22] Varadi M., Anyango S., Deshpande M. *et al.* (2022),
   *AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models*,
   `Nucleic Acids Research <https://academic.oup.com/nar/article/50/D1/D439/6430488>`__.

.. [Lau23] Lau A.M., Kandathil S.M., Jones D.T. (2023),
   *Merizo: a rapid and accurate protein domain segmentation method using invariant point attention*,
   `Nature Communications <https://www.nature.com/articles/s41467-023-43934-4>`__.

.. [Wells24] Wells J., Hawkins-Hooker A., Bordin N., Sillitoe I., Paige B., Orengo C.A. (2024),
   *Chainsaw: protein domain segmentation with fully convolutional neural networks*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/40/5/btae296/7667299>`__.

.. [Verwimp25] Verwimp S., Lavigne R., Lood C., van Noort V. (2025),
   *AFragmenter: schema-free, tuneable protein domain segmentation for AlphaFold protein structures*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/41/11/btaf588/8303958>`__.

Machine Learning
----------------
.. [Hastie09] Hastie, Tibshirani, and Friedman (2009),
   *The Elements of Statistical Learning*,
   `Springer <https://www.springer.com/gp/book/9780387848570>`__.

.. [MilliganCooper88] Milligan and Cooper (1988),
   *A study of standardization of variables in cluster analysis*,
   `Journal of Classification <https://doi.org/10.1007/BF01897163>`__.

.. [Eisen98] Eisen *et al.* (1998),
   *Cluster analysis and display of genome-wide expression patterns*,
   `PNAS <https://doi.org/10.1073/pnas.95.25.14863>`__.

Positive-Unlabeled Learning
---------------------------
.. [ElkanNoto08] Elkan and Noto (2008),
   *Learning classifiers from only positive and unlabeled data*,
   `KDD <https://doi.org/10.1145/1401890.1401920>`__.

.. [BekkerDavis20] Bekker and Davis (2020),
   *Learning from positive and unlabeled data: a survey*,
   `Machine Learning <https://doi.org/10.1007/s10994-020-05877-5>`__.

Explainable AI
--------------
.. [Lundberg20] Lundberg *et al.* (2020),
   *From local explanations to global understanding with explainable AI for trees*,
   `Nature Machine Intelligence <https://www.nature.com/articles/s42256-019-0138-9>`__.

Protein Design and Engineering
------------------------------
.. [Deb02] Deb *et al.* (2002),
   *A fast and elitist multiobjective genetic algorithm: NSGA-II*,
   `IEEE Transactions on Evolutionary Computation <https://doi.org/10.1109/4235.996017>`__.

.. [Yang19] Yang, Wu and Arnold (2019),
   *Machine-learning-guided directed evolution for protein engineering*,
   `Nature Methods <https://doi.org/10.1038/s41592-019-0496-6>`__.

.. [Wittmann21] Wittmann, Johnston, Wu and Arnold (2021),
   *Advances in machine learning for directed evolution*,
   `Current Opinion in Structural Biology <https://doi.org/10.1016/j.sbi.2021.01.008>`__.

.. [Dauparas22] Dauparas *et al.* (2022),
   *Robust deep learning-based protein sequence design using ProteinMPNN*,
   `Science <https://doi.org/10.1126/science.add2187>`__.

.. [Watson23] Watson *et al.* (2023),
   *De novo design of protein structure and function with RFdiffusion*,
   `Nature <https://doi.org/10.1038/s41586-023-06415-8>`__.

.. [Yang26] Yang *et al.* (2026),
   *The past, present and future of de novo protein design*,
   `Nature <https://doi.org/10.1038/s41586-026-10328-7>`__.

Datasets and Benchmarks
-----------------------
.. [Cheng06] Cheng *et al.* (2006),
   *Large-scale prediction of disulphide bridges using kernel methods, two-dimensional recursive neural networks, and weighted graph matching*,
   `Proteins: Structure, Function, Bioinformatics <https://onlinelibrary.wiley.com/doi/10.1002/prot.20787>`__.

.. [Kawashima08] Kawashima *et al.* (2008),
    *AAindex: Amino aid index database, progress report 2008*
    `Nucleic Acids Research <https://academic.oup.com/nar/article/36/suppl_1/D202/2508449>`__.

.. [Magnan09] Magnan, Randall, and Baldi (2009),
   *SOLpro: Accurate sequence-based prediction of protein solubility*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/25/17/2200/211163>`__.

.. [Galiez16] Galiez *et al.* (2016),
   *VIRALpro: A tool to identify viral capsid and tail sequences*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/32/9/1405/1743663>`__.

.. [Song18] Song *et al.* (2018),
   *PROSPERous: High-throughput prediction of substrate cleavage sites for 90 proteases with improved accuracy*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/34/4/684/4562332>`__.

.. [Shen19] Shen *et al.* (2019),
   *Identification of protein subcellular localization via integrating evolutionary and physicochemical information into Chou’s general PseAAC*,
   `Journal of Theoretical Biology <https://pubmed.ncbi.nlm.nih.gov/30452958/>`__.

.. [Tang20] Tang *et al.* (2020),
    *IDP-Seq2Seq: Identification of intrinsically disordered regions based on sequence to sequence learning*,
    `Bioinformatics <https://academic.oup.com/bioinformatics/article/36/21/5177/5875603>`__.

.. [Teng21] Teng *et al.* (2021),
   *ReRF-Pred: Predicting amyloidogenic regions of proteins based on pseudo amino acid composition and tripeptide composition*,
   `BMC Bioinformatics <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04446-4>`__.

.. [Yang21] Yang *et al.* (2021),
   *Granular multiple kernel learning for identifying RNA-binding protein residues via integrating sequence and structure information*,
   `Neural Computation and Applications <https://dl.acm.org/doi/10.1007/s00521-020-05573-4>`__.

Sampling Strategies
-------------------
.. [Boyd10Cascleave] Boyd *et al.* (2010),
   *Cascleave: towards more accurate prediction of caspase substrate cleavage sites*,
   `Bioinformatics <https://doi.org/10.1093/bioinformatics/btq043>`__.

.. [Song12] Song *et al.* (2012),
   *PROSPER: an integrated feature-based tool for predicting protease substrate cleavage sites*,
   `PLoS ONE <https://doi.org/10.1371/journal.pone.0050300>`__.

.. [Fu14ScreenCap3] Fu *et al.* (2014),
   *ScreenCap3: improving prediction of caspase-3 cleavage sites using experimentally verified non-cleavage sites*,
   `Proteomics <https://doi.org/10.1002/pmic.201300525>`__.

.. [Rawlings16] Rawlings *et al.* (2016),
   *Peptidase specificity from the substrate cleavage collection in the MEROPS database and a tool to measure cleavage site conservation*,
   `Biochimie <https://doi.org/10.1016/j.biochi.2015.10.003>`__.

.. [Li20Procleave] Li *et al.* (2020),
   *Procleave: predicting protease-specific substrate cleavage sites by combining sequence and structural information*,
   `Genomics, Proteomics & Bioinformatics <https://doi.org/10.1016/j.gpb.2019.08.002>`__.

.. [LiuDeber99] Liu L.-P., Deber C.M. (1999),
   *Combining hydrophobicity and helicity: a novel approach to membrane protein structure prediction*,
   `Bioorganic & Medicinal Chemistry <https://www.sciencedirect.com/science/article/pii/S0968089698002338>`__.

Use Cases
---------
- :doc:`Charting γ-secretase substrates by explainable AI </generated/use_case1_gamma_secretase>`
  reproduces [Breimann25]_.

Further Information
-------------------
