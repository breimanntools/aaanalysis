CPP: Identifying Physicochemical Signatures
===========================================

The central algorithm of the AAanalysis framework is Comparative Physicochemical Profiling (**CPP**), a sequence-based
feature engineering algorithm for interpretable protein prediction [Breimann24c]_. CPP enables the identification
of physicochemical signatures underlying biological recognition processes. It thereby extends rational protein
biology beyond mere sequence motifs.

The core idea of CPP is its feature concept:

.. figure:: /_artwork/schemes/scheme_CPP1.png

   Scheme of CPP feature (**Part-Split-Scale** combination) with example of feature creation, from [Breimann24c]_.

All possible parts are sub-parts or combinations of the **Target Middle Domain (TMD)**,
**Juxta Middle Domain N-terminal (JMD-N)**, and **Juxta Middle Domain N-terminal (JMD-C)**.

.. figure:: /_artwork/schemes/scheme_CPP2.png

   Scheme of sequence **Parts** comprising three basic parts from which each other can be derived from ([Breimann24c]_):
   target middle domain (TMD), N- and C-terminal juxta middle domain (JMD-N and JMD-C).

These names were generalized from the first application of CPP on predicting substrates of γ-secretase,
which is a pivotal intramembrane protease implicated in cancer and Alzheimer´s disease. γ-Secretases cleaves its
substrates within their transmembrane domain (TMD) and their N- and C-terminal juxtamembrane domains (JMDs)
are of high importance for recognition. The three different split types (**Segment**, **Pattern**, and **PeriodicPattern**)
are exemplified for the two prominent γ-secretase substrates: the amyloid precursor protein (APP) and NOTCH1:

.. figure:: /_artwork/schemes/scheme_CPP3.png

   Scheme of part **Splits**, exemplifying the three split types ([Breimann24c]_): segments, patterns, and periodic patterns.

CPP uses by default 120 continuous (segments) and 210 discontinuous (patterns and periodic patterns) splits:

.. figure:: /_artwork/schemes/scheme_CPP4.png

   Overview of **Split** classification and their count using default CPP settings, from [Breimann24c]_.

Scales can be chosen from **AAontology**, our two-level scale classification, based on their category or subcategory
classification. To then select a redundancy-reduced scale set, AAanalysis provides the **AAclust** clustering wrapper.

