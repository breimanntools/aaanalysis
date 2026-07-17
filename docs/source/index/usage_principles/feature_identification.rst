.. _usage_principles_feature_identification:

CPP: Identifying Physicochemical Signatures
===========================================

The central algorithm of the AAanalysis framework is Comparative Physicochemical Profiling (:class:`~aaanalysis.CPP`), a sequence-based
feature engineering algorithm for interpretable protein prediction [Breimann25]_. CPP enables the identification
of physicochemical signatures underlying biological recognition processes. It thereby extends rational protein
biology beyond mere sequence motifs.

.. admonition:: Provided by
   :class: note

   In AAanalysis this is the :class:`~aaanalysis.CPP` class, with
   :class:`~aaanalysis.SequenceFeature` for building parts and splits and
   :class:`~aaanalysis.CPPPlot` for the figures. The golden pipeline
   :func:`~aaanalysis.pipe.find_features` automates the search. See the
   :ref:`API reference <api>`, the :ref:`tutorials <tutorials>`, and the
   :ref:`Evaluation Regimes <eval_feature_selection>` chapter for how to read a
   feature-selection score.

The core idea of CPP is its feature concept:

.. figure:: /_artwork/schemes/scheme_CPP1.png

   Scheme of CPP feature (**Part-Split-Scale** combination) with example of feature creation, from [Breimann25]_.

All possible parts are sub-parts or combinations of the **Target Middle Domain (TMD)**,
**Juxta Middle Domain N-terminal (JMD-N)**, and **Juxta Middle Domain N-terminal (JMD-C)**.

.. figure:: /_artwork/schemes/scheme_CPP2.png

   Scheme of sequence **Parts** comprising three basic parts from which each other can be derived from ([Breimann25]_):
   target middle domain (TMD), N- and C-terminal juxta middle domain (JMD-N and JMD-C).

These names were generalized from the first application of CPP on predicting substrates of γ-secretase,
which is a pivotal intramembrane protease implicated in cancer and Alzheimer´s disease. γ-Secretases cleaves its
substrates within their transmembrane domain (TMD) and their N- and C-terminal juxtamembrane domains (JMDs)
are of high importance for recognition. The three different split types (**Segment**, **Pattern**, and **PeriodicPattern**)
are exemplified for the two prominent γ-secretase substrates: the amyloid precursor protein (APP) and NOTCH1:

.. figure:: /_artwork/schemes/scheme_CPP3.png

   Scheme of part **Splits**, exemplifying the three split types ([Breimann25]_): segments, patterns, and periodic patterns.

CPP uses by default 120 continuous (segments) and 210 discontinuous (patterns and periodic patterns) splits:

.. figure:: /_artwork/schemes/scheme_CPP4.png

   Overview of **Split** classification and their count using default CPP settings, from [Breimann25]_.

Scales can be chosen from **AAontology**, our two-level scale classification, based on their category or subcategory
classification. To then select a redundancy-reduced scale set, AAanalysis provides the :class:`~aaanalysis.AAclust` clustering wrapper.

Once features are identified, the number a CPP or model score reports depends on *how* it was
produced. The default exploratory search ranks features but does not estimate generalization,
while an honest nested regime, a final refit, and an external test set each answer a different
question. The :func:`~aaanalysis.pipe.find_features` ``selection_scope`` knob moves feature
selection in and out of the cross-validation to switch between them. See the
:ref:`Evaluation Regimes <eval_feature_selection>` chapter for how to read each score.

.. toctree::
   :maxdepth: 1

   /index/evaluation/eval_feature_selection

