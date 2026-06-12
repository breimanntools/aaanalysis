.. _glossary:

Glossary
========

The canonical vocabulary of AAanalysis. Terms are short and opinionated; use
them consistently so code, docstrings, and tutorials read the same way. Terms
defined here can be cross-referenced from anywhere in the documentation with
``:term:`` (e.g. ``:term:`dict_num```).

.. Maintainer note: keep this in sync with the project glossary in CONTEXT.md
   (the deeper, internal source of truth). Add a term here when it appears in a
   public docstring or user-facing page.

Sequences & data objects
-------------------------

.. glossary::

   df_seq
      Sequence table — one row per protein or region. Columns: ``entry``,
      ``sequence``, ``label``, and, for domain-level tasks, the TMD bounds
      ``tmd_start`` / ``tmd_stop``.

   df_parts
      Wide table with one column per :term:`part` (``tmd``, ``jmd_n``,
      ``jmd_c``, …), produced by ``SequenceFeature.get_df_parts``.

   df_feat
      Ranked feature table: ``feature`` id, ``abs_auc``, ``mean_dif``,
      ``p_val``, ``positions``, ``scale``, ``category``.

   entry
      Unique identifier of a sequence — the key/index of :term:`df_seq`.

   part
      A named region of a sequence over which a :term:`split` operates and a
      :term:`scale` is averaged; the ``PART`` field of a feature id
      (``PART-SPLIT-SCALE``). The default vocabulary is TMD-centric (``jmd_n`` /
      ``tmd`` / ``jmd_c`` and composites); name parts after the
      :term:`prediction level` when that fits better.

   scale
      A mapping from each amino acid to a real number — a physicochemical
      property. :term:`AAontology` ships ~600 curated scales.

The CPP feature model
---------------------

.. glossary::

   feature
      A ``(Part × Split × Scale)`` triple, written ``PART-SPLIT-SCALE`` — the
      atomic, residue-grounded, interpretable unit that CPP ranks.

   split
      How a scale is read across a :term:`part`: **Segment** (a contiguous
      block), **Pattern** (fixed positions counted from a terminus), or
      **PeriodicPattern** (periodic positions, e.g. ``i, i+3/4`` for an
      α-helical face).

   CPP
      Comparative Physicochemical Profiling — discovers ranked
      ``Part × Split × Scale`` features that distinguish a :term:`test group`
      from a :term:`reference group`.

   AAontology
      A two-level taxonomy of amino-acid scales; CPP uses its categories to
      organize and rank features.

   compositional vs positional
      Whether ``split_kws`` yields one whole-part average (*compositional*) or
      position-resolved sub-segments and patterns (*positional*). It is not a
      flag — it emerges from the chosen splits.

Prediction tasks
----------------

.. glossary::

   prediction level
      Residue (``AA_*``), domain (``DOM_*``), or protein (``SEQ_*``) — the unit
      a task predicts at. A convenient proxy for the two axes that actually
      define a task: the :term:`unit of comparison` and the
      :term:`reference construction`.

   unit of comparison
      What CPP profiles for a task — a :term:`window` (residue level), a
      part-set (domain level), or the whole chain (protein level). One of the
      two axes that define a use-case class.

   reference construction
      How the contrasting set is built — labeled A-vs-B groups, non-site /
      non-cleaved windows, an unlabeled pool, or a composition-matched shuffled
      background. The second task-defining axis.

   test group
      The set CPP profiles, contrasted against the :term:`reference group`. A
      feature's mean difference (``mean_dif``) is computed as *test − reference*
      (``abs_auc`` measures the separation magnitude). For multi-class, each
      class is the test group in turn versus the rest as the reference group.

   reference group
      The contrasting set a :term:`test group` is profiled against — what
      :term:`reference construction` produces.

CPP modes
---------

.. glossary::

   determinant discovery
      Using CPP with **no prediction target**: contrast two groups to surface
      *what physicochemically distinguishes them*, interpreted via
      :term:`AAontology`. CPP's purest, most interpretable use.

   design / engineering
      Inverting prediction: measure how a mutation shifts a sequence's CPP
      feature profile (:term:`ΔCPP`) and use that to steer a sequence toward a
      target profile (``AAMut`` / ``SeqMut``). Deliberately model-free.

   ΔCPP
      The change in a sequence's CPP feature values caused by a mutation — the
      model-free signal that :term:`design / engineering` ranks and optimizes.

Numerical features
------------------

.. glossary::

   dict_num
      A mapping ``{entry: ndarray (L, D)}`` of per-residue numerical values —
      e.g. protein-language-model embeddings, structural descriptors, or PTM
      annotations.

   pseudo-scale
      A single column of a :term:`dict_num` treated like a :term:`scale`,
      letting CPP profile any per-residue numerical signal, not just
      amino-acid scales.

   numerical CPP
      ``CPP.run_num`` — the numerical-mode pipeline that profiles
      :term:`dict_num` inputs (sliced to :term:`part`-sets) instead of
      amino-acid scale look-ups. Generalizes CPP beyond physicochemical scales.

Models & explainability
------------------------

.. glossary::

   feature importance
      An **unsigned, group-level** ranking of how much each :term:`feature`
      contributes to a model (e.g. ``TreeModel`` Monte-Carlo importance).

   feature impact
      A **signed, per-sample** attribution of how each :term:`feature` pushes a
      single prediction (e.g. ``ShapModel``; visualized via ``shap_plot``).

Reducing features
-----------------

.. glossary::

   redundancy reduction
      Clustering correlated amino-acid scales and keeping one representative per
      cluster (``AAclust``) to obtain a redundancy-reduced scale set.

   medoid
      The representative scale of an ``AAclust`` cluster; the
      redundancy-reduced set is the set of medoids.

   feature selection
      Choosing an informative subset of features — e.g. recursive feature
      elimination (RFE) inside ``TreeModel``.

   feature pruning
      Dropping correlated or uninformative features before modeling
      (``NumericalFeature.filter_correlation``).

   feature simplification
      ``CPP.simplify`` — swapping features onto fewer, more interpretable scales
      without retraining.

PU learning
-----------

.. glossary::

   PU labels
      ``dPULearn`` input labels: ``1`` = positive, ``2`` = unlabeled. The output
      adds ``0`` = :term:`reliable negative`.

   reliable negative
      An unlabeled sample that ``dPULearn`` identifies as confidently negative
      (output label ``0``), drawn from the unlabeled pool.

Window sampling
---------------

.. glossary::

   window
      A fixed-length residue stretch sampled around a position of interest — the
      residue-level :term:`unit of comparison`.

   P1 anchor
      The source/anchor position of a :term:`window` (the Schechter–Berger
      ``P1``), about which test and reference windows are defined.

   reference window
      A background :term:`window` (non-site, shuffled, or distance-banded)
      contrasted against test windows.

Class conventions
-----------------

.. glossary::

   Wrapper
      An scikit-learn-style class implementing ``.fit`` / ``.predict`` /
      ``.eval`` and setting trailing ``*_`` attributes after ``fit``.

   Tool
      A pipeline-style class implementing ``.run`` / ``.eval``.

   Plot class
      A ``*Plot`` mirror of an analytical class — same arguments, visualization
      only (e.g. ``CPPPlot`` mirrors ``CPP``).
