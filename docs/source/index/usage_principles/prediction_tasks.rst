..
   Developer Notes:
   Concept-overview page (issue #21). The entry point that answers "which task do
   I have, and which AAanalysis workflow solves it?". It hosts the prediction-task
   class table and routes each task to its workflow / protocol (#35). Canonical
   vocabulary lives in the Glossary (and, deeper, in CONTEXT.md); the taxonomy is
   recorded in ADR-0022. Keep the table's two lead columns — unit of comparison and
   reference construction — as the load-bearing axes; the level is the label.
..

.. _prediction_tasks:

Prediction tasks: which task, which workflow
============================================

Most users arrive with a biological question and one practical worry: *which
AAanalysis workflow solves it?* This page is the map. It sorts protein-prediction
tasks into a small taxonomy, then routes each one to its dataset, its CPP setup,
and the classes that carry it out. Use it as the front door to the
:ref:`tutorials <tutorials>` (which teach one function at a time) and the
:ref:`protocols <protocols>` (which walk an end-to-end workflow).

The two axes that actually define a task
----------------------------------------

It is tempting to organize tasks by *biological scale* alone — residue, domain,
protein. That label is useful shorthand, and AAanalysis encodes it directly in the
dataset name prefixes (``AA_*``, ``DOM_*``, ``SEQ_*``; see ``load_dataset``).
But the scale is only a **proxy**. What genuinely
determines how you set CPP up are two axes:

- **Unit of comparison** — the part CPP profiles. A fixed-length
  :term:`window` (residue level), a :term:`part`-set such as
  ``jmd_n`` / ``tmd`` / ``jmd_c`` (domain level), or the whole chain
  (protein level).
- **Reference construction** — how the contrasting set is built. Labeled
  A-vs-B groups, non-site / non-cleaved windows, an unlabeled pool, or a
  composition-matched shuffled background. CPP always reads out a
  :term:`test group` against a :term:`reference group`, and a feature's
  ``mean_dif`` is *test − reference*.

The :term:`prediction level` is the convenient label; these two axes are the
substance. The table below leads with them.

The prediction-task table
--------------------------

.. list-table:: AAanalysis prediction tasks — by unit of comparison and reference construction
   :header-rows: 1
   :widths: 16 20 24 10 14 16
   :class: longtable

   * - Task
     - Unit of comparison
     - Reference construction
     - Dataset prefix
     - CPP strategy
     - Typical classes
   * - **Residue · single-residue**

       (e.g. a PTM or modified site)
     - One :term:`window` centered *on* a residue (odd ``aa_window_size``)
     - Site windows vs non-site windows (or an unlabeled residue pool)
     - ``AA_``
     - Positional
     - ``AAWindowSampler``, ``CPP``, ``TreeModel``
   * - **Residue · between-residues**

       (e.g. a cleavage / scissile bond)
     - One :term:`window` spanning a bond ``P1│P1′`` (even ``aa_window_size``)
     - Cleaved windows vs non-cleaved windows
     - ``AA_``
     - Positional
     - ``AAWindowSampler``, ``CPP``, ``TreeModel``
   * - **Domain**

       (a defined sub-region)
     - A :term:`part`-set from ``tmd_start`` / ``tmd_stop``
       (``jmd_n`` / ``tmd`` / ``jmd_c``)
     - Labeled A-vs-B groups (e.g. substrate vs non-substrate)
     - ``DOM_``
     - Compositional **and** positional
     - ``SequenceFeature``, ``CPP``, ``TreeModel``
   * - **Protein**

       (the whole chain)
     - The whole chain as a single part
     - Labeled A-vs-B groups of proteins
     - ``SEQ_``
     - Compositional
     - ``CPP``, ``TreeModel``
   * - **Determinant discovery**

       (cross-cutting; no prediction target)
     - Any unit — window, part-set, or chain
     - Two groups contrasted to surface *what distinguishes them*
       (interpreted via :term:`AAontology`)
     - ``AA_`` / ``DOM_`` / ``SEQ_``
     - Compositional or positional
     - ``CPP``, ``CPPPlot``
   * - **Design / engineering**

       (cross-cutting; inverts prediction)
     - A sequence profiled against a target CPP profile
     - A target / reference profile the sequence is moved toward (``ΔCPP``)
     - ``AA_`` / ``DOM_`` / ``SEQ_``
     - Compositional and positional
     - ``AAMut``, ``SeqMut``
   * - **Relational / interaction**

       (scope boundary — not a level)
     - Interface **segments** only (a part-set on each partner)
     - Within-AAanalysis only for segments; pairwise contacts hand off
     - Interface segments via ``DOM_`` / ``AA_``
     - Positional (segments)
     - Out of scope: structure / PLM tooling

Reading the table
-----------------

**The three levels.** :term:`Residue level <residue level>`, :term:`domain level`,
and :term:`protein level` map one-to-one onto the ``AA_`` / ``DOM_`` / ``SEQ_``
dataset prefixes. "Protein level" is the user-facing name of the ``SEQ_``
(sequence) prefix; *sequence* stays reserved for the amino-acid string itself.
The residue level carries **two sub-modes** that differ only by window parity:
*single-residue* (odd window, a site on a residue) and *between-residues* (even
window, a bond between two residues). They share the windowing machinery, so they
are sub-modes of one level rather than two levels.

**The two cross-cutting rows.** :term:`Determinant discovery` and
:term:`design / engineering` are not levels — they apply *at any level* and run
in opposite directions. Determinant discovery asks *what physicochemically
distinguishes two groups* (CPP's purest, most interpretable use, with no
prediction target). Design / engineering inverts that: it measures how a mutation
shifts a sequence's CPP profile (:term:`ΔCPP`) and steers a sequence toward a
target. Both showcase the interpretability edge, so they are first-class rows.

**The boundary row.** :term:`Relational / interaction <relational / interaction>`
tasks — PPI interfaces and residue–residue contacts — are listed to be honest
about the taxonomy's edge. AAanalysis profiles interface **segments**; long-range
pairwise contacts and PPI-pair prediction are **out of scope** and hand off to
structure / PLM tooling. It is a documented boundary, **not** a fourth level.

CPP strategy in one line
------------------------

The **CPP strategy** column is :term:`compositional vs positional`, and it is not
a parameter — it *emerges* from ``split_kws``. A single whole-part average
(``Segment(1,1)``) is **compositional** (composition-like, position-agnostic);
sub-segments, ``Pattern``, or ``PeriodicPattern`` are **positional**. The strategy
tracks the level: compositional suits the protein level, positional suits the
residue level, and the domain level uses both. The deeper recipes live in the
dedicated CPP-strategies guide.

Where to go next
----------------

.. seealso::

   - **Run it:** the minimal end-to-end notebook
     :doc:`A minimal CPP analysis </generated/tutorial0_minimal>` loads a
     domain-level dataset, runs CPP, and reads out the signature in a few cells.
   - **Workflows:** the :ref:`protocols <protocols>` catalog turns each task in
     this table into a start-to-finish workflow.
   - **Mechanics:** the per-function :ref:`tutorials <tutorials>` cover
     ``CPP``, ``SequenceFeature``, ``AAclust``, and the rest.
   - **Vocabulary:** every term used here is defined in the
     :ref:`glossary <glossary>`. The taxonomy itself is recorded in ADR-0022.
