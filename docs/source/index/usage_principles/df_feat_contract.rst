.. _df_feat_contract:

df_feat: the CPP Output Contract
================================

The feature DataFrame ``df_feat`` returned by ``CPP.run`` is the primary
output other tools build on. To make that boundary safe to depend on, its schema
is a **documented, test-guarded contract**: each consumer reads columns by their
documented name and type, and a schema-stability test fails if a contracted column
is renamed or removed, a dtype changes, or the feature-id format changes.

``df_feat`` follows a **standardized, deterministic column order**. The columns below
are the *canonical lower bound* — every ``CPP.run`` output carries them, always in this
order. Optional and dynamic columns (a test-dependent p-value variant, diagnostic
residue columns, and the explainable-AI columns appended by ``TreeModel`` /
``ShapModel``) are appended after ``positions`` in a stable order, so the canonical order
is a lower bound, never a restriction.

Feature id grammar
------------------

The ``feature`` column is an opaque ``PART-SPLIT-SCALE`` string, for example
``TMD_C_JMD_C-Segment(3,4)-KLEP840101``:

- **PART** — the sequence part (e.g. ``TMD``, ``JMD_N``, or a compound part such as
  ``TMD_C_JMD_C``).
- **SPLIT** — the split selector, one of ``Segment(...)``, ``Pattern(...)``, or
  ``PeriodicPattern(...)``.
- **SCALE** — the AAontology scale id (e.g. ``KLEP840101``).

Split the id with the canonical parser ``aa.utils.split_feat_id(feat_id)`` (returns
``(part, split, scale_id)``) rather than parsing it by hand, so the grammar stays in one
place.

Column schema
-------------

Required columns are present in every ``CPP.run`` output; optional columns appear
depending on settings or are appended downstream.

.. list-table::
   :header-rows: 1
   :widths: 22 8 9 9 52

   * - Column
     - Type
     - Required
     - Nullable
     - Description
   * - ``feature``
     - str
     - yes
     - no
     - Opaque ``PART-SPLIT-SCALE`` feature id; split with ``split_feat_id``.
   * - ``category``
     - str
     - yes
     - no
     - AAontology scale category of the feature's scale.
   * - ``subcategory``
     - str
     - yes
     - no
     - AAontology scale subcategory.
   * - ``scale_name``
     - str
     - yes
     - no
     - Human-readable scale name.
   * - ``scale_description``
     - str
     - yes
     - no
     - One-sentence scale description.
   * - ``abs_auc``
     - float
     - yes
     - no
     - Absolute adjusted AUC, range [-0.5, 0.5]; primary feature ranking statistic.
   * - ``abs_mean_dif``
     - float
     - yes
     - no
     - Absolute mean difference between test and reference group, range [0, 1].
   * - ``mean_dif``
     - float
     - yes
     - no
     - Signed mean difference (test - reference), range [-1, 1]; the sign gives the direction.
   * - ``std_test``
     - float
     - yes
     - no
     - Standard deviation of the feature in the test group.
   * - ``std_ref``
     - float
     - yes
     - no
     - Standard deviation of the feature in the reference group.
   * - ``p_val_mann_whitney``
     - float
     - yes
     - no
     - Mann-Whitney U p-value (default, non-parametric). Named ``p_val_ttest_indep`` instead when ``parametric=True``.
   * - ``p_val_fdr_bh``
     - float
     - yes
     - no
     - Benjamini-Hochberg FDR-corrected p-value.
   * - ``positions``
     - str
     - yes
     - no
     - Comma-separated 1-based residue positions the feature spans.
   * - ``p_val_ttest_indep``
     - float
     - no
     - no
     - Independent t-test p-value; replaces ``p_val_mann_whitney`` when ``parametric=True``.
   * - ``amino_acids_test``
     - str
     - no
     - no
     - Amino acids at the feature positions in the test group (diagnostic).
   * - ``amino_acids_ref``
     - str
     - no
     - no
     - Amino acids at the feature positions in the reference group (diagnostic).
   * - ``feature_description``
     - str
     - no
     - yes
     - Optional readable one-sentence feature description.
   * - ``feat_importance``
     - float
     - no
     - no
     - Feature importance from ``TreeModel.fit`` (post-fit).
   * - ``feat_importance_std``
     - float
     - no
     - no
     - Standard deviation of the feature importance across CV rounds (post-fit).
   * - ``feat_impact``
     - float
     - no
     - no
     - SHAP-based signed feature impact from ``ShapModel`` (post-fit, pro).
   * - ``feat_impact_std``
     - float
     - no
     - no
     - Standard deviation of the feature impact (post-fit, pro).

Per-residue positions
---------------------

The ``positions`` column encodes the residue positions a feature spans as a
comma-separated list of **1-based** indices into the sequence parts. Downstream tools
that map features back to single residues (for per-residue scoring) parse this column;
its 1-based, comma-separated format is part of the contract.

Stability notes
---------------

- The contract is pinned to **column-name strings**; depend on those names, not on
  column positions.
- The canonical column set is a lower bound: new optional columns may be appended in a
  stable order without breaking the contract, but a required column is never renamed or
  removed without a major-version change.
