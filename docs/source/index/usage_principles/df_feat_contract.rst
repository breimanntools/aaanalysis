.. _df_feat_contract:

df_feat: the CPP Output Contract
================================

The feature DataFrame ``df_feat`` returned by ``CPP.run`` is the primary
output other tools build on. To make that boundary safe to depend on, its schema
is a **documented, test-guarded contract**: each consumer reads columns by their
documented name and type, and a schema-stability test fails if a contracted column
is renamed or removed, a dtype changes, or the feature-id format changes.

``df_feat`` follows a **standardized, deterministic column order**. The columns listed
in the :ref:`Data Dictionary <df_schemas>` are the *canonical lower bound* — every
``CPP.run`` output carries them, always in this order. Optional and dynamic columns (a
test-dependent p-value variant, diagnostic residue columns, and the explainable-AI
columns appended by ``TreeModel`` / ``ShapModel``) are appended after ``positions`` in a
stable order, so the canonical order is a lower bound, never a restriction.

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

The full, test-guarded column list — every column with its dtype, required / nullable / unique flags, ranges, and an example — lives in the :ref:`Data Dictionary <df_schemas>` (the ``df_feat`` entry), so the column set is documented in exactly one place. This page covers the rest of the contract: the feature-id grammar above, the ``positions`` encoding below, and the stability policy.

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
