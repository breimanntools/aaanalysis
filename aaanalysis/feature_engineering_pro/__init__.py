"""
Pro feature-engineering plots: CPP feature impact on 3D structure (``pro`` extra).

Public objects: CPPStructurePlot.
Gated behind the ``pro`` extra (needs ``biopython``; ``py3Dmol`` for the
interactive backend, with a matplotlib fallback otherwise). Paints the per-residue
CPP / CPP-SHAP feature impact from a ``df_feat`` onto a protein structure, reusing
the shared CPP position backend (``feature_engineering``) and the structure parser
(``data_handling_pro``). Imported lazily from the top-level package and replaced by
an install-hint stub when ``biopython`` is absent.

See ``.claude/rules/pro-core-boundary.md`` for the pro/core boundary, ``CONTEXT.md``
for domain terms (CPPStructurePlot, StructureView).
"""
from ._cpp_structure_plot import CPPStructurePlot

__all__ = [
    "CPPStructurePlot",
]
