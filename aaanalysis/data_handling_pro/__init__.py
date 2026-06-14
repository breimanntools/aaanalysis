"""
Pro data handling: structure- and annotation-based preprocessing (``pro`` extra).

Public objects: StructurePreprocessor, AnnotationPreprocessor.
Gated behind the ``pro`` extra — ``StructurePreprocessor`` needs DSSP + biopython,
``AnnotationPreprocessor`` fetches UniProt annotations via ``requests``. Off the core
``load → CPP → model`` flow; both enrich ``df_seq`` with structure / annotation columns.
Imported lazily from the top-level ``aaanalysis`` package and replaced by an
install-hint stub when the extra is absent.

See ``.claude/rules/pro-core-boundary.md`` for the pro/core boundary, ``CONTEXT.md``
for domain terms (structure-/annotation-based feature engineering).
"""
from ._struct_preproc import StructurePreprocessor
from ._annot_preproc import AnnotationPreprocessor

__all__ = [
    "StructurePreprocessor",
    "AnnotationPreprocessor",
]
