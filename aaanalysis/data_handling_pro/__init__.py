"""
Pro data handling: structure- and annotation-based preprocessing (``pro`` extra).

Public objects: StructurePreprocessor, AnnotationPreprocessor.
Gated behind the ``pro`` extra — ``StructurePreprocessor`` reads PDB / CIF / AlphaFold
structure files (DSSP geometry, PAE, domains) and ``AnnotationPreprocessor`` fetches
UniProt PTM / functional-site annotations via ``requests``. Both follow an instance-based
``encode_*`` pattern that yields the ``[0, 1]``-normalized per-residue ``dict_num``
consumed by ``CPP.run_num`` (stackable via ``combine_dict_nums``); off the core
``load → CPP → model`` flow. Imported lazily from the top-level ``aaanalysis`` package
and replaced by an install-hint stub when the extra is absent.

See ``.claude/rules/pro-core-boundary.md`` for the pro/core boundary, ``CONTEXT.md``
for domain terms (structure-/annotation-based feature engineering).
"""
from ._struct_preproc import StructurePreprocessor
from ._annot_preproc import AnnotationPreprocessor

__all__ = [
    "StructurePreprocessor",
    "AnnotationPreprocessor",
]
