"""
Data loading and sequence/embedding preprocessing — the package's data entry point.

Public objects: load_dataset, load_scales, load_features, get_labels, read_fasta,
to_fasta, SequencePreprocessor, EmbeddingPreprocessor, combine_dict_nums.
Produces the core data objects the rest of the pipeline consumes: ``load_dataset``
yields ``df_seq``, ``load_scales`` yields ``df_scales`` (fed to
``feature_engineering.AAclust`` / ``CPP``), ``load_features`` yields a reference
``df_feat``, and ``read_fasta`` / ``to_fasta`` handle FASTA I/O. ``SequencePreprocessor``
turns raw sequences into windows / numeric encodings; ``EmbeddingPreprocessor`` normalizes
protein-language-model embeddings into the per-residue ``dict_num`` consumed by
``CPP.run_num`` (``combine_dict_nums`` stacks such tensors).

See ``.claude/rules/code-conventions.md`` and ``frontend-backend.md`` for conventions,
``CONTEXT.md`` for domain terms (df_seq, scale set, embedding-based feature engineering).
"""
from ._load_dataset import load_dataset
from ._load_scales import load_scales
from ._load_features import load_features
from ._get_labels import get_labels
from ._read_fasta import read_fasta
from ._to_fasta import to_fasta
from ._seq_preproc import SequencePreprocessor
from ._embed_preproc import EmbeddingPreprocessor
from ._combine_dict_nums import combine_dict_nums

__all__ = [
    "load_dataset",
    "load_scales",
    "load_features",
    "get_labels",
    "read_fasta",
    "to_fasta",
    "SequencePreprocessor",
    "EmbeddingPreprocessor",
    "combine_dict_nums",
]
