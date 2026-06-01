"""
This is a script for the backend of EmbeddingPreprocessor.encode.

Raw per-residue PLM embeddings are unbounded floats; ``CPP.run_num`` expects
per-residue values in ``[0, 1]`` (the StructurePreprocessor / AnnotationPreprocessor
normalization convention). This module fits a per-dimension normalizer over the whole
corpus (all residues of all proteins) and transforms every entry's ``(L, D)`` tensor
into a ``[0, 1]``-normalized ``dict_num``. The fitted parameters are returned so the
exact same transform can be reapplied to new proteins.
"""
import numpy as np


# I Helper Functions
def _stack_residues_(dict_num=None, entries=None):
    """Vertically stack every entry's (L, D) array into one (sum_L, D) matrix."""
    return np.vstack([np.asarray(dict_num[e], dtype=np.float64) for e in entries])


def _fit_minmax_(stacked=None):
    """Per-dimension min/max over all residues. Returns dict with 'lo','hi'."""
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    return {"method": "minmax", "lo": lo, "hi": hi}


def _fit_quantile_(stacked=None, clip=(1.0, 99.0)):
    """Per-dimension robust min/max at the given percentiles. Returns 'lo','hi'."""
    q_lo, q_hi = clip
    lo = np.percentile(stacked, q_lo, axis=0)
    hi = np.percentile(stacked, q_hi, axis=0)
    return {"method": "quantile", "lo": lo, "hi": hi, "clip": (q_lo, q_hi)}


def _fit_sigmoid_(stacked=None):
    """Per-dimension mean/std for a logistic squash. Returns 'mean','std'."""
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return {"method": "sigmoid", "mean": mean, "std": std}


def _transform_one_(emb=None, params=None):
    """Apply the fitted per-dimension normalizer to one (L, D) array -> [0, 1]."""
    emb = np.asarray(emb, dtype=np.float64)
    method = params["method"]
    if method in ("minmax", "quantile"):
        lo, hi = params["lo"], params["hi"]
        span = hi - lo
        # Constant dimensions (span == 0) map to 0.0 rather than dividing by zero.
        safe = np.where(span > 0, span, 1.0)
        out = (emb - lo) / safe
        out = np.where(span > 0, out, 0.0)
        return np.clip(out, 0.0, 1.0)
    # sigmoid: z-score per dim, then logistic -> open interval (0, 1)
    mean, std = params["mean"], params["std"]
    safe = np.where(std > 0, std, 1.0)
    z = (emb - mean) / safe
    z = np.where(std > 0, z, 0.0)
    return 1.0 / (1.0 + np.exp(-z))


# II Main Functions
def encode_(dict_num=None, entries=None, method="minmax", clip=(1.0, 99.0)):
    """Fit a per-dimension [0, 1] normalizer over the corpus and transform every entry.

    All residues of all ``entries`` are pooled to fit one normalizer per embedding
    dimension, so the transform is consistent across proteins (and dataset-dependent).
    Returns ``(dict_num_norm, params)`` where ``dict_num_norm`` maps each entry to its
    ``[0, 1]``-normalized ``(L, D)`` array and ``params`` are the fitted parameters
    (for reapplying the identical transform to new data).
    """
    stacked = _stack_residues_(dict_num=dict_num, entries=entries)
    if method == "minmax":
        params = _fit_minmax_(stacked=stacked)
    elif method == "quantile":
        params = _fit_quantile_(stacked=stacked, clip=clip)
    elif method == "sigmoid":
        params = _fit_sigmoid_(stacked=stacked)
    else:  # defensive: frontend validates, but keep the invariant explicit
        raise ValueError(f"'method' ({method}) should be one of 'minmax', 'quantile', 'sigmoid'.")
    dict_num_norm = {e: _transform_one_(emb=dict_num[e], params=params) for e in entries}
    return dict_num_norm, params
