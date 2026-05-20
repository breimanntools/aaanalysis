"""
This is a script for the backend of EmbeddingPreprocessor.build_pseudo_scales.

Pseudo-scales are computed by context-free averaging of per-residue embedding
values across occurrences of each canonical AA in the input corpus. The
result is a (20, D) matrix that mirrors the shape of an AAontology
``df_scales`` and feeds ``cluster_pseudo_scales`` to derive pseudo-categories.
"""
import numpy as np
import pandas as pd


# I Helper Functions
# (no helpers — single-function module)


# II Main Functions
# TODO provide option also to return means and stds (both could be usefull for different purposes, e.g. clustering vs normalization)
def build_pseudo_scales_(df_seq=None, embeddings=None, list_aa=None, col_entry=None, col_seq=None):
    """Compute context-free per-AA averages of embedding dimensions.

    For each canonical AA letter and each embedding dimension d, accumulates
    ``embedding[i, d]`` over all residues i in df_seq where ``seq[i] == aa``,
    then divides by the count. Non-canonical residues (not in ``list_aa``) are
    skipped. AAs with zero occurrences become NaN rows.
    """
    n_aa = len(list_aa)
    aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}

    entries = df_seq[col_entry].tolist()
    D = embeddings[entries[0]].shape[1]

    sums = np.zeros((n_aa, D), dtype=np.float64)
    counts = np.zeros(n_aa, dtype=np.int64)

    for entry, seq in zip(df_seq[col_entry], df_seq[col_seq]):
        emb = embeddings[entry]
        # Map each character to its AA index, -1 for non-canonical
        aa_idx = np.fromiter((aa_to_idx.get(c, -1) for c in seq), dtype=np.int64, count=len(seq))
        mask = aa_idx >= 0
        if not mask.any():
            continue
        valid_idx = aa_idx[mask]
        valid_emb = emb[mask]
        # np.add.at handles repeated indices correctly (unbuffered scatter-add)
        np.add.at(sums, valid_idx, valid_emb)
        np.add.at(counts, valid_idx, 1)

    means = np.full((n_aa, D), np.nan, dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero, np.newaxis]
    return means
