"""
This is a script for the backend of EmbeddingPreprocessor.build_pseudo_scales.

Pseudo-scales are computed by context-free averaging of per-residue embedding
values across occurrences of each canonical AA in the input corpus. The
result is a (20, D) matrix that mirrors the shape of an AAontology
``df_scales`` and feeds ``cluster_pseudo_scales`` to derive pseudo-categories.
"""
import numpy as np


# I Helper Functions
# (no helpers — single-function module)


# II Main Functions
def build_pseudo_scales_(df_seq=None, embeddings=None, list_aa=None, col_entry=None, col_seq=None,
                         return_std=False):
    """Compute context-free per-AA averages (and optionally stds) of embedding dimensions.

    For each canonical AA letter and each embedding dimension d, accumulates
    ``embedding[i, d]`` over all residues i in df_seq where ``seq[i] == aa``,
    then divides by the count. Non-canonical residues (not in ``list_aa``) are
    skipped. AAs with zero occurrences become NaN rows.

    When ``return_std=True``, also returns the per-AA population std computed in
    a single pass via sum-of-squares. AAs occurring exactly once get std=0;
    absent AAs get NaN.
    """
    n_aa = len(list_aa)
    aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}

    entries = df_seq[col_entry].tolist()
    D = embeddings[entries[0]].shape[1]

    sums = np.zeros((n_aa, D), dtype=np.float64)
    counts = np.zeros(n_aa, dtype=np.int64)
    sums_sq = np.zeros((n_aa, D), dtype=np.float64) if return_std else None

    for entry, seq in zip(df_seq[col_entry], df_seq[col_seq]):
        emb = embeddings[entry]
        # Map each character to its AA index, -1 for non-canonical
        aa_idx = np.fromiter((aa_to_idx.get(c, -1) for c in seq), dtype=np.int64, count=len(seq))
        mask = aa_idx >= 0
        if not mask.any():
            continue
        valid_idx = aa_idx[mask]
        valid_emb = emb[mask].astype(np.float64, copy=False)
        # np.add.at handles repeated indices correctly (unbuffered scatter-add)
        np.add.at(sums, valid_idx, valid_emb)
        np.add.at(counts, valid_idx, 1)
        if return_std:
            np.add.at(sums_sq, valid_idx, valid_emb * valid_emb)

    means = np.full((n_aa, D), np.nan, dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero, np.newaxis]
    if not return_std:
        return means

    stds = np.full((n_aa, D), np.nan, dtype=np.float64)
    if nonzero.any():
        var = sums_sq[nonzero] / counts[nonzero, np.newaxis] - means[nonzero] ** 2
        # Clip tiny negatives from floating-point error before sqrt
        var = np.clip(var, 0.0, None)
        stds[nonzero] = np.sqrt(var)
    return means, stds
