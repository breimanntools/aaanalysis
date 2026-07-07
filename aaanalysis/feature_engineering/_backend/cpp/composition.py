"""
This is a script for the backend of ``CPP.run_composit`` (composition features as a ``df_feat``).

Amino-acid composition (AAC) is handled *positionally* in the frontend — a one-hot identity scale set
with the whole-part ``Segment(1,1)`` split run through the full CPP pipeline, so it yields a genuine,
feature-map-able ``df_feat``. This module builds the **non-positional** k-mer composition ``df_feat``
(``k >= 2``): a dipeptide / k-mer is a property of an adjacent residue *tuple*, not a per-residue scale,
so it cannot carry a position. The k-mer composition matrix is scored with CPP's own discriminative
statistics (``add_stat_``: adjusted AUC, mean difference, test/ref std, p-value) and filtered by
adjusted AUC (top ``n_filter``), a min-occurrence guard, and optional correlation dedup.
"""
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from .sequence_feature import get_kmer_composition_, get_composition_scales_
from ._utils_feature_stat import add_stat_
from ..feature_filter import filter_correlation_


def get_kmer_composit_df_feat_(df_parts=None, labels=None, k=2, label_test=1, label_ref=0,
                               n_filter=100, max_cor=None, min_count=1, parametric=False, n_jobs=1):
    """Non-positional k-mer composition ``df_feat`` with CPP discriminative stats + filtering (``k >= 2``).

    Returns a ``df_feat``-shaped table (``feature`` = k-mer, ``category`` / ``subcategory`` = residue
    class, plus the CPP statistic columns from ``add_stat_``), ranked best-first by ``abs_auc``. Note
    it has no ``positions`` (a k-mer is position-less), so it is not drawn by the CPP feature map.
    """
    _df_scales, df_cat = get_composition_scales_(k=k)           # df_scales is None for k >= 2
    X, _n_kmers = get_kmer_composition_(df_parts=df_parts, k=k)
    y = np.asarray(labels)
    keep_rows = np.isfinite(X).all(axis=1)                     # spans shorter than k -> all-NaN rows
    Xk, yk = X[keep_rows], y[keep_rows]
    eligible = (Xk > 0).sum(axis=0) >= max(1, int(min_count))  # min-occurrence guard
    df = pd.DataFrame({ut.COL_FEATURE: df_cat[ut.COL_SCALE_ID].to_numpy(),
                       ut.COL_CAT: df_cat[ut.COL_CAT].to_numpy(),
                       ut.COL_SUBCAT: df_cat[ut.COL_SUBCAT].to_numpy(),
                       ut.COL_SCALE_NAME: df_cat[ut.COL_SCALE_NAME].to_numpy()})
    df = add_stat_(df=df, X=Xk, labels=yk, parametric=parametric, label_test=label_test,
                   label_ref=label_ref, n_jobs=n_jobs)
    df, Xk = df[eligible].reset_index(drop=True), Xk[:, eligible]
    order = np.argsort(-df[ut.COL_ABS_AUC].to_numpy())         # best AUC first (scale-free across k)
    df, Xk = df.iloc[order].reset_index(drop=True), Xk[:, order]
    if max_cor is not None:
        keep = np.asarray(filter_correlation_(Xk, max_cor=max_cor), dtype=bool)
        df = df[keep].reset_index(drop=True)
    return df.head(int(n_filter)).reset_index(drop=True)
