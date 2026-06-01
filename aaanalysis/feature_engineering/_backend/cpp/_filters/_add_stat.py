"""
This is a script for the backend of CPP's numerical-mode full-statistics stage:
``add_stat`` augments the surviving feature DataFrame with AUC, mean
difference, p-values, and FDR-adjusted p-values using a **pre-cached** per-sample
feature-value matrix from ``_filters._stat_filter.pre_filtering_info``.

Skips the duplicate per-feature compute of a naive ``get_feature_matrix_``
loop: the cached survivor matrix from ``pre_filtering_info`` is sufficient
because both compute paths reduce to ``mean(scale_matrix[aa_idx_in_segment, d])``
over the same residues. The Cython kernel in ``_filters_c/_inner.pyx``
preserves the bit-exact ``np.mean`` summation tree (8-way unrolled pairwise
summation, ``np.round(_, 5)`` boundary) so Mann-Whitney p-values land on the
same ranks as a reference numpy implementation.
"""
import numpy as np

import aaanalysis.utils as ut
from .._utils_feature_stat import add_stat_


# I Helper Functions
# (no helpers — single-function module wrapping shared utilities)


# II Main Functions
def add_stat(df_feat=None, X_cached=None, labels=None, parametric=False,
                 label_test=1, label_ref=0, n_jobs=None, vectorized=True):
    """Add summary statistics from a pre-cached (n_samples, n_pre_filter) feature matrix.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Surviving features after ``pre_filtering`` (row order = column order of ``X_cached``).
    X_cached : np.ndarray, shape (n_samples, n_features_kept)
        Pre-cached feature values from ``pre_filtering_info``, already
        column-sliced to match ``df_feat[ut.COL_FEATURE]``.

    Notes
    -----
    Matches the rounding behavior of legacy ``_filters._add_stat``: legacy
    ``get_feature_matrix_`` calls ``_feature_value`` which applies
    ``np.round(..., 5)`` to every per-sample value before stats are computed
    (``utils_feature.py:157``). For bit-identical parity, we apply the same
    5-decimal rounding to the cached matrix here before handing it to
    ``add_stat_``.
    """
    # Match legacy precision: get_feature_matrix_ rounds per-sample values to 5 decimals.
    X_cached = np.round(X_cached, 5)
    df_feat = add_stat_(df=df_feat, X=X_cached, labels=labels, parametric=parametric,
                        label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                        vectorized=vectorized)
    return df_feat
