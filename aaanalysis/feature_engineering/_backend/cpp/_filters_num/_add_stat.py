"""
This is a script for the backend of CPP's numerical-mode full-statistics stage:
``add_stat_num`` augments the surviving feature DataFrame with AUC, mean
difference, p-values, and FDR-adjusted p-values using a **pre-cached** per-sample
feature-value matrix from ``_filters_num._stat_filter.pre_filtering_info_num``.

Eliminates the duplicate per-feature compute that the legacy ``_filters._add_stat``
incurs via ``get_feature_matrix_`` (which iterates the split+scale lookup per
feature in Python). The cached matrix has full numerical equivalence in
seq-mode because both paths reduce to ``mean(scale_matrix[aa_idx_in_segment, d])``
over the same residues — see ``docs/adr/0001-cpp-run-num.md`` for the parity
contract.
"""
import numpy as np

import aaanalysis.utils as ut
from .._utils_feature_stat import add_stat_


# I Helper Functions
# (no helpers — single-function module wrapping shared utilities)


# II Main Functions
def add_stat_num(df_feat=None, X_cached=None, labels=None, parametric=False,
                 label_test=1, label_ref=0, n_jobs=None, vectorized=True):
    """Add summary statistics from a pre-cached (n_samples, n_pre_filter) feature matrix.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Surviving features after ``pre_filtering`` (row order = column order of ``X_cached``).
    X_cached : np.ndarray, shape (n_samples, n_features_kept)
        Pre-cached feature values from ``pre_filtering_info_num``, already
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
