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
from .._utils_feature_stat import add_stat_, _p_correction


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


def apply_pooled_fdr(df_feat=None, parametric=False):
    """Recompute the BH FDR column once over the pooled p-values of ``df_feat``.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Feature DataFrame whose raw p-value column was filled by ``add_stat``,
        concatenated across all batches of a batched orchestration.
    parametric : bool, default=False
        Whether the raw p-value column is the t-test (``True``) or the
        Mann-Whitney U (``False``) column.

    Returns
    -------
    df_feat : pd.DataFrame
        Copy of the input with ``p_val_fdr_bh`` corrected over every row at once.

    Notes
    -----
    The Benjamini-Hochberg correction is a property of the whole candidate set:
    the adjusted p-value of a feature depends on how many features were tested
    with it. A batched orchestration that calls ``add_stat`` per batch therefore
    corrects each batch against its own size, which makes the reported
    ``p_val_fdr_bh`` depend on an arbitrary partition. Pooling the raw p-values
    and correcting once reproduces the single-pass result exactly.
    """
    p_str = ut.COL_PVAL_TTEST if parametric else ut.COL_PVAL_MW
    df_feat = df_feat.copy()
    df_feat[ut.COL_PVAL_FDR] = _p_correction(p_vals=df_feat[p_str].to_list())
    return df_feat
