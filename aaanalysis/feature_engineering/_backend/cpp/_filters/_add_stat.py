"""
This is a script for the backend of CPP's full-statistics stage:
``add_stat`` augments the surviving feature DataFrame with AUC, mean
difference, p-values, and FDR-adjusted p-values.
"""
import aaanalysis.utils as ut
from ..utils_feature import get_feature_matrix_
from .._utils_feature_stat import add_stat_


# I Helper Functions
# (no helpers — single-function module wrapping shared utilities)


# II Main Functions
def add_stat(df_feat=None, df_parts=None, df_scales=None, labels=None, parametric=False, accept_gaps=False,
             label_test=1, label_ref=0, n_jobs=None, vectorized=True):
    """Add summary statistics for each feature to DataFrame."""
    features = list(df_feat[ut.COL_FEATURE])
    X = get_feature_matrix_(features=features, df_parts=df_parts, df_scales=df_scales, accept_gaps=accept_gaps, n_jobs=n_jobs)
    df_feat = add_stat_(df=df_feat, X=X, labels=labels, parametric=parametric,
                        label_test=label_test, label_ref=label_ref, n_jobs=n_jobs, vectorized=vectorized)
    return df_feat
