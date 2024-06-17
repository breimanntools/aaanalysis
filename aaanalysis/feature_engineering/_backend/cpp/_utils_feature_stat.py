"""
This is a script for utility feature statistics functions for CPP and SequenceFeature objects and backend.
"""
import numpy as np
from scipy import stats

import aaanalysis.utils as ut


# I Helper Functions
# Benjamini Hochberg correction
def _bh_corrected_pvalues(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    ranks = np.arange(1, n+1)
    corrected_pvals = sorted_pvals * n / ranks
    corrected_pvals[corrected_pvals > 1] = 1  # p-values cannot exceed 1
    # Reorder to the original order
    corrected_pvals_original_order = np.empty(n, dtype=float)
    corrected_pvals_original_order[sorted_indices] = corrected_pvals
    return corrected_pvals_original_order


# Summary and test statistics for feature matrix based on classification by labels
def _mean_dif(X=None, labels=None, label_test=1, label_ref=0):
    """ Get mean difference for values in X (feature matrix) based on y (labels)"""
    mask_test = [x == label_test for x in labels]
    mask_ref = [x == label_ref for x in labels]
    mean_difs = np.mean(X[mask_test], axis=0) - np.mean(X[mask_ref], axis=0)
    return mean_difs


def _std(X=None, labels=None, group=1):
    """Get standard deviation (std) for datasets points with group label"""
    mask = [x == group for x in labels]
    group_std = np.std(X[mask], axis=0)
    return group_std


def _p_correction(p_vals=None):
    """Correct p-values"""
    # Exclude nan from list of corrected p-values
    p_vals_without_na = [p for p in p_vals if str(p) != "nan"]
    p_corrected_without_na = _bh_corrected_pvalues(p_vals_without_na)
    # Include nan in list of corrected p-values
    p_corrected = []
    i = 0
    for p in p_vals:
        if str(p) != "nan":
            p_corrected.append(p_corrected_without_na[i])
            i += 1
        else:
            p_corrected.append(np.nan)
    return p_corrected


def _mean_stat(X=None, labels=None, parametric=False, p_cor=False, label_test=1, label_ref=0):
    """Statistical comparison of central tendency between two groups for each feature"""
    mask_test = [x == label_test for x in labels]
    mask_ref = [x == label_ref for x in labels]
    if parametric:
        p_vals = stats.ttest_ind(X[mask_test], X[mask_ref], nan_policy="omit")[1]
        p_str = "p_val_ttest_indep"
    else:
        t = lambda x1, x2: stats.mannwhitneyu(x1, x2, alternative="two-sided")[1]  # Test statistic
        c = lambda x1, x2: np.mean(x1) != np.mean(x2) or np.std(x1) != np.std(x2)  # Test condition
        p_vals = np.round([t(col[mask_test], col[mask_ref]) if c(col[mask_test], col[mask_ref]) else 1 for col in X.T], 10)
        p_str = "p_val_mann_whitney"
    if p_cor:
        p_vals = _p_correction(p_vals=p_vals)
        p_str = "p_val_fdr_bh"
    return p_vals, p_str


# II Main Functions
def add_stat_(df=None, X=None, labels=None, parametric=False, label_test=1, label_ref=0):
    """Add summary statistics of feature matrix (X) for given labels (y) to df"""
    df = df.copy()
    columns_input = list(df)
    args_labels = dict(labels=labels, label_test=label_test, label_ref=label_ref)
    df[ut.COL_ABS_AUC] = abs(ut.auc_adjusted_(X=X, labels=labels, label_test=label_test))
    df[ut.COL_MEAN_DIF] = _mean_dif(X=X, **args_labels)
    if ut.COL_ABS_MEAN_DIF not in list(df):
        df[ut.COL_ABS_MEAN_DIF] = abs(_mean_dif(X=X, **args_labels))
    df[ut.COL_STD_TEST] = _std(X=X, labels=labels, group=label_test)
    df[ut.COL_STD_REF] = _std(X=X, labels=labels, group=label_ref)
    p_val, p_str = _mean_stat(X=X, parametric=parametric, **args_labels)
    df[p_str] = p_val
    p_val_fdr, p_str_fdr = _mean_stat(X=X, parametric=parametric, p_cor=True, **args_labels)
    df[p_str_fdr] = p_val_fdr
    cols_stat = [ut.COL_ABS_AUC,
                 ut.COL_ABS_MEAN_DIF, ut.COL_MEAN_DIF,
                 ut.COL_STD_TEST, ut.COL_STD_REF,
                 p_str, p_str_fdr]
    cols_in = [x for x in columns_input if x not in cols_stat and x != ut.COL_FEATURE]
    columns = [ut.COL_FEATURE] + cols_in + cols_stat
    df = df[columns]
    cols_round = [x for x in cols_stat if x not in [p_str, p_str_fdr]]
    df[cols_round] = df[cols_round].round(6)
    return df