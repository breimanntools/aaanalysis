"""
This is a script for utility feature statistics functions for CPP and SequenceFeature objects and backend.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
from statsmodels.stats.multitest import multipletests

import aaanalysis.utils as ut


# I Helper Functions
# Summary and test statistics for feature matrix based on classification by labels
def _mean_dif(X=None, y=None):
    """ Get mean difference for values in X (feature matrix) based on y (labels)"""
    mask_0 = [x == 0 for x in y]
    mask_1 = [x == 1 for x in y]
    mean_difs = np.mean(X[mask_1], axis=0) - np.mean(X[mask_0], axis=0)
    return mean_difs


def _std(X=None, y=None, group=1):
    """Get standard deviation (std) for data sets points with group label"""
    mask = [x == group for x in y]
    group_std = np.std(X[mask], axis=0)
    return group_std


def _auc(X=None, y=None):
    """Get adjusted Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    comparing, for each feature, groups (given by y (labels)) by feature values in X (feature matrix).
    """
    # Multiprocessing for AUC computation
    auc = np.apply_along_axis((lambda x: roc_auc_score(y, x) - 0.5), 0, X)
    auc = np.round(auc, 3)
    return auc

def _p_correction(p_vals=None, p_cor="fdr_bh"):
    """Correct p-values"""
    # Exclude nan from list of corrected p-values
    p_vals_without_na = [p for p in p_vals if str(p) != "nan"]
    p_corrected_without_na = list(multipletests(p_vals_without_na, method=p_cor)[1])
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


def _mean_stat(X=None, y=None, parametric=False, p_cor=None):
    """Statistical comparison of central tendency between two groups for each feature"""
    mask_0 = [True if x == 0 else False for x in y]
    mask_1 = [True if x == 1 else False for x in y]
    if parametric:
        p_vals = stats.ttest_ind(X[mask_1], X[mask_0], nan_policy="omit")[1]
        p_str = "p_val_ttest_indep"
    else:
        t = lambda x1, x2: stats.mannwhitneyu(x1, x2, alternative="two-sided")[1]  # Test statistic
        c = lambda x1, x2: np.mean(x1) != np.mean(x2) or np.std(x1) != np.std(x2)  # Test condition
        p_vals = np.round([t(col[mask_1], col[mask_0]) if c(col[mask_1], col[mask_0]) else 1 for col in X.T], 10)
        p_str = "p_val_mann_whitney"
    if p_cor is not None:
        p_vals = _p_correction(p_vals=p_vals, p_cor=p_cor)
        p_str = "p_val_" + p_cor
    return p_vals, p_str


# II Main Functions
def add_stat_(df=None, X=None, y=None, parametric=False):
    """Add summary statistics of feature matrix (X) for given labels (y) to df"""
    df = df.copy()
    columns_input = list(df)
    df[ut.COL_ABS_AUC] = abs(_auc(X=X, y=y))
    df[ut.COL_MEAN_DIF] = _mean_dif(X=X, y=y)
    if ut.COL_ABS_MEAN_DIF not in list(df):
        df[ut.COL_ABS_MEAN_DIF] = abs(_mean_dif(X=X, y=y))
    df[ut.COL_STD_TEST] = _std(X=X, y=y, group=1)
    df[ut.COL_STD_REF] = _std(X=X, y=y, group=0)
    p_val, p_str = _mean_stat(X=X, y=y, parametric=parametric)
    df[p_str] = p_val
    p_val_fdr, p_str_fdr = _mean_stat(X=X, y=y, parametric=parametric, p_cor="fdr_bh")
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