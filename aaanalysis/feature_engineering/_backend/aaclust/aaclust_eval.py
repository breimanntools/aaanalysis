"""
This is a script for the backend of the AAclust.eval method.
"""
import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from aaanalysis.utils import bic_score_

import aaanalysis.utils as ut


# I Helper Functions
def _evaluate_clustering(X, labels=None):
    """Evaluate clustering results using BIC, CH, SC scores"""
    # Extreme-but-finite feature values can overflow inside the metric reductions
    # (e.g. ``(mean_k - mean) ** 2`` -> inf -> ``inf - inf`` -> NaN in a reduce). The
    # resulting NaN/inf scores are handled by the caller (NaN -> 0/-1 + warn_ch/warn_sc),
    # so silence the low-level numpy floating-point noise here (np.errstate gates emission
    # only; the computed values are unchanged).
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Bayesian Information Criterion
        bic = bic_score_(X, labels)
        # Calinski-Harabasz Index
        ch = calinski_harabasz_score(X, labels)
        # Silhouette Coefficient
        sc = silhouette_score(X, labels)
    return bic, ch, sc


# II Main function
@ut.catch_runtime_warnings()
def evaluate_clustering(X, list_labels=None, names_datasets=None):
    """Evaluate sets of clustering labels"""
    list_evals = []
    warn_ch = False
    warn_sc = False
    for labels in list_labels:
        n_clusters = len(set(labels))
        bic, ch, sc = _evaluate_clustering(X, labels=labels)
        if np.isnan(ch):
            ch = 0
            warn_ch = True
        if np.isnan(sc):
            sc = -1
            warn_sc = True
        list_evals.append([n_clusters, bic, ch, sc])
    # Create the DataFrame
    df_eval = pd.DataFrame(list_evals, columns=ut.COLS_EVAL_AACLUST)
    df_eval = ut.add_names_to_df_eval(df_eval=df_eval, names=names_datasets)
    return df_eval, warn_ch, warn_sc

