"""
This is a script for the backend of the AAclust.eval method.
"""
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from aaanalysis.utils import bic_score_

import aaanalysis.utils as ut


# I Helper Functions


# II Main function
@ut.catch_runtime_warnings()
def evaluate_clustering(X, labels=None):
    """Evaluate clustering results using BIC, CH, SC scores"""
    # Bayesian Information Criterion
    bic = bic_score_(X, labels)
    # Calinski-Harabasz Index
    ch = calinski_harabasz_score(X, labels)
    # Silhouette Coefficient
    sc = silhouette_score(X, labels)
    return bic, ch, sc
