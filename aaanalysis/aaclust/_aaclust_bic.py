"""
This is a script for computing the Bayesian Information Criterion (BIC).
"""
import time
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance
from aaanalysis.aaclust._aaclust import compute_centers

# I Helper Functions

# II Main function
def bic_score(X, labels=None):
    """Computes the BIC metric for given clusters.

    See also
    --------
    https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    """
    # Check if labels match to number of clusters
    n_classes = len(set(labels))
    n_samples, n_features = X.shape
    if n_classes >= n_samples:
        raise ValueError(f"Number of classes in 'labels' ({n_classes}) must be smaller than n_samples ({n_samples})")
    if n_features == 0:
        raise ValueError(f"'n_features' should not be 0")
    # Map labels to increasing order starting with 0
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    labels = inverse
    centers, center_labels = compute_centers(X, labels=labels)
    size_clusters = np.bincount(labels)
    # Compute variance over all clusters beforehand
    list_masks = [[i == label for i in labels] for label in center_labels]
    sum_squared_dist = sum(
        [sum(distance.cdist(X[list_masks[i]], [centers[i]], 'euclidean') ** 2) for i in range(n_classes)])
    # Compute between-cluster variance
    denominator = (n_samples - n_classes) * n_features
    bet_clu_var = (1.0 / denominator) * sum_squared_dist
    if bet_clu_var == 0:
        raise ValueError("The between-cluster variance should not be 0")
    # Compute BIC
    const_term = 0.5 * n_classes * np.log(n_samples) * (n_features + 1)
    bic_components = []
    for i in range(n_classes):
        component = (size_clusters[i] * np.log(size_clusters[i]) - size_clusters[i] * np.log(n_samples) - (
                    (size_clusters[i] * n_features) / 2) * np.log(2 * np.pi * bet_clu_var) - (
                                 (size_clusters[i] - 1) * n_features / 2))
        bic_components.append(component)
    bic = np.sum(bic_components) - const_term
    return bic
