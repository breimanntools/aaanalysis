"""
This is a script for computing the Bayesian Information Criterion (BIC).
"""
import time
import pandas as pd
import numpy as np

# I Helper Functions
def bic_score(X, labels=None, centers=None, center_labels=None, n_clust=None):
    """Computes the BIC metric for given clusters

    See also
    --------
    https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    """
    # Check if labels match to number of clusters
    if len(set(labels)) != n_clust:
        return np.NaN
    if 0 not in labels:
        labels = np.array([x-1 for x in labels])    # Adjust labels starting at 0 for bincount
        if center_labels is not None:
            center_labels = np.array([x-1 for x in center_labels])
        if min(labels) != 0:
            print(min(labels))
    size_clusters = np.bincount(labels)
    n_samples, n_features = X.shape
    # Compute variance over all clusters beforehand
    if center_labels is None:
        center_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in center_labels]
    sum_squared_dist = sum([sum(distance.cdist(X[list_masks[i]], [centers[i]], 'euclidean')**2) for i in range(n_clust)])
    cl_var = (1.0 / (n_samples - n_clust) / n_features) * sum_squared_dist
    const_term = 0.5 * n_clust * np.log(n_samples) * (n_features + 1)
    # Compute BIC
    bic = np.sum([size_clusters[i] * np.log(size_clusters[i]) -
                  size_clusters[i] * np.log(n_samples) -
                  ((size_clusters[i] * n_features) / 2) * np.log(2*np.pi*cl_var) -
                  ((size_clusters[i] - 1) * n_features / 2)
                  for i in range(n_clust)]) - const_term
    return bic
