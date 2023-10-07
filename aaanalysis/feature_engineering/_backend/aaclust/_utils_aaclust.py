"""
This is a script for utility functions for AAclust object and backend .
"""
import numpy as np
from collections import OrderedDict


# II Main Functions
def _cluster_center(X):
    """Compute cluster center (i.e., arithmetical mean over all data points/observations of a cluster)"""
    return X.mean(axis=0)[np.newaxis, :]


def _cluster_medoid(X):
    """Obtain cluster medoids (i.e., scale closest to cluster center used as representative scale for a cluster)"""
    # Create new array with cluster center and given array
    center_X = np.concatenate([_cluster_center(X), X], axis=0)
    # Get index for scale with the highest correlation with cluster center
    ind_max = np.corrcoef(center_X)[0, 1:].argmax()
    return ind_max


def _compute_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    center_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[True if i == label else False for i in labels] for label in center_labels]
    centers = np.concatenate([_cluster_center(X[mask]) for mask in list_masks]).round(3)
    return centers, np.array(center_labels)




