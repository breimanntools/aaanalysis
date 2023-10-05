"""
This is a script for utility functions for aaclust object.
"""
import numpy as np


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




