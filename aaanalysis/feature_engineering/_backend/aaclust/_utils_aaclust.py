"""
This is a script for utility functions for AAclust object and backend.
"""
import numpy as np
from collections import OrderedDict
from sklearn.metrics import pairwise_distances

# II Main Functions
def _cluster_center(X):
    """Compute cluster center (i.e., arithmetical mean over all data points/observations of a cluster)"""
    return X.mean(axis=0)[np.newaxis, :]


def _cluster_medoid(X, metric="correlation"):
    """Obtain cluster medoids (i.e., scale closest to cluster center used as representative scale for a cluster)"""
    center = _cluster_center(X).reshape(1, -1)
    if metric == "correlation":
        # Create new array with cluster center and given array
        center_X = np.concatenate([_cluster_center(X), X], axis=0)
        # Get index for scale with the highest correlation with cluster center
        medoid_index = np.corrcoef(center_X)[0, 1:].argmax()
    else:
        # Calculating pairwise distances from center to all points in X
        distances = pairwise_distances(center, X, metric=metric)
        # Finding the index of the point with the minimum distance to the center
        medoid_index = np.argmin(distances)
    return medoid_index


def _dist_to_medoids(X, labels=None, medoid_ind=None, labels_medoids=None, metric="correlation"):
    """Distance (under metric) of each sample in X to its cluster's medoid; medoids get 0.0.

    For ``metric='correlation'`` the distance is ``1 - Pearson(sample, medoid)``; otherwise the
    pairwise distance from the medoid is used (mirrors the metric handling of `_cluster_medoid`).
    """
    labels = np.asarray(labels)
    medoid_ind = np.asarray(medoid_ind)
    labels_medoids = np.asarray(labels_medoids)
    label_to_medoid = {lab: int(medoid_ind[j]) for j, lab in enumerate(labels_medoids)}
    n = len(labels)
    dist = np.zeros(n, dtype=float)
    medoid_of = np.array([label_to_medoid[lab] for lab in labels])
    if metric == "correlation":
        # Row-wise Pearson(sample, its medoid), vectorized. Mathematically the same value
        # as np.corrcoef per pair (the ddof cancels in the ratio); differs only at the ULP
        # level from the summation order, and the medoid's own row is forced to 0.0.
        Xm = X[medoid_of]
        Xc = X - X.mean(axis=1, keepdims=True)
        Mc = Xm - Xm.mean(axis=1, keepdims=True)
        num = np.einsum("ij,ij->i", Xc, Mc)
        den = np.sqrt(np.einsum("ij,ij->i", Xc, Xc) * np.einsum("ij,ij->i", Mc, Mc))
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = 1.0 - num / den
        dist[np.arange(n) == medoid_of] = 0.0
    else:
        for i in range(n):
            m = medoid_of[i]
            if i == m:
                continue  # medoid's distance to itself is 0.0
            dist[i] = pairwise_distances(X[m].reshape(1, -1), X[i].reshape(1, -1), metric=metric)[0, 0]
    return dist


def _compute_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    labels_centers = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in labels_centers]
    centers = np.concatenate([_cluster_center(X[mask]) for mask in list_masks]).round(3)
    labels_centers = np.array(labels_centers)
    return centers, labels_centers


def _compute_medoids(X, labels=None, metric="correlation"):
    """Obtain cluster medoids and their labels"""
    unique_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in unique_labels]
    # Calculating medoid for each mask using specified metric
    list_ind_max = [_cluster_medoid(X[mask], metric=metric) for mask in list_masks]
    indices = np.array(range(0, len(labels)))
    # Finding global indices of medoids
    medoid_ind = [indices[m][i] for m, i in zip(list_masks, list_ind_max)]
    # Finding labels and data of medoids
    labels_medoids = np.array([labels[i] for i in medoid_ind])
    medoids = np.array([X[i, :] for i in medoid_ind])
    return medoids, labels_medoids, medoid_ind
