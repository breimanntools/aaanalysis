"""
This is a script for utility functions for statistical measures.
"""
import numpy as np
from scipy.stats import entropy, gaussian_kde
from collections import OrderedDict
from scipy.spatial import distance
from joblib import Parallel, delayed
import os
from scipy.stats import rankdata

DTYPE = np.float64


# AUC adjusted
def _compute_auc_sorted(X, labels):
    """Compute AUC for a subset of features using the same ranking approach as roc_auc_score."""
    n_samples, n_features = X.shape
    auc_values = np.empty(n_features, dtype=np.float64)
    for j in range(n_features):
        # Rank the feature values, handling ties properly
        ranks = rankdata(X[:, j])  # Average ranking for ties
        pos = np.sum(labels)
        neg = n_samples - pos
        if pos == 0 or neg == 0:
            auc_values[j] = 0.5  # Undefined AUC when only one class is present
            continue
        # Compute AUC using the Mann-Whitney U statistic
        rank_sum_pos = np.sum(ranks[labels == 1])
        auc_values[j] = (rank_sum_pos - (pos * (pos + 1) / 2)) / (pos * neg)
    return np.round(auc_values - 0.5, 3)


def auc_adjusted_(X=None, labels=None, label_test=1, n_jobs=None):
    """Get adjusted ROC AUC with pre-sorting and parallel computation."""
    # Get binary labels and precompute ranks for all features
    labels_binary = np.array([int(y == label_test) for y in labels], dtype=DTYPE)
    ranked_X = np.apply_along_axis(rankdata, 0, X)

    if n_jobs is None:
        n_jobs = min(os.cpu_count(), max(int(X.shape[1] / 10), 1))

    if n_jobs == 1:
        return _compute_auc_sorted(ranked_X, labels_binary)

    feature_chunks = np.array_split(np.arange(X.shape[1]), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_compute_auc_sorted)(ranked_X[:, chunk], labels_binary)
                           for chunk in feature_chunks)
    auc_values = np.concatenate(results)
    return auc_values


# Bayesian Information Criterion for clusters
def _cluster_center(X):
    """Compute cluster center (i.e., arithmetical mean over all data points/observations of a cluster)"""
    return X.mean(axis=0)[np.newaxis, :]


def _compute_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    labels_centers = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in labels_centers]
    centers = np.concatenate([_cluster_center(X[mask]) for mask in list_masks]).round(3)
    labels_centers = np.array(labels_centers)
    return centers, labels_centers


def bic_score_(X, labels=None):
    """Computes the Bayesian Information Criterion (BIC) metric for given clusters."""
    epsilon = 1e-10  # prevent division by zero

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
    centers, center_labels = _compute_centers(X, labels=labels)
    size_clusters = np.bincount(labels)

    # Compute variance over all clusters
    list_masks = [labels == label for label in center_labels]
    sum_squared_dist = sum([sum(distance.cdist(X[mask], [center], 'euclidean') ** 2) for mask, center in zip(list_masks, centers)])

    # Compute between-cluster variance
    denominator = max((n_samples - n_classes) * n_features, epsilon)
    bet_clu_var = max((1.0 / denominator) * sum_squared_dist, epsilon)

    # Compute BIC components
    const_term = 0.5 * n_classes * np.log(n_samples) * (n_features + 1)

    log_size_clusters = np.log(size_clusters + epsilon)
    log_n_samples = np.log(n_samples + epsilon)
    log_bcv = np.log(2 * np.pi * bet_clu_var)

    bic_components = size_clusters * (log_size_clusters - log_n_samples) - 0.5 * size_clusters * n_features * log_bcv - 0.5 * (size_clusters - 1) * n_features
    bic = np.sum(bic_components) - const_term
    return bic


# Kullback-Leibler Divergence
def _comp_kld_for_feature(args):
    """Compute KLD for a single feature."""
    x1, x2 = args
    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)
    xmin = min(x1.min(), x2.min())
    xmax = max(x1.max(), x2.max())
    x = np.linspace(xmin, xmax, 1000)
    density1 = kde1(x)
    density2 = kde2(x)
    return entropy(density1, density2)


def kullback_leibler_divergence_(X=None, labels=None, label_test=0, label_ref=1):
    """Calculate the average Kullback-Leibler Divergence (KLD) for each feature."""
    mask_test = np.asarray([x == label_test for x in labels])
    mask_ref = np.asarray([x == label_ref for x in labels])
    X1 = X[mask_ref]
    X2 = X[mask_test]
    # Prepare arguments for each feature
    args = [(X1[:, i], X2[:, i]) for i in range(X.shape[1])]
    # Compute KLD for each feature
    kld = np.array([_comp_kld_for_feature(arg) for arg in args])
    return kld
