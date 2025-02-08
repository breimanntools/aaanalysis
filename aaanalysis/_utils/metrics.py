"""
This is a script for utility functions for statistical measures.
"""
import numpy as np
from scipy.stats import entropy, gaussian_kde
from collections import OrderedDict
from scipy.spatial import distance
from joblib import Parallel, delayed
import os

DTYPE = np.float64


# AUC adjusted
def _pre_sort(X):
    """Pre-sort X in descending order across all features."""
    # Sort each column (feature-wise) in descending order
    sorted_indices = np.argsort(-X, axis=0)
    return sorted_indices


def _compute_auc_sorted(sorted_indices, y_true):
    """Compute AUC using pre-sorted indices for all features."""
    n_samples, n_features = sorted_indices.shape
    auc_values = np.empty(n_features, dtype=DTYPE)
    for j in range(n_features):
        # Reorder labels using precomputed indices
        y_true_sorted = np.take(y_true, sorted_indices[:, j])
        pos = np.sum(y_true_sorted)
        neg = n_samples - pos
        if pos == 0 or neg == 0:
            auc_values[j] = 0.5  # AUC is undefined when there's only one class
            continue
        cum_pos = np.cumsum(y_true_sorted)  # Cumulative sum of positive samples
        auc_values[j] = np.sum(cum_pos * (1 - y_true_sorted)) / (pos * neg)
    return auc_values


def auc_adjusted_(X=None, labels=None, label_test=1, n_jobs=None):
    """Get adjusted ROC AUC with pre-sorting and parallel computation."""
    labels_binary = np.array([int(y == label_test) for y in labels], dtype=DTYPE)
    # Determine the number of parallel jobs
    if n_jobs is None:
        n_jobs = min(os.cpu_count(), max(int(X.shape[1] / 10), 1))
    # Step 1: Pre-sort feature values once for all features
    sorted_indices = _pre_sort(X)
    # Step 2: Compute AUC using pre-sorted indices
    if n_jobs == 1:
        auc_values = _compute_auc_sorted(sorted_indices, labels_binary)
    else:
        # Split features into chunks for parallel processing
        feature_chunks = np.array_split(np.arange(X.shape[1]), n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_auc_sorted)(sorted_indices[:, chunk], labels_binary) for chunk in feature_chunks)
        auc_values = np.concatenate(results)
    return np.round(auc_values - 0.5, 3)


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
