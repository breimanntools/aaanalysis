"""
This is a script for utility functions for statistical measures.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy, gaussian_kde
from collections import OrderedDict
from scipy.spatial import distance


# AUC adjusted
def auc_adjusted_(X=None, labels=None, label_test=1):
    """Get adjusted Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    comparing, for each feature, groups (given by y (labels)) by feature values in X (feature matrix).
    """
    # Convert labels to binary format (1 or 0)
    labels_binary = [1 if y == label_test else 0 for y in labels]
    auc = np.apply_along_axis((lambda x: roc_auc_score(labels_binary, x) - 0.5), 0, X)
    auc = np.round(auc, 3)
    return auc


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
