"""
This is a script for backend of the dPULearn.eval method.
"""
import aaanalysis.utils as ut

import numpy as np
from scipy.stats import entropy, gaussian_kde

# I Helper functions
# Test homogeneity in identified negatives
def _comp_cv(X):
    """Calculate the Coefficient of Variation (CV) of the given X."""
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0)
    cvs = np.divide(std_devs, means, out=np.zeros_like(std_devs), where=means != 0)
    avg_cv = np.mean(cvs)
    return avg_cv


def _comp_entropy(X):
    """Calculate the Entropy of the given X."""
    # Normalize each feature to represent probabilities
    X_prob = X / np.sum(X, axis=0)
    entropies = np.apply_along_axis(entropy, 0, X_prob)
    avg_entropy = np.mean(entropies)
    return avg_entropy

# Test similarity between identified negatives and other classes
def _comp_auc(X=None, labels=None, label_test=0, label_ref=1):
    """Calculate the adjusted Area Under the Curve (AUC) of the given data."""
    X = X.copy()
    # Create a mask for the test and reference labels
    mask = np.logical_or(labels == label_test, labels == label_ref)
    # Filter X and labels
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    # Convert labels to binary format (0 or 1)
    labels_binary = np.where(labels_filtered == label_test, 0, 1)
    auc_vals = abs(ut.auc_adjusted(X=X_filtered, y=labels_binary))
    # Compute the average AUC
    avg_auc_val = np.mean(auc_vals)
    return avg_auc_val


def _compute_kld_for_feature(args):
    """Helper function to compute KLD for a single feature."""
    x1, x2 = args
    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)
    xmin = min(x1.min(), x2.min())
    xmax = max(x1.max(), x2.max())
    x = np.linspace(xmin, xmax, 1000)
    density1 = kde1(x)
    density2 = kde2(x)
    return entropy(density1, density2)


def _comp_kld(X, labels, label_test=0, label_ref=1):
    """Calculate the average Kullback-Leibler Divergence (KLD) for each feature."""
    mask_test = labels == label_test
    mask_ref = labels == label_ref
    X1 = X[mask_ref]
    X2 = X[mask_test]
    # Prepare arguments for each feature
    args = [(X1[:, i], X2[:, i]) for i in range(X.shape[1])]
    # Compute KLD for each feature
    kld_values = np.array([_compute_kld_for_feature(arg) for arg in args])
    # Compute the average KLD
    avg_kld_val = np.mean(kld_values)
    return avg_kld_val


# II Main functions
def eval_homogeneity(X=None, labels=None, label_test=0):
    """Compute two homogeneity measures of coefficient if variation (CV) and entropy"""
    mask = np.asarray(labels) == label_test
    print("hit")
    print(mask)
    cv_val = _comp_cv(X[mask])
    entropy_val = _comp_entropy(X[mask])
    return cv_val, entropy_val


def eval_distribution_alignment(X=None, labels=None):
    """Compute the similarity between identified negatives and the other dataset classes (positives, unlabeled)"""
    label_test = 0
    list_label_ref = [x for x in set(labels) if x in [1, 2]]
    list_eval = []
    # Perform tests
    for label_ref in list_label_ref:
        auc_val = _comp_auc(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
        kld_val = _comp_kld(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
        list_eval.extend([auc_val, kld_val])
    return list_eval


def eval_distribution_alignment_X_neg(X=None, labels=None, X_neg=None):
    """Compute the similarity between identified negatives and ground-truth negatives"""
    label_test = 0
    label_ref = 1  # temporary label for ground-truth negatives for comparison
    mask_test = np.asarray(labels) == label_test
    # Create a combined dataset and labels for identified and ground-truth negatives
    X_test = X[mask_test]
    X_combined = np.vstack([X_test, X_neg])
    labels_combined = np.array([label_test] * len(X_test) + [label_ref] * len(X_neg))
    # Perform tests
    auc_val = _comp_auc(X=X_combined, labels=labels_combined, label_test=label_test, label_ref=label_ref)
    kld_val = _comp_kld(X=X_combined, labels=labels_combined, label_test=label_test, label_ref=label_ref)
    return auc_val, kld_val
