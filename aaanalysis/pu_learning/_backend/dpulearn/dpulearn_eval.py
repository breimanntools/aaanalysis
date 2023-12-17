"""
This is a script for backend of the dPULearn.eval method.
"""
import numpy as np

import aaanalysis.utils as ut


# I Helper functions
# Test homogeneity in identified negatives
def _comp_std(X):
    """Calculate the Coefficient of Variation (CV) of the given X."""
    std_devs = np.std(X, axis=0)
    avg_std = np.mean(std_devs)
    return avg_std

def _comp_iqr(X):
    """
    Calculate the average inter quantile range (IQR) of the given X.
    """
    q1 = np.percentile(X, 25, axis=0)
    q2 = np.percentile(X, 75, axis=0)
    # Calculate the IQR for each feature
    iqr_values = q2 - q1
    # Calculate the average IQR across all features
    avg_iqr = np.mean(iqr_values)
    return avg_iqr


# Test similarity between identified negatives and other classes
def _comp_auc(X=None, labels=None, label_test=0, label_ref=1):
    """Calculate the adjusted Area Under the Curve (AUC) of the given data."""
    X = X.copy()
    # Create a mask for the test and reference labels
    mask = np.asarray([l in [label_test, label_ref] for l in labels])
    # Filter X and labels
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    # Convert labels to binary format (0 or 1)
    labels_binary = np.where(labels_filtered == label_test, 0, 1)
    auc_abs_vals = abs(ut.auc_adjusted_(X=X_filtered, labels=labels_binary))
    # Compute the average AUC
    avg_auc_abs = np.mean(auc_abs_vals)
    return avg_auc_abs


def _comp_kld(X, labels, label_test=0, label_ref=1):
    """Calculate the average Kullback-Leibler Divergence (KLD) for each feature."""
    kld_values = ut.kullback_leibler_divergence_(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
    # Compute the average KLD
    avg_kld = np.mean(kld_values)
    return avg_kld


# II Main functions
def eval_homogeneity(X=None, labels=None, label_test=0):
    """Compute two homogeneity measures of coefficient if variation (CV) and entropy"""
    mask = np.asarray([l == label_test for l in labels])
    avg_std = _comp_std(X[mask])
    avg_iqr = _comp_iqr(X[mask])
    return avg_std, avg_iqr


def eval_distribution_alignment(X=None, labels=None, label_test=0, label_ref=1):
    """Compute the similarity between identified negatives and the other dataset classes (positives, unlabeled)"""
    # Perform tests
    avg_auc_abs = _comp_auc(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
    avg_kld = _comp_kld(X=X, labels=labels, label_test=label_test, label_ref=label_ref)
    return avg_auc_abs, avg_kld


def eval_distribution_alignment_X_neg(X=None, labels=None, X_neg=None):
    """Compute the similarity between identified negatives and ground-truth negatives"""
    label_test = 0
    label_ref = 1  # temporary label for ground-truth negatives for comparison
    mask_test = np.asarray([l == label_test for l in labels])
    # Create a combined dataset and labels for identified and ground-truth negatives
    X_test = X[mask_test]
    X_combined = np.vstack([X_test, X_neg])
    labels_combined = np.array([label_test] * len(X_test) + [label_ref] * len(X_neg))
    # Perform tests
    avg_auc = _comp_auc(X=X_combined, labels=labels_combined, label_test=label_test, label_ref=label_ref)
    avg_kld = _comp_kld(X=X_combined, labels=labels_combined, label_test=label_test, label_ref=label_ref)
    return avg_auc, avg_kld
