"""
This is a script for the backend of the dPULearn.eval() method.
"""
import numpy as np
import pandas as pd

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
    Calculate the average interquartile range (IQR) of the given X.
    """
    q1 = np.percentile(X, 25, axis=0)
    q2 = np.percentile(X, 75, axis=0)
    # Calculate the IQR for each feature
    iqr_values = q2 - q1
    # Calculate the average IQR across all features
    avg_iqr = np.mean(iqr_values)
    return avg_iqr


# Test similarity between identified negatives and other classes
def _comp_auc(X=None, labels=None, label_test=0, label_ref=1, n_jobs=None):
    """Calculate the adjusted Area Under the Curve (AUC) of the given data."""
    X = X.copy()
    # Create a mask for the test and reference labels
    mask = np.asarray([l in [label_test, label_ref] for l in labels])
    # Compute AUC
    auc_abs_vals = abs(ut.auc_adjusted_(X=X[mask], labels=labels[mask], label_test=label_test, n_jobs=n_jobs))
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
def _eval_homogeneity(X=None, labels=None, label_test=0):
    """Compute two homogeneity measures of coefficient if variation (CV) and entropy"""
    mask = np.asarray([l == label_test for l in labels])
    avg_std = _comp_std(X[mask])
    avg_iqr = _comp_iqr(X[mask])
    return avg_std, avg_iqr


def _eval_distribution_alignment(X=None, labels=None, label_test=0, label_ref=1, comp_kld=True, n_jobs=None):
    """Compute the similarity between identified negatives and the other dataset classes (positives, unlabeled)"""
    # Perform tests
    avg_auc_abs = _comp_auc(X=X, labels=labels, label_test=label_test, label_ref=label_ref, n_jobs=n_jobs)
    avg_kld = _comp_kld(X=X, labels=labels, label_test=label_test, label_ref=label_ref) if comp_kld else None
    return avg_auc_abs, avg_kld


def _eval_distribution_alignment_X_neg(X=None, labels=None, X_neg=None, comp_kld=True, n_jobs=None):
    """Compute the similarity between identified negatives and ground-truth negatives"""
    label_test = 0
    label_ref = 1  # temporary label for ground-truth negatives for comparison
    mask_test = np.asarray([l == label_test for l in labels])
    # Create a combined dataset and labels for identified and ground-truth negatives
    X_test = X[mask_test]
    X_combined = np.vstack([X_test, X_neg])
    labels_combined = np.array([label_test] * len(X_test) + [label_ref] * len(X_neg))
    # Perform tests
    avg_auc = _comp_auc(X=X_combined, labels=labels_combined, label_test=label_test, label_ref=label_ref, n_jobs=n_jobs)
    avg_kld = _comp_kld(X=X_combined, labels=labels_combined, label_test=label_test, label_ref=label_ref) if comp_kld else None
    return avg_auc, avg_kld


@ut.catch_runtime_warnings()
def eval_identified_negatives(X=None, list_labels=None, names_datasets=None, X_neg=None, comp_kld=True, n_jobs=None):
    """Evaluate set of identified negatives for homogeneity and alignment with other datasets"""
    unl_in = False
    list_evals = []
    for labels in list_labels:
        # Evaluate homogeneity
        n_rel_neg = sum(labels == 0)
        avg_std, avg_iqr = _eval_homogeneity(X=X, labels=labels)
        # Evaluate distribution alignment with positives
        args = dict(X=X, labels=labels, comp_kld=comp_kld)
        avg_auc_abs, avg_kld = _eval_distribution_alignment(**args, label_test=0, label_ref=1, n_jobs=n_jobs)
        list_eval = [n_rel_neg, avg_std, avg_iqr, avg_auc_abs, avg_kld]
        # Evaluate distribution alignment with unlabeled
        if 2 in labels:
            avg_auc_abs, avg_kld = _eval_distribution_alignment(**args, label_test=0, label_ref=2, n_jobs=n_jobs)
            list_eval += [avg_auc_abs, avg_kld]
            unl_in = True
        # Evaluate distribution alignment with ground-truth negatives, if provided
        if X_neg is not None:
            avg_auc_abs, avg_kld = _eval_distribution_alignment_X_neg(**args, X_neg=X_neg, n_jobs=n_jobs)
            list_eval += [avg_auc_abs, avg_kld]
        # Remove kld if disabled
        list_eval = [x for x in list_eval if x is not None]
        list_evals.append(list_eval)
    # Define column names based on the evaluations performed
    cols_eval = ut.COLS_EVAL_DPULEARN[0:4]
    if comp_kld:
        cols_eval.append(ut.COL_AVG_KLD_POS)
    if unl_in:
        cols_eval.append(ut.COL_AVG_ABS_AUC_UNL)
        if comp_kld:
            cols_eval.append(ut.COL_AVG_KLD_UNL)
    if X_neg is not None:
        cols_eval.append(ut.COL_AVG_ABS_AUC_NEG)
        if comp_kld:
            cols_eval.append(ut.COL_AVG_KLD_NEG)
    # Create the DataFrame
    df_eval = pd.DataFrame(list_evals, columns=cols_eval).round(4)
    df_eval = ut.add_names_to_df_eval(df_eval=df_eval, names=names_datasets)
    return df_eval