"""
This is a script for the AAclust().compute_correlation().
"""

import numpy as np
from aaanalysis.aaclust._aaclust import _cluster_center, min_cor_all


def compute_correlation(X, X_ref, labels=None, labels_ref=None, n=3, positive=True, on_center=False):
    """
    Computes the Pearson correlation of given data with reference data.

    Parameters
    ----------
    X : `array-like, shape (n_samples, n_features)`
        Feature matrix. Rows correspond to scales and columns to amino acids.
    X_ref : `array-like, shape (n_samples, n_features)`
        Feature matrix of reference data.
    labels : `array-like, shape (n_samples, )`
        Cluster labels for each sample in ``X``.
    labels_ref  : `array-like, shape (n_samples, )`
        Cluster labels for the reference data.
    n
        Number of top centers to consider based on correlation strength.
    positive
        If True, considers positive correlations. Else, negative correlations.
    on_center
        If True, correlation is computed with cluster centers. Otherwise, with all cluster members.

    Returns
    -------
    list_top_center_name_corr : list
        Names and correlations of centers having the strongest (positive/negative) correlation with test data samples.
    """
    names_ref = list(dict.fromkeys(labels_ref))
    masks_ref = [[i == label for i in labels_ref] for label in names_ref]
    if on_center:
        # Get centers for all clusters in reference data
        centers = np.concatenate([_cluster_center(X_ref[mask]) for mask in masks_ref], axis=0)
        # Compute correlation of test data with centers
        Xtest_centers = np.concatenate([X, centers], axis=0)
        n_test = X.shape[0]
        X_corr = np.corrcoef(Xtest_centers)[:n_test, n_test:]
    else:
        masks_test = [[True if i == j else False for j in range(0, len(labels))] for i, _ in enumerate(labels)]
        # Compute minimum correlation of test data with each group of reference data
        X_corr = np.array(
            [[min_cor_all(np.concatenate([X[mask_test], X_ref[mask_ref]], axis=0)) for mask_ref in masks_ref] for
             mask_test in masks_test])
    # Get index for n centers with highest/lowest correlation for each scale
    if positive:
        list_top_center_ind = X_corr.argsort()[:, -n:][:, ::-1]
    else:
        list_top_center_ind = X_corr.argsort()[:, :n]
    # Get name and correlation for centers correlating strongest (positive/negative) with test data samples
    list_top_center_name_corr = []
    for i, ind in enumerate(list_top_center_ind):
        top_corr = X_corr[i, :][ind]
        top_names = [names_ref[x] for x in ind]
        str_corr = ";".join([f"{name} ({round(corr, 3)})" for name, corr in zip(top_names, top_corr)])
        list_top_center_name_corr.append(str_corr)
    return list_top_center_name_corr