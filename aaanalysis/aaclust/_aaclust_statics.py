"""
This is a script for the static methods of the AAclust class.
"""
from collections import OrderedDict
import numpy as np

from aaanalysis.aaclust._aaclust import _cluster_center, _cluster_medoid, min_cor_all
# I Helper Functions


# II Main Functions
def compute_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    center_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[True if i == label else False for i in labels] for label in center_labels]
    centers = np.concatenate([_cluster_center(X[mask]) for mask in list_masks]).round(3)
    return centers, center_labels


def compute_medoids(X, labels=None):
    """Obtain cluster medoids and their labels"""
    unique_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[True if i == label else False for i in labels] for label in unique_labels]
    list_ind_max = [_cluster_medoid(X[mask]) for mask in list_masks]
    indices = np.array(range(0, len(labels)))
    medoid_ind = [indices[m][i] for m, i in zip(list_masks, list_ind_max)]
    medoid_labels = [labels[i] for i in medoid_ind]
    medoids = np.array([X[i, :] for i in medoid_ind])
    return medoids, medoid_labels, medoid_ind


def compute_corr(X, X_ref, labels=None, labels_ref=None, n=3, positive=True, on_center=False, except_unclassified=True):
    """Computes the correlation of given data with reference data."""
    # Delete
    if except_unclassified:
        names_ref = list(dict.fromkeys(labels_ref))
    else:
        names_ref = [x for x in list(dict.fromkeys(labels_ref)) if "unclassified" not in x.lower()]
    masks_ref = [[True if i == label else False for i in labels_ref] for label in names_ref]
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
