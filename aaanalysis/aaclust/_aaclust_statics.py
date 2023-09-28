"""
This is a script for the static methods of the AAclust class.
"""
from collections import OrderedDict
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
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

def compute_corr(X, X_ref, labels=None, labels_ref=None, n=3, positive=True, on_center=False):
    """Computes Pearson correlation of given data with reference data."""
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

# Obtain cluster names
def _get_cluster_names(list_names=None, name_medoid=None, name_unclassified="Unclassified"):
    """
    Get list of cluster names sorted based on following criteria (descending order):
        a) Frequency of term (most frequent term is preferred)
        b) Term is the name or a sub-name of the given medoid
        c) Length of term (shorter terms are preferred)
    If cluster consists of only one term, the name will be 'unclassified ('category name')'
    """
    def remove_2nd_info(name_):
        """Remove information given behind comma"""
        if "," in name_:
            name_ = name_.split(",")[0]
            if "(" in name_:
                name_ += ")"  # Close parenthesis if interpreted by deletion
        return name_
    # Filter categories (Remove unclassified scales and secondary infos)
    list_names = [remove_2nd_info(x) for x in list_names if ut.STR_UNCLASSIFIED not in x]
    # Create list of shorter names not containing information given in parentheses
    list_short_names = [x.split(" (")[0] for x in list_names if " (" in x]
    if len(list_names) > 1:
        list_names.extend(list_short_names)
        # Obtain information to check criteria for sorting scale names
        df_counts = pd.Series(list_names).value_counts().to_frame().reset_index()   # Compute frequencies of names
        df_counts.columns = ["name", "count"]
        df_counts["medoid"] = [True if x in name_medoid else False for x in df_counts["name"]]  # Name in medoid
        df_counts["length"] = [len(x) for x in df_counts["name"]]      # Length of name
        # Sort names based on given criteria
        df_counts = df_counts.sort_values(by=["count", "medoid", "length"], ascending=[False, False, True])
        names_cluster = df_counts["name"].tolist()
    else:
        names_cluster = [name_unclassified]
    return names_cluster

def name_clusters(X, labels=None, names=None):
    """"""
    medoids, medoid_labels, medoid_ind = compute_medoids(X, labels=labels)
    dict_medoids = dict(zip(medoid_labels, medoid_ind))
    # Get cluster labels sorted in descending order of frequency
    labels_sorted = pd.Series(labels).value_counts().index
    # Assign names to cluster
    dict_cluster_names = {}
    for clust in labels_sorted:
        name_medoid = names[dict_medoids[clust]]
        list_names = [names[i] for i in range(0, len(names)) if labels[i] == clust]
        names_cluster = _get_cluster_names(list_names=list_names, name_medoid=name_medoid,
                                           name_unclassified=ut.STR_UNCLASSIFIED)
        assigned = False
        for name in names_cluster:
            if name not in dict_cluster_names.values() or name == ut.STR_UNCLASSIFIED:
                dict_cluster_names[clust] = name
                assigned = True
                break
        if not assigned:
            dict_cluster_names[clust] = ut.STR_UNCLASSIFIED
    cluster_names = [dict_cluster_names[label] for label in labels]
    return cluster_names