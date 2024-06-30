"""
This is a script for the backend of various AAclust methods.
"""
import pandas as pd
import numpy as np

import aaanalysis.utils as ut
from ._utils_aaclust import _compute_centers, _compute_medoids


# I Helper function
# Name clusters
def _get_cluster_names(list_names=None, name_medoid=None,
                       name_unclassified="Unclassified",
                       shorten_names=True):
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
        if shorten_names:
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


# Compute correlation
def _sort_X_labels_names(X, labels=None, names=None):
    """Sort labels"""
    sorted_order = np.argsort(labels)
    labels = np.array([labels[i] for i in sorted_order])
    X = X[sorted_order]
    if names:
        names = [names[i] for i in sorted_order]
    return X, labels, names


def _get_df_corr(X=None, X_ref=None):
    """Get df with correlations"""
    # Temporary labels to avoid any confusion with potential duplicates
    X_labels = range(len(X))
    X_ref_labels = range(len(X), len(X) + len(X_ref))
    combined = np.vstack((X, X_ref))
    df_corr_full = pd.DataFrame(combined.T).corr()
    # Select only the rows corresponding to X and columns corresponding to X_ref
    df_corr = df_corr_full.loc[X_labels, X_ref_labels]
    return df_corr


# II Main Functions
def compute_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    # Function in utilis for not breaking dependency rules:
    # Backend functions should only depend on backend utility functions
    return _compute_centers(X, labels=labels)


def compute_medoids(X, labels=None, metric="correlation"):
    """Obtain cluster medoids and their labels"""
    if metric is None:
        metric = "correlation"
    # Function in utilis for not breaking dependency rules:
    # Backend functions should only depend on backend utility functions
    return _compute_medoids(X, labels=labels, metric=metric)


def name_clusters(X, labels=None, names=None, shorten_names=True):
    """Create ordered list of cluster names"""
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
                                           name_unclassified=ut.STR_UNCLASSIFIED,
                                           shorten_names=shorten_names)
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


def compute_correlation(X, X_ref=None, labels=None, labels_ref=None, names=None, names_ref=None):
    """Computes Pearson correlation of given data with reference data."""
    # Sort based on labels
    X, labels_sorted, names_sorted = _sort_X_labels_names(X, labels=labels, names=names)
    if X_ref is not None:
        X_ref, labels_ref, names_ref = _sort_X_labels_names(X_ref, labels=labels_ref, names=names_ref)
    # Compute correlations
    if X_ref is None:
        df_corr = pd.DataFrame(X.T).corr()
    else:
        df_corr = _get_df_corr(X=X, X_ref=X_ref)
    # Replace indexes and columns with names or labels
    df_corr.index = names_sorted if names else labels_sorted
    if X_ref is None:
        df_corr.columns = names_sorted if names else labels_sorted
    else:
        df_corr.columns = names_ref if names_ref else labels_ref
    return df_corr, labels_sorted
