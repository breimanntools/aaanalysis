"""
This is a script for the backend of the AAclust.fit() method.

The fit functions performs the AAclust algorithm consisting of three steps:

1. Estimate lower bound for n_clusters
2. Optimization of n_clusters
3. Merge clusters

"""
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import pairwise_distances

import aaanalysis.utils as ut
from ._utils_aaclust import _cluster_center

# I Helper Functions
def min_cor_center(X):
    """Get minimum for correlation of all columns with cluster center, defined as the mean values
    for each amino acid over all scales."""
    # Create new matrix including cluster center
    center_X = np.concatenate([_cluster_center(X), X], axis=0)
    # Get minimum correlation with mean values
    min_cor = np.corrcoef(center_X)[0, ].min()
    return min_cor


def min_cor_all(X):
    """Get minimum for pair-wise correlation of all columns in given matrix."""
    # Get minimum correlations minimum/ maximum distance for pair-wise comparisons
    min_cor = np.corrcoef(X).min()
    return min_cor


def get_min_cor(X, labels=None, on_center=True):
    """Compute minimum pair-wise correlation or correlation with cluster center for each cluster label
    and return minimum of obtained cluster minimums."""
    f = min_cor_center if on_center else min_cor_all
    if labels is None:
        return f(X)
    # Minimum correlations for each cluster (with center or all scales)
    unique_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in unique_labels]
    list_min_cor = [f(X[mask]) for mask in list_masks]
    # Minimum for all clusters
    min_cor = min(list_min_cor)
    return min_cor


def get_max_dist(X, on_center=True, metric="euclidean"):
    """Get maximum distance on center or all scales"""
    # Maximum distance for cluster
    if on_center:
        # Create new matrix including cluster center
        center_X = np.concatenate([_cluster_center(X), X], axis=0)
        # Get maximum distance with mean values
        max_dist = pairwise_distances(center_X, metric=metric)[0, ].max()
    else:
        # Get maximum distance for pair-wise comparisons
        max_dist = pairwise_distances(X, metric=metric).max()
    return max_dist


# II Main Functions
# 1. Step (Estimation of n clusters)
@ut.catch_convergence_warning()
def _estimate_lower_bound_n_clusters(X, model=None, model_kwargs=None, min_th=0.3, on_center=True):
    """
    Estimate the lower bound of the number of clusters (k).

    This function estimates the lower bound of the number of clusters by testing a range
    between 10% and 90% of all observations, incrementing in 10% steps.
    """
    f = lambda c: get_min_cor(X, labels=model(n_clusters=c, **model_kwargs).fit(X).labels_, on_center=on_center)
    # Create range between 10% and 90% of all scales (10% steps) as long as minimum correlation is lower than threshold
    n_samples, n_features = X.shape
    nclust_mincor = [(1, f(1))]
    step_number = 40
    for i in range(1, step_number, 1):
        n_clusters = max(1, int(n_samples*i/step_number))    # n cluster in 2.5% steps
        min_cor = f(n_clusters)
        if min_cor < min_th:   # Save only lower bounds
            nclust_mincor.append((n_clusters, min_cor))
        else:
            break
    # Select second highest lower bound (highest lower bound is faster but might surpass true bound)
    nclust_mincor.sort(key=lambda x: x[0], reverse=True)
    n_clusters = nclust_mincor[1][0] if len(nclust_mincor) > 1 else nclust_mincor[0][0]  # Otherwise, only existing one
    return n_clusters


@ut.catch_runtime_warnings()
def estimate_lower_bound_n_clusters(X, model=None, model_kwargs=None, min_th=0.6, on_center=True):
    """Wrapper for _estimate_lower_bound_n_clusters to catch convergence warnings"""
    try:
        n_clusters = _estimate_lower_bound_n_clusters(X, model=model, model_kwargs=model_kwargs,
                                                      min_th=min_th, on_center=on_center)
    except ut.ClusteringConvergenceException as e:
        n_clusters = e.distinct_clusters
    return n_clusters


# 2. Step (Optimization of n clusters)
@ut.catch_convergence_warning()
def _optimize_n_clusters(X, model=None, model_kwargs=None, n_clusters=None, min_th=0.3, on_center=True):
    """
    Optimize the number of clusters using a recursive algorithm.

    This function performs clustering in a recursive manner (through a while loop) to ensure
    that the minimum within-cluster correlation is achieved for all clusters. It is an efficiency
    optimized version of a step-wise algorithm where the `n_clusters` is incrementally increased
    until a stop condition is met.
    """
    n_samples, n_features = X.shape
    f = lambda c: get_min_cor(X, labels=model(n_clusters=c, **model_kwargs).fit(X).labels_, on_center=on_center)
    min_cor = f(n_clusters)
    # Recursive optimization of n_clusters via step-wise increase starting from lower bound
    step = max(1, min(int(n_samples/10), 5))    # Step size between 1 and 5
    while min_cor < min_th and n_clusters < n_samples:    # Stop condition of clustering
        n_clusters = min(n_clusters+step, n_samples) # Maximum of n_samples is allowed
        min_cor = f(n_clusters)
        # Exceeding of threshold -> Conservative adjustment of clustering parameters to meet true optimum
        if min_cor >= min_th and step != 1:
            n_clusters = max(1, n_clusters - step * 2)
            step = 1
            min_cor = f(n_clusters)
    return n_clusters


@ut.catch_runtime_warnings()
def optimize_n_clusters(X, model=None, model_kwargs=None, n_clusters=None, min_th=0.5, on_center=True):
    """Wrapper for _optimize_n_clusters to catch convergence warnings"""
    try:
        n_clusters = _optimize_n_clusters(X, model=model, model_kwargs=model_kwargs,
                                          n_clusters=n_clusters, min_th=min_th, on_center=on_center)
    except ut.ClusteringConvergenceException as e:
        n_clusters = e.distinct_clusters
    return n_clusters


# 3. Step (Merging)
def _get_min_cor_cluster(X, labels=None, label_cluster=None, on_center=True):
    """Get min_cor for single cluster"""
    mask = [l == label_cluster for l in labels]
    min_cor = get_min_cor(X[mask], on_center=on_center)
    return min_cor


def _get_quality_measure(X, metric=None, labels=None, label_cluster=None, on_center=True):
    """Get quality measure single cluster given by feature matrix X, labels, and label of cluster"""
    mask = [l == label_cluster for l in labels]
    if metric == ut.METRIC_CORRELATION:
        return get_min_cor(X[mask], on_center=on_center)
    else:
        return get_max_dist(X[mask], on_center=on_center, metric=metric)


def _get_best_cluster(dict_clust_qm=None, metric=None):
    """Get cluster with the best quality measure: either highest minimum Pearson correlation
    or lowest distance measure"""
    if metric == ut.METRIC_CORRELATION:
        return max(dict_clust_qm, key=dict_clust_qm.get)
    else:
        return min(dict_clust_qm, key=dict_clust_qm.get)


def merge_clusters(X, n_max=5, labels=None, min_th=0.5, on_center=True, metric="correlation"):
    """
    Merge small clusters into other clusters optimizing a given quality measure.

    This function merges clusters with sizes less than or equal to `n_max` into other clusters
    based on a specified quality measure (Pearson correlation or a distance metric).
    Merging is conducted only if the new assignment meets a minimum within-cluster Pearson
    correlation threshold defined by `min_th`.
    """
    unique_labels = list(OrderedDict.fromkeys(labels))
    for n in range(1, n_max):
        s_clusters = [x for x in unique_labels if labels.count(x) == n]   # Smallest clusters
        b_clusters = [x for x in unique_labels if labels.count(x) > n]    # Bigger clusters (all others)
        # Assign scales from smaller clusters to cluster by optimizing for quality measure
        for s_clust in s_clusters:
            dict_clust_qm = {}  # Cluster to quality measure
            for b_clust in b_clusters:
                labels_ = [x if x != s_clust else b_clust for x in labels]
                args = dict(labels=labels_, label_cluster=b_clust, on_center=on_center)
                min_cor = _get_min_cor_cluster(X, **args)
                if min_cor >= min_th:
                    dict_clust_qm[b_clust] = _get_quality_measure(X, **args, metric=metric)
            if len(dict_clust_qm) > 0:
                b_clust_best = _get_best_cluster(dict_clust_qm=dict_clust_qm, metric=metric)
                labels = [x if x != s_clust else b_clust_best for x in labels]
    # Update labels (cluster labels are given in descending order of cluster size)
    sorted_labels = pd.Series(labels).value_counts().index  # sorted in descending order of size
    dict_update = {label: i for label, i in zip(sorted_labels, range(0, len(set(labels))))}
    labels = [dict_update[label] for label in labels]
    return labels

