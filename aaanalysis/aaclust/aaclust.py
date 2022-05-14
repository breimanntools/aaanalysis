"""
This is a script for the AAclust clustering wrapper method.
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

import aaanalysis.aaclust._utils as ut


# I Helper Functions
# Obtain centroids and medoids
def cluster_center(X):
    """Compute cluster center (i.e., arithmetical mean over all data points/observations of a cluster)"""
    return X.mean(axis=0)[np.newaxis, :]


def get_cluster_centers(X, labels=None):
    """Obtain cluster centers and their labels"""
    center_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[True if i == label else False for i in labels] for label in center_labels]
    centers = np.concatenate([cluster_center(X[mask]) for mask in list_masks]).round(3)
    return centers, center_labels


def _cluster_medoid(X):
    """Obtain cluster medoids (i.e., scale closest to cluster center used as representative scale for a cluster)"""
    # Create new array with cluster center and given
    center_X = np.concatenate([cluster_center(X), X], axis=0)
    # Get index for scale with highest correlation with cluster center
    ind_max = np.corrcoef(center_X)[0, 1:].argmax()
    return ind_max


def get_cluster_medoids(X, labels=None):
    """Obtain cluster medoids and their labels"""
    unique_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[True if i == label else False for i in labels] for label in unique_labels]
    list_ind_max = [_cluster_medoid(X[mask]) for mask in list_masks]
    indices = np.array(range(0, len(labels)))
    medoid_ind = [indices[m][i] for m, i in zip(list_masks, list_ind_max)]
    medoid_labels = [labels[i] for i in medoid_ind]
    medoids = np.array([X[i, :] for i in medoid_ind])
    return medoids, medoid_labels, medoid_ind


# Compute minimum correlation on center or all scales
def _min_cor_center(X):
    """Get minimum for correlation of all columns with cluster center, defined as the mean values
    for each amino acid over all scales."""
    # Create new matrix including cluster center
    center_X = np.concatenate([cluster_center(X), X], axis=0)
    # Get minimum correlation with mean values
    min_cor = np.corrcoef(center_X)[0, ].min()
    return min_cor


def _min_cor_all(X):
    """Get minimum for pair-wise correlation of all columns in given matrix."""
    # Get minimum correlations minimum/ maximum distance for pair-wise comparisons
    min_cor = np.corrcoef(X).min()
    return min_cor


def get_min_cor(X, labels=None, on_center=True):
    """Compute minimum pair-wise correlation or correlation with cluster center for each cluster label
    and return minimum of obtained cluster minimums."""
    f = _min_cor_center if on_center else _min_cor_all
    if labels is None:
        return f(X)
    # Minimum correlations for each cluster (with center or all scales)
    unique_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[True if i == label else False for i in labels] for label in unique_labels]
    list_min_cor = [f(X[mask]) for mask in list_masks]
    # Minimum for all clusters
    min_cor = min(list_min_cor)
    return min_cor


# Get maximum distance on center or all scales
def get_max_dist(X, on_center=True, metric="euclidean"):
    """"""
    # Maximum distance for cluster
    if on_center:
        # Create new matrix including cluster center
        center_X = np.concatenate([cluster_center(X), X], axis=0)
        # Get maximum distance with mean values
        max_dist = pairwise_distances(center_X, metric=metric)[0, ].max()
    else:
        # Get maximum distance for pair-wise comparisons
        max_dist = pairwise_distances(X, metric=metric).max()
    return max_dist


# II Main Functions
# AAclust algorithm steps (estimate lower bound for n_clusters -> optimization of n_clusters -> merge clusters)
# 1. Step (Estimation of n clusters)
def estimate_lower_bound_n_clusters(X, model=None, model_kwargs=None, min_th=0.6, on_center=True):
    """Estimation of lower bound of n_clusters (k) via testing range between 10% and 90%
        of all observations in in 10% steps.
    :arg X: feature matrix
    :arg model: k-based clustering model
    :arg model_kwargs: keyword arguments of model
    :arg min_th: minimum threshold of within-cluster Pearson correlation
    :arg on_center: whether minimum correlation is computed for all observations within a cluster
        or for the cluster center
    :returns n_clusters: estimated number of clusters (k)"""
    f = lambda c: get_min_cor(X, labels=model(n_clusters=c, **model_kwargs).fit(X).labels_, on_center=on_center)
    # Create range between 10% and 90% of all scales (10% steps) as long as minimum correlation is lower than threshold
    n_samples, n_features = X.shape
    nclust_mincor = [(1, f(1))]
    for i in range(1, 10, 1):
        n_clusters = max(1, int(n_samples*i/10))    # n cluster in 10% steps
        min_cor = f(n_clusters)
        if min_cor < min_th:   # Save only lower bounds
            nclust_mincor.append((n_clusters, min_cor))
        else:
            break
    # Select second highest lower bound (highest lower bound is faster but might surpass true bound)
    nclust_mincor.sort(key=lambda x: x[1], reverse=True)
    n_clusters = nclust_mincor[1][0] if len(nclust_mincor) > 1 else nclust_mincor[0][0]  # Otherwise, only existing one
    return n_clusters


# 2. Step (Optimization of n clusters)
def optimize_n_clusters(X, model=None, model_kwargs=None, n_clusters=None, min_th=0.5, on_center=True):
    """Optimization of number of clusters. Recursive clustering (via while loop) such that
    minimal within-cluster correlation is fulfilled for all clusters. This clustering algorithm
    is an efficiency optimized version of a step wise algorithm, where each round the n_clusters
     is increased until a stop condition is reached.
    :arg X: feature matrix
    :arg model: k-based clustering model
    :arg model_kwargs: keyword arguments of model
    :arg n_clusters: estimated number of clusters (k)
    :arg min_th: minimum threshold of within-cluster Pearson correlation
    :arg on_center: whether minimum correlation is computed for all observations within a cluster
        or for the cluster center
    :returns n_clusters: optimized number of clusters (k)"""
    n_samples, n_features = X.shape
    f = lambda c: get_min_cor(X, labels=model(n_clusters=c, **model_kwargs).fit(X).labels_, on_center=on_center)
    min_cor = f(n_clusters)
    # Recursive optimization of n_clusters via step wise increase starting from lower bound
    step = max(1, min(int(n_samples/10), 5))    # Step size between 1 and 5
    while min_cor < min_th and n_clusters < n_samples:    # Stop condition of clustering
        n_clusters = min(n_clusters+step, n_samples) # Maximum of of n_samples is allowed
        min_cor = f(n_clusters)
        # Exceeding of threshold -> Conservative adjustment of clustering parameters to meet true optimum
        if min_cor >= min_th and step != 1:
            n_clusters = max(1, n_clusters - step * 2)
            step = 1
            min_cor = f(n_clusters)
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
    """Get cluster with best quality measure: either highest minimum Pearson correlation
    or lowest distance measure"""
    if metric == ut.METRIC_CORRELATION:
        return max(dict_clust_qm, key=dict_clust_qm.get)
    else:
        return min(dict_clust_qm, key=dict_clust_qm.get)


def merge_clusters(X, n_max=5, labels=None, min_th=0.5, on_center=True, metric="correlation"):
    """Merge small clusters (defined by n_max) into other clusters by optimizing quality measure
    (either Pearson correlation or distance metric such as Euclidean distance) given by metric.
    Merging is performed only if result of new assignment fulfills given minimum within-cluster Pearson
    correlation threshold (min_th).
    :arg X: feature matrix
    :arg n_max: maximum cluster size for small clusters
    :arg labels: initial cluster labels for observations
    :arg min_th: minimum threshold of within-cluster Pearson correlation
    :arg on_center: whether minimum correlation is computed for all observations within a cluster
        or for the cluster center
    :arg metric: quality measure on which merging should be optimized (either maximum correlation or minimum distance)
    :returns labels: cluster labels for observations after merging"""
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


# AAclust naming
def get_names_cluster(list_names=None, name_medoid=None, name_unclassified="Unclassified"):
    """Get list of cluster names sorted based on following criteria (descending order):
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
    list_names = [remove_2nd_info(x) for x in list_names if "Unclassified" not in x]
    # Create list of shorter names not containing information given in parenthesis
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


class AAclust:
    """k-free wrapper for clustering algorithm to create redundancy reduced sets of numerical scales"""
    def __init__(self, model=None, model_kwargs=None, name_unclassified="unclassified", verbose=False):
        # Model parameters
        if model is None:
            model = KMeans
        self.model = model
        if model_kwargs is None:
            model_kwargs = dict()
        model_kwargs = ut.check_model(model=self.model, model_kwargs=model_kwargs)
        self._model_kwargs = model_kwargs
        # AAclust clustering settings
        self._name_unclassified = name_unclassified
        self._verbose = verbose
        # Output parameters (will be set during model fitting)
        self.n_clusters = None  # Number of by AAclust obtained clusters
        self.labels_ = None     # Cluster labels in order of samples in feature matrix
        self.centers_ = None    # Mean scales for each cluster
        self.center_labels_ = None
        self.medoids_ = None
        self.medoid_labels_ = None
        self.medoid_ind_ = None

    # Clustering method
    def fit(self, X, names=None, min_th=0.5, n_clusters=None, on_center=True, merge=True, merge_metric="correlation"):
        """Get hierarchy of cluster categories based on list of minimum correlation thresholds."""
        # Check input
        ut.check_min_th(min_th=min_th)
        merge_metric = ut.check_merge_metric(merge_metric=merge_metric)
        X, names = ut.check_feat_matrix(X=X, names=names)
        args = dict(model=self.model, model_kwargs=self._model_kwargs, min_th=min_th, on_center=on_center)
        # Clustering using given clustering models
        if n_clusters is not None:
            labels = self.model(n_clusters=n_clusters, **self._model_kwargs).fit(X).labels_.tolist()
        # Clustering using AAclust algorithm
        else:
            # Estimation of lower bound of number of clusters via testing range between 10% and 90% of all scales
            n_clusters_lb = estimate_lower_bound_n_clusters(X, **args)
            # Optimization of number of clusters by recursive clustering
            n_clusters = optimize_n_clusters(X, n_clusters=n_clusters_lb, **args)
            # Cluster merging: assign scales from small clusters to other cluster with highest minimum correlation
            labels = self.model(n_clusters=n_clusters, **self._model_kwargs).fit(X).labels_.tolist()
            if merge:
                labels = merge_clusters(X, labels=labels, min_th=min_th, on_center=on_center, metric=merge_metric)
        # Obtain cluster centers and medoids
        medoids, medoid_labels, medoid_ind = get_cluster_medoids(X, labels=labels)
        centers, center_labels = get_cluster_centers(X, labels=labels)
        # Save results
        self.n_clusters = len(set(labels))
        self.labels_ = np.array(labels)
        self.centers_ = centers
        self.center_labels_ = center_labels
        self.medoids_ = medoids # Representative scales
        self.medoid_labels_ = medoid_labels
        self.medoid_ind_ = medoid_ind   # Index of medoids
        # Return labels of medoid if y is given
        if names is not None:
            names_medoid = [names[i] for i in medoid_ind]
            return names_medoid

    def cluster_naming(self, names=None, labels=None, name_unclassified="Unclassified"):
        """Change cluster labels by scale names"""
        if type(names) is not list:
            raise ValueError("'names' must be list")
        if labels is None:
            labels = self.labels_
        dict_medoids = dict(zip(self.medoid_labels_, self.medoid_ind_))
        # Get cluster labels sorted in descending order of frequency
        labels_sorted = pd.Series(labels).value_counts().index
        # Assign names to cluster
        dict_cluster_names = {}
        for clust in labels_sorted:
            name_medoid = names[dict_medoids[clust]]
            list_names = [names[i] for i in range(0, len(names)) if labels[i] == clust]
            names_cluster = get_names_cluster(list_names=list_names,
                                              name_medoid=name_medoid,
                                              name_unclassified=name_unclassified)
            assigned = False
            for name in names_cluster:
                if name not in dict_cluster_names.values() or name == name_unclassified:
                    dict_cluster_names[clust] = name
                    assigned = True
                    break
            if not assigned:
                dict_cluster_names[clust] = name_unclassified
        cluster_names = [dict_cluster_names[label] for label in labels]
        return cluster_names

    # TODO check if necessary
    @staticmethod
    def get_cluster_centers(X, labels=None):
        """"""
        centers, center_labels = get_cluster_centers(X, labels=labels)
        return centers, center_labels

    # TODO check if necessary
    @staticmethod
    def get_cluster_medoids(X, labels=None):
        """"""
        medoids, medoid_labels, medoid_ind = get_cluster_medoids(X, labels=labels)
        return medoids, medoid_labels, medoid_ind

    @staticmethod
    def correlation(X_test, X_ref, labels_test=None, labels_ref=None, n=3, positive=True,
                    on_center=False, except_unclassified=True):
        """Compute cluster centers from reference list and correlation of test array with centers"""
        # Check input
        X_test, labels_test = ut.check_feat_matrix(X=X_test, names=labels_test)
        X_ref, labels_ref = ut.check_feat_matrix(X=X_ref, names=labels_ref)
        if except_unclassified:
            names_ref = list(dict.fromkeys(labels_ref))
        else:
            names_ref = [x for x in list(dict.fromkeys(labels_ref)) if "unclassified" not in x.lower()]
        masks_ref = [[True if i == label else False for i in labels_ref] for label in names_ref]
        if on_center:
            # Get centers for all clusters in reference data
            centers = np.concatenate([cluster_center(X_ref[mask]) for mask in masks_ref], axis=0)
            # Compute correlation of test data with centers
            Xtest_centers = np.concatenate([X_test, centers], axis=0)
            n_test = X_test.shape[0]
            X_corr = np.corrcoef(Xtest_centers)[:n_test, n_test:]
        else:
            masks_test = [[True if i == j else False for j in range(0, len(labels_test))]
                          for i, _ in enumerate(labels_test)]
            # Compute minimum correlation of test data with each group of reference data
            X_corr = np.array([[_min_cor_all(np.concatenate([X_test[mask_test], X_ref[mask_ref]], axis=0))
                                for mask_ref in masks_ref ] for mask_test in masks_test])
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
