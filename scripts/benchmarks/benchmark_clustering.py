"""
This is a script for benchmarking AAclust clustering against clustering models not using
    number of clusters k as parameter.
"""
import time
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance

from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS, Birch, \
    KMeans,AgglomerativeClustering, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from aaclust import AAclust
import scripts._utils as ut


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe

# General Columns and strings
COL_SCALE_ID = "scale_id"
COL_CAT = "category"
COL_SUBCAT = "subcategory"

# I Helper Functions
# Feature matrix
def get_feat_matrix(df_cat=None, df_scales=None, return_col=COL_SCALE_ID, cat=None):
    """"""
    if cat is not None:
        df_cat = df_cat[df_cat[COL_CAT] == cat]
    scales = df_cat[return_col].to_list()
    X = np.array(df_scales[scales]).T
    labels = list(df_cat[return_col])
    return X, labels

# Data processing
def merge_noise_to_cluster(labels=None):
    """Merge noise to one cluster"""
    k = len(set(labels)) - 1
    for i, j in enumerate(labels):
        if j == -1:
            labels[i] = k
    return labels


# Clustering evaluation
def _compute_bic(X, labels=None, centers=None, center_labels=None, n_clust=None):
    """Computes the BIC metric for given clusters

    See also
    --------
    https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    """
    # Check if labels match to number of clusters
    if len(set(labels)) != n_clust:
        return np.NaN
    size_clusters = np.bincount(labels)
    n_samples, n_features = X.shape
    # Compute variance over all clusters beforehand

    if center_labels is None:
        center_labels = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in center_labels]
    sum_squared_dist = sum([sum(distance.cdist(X[list_masks[i]], [centers[i]], 'euclidean')**2) for i in range(n_clust)])
    cl_var = (1.0 / (n_samples - n_clust) / n_features) * sum_squared_dist
    const_term = 0.5 * n_clust * np.log(n_samples) * (n_features + 1)
    # Compute BIC
    bic = np.sum([size_clusters[i] * np.log(size_clusters[i]) -
                  size_clusters[i] * np.log(n_samples) -
                  ((size_clusters[i] * n_features) / 2) * np.log(2*np.pi*cl_var) -
                  ((size_clusters[i] - 1) * n_features / 2)
                  for i in range(n_clust)]) - const_term
    return bic

def _cluster_evaluation(X, labels=None, n_clust=None, centers=None, centers_labels=None):
    """Compute BIC, silhouette score, and Calinski Harabasz score to evaluate clustering"""
    # Compute performance metrics
    bic = _compute_bic(X, labels=labels, n_clust=n_clust, centers=centers, center_labels=centers_labels)
    ss = silhouette_score(X, labels=labels)
    ch = calinski_harabasz_score(X, labels=labels)
    return bic, ss, ch


def get_eval(X, labels=None):
    """Get evaluation for clustering result"""
    mask = [x!=-1 for x in labels]
    X_ = X[mask]
    labels = labels[mask]
    set_labels = set(labels)
    n_clust = len(set_labels)
    centers = [X_[[i == j for j in labels]].mean(axis=0) for i in set_labels]
    bic, ss, ch = _cluster_evaluation(X_, labels=labels, n_clust=n_clust, centers=centers, centers_labels=set_labels)
    return bic, ss, ch, len(labels), len(set_labels)


def _score_ranking(df=None, cols_scores=None):
    """Obtain average ranking for given list of scores"""
    mean_rank = df[cols_scores].round(5).rank(ascending=False).mean(axis=1).rank(method="dense")
    return mean_rank


# II Main Functions
# Benchmark aaanalysis classification
def benchmark_aaclust_clustering():
    """Initial benchmarking to compare AAclust against clustering models without k as parameter"""
    df_cat = pd.read_excel(ut.FOLDER_DATA + "scale_classification.xlsx")
    df_scales = pd.read_excel(ut.FOLDER_DATA + "scales.xlsx", index_col=0)
    X, scales = get_feat_matrix(df_cat=df_cat.copy(),
                                df_scales=df_scales.copy())
    list_results = []
    # Benchmark clustering
    list_pred = ["Birch", "AffinityProp"]
    list_models = [(DBSCAN, dict(min_samples=2), "DBSCAN"),
                   (OPTICS, dict(min_samples=2), "OPTICS"),
                   (Birch, dict(n_clusters=None), "Birch"),
                   (AffinityPropagation, dict(), "AffinityProp")]
    for model, params, name in list_models:
        m = model(**params).fit(X)
        t0 = time.time()
        labels = m.predict(X) if name in list_pred else m.labels_
        t1 = time.time() - t0
        list_results.append([False, name, *get_eval(X, labels=labels), t1])
        if -1 in labels:
            labels = merge_noise_to_cluster(labels=labels)
            name += "_with_noise"
            list_results.append([False, name, *get_eval(X, labels=labels), t1])

    # AAclust clustering
    list_models = [(AgglomerativeClustering, dict(linkage="ward"), "Agglomerative_ward"),
                   (AgglomerativeClustering, dict(linkage="average"), "Agglomerative_average"),
                   (KMeans, dict(random_state=42), "KMeans"),
                   (MiniBatchKMeans, dict(random_state=42), "MiniBatchKMeans"),
                   (SpectralClustering, dict(), "Spectral")]

    for model, params, name in list_models:
        aac = AAclust(model=model, model_kwargs=params)
        for on_center in [True, False]:
            for merge in ["correlation", "euclidean", False]:
                merge_metric = merge if merge else None
                args = dict(on_center=on_center, min_th=0.3, merge=merge, merge_metric=merge_metric)
                t0 = time.time()
                aac.fit(X,  **args)
                labels = aac.labels_
                t1 = time.time() - t0
                name_ = name
                if on_center:
                    name_ += "+center"
                if merge:
                    name_ += f"+merge({merge_metric})"
                list_results.append([True, name_, *get_eval(X, labels=labels), t1])
        df_bench = pd.DataFrame(list_results, columns=["AAclust", "name", "BIC", "SS", "CH", "n_scales", "n_clusters", "run_time"]).round(6)
        df_bench["rank"] = _score_ranking(df=df_bench, cols_scores=["BIC", "SS", "CH"])
        file_out = "AAclust_benchmark_clustering.xlsx"
        df_bench.to_excel(ut.FOLDER_RESULTS + file_out, index=False)


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    benchmark_aaclust_clustering()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
