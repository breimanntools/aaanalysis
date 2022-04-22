"""
This is a script for benchmarking AAclust clustering against clustering models not using
    number of clusters k as parameter.
"""
import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS, Birch, \
    KMeans,AgglomerativeClustering, MiniBatchKMeans, SpectralClustering

import scripts._utils as ut
from aaclust import AAclust

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions
def merge_noise_to_cluster(labels=None):
    """Merge noise to one cluster"""
    k = len(set(labels)) - 1
    for i, j in enumerate(labels):
        if j == -1:
            labels[i] = k
    return labels

def get_eval(X, labels=None):
    """Get evaluation for clustering result"""
    mask = [x!=-1 for x in labels]
    X_ = X[mask]
    labels = labels[mask]
    set_labels = set(labels)
    n_clust = len(set_labels)
    centers = [X_[[i == j for j in labels]].mean(axis=0) for i in set_labels]
    bic, ss, ch = ut.cluster_evaluation(X_, labels=labels, n_clust=n_clust, centers=centers, centers_labels=set_labels)
    return bic, ss, ch, len(labels), len(set_labels)

# II Main Functions
def benchmark_aaclust():
    """Initial benchmarking to compare AAclust against clustering models without k as parameter"""
    df_cat = pd.read_excel(ut.FOLDER_DATA + "scale_classification.xlsx")
    df_scales = pd.read_excel(ut.FOLDER_DATA + "scales.xlsx", index_col=0)
    X, scales = ut.get_feat_matrix(df_cat=df_cat.copy(),
                                   df_scales=df_scales.copy(),
                                   unclassified_in=True)
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
        df_eval = pd.DataFrame(list_results, columns=["AAclust", "name", "BIC", "SS", "CH", "n_scales", "n_clusters", "run_time"]).round(6)
        df_eval[ut.COL_RANK] = ut.score_ranking(df=df_eval, cols_scores=ut.LIST_SCORES)
        df_eval.to_excel(ut.FOLDER_RESULTS + ut.FILE_BENCHMARKING)

# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    benchmark_aaclust()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
