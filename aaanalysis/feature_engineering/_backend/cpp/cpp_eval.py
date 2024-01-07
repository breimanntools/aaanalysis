"""
This is a script for the backend of the CPP.eval method.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

import aaanalysis.utils as ut
from .utils_feature import get_feature_matrix_


# Helper function
def get_min_cor(X, labels=None):
    """Compute minimum pair-wise correlation for each cluster label and return minimum of obtained cluster minimums."""
    # Minimum correlations for each cluster (with center or all scales)
    list_masks = [[i == label for i in labels] for label in set(labels)]
    f = lambda x: np.corrcoef(x).min()
    list_min_cor = [f(X[mask]) for mask in list_masks]
    # Minimum for all clusters
    min_cor = min(list_min_cor)
    return min_cor


def get_best_n_clusters(X=None, min_th=0.3):
    """Obtain the best number of clusters based on internal cluster minimum correlation."""
    random_state = ut.options["random_state"]
    n_features, _ = X.shape
    max_n = min([100, n_features-1])
    best_n_clusters = max_n
    for n_clusters in range(2, max_n):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = kmeans.fit_predict(X)
            min_cor = get_min_cor(X, labels=labels)
        # Check if the minimum correlation in all clusters is at least 0.3
        if min_cor >= min_th:
            best_n_clusters = n_clusters
            break
    return best_n_clusters


def get_features_per_cluster_stats(X=None, n_clusters=2):
    """Calculate the average number of features per cluster and its standard deviation."""
    random_state = ut.options["random_state"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(X)
    # Count features in each cluster
    feat_per_cluster = [np.sum(labels == i) for i in range(n_clusters)]
    # Calculate average and standard deviation
    avg_feat_per_cluster = round(np.mean(feat_per_cluster), 2)
    std_feat_per_cluster = round(np.std(feat_per_cluster), 2)
    return avg_feat_per_cluster, std_feat_per_cluster


# II Main function
@ut.catch_runtime_warnings()
def evaluate_features(list_df_feat=None, names_feature_sets=None,
                      list_df_parts=None, df_scales=None, accept_gaps=False,
                      n_jobs=1, min_th=0.0):
    """Evaluate sets of clustering labels"""
    list_evals = []
    for df_feat, df_parts in zip(list_df_feat, list_df_parts):
        eval_dict = {}
        # Get number features per of category
        eval_dict[ut.COL_N_FEAT] = (len(df_feat), [len(df_feat[df_feat[ut.COL_CAT] == cat]) for cat in ut.COLS_CAT])
        # Compute CPP statistics
        eval_dict[ut.COL_AVG_ABS_AUC] = df_feat[ut.COL_ABS_AUC].mean()
        eval_dict[ut.COL_MAX_ABS_AUC] = df_feat[ut.COL_ABS_AUC].max()
        eval_dict[ut.COL_AVG_MEAN_DIF] = (round(df_feat[df_feat[ut.COL_MEAN_DIF] > 0][ut.COL_MEAN_DIF].mean(), 3),
                                          round(df_feat[df_feat[ut.COL_MEAN_DIF] < 0][ut.COL_MEAN_DIF].mean(), 3))
        eval_dict[ut.COL_AVG_STD_TEST] = df_feat[ut.COL_STD_TEST].mean()
        # Obtain optimal number of clusters
        features = list(df_feat[ut.COL_FEATURE])
        X = get_feature_matrix_(features=features, df_parts=df_parts, df_scales=df_scales,
                                accept_gaps=accept_gaps, n_jobs=n_jobs)
        # Get n_clusters for features
        best_n_clust = get_best_n_clusters(X=X.T, min_th=min_th)
        eval_dict[ut.COL_N_CLUST] = best_n_clust
        avg_feat_per_cluster, std_feat_per_cluster = get_features_per_cluster_stats(X=X.T,
                                                                                    n_clusters=best_n_clust)
        eval_dict[ut.COL_AVG_N_FEAT_PER_CLUST] = avg_feat_per_cluster
        eval_dict[ut.COL_STD_N_FEAT_PER_CLUST] = std_feat_per_cluster
        # Append this dictionary to the list_evals
        list_evals.append(eval_dict)
    # Create the DataFrame
    df_eval = pd.DataFrame(list_evals).round(3)
    df_eval = ut.add_names_to_df_eval(df_eval=df_eval, names=names_feature_sets)
    return df_eval
