"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

import aaanalysis as aa
from aaanalysis.aaclust.aaclust import get_min_cor, estimate_lower_bound_n_clusters, \
    optimize_n_clusters, merge_clusters, AAclust

import tests._utils as ut

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe

# TODO change to proper test (CPP)


# I Helper Functions
def get_feat_matrix(df_cat=None, df_scales=None, unclassified_in=True, return_col=ut.COL_SCALE_ID, cat=None):
    """"""
    if cat is not None:
        df_cat = df_cat[df_cat[ut.COL_CAT] == cat]
    if unclassified_in:
        scales = df_cat[return_col].to_list()
    else:
        mask = (~df_cat[ut.COL_SUBCAT].str.contains("Unclassified")) & (df_cat[ut.COL_CAT] != "Others")
        df_cat = df_cat[mask]
        scales = df_cat[ut.COL_SCALE_ID].to_list()
    X = np.array(df_scales[scales]).T
    labels = list(df_cat[return_col])
    return X, labels


def get_data():
    """"""
    df_cat = aa.load_scales(name="scale_classification")
    df_scales = aa.load_scales(name="scales")
    X, scales = get_feat_matrix(df_cat=df_cat.copy(),
                                df_scales=df_scales.copy(),
                                unclassified_in=True)
    return X


def get_model():
    """"""
    model_kwargs=dict()
    model = AgglomerativeClustering
    return model, model_kwargs


# II Main Functions
def test_steps():
    """"""
    X = get_data()
    model, model_kwargs = get_model()
    args = dict(X=X, model=model, model_kwargs=model_kwargs, min_th=0.3, on_center=False)
    k = estimate_lower_bound_n_clusters(**args)
    k = optimize_n_clusters(**args, n_clusters=k)
    labels = model(n_clusters=k, **model_kwargs).fit(X).labels_.tolist()
    print(len(set(labels)))
    labels_ = merge_clusters(X, labels=labels, min_th=0.3, on_center=False)
    print(len(set(labels_)))
    print(get_min_cor(X, labels=labels_, on_center=False))


def test_aaclust():
    """"""
    X = get_data()
    model, model_kwargs = get_model()
    aac = AAclust(model=model, model_kwargs=model_kwargs)
    args = dict(on_center=False, min_th=0.3, merge=True, merge_metric="euclidean")
    aac.fit(X,  **args)


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    test_steps()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
