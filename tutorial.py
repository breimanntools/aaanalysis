"""
This is an example script for AAclust
"""
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import aaanalysis as aa

pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


def tutorial_for_aaclust_and_cpp():
    # Load scale data
    df_scales = aa.load_scales()
    df_cat = aa.load_scales(name="scale_categories")

    # Select scales using AAclust
    aac = aa.AAclust(model=AgglomerativeClustering, model_kwargs=dict(linkage="ward"))
    X = np.array(df_scales).T
    scales = aac.fit(X, n_clusters=100, names=list(df_scales))   # Number of clusters = number of selected scales
    df_cat = df_cat[df_cat["scale_id"].isin(scales)]
    df_scales = df_scales[scales]
    # Load training data
    df_info = aa.load_dataset()
    print(df_info)
    df = aa.load_dataset(name="DISULFIDE_SEQ", min_len=200)

    # Feature Engineering
    y = list(df["label"])
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df, jmd_n_len=0, jmd_c_len=0, list_parts=["tmd"], ext_len=0)
    split_kws = sf.get_split_kws(n_split_max=1, split_types=["Segment"])
    args = dict(df_scales=df_scales, df_parts=df_parts, accept_gaps=True)
    cpp = aa.CPP(df_cat=df_cat, **args, split_kws=split_kws)
    df_feat = cpp.run(labels=y, tmd_len=200, n_processes=1)
    print(df_feat)
    X = sf.feat_matrix(**args, features=df_feat["feature"])
    # ML
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, scoring="accuracy", cv=5, n_jobs=8)
    print(np.mean(cv))


if __name__ == '__main__':
    tutorial_for_aaclust_and_cpp()
