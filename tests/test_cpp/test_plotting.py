"""
This is a script for ...
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import aaanalysis as aa

import tests._utils as ut

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
def test_cpp():

    # Load training data
    df_info = aa.load_dataset()
    df = aa.load_dataset(name="SEQ_DISULFIDE", min_len=300, n=100)
    print(df)
    # Load scales and scale categories from AAanalysis
    df_scales = aa.load_scales()
    df_cat = aa.load_scales(name="scale_classification")
    # Select scales using AAclust
    aac = aa.AAclust(model=AgglomerativeClustering, model_kwargs=dict(linkage="ward"))
    X = np.array(df_scales).T
    scales = aac.fit(X, n_clusters=10, names=list(df_scales))   # Number of clusters = number of selected scales (100 is recommended)
    df_cat = df_cat[df_cat["scale_id"].isin(scales)]
    df_scales = df_scales[scales]

    # Feature Engineering
    y = list(df["label"])
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df, jmd_n_len=50, jmd_c_len=50)
    args = dict(df_scales=df_scales, df_parts=df_parts, accept_gaps=True)
    # Small set of features (300 features created)
    split_kws = sf.get_split_kws(n_split_max=1, split_types=["Segment"])
    cpp = aa.CPP(df_cat=df_cat, **args, split_kws=split_kws)
    df_feat = cpp.run(labels=y, tmd_len=20, n_processes=8, n_filter=100)
    print(df_feat)
    X = sf.feat_matrix(**args, features=df_feat["feature"])
    # ML evaluation
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, scoring="accuracy", cv=5, n_jobs=8) # Set n_jobs=1 to disable multi-processing
    print(f"Mean accuracy of {round(np.mean(cv), 2)}")

    # Default set of features (around 100.000 features created)
    split_kws = sf.get_split_kws()
    cpp = aa.CPP(df_cat=df_cat, **args, split_kws=split_kws)
    df_feat = cpp.run(labels=y, tmd_len=20, n_processes=8, n_filter=100)
    df_feat.to_excel(ut.FOLDER_DATA + "cpp_features.xlsx", index=False)

    X = sf.feat_matrix(**args, features=df_feat["feature"])
    # ML evaluation
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, scoring="accuracy", cv=5, n_jobs=1)  # Set n_jobs=1 to disable multi-processing
    print(f"Mean accuracy of {round(np.mean(cv), 2)}")


def test_plotting():
    """"""
    tmd = "M" * 50
    jmd_n = "L" * 10
    df_feat = pd.read_excel(ut.FOLDER_DATA + "cpp_features.xlsx")
    cpp = aa.CPP()
    ylabel = "\nTest\nT" * 2

    cpp.plot_profile(df_feat=df_feat, tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_n, figsize=(8, 5))

    plt.ylabel(ylabel)
    cpp.update_seq_size()
    plt.tight_layout()
    plt.savefig(ut.FOLDER_DATA + "cpp_plot.pdf", dpi=100, bbox_inches="tight")
    plt.close()

    cpp.plot_heatmap(df_feat=df_feat, tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_n, figsize=(8, 5))
    plt.ylabel(ylabel)
    cpp.update_seq_size()
    plt.tight_layout()
    plt.savefig(ut.FOLDER_DATA + "cpp_plot_heat.pdf", dpi=100, bbox_inches="tight")


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    #test_cpp()
    test_plotting()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()

