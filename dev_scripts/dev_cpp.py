"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aaanalysis as aa
import seaborn as sns
import dev_scripts._utils as ut


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions
# TODO get final plots (feature map, profile, ranking, heatmap)

# II Main Functions
def check_cpp():
    """
    """
    sf = aa.SequenceFeature()
    # TODO adjust for GSEC_SUB_SEQ # TODO adjust for short peptides # TOD0 remove gaps
    df_seq = aa.load_dataset(name='SEQ_DISULFIDE', min_len=100, non_canonical_aa="remove")
    print(df_seq)
    labels = list(df_seq["label"])
    sns.histplot(list([len(x) for x in df_seq["sequence"]]))
    plt.show()
    df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10)
    split_kws = sf.get_split_kws(n_split_min=1, n_split_max=3,
                                 split_types=["Segment", "PeriodicPattern"])
    df_scales = aa.load_scales(unclassified_in=False).sample(n=10, axis=1)
    print(df_scales)
    print(split_kws)
    print(df_parts)
    cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
    df_feat = cpp.run(labels=labels)    # TODO Where is n_split
    cpp.plot_heatmap(df_feat=df_feat)
    plt.tight_layout()
    plt.show()


def sub_pred():
    """"""
    df_feat = pd.read_excel(ut.FOLDER_DATA + "cpp_features_ranked_TMHMM.xlsx")
    df_sub = pd.read_excel(ut.FOLDER_DATA + "SUB_TMHMM.xlsx")
    df_sub = df_sub[df_sub["class"] == "SUBEXPERT"]
    df_nonsub = pd.read_excel(ut.FOLDER_DATA + "NONSUB_TMHMM.xlsx")
    _list_rel_neg = df_nonsub[df_nonsub["class"] != "NONSUB"]["name"].to_list()
    df_nonsub = df_nonsub[df_nonsub["class"] == "NONSUB"]
    df_others = pd.read_excel(ut.FOLDER_DATA + "OTHERS_TMHMM.xlsx")
    df_seq = pd.concat([df_sub, df_others, df_nonsub])
    sf = aa.SequenceFeature()
    df_scales = aa.load_scales()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    list_feat = df_feat["feature"].to_list()
    X = sf.feat_matrix(df_parts=df_parts, df_scales=df_scales, features=list_feat, n_jobs=1)
    labels = [1] * len(df_sub) + [2] * len(df_others) + [3] * len(df_nonsub)
    n_neg = len(df_sub) - len(df_nonsub)
    dpul = aa.dPULearn()

    df_seq = dpul.fit(X=X, df_seq=df_seq, labels=labels, n_neg=n_neg)
    labels = dpul.labels_
    print(labels)
    df_seq.insert(2, "in_NONSUBPRED", [n in _list_rel_neg for n in df_seq["name"]])

    df = df_seq[df_seq["class"].str.contains("REL_NEG")].sort_values(by="class")
    list_non = df["name"].to_list()
    print([x for x in _list_rel_neg if x not in list_non])
    print([x for x in list_non if x not in _list_rel_neg])

    aa.plot_settings(font_scale=1.1)
    sns.scatterplot(data=df_seq, x="PC1 (56.1%)", y="PC2 (7.4%)", hue="class", legend=False)
    sns.despine()
    plt.tight_layout()
    plt.show()
    df_seq = df_seq[~df_seq["class"].isin(["OTHERS", "REL_NEG_PC1", "REL_NEG_PC2"])]
    sns.scatterplot(data=df_seq, x="PC3 (2.9%)", y="PC4 (2.8%)", hue="class", legend=False)
    sns.despine()
    plt.tight_layout()
    plt.show()


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    sub_pred()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
