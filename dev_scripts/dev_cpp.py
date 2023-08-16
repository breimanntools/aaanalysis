"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aaanalysis as aa
import seaborn as sns

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions

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


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    check_cpp()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
