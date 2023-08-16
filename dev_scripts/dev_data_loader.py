"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np

import aaanalysis as aa

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
def check_data_loading():
    """"""
    # TODO remove sequence with non-standard amino acids
    # TODO remove sequnces with min_length
    df_data = aa.load_dataset()
    df_scales = aa.load_scales()
    df_cat = aa.load_scales(name="scale_classification")
    df_eval = aa.load_scales(name="top60_eval")
    for i in range(0, 10):
        top_id = df_eval.sort_values(by="SEQ_AMYLO", ascending=False).index[i]
        df_top60 = aa.load_scales(name="top60")
        list_scales = df_top60.columns[df_top60.loc[top_id] == 1].to_list()
        _df_cat = df_cat[df_cat["scale_id"].isin(list_scales)]
        _df_scales = df_scales[list_scales]


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    check_data_loading()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
