"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
import seaborn as sns

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
def plot_correlation(df_corr=None):
    """"""
    ax = sns.heatmap(df_corr, cmap="viridis", vmin=-1, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


# III Test/Caller Functions


# IV Main

