"""
This is a script for the backend of CPP's pre-filtering threshold stage:
``pre_filtering`` drops features above ``max_std_test``, sorts by
absolute mean difference, and returns the top ``n`` features.
"""
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
# (no helpers — single-function module)


# II Main Functions
def pre_filtering(df=None, features=None, abs_mean_dif=None, std_test=None, max_std_test=0.2, n=10000, accept_gaps=False):
    """CPP pre-filtering based on thresholds."""
    if df is None:
        df = pd.DataFrame(zip(features, abs_mean_dif, std_test), columns=[ut.COL_FEATURE, ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST])
    df = df[df[ut.COL_STD_TEST] <= max_std_test]
    if accept_gaps:
        df = df[~df[ut.COL_ABS_MEAN_DIF].isna()]
    df = df.sort_values(by=[ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST, ut.COL_FEATURE], ascending=[False, True, True])
    df = df.reset_index(drop=True).head(n)
    return df
