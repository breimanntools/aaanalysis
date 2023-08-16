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
def check_aaclust():
    """"""
    aac = aa.AAclust()
    X = np.array(aa.load_scales()).T
    print(X)
    aac.fit(X, n_clusters=4)
    print(aac.centers_)


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    check_aaclust()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
