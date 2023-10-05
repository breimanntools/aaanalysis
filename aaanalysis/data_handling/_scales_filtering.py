"""
This is a script for filtering scales from df_scales, such as using correlation or subcategory coverage.
"""
import time
import pandas as pd
import numpy as np

import aaanalysis as aa
import aaanalysis.utils as ut

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
# TODO
def scales_correlation():
    """Filter scales based on their pair-wise correlation."""

# TODO
def scales_coverage():
    """Compute the AAontology subcategory coverage of a scale set"""


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()

    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
