"""
This is a script for the frontend of the SeqMutPlot class
"""
import aaanalysis.utils as ut


# I Helper Functions
# Check functions


# II Main Functions
class SeqMutPlot:
    """
    Plot SeqMut results.
    """
    def __init__(self, verbose=False, df_scales=None):
        self.verbose = verbose
        self.df_scales = df_scales

    # Main method
