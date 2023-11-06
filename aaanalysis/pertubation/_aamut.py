"""
This is a script for the frontend of the AAmut class for analysing the effect of amino acid changes
for given property scales."""
import aaanalysis.utils as ut


# I Helper Functions
# Check functions


# II Main Functions
class AAMut:
    """
    A class for analyzing the impact of amino acid substitutions on various property scales.
    """
    def __init__(self, verbose=False, df_scales=None):
        self.verbose = verbose
        self.df_scales = df_scales

    # Main method
    def fit(self, name=None, from_aa=None, to_aa=None):
        """Compute difference between amino acids for given property scale. """

    def eval(self):
        """"""  # TODO add evaluation function
