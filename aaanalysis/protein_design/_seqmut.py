"""
This is a script for the frontend of the SeqMut class
"""
import aaanalysis.utils as ut


# I Helper Functions
# Check functions


# II Main Functions
class SeqMut:
    """
    UNDER CONSTRUCTION - Sequence Mutator (SeqMut) class for analyzing the impact of amino acid substitutions
    in protein sequences.
    """
    def __init__(self, verbose=False, df_scales=None):
        self._verbose = ut.check_verbose(verbose)
        self.df_scales = df_scales

    # Main method
    def fit(self, name=None, from_aa=None, to_aa=None):
        """Compute difference between amino acids for given property scale. """

    def eval(self):
        """"""
