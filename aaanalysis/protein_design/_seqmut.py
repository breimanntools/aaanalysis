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
    def __init__(self, verbose=False):
        self._verbose = ut.check_verbose(verbose)

    # Main method
    def fit(self, name=None, from_aa=None, to_aa=None):
        """Compute difference between amino acids for given property scale. """

    def scan(self, df_seq=None, jmd_n_len=10, jmd_c_len=10, show_delta=False):
        """
        Make single position mutational scan for all existing amino acids for each sequence in df_seq
        using the average prediction for each model (already fitted).
        """
