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

    .. versionadded:: 1.0.0

    """
    def __init__(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.
        """
        self._verbose = ut.check_verbose(verbose)

    # Main method
    def fit(self, name=None, from_aa=None, to_aa=None):
        """Compute the difference between amino acids for a given property scale.

        .. note::
           Not yet implemented — :class:`SeqMut` is under construction.
        """
        raise NotImplementedError("SeqMut is under construction.")

    def scan(self, df_seq=None, jmd_n_len=10, jmd_c_len=10, show_delta=False):
        """Run a single-position mutational scan over every amino acid for each sequence.

        .. note::
           Not yet implemented — :class:`SeqMut` is under construction.
        """
        raise NotImplementedError("SeqMut is under construction.")
