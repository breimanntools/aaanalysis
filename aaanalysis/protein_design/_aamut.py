"""
This is a script for the frontend of the AAMut class for analysing the effect of amino acid changes
for given property scales."""
import aaanalysis.utils as ut


# I Helper Functions
# Check functions


# II Main Functions
class AAMut:
    """
    UNDER CONSTRUCTION - Amino Acid Mutator (AAMut) class for analyzing the impact of amino acid substitutions
    on amino acid scales.

    .. versionadded:: 1.0.0

    """
    def __init__(self, verbose=False, df_scales=None):
        """
        Parameters
        ----------
        verbose : bool, default=False
            If ``True``, verbose outputs are enabled.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of amino acid scales. Default from :func:`load_scales`.
        """
        self._verbose = ut.check_verbose(verbose)
        self.df_scales = df_scales

    # Main method
    def fit(self, name=None, from_aa=None, to_aa=None):
        """Compute the difference between amino acids for a given property scale.

        .. note::
           Not yet implemented — :class:`AAMut` is under construction.
        """
        raise NotImplementedError("AAMut is under construction.")

    def eval(self):
        """Evaluate the amino acid substitution impact across scales.

        .. note::
           Not yet implemented — :class:`AAMut` is under construction.
        """
        raise NotImplementedError("AAMut is under construction.")
