"""
This is a script for the frontend of the AAMut class
"""
import aaanalysis.utils as ut


# I Helper Functions
# Check functions


# II Main Functions
class AAMutPlot:
    """
    UNDER CONSTRUCTION - Plotting class for ``AAMut`` (Amino Acid Mutator).

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
    def substitution_matrix(self):
        """Plot the amino acid substitution impact matrix.

        .. note::
           Not yet implemented — :class:`AAMutPlot` is under construction.
        """
        raise NotImplementedError("AAMutPlot is under construction.")

    def scale_ranking(self):
        """Plot the per-scale ranking of substitution impact.

        .. note::
           Not yet implemented — :class:`AAMutPlot` is under construction.
        """
        raise NotImplementedError("AAMutPlot is under construction.")

    def aa_comparison(self):
        """Plot the pairwise amino acid comparison.

        .. note::
           Not yet implemented — :class:`AAMutPlot` is under construction.
        """
        raise NotImplementedError("AAMutPlot is under construction.")
