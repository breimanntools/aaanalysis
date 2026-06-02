"""
This is a script for the frontend of the SeqMutPlot class
"""
import aaanalysis.utils as ut


# I Helper Functions
# Check functions


# II Main Functions
class SeqMutPlot:
    """
    UNDER CONSTRUCTION - Plotting class for ``SeqMut`` (Sequence Mutator).

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
    def mutation_landscape(self):
        """Plot the per-position mutation-impact landscape across a sequence.

        .. note::
           Not yet implemented — :class:`SeqMutPlot` is under construction.
        """
        raise NotImplementedError("SeqMutPlot is under construction.")

    def residue_mutation_impact(self):
        """Plot the mutation impact for a single residue across substitutions.

        .. note::
           Not yet implemented — :class:`SeqMutPlot` is under construction.
        """
        raise NotImplementedError("SeqMutPlot is under construction.")
