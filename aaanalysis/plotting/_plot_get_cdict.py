"""
This is a script for frontend of the plotting utility function to obtain an AAanalysis color dict.
The backend is in general utility module to provide function to remaining AAanalysis modules.
"""
from aaanalysis import utils as ut


# Main function
def plot_get_cdict(name: str = "DICT_COLOR") -> dict:
    """
    Get color dictionaries specified for AAanalysis.

    Parameters
    ----------
    name : {'DICT_COLOR', 'DICT_CAT'}, default='DICT_COLOR'
        The name of the AAanalysis color dictionary.

         - ``DICT_COLOR``: Dictionary with default colors for plots.
         - ``DICT_CAT``: Dictionary with default colors for scale categories.

    Returns
    -------
    dict_color
       AAanalysis color dictionary.

    See Also
    --------
    * `Plotting Prelude <plotting_prelude.html>`_.

    Examples
    --------
    .. include:: examples/plot_get_cdict.rst
    """
    # Check input
    list_names = [ut.STR_DICT_COLOR, ut.STR_DICT_CAT]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    # Get plotting dictionary
    dict_color = ut.plot_get_cdict_(name=name)
    return dict_color
