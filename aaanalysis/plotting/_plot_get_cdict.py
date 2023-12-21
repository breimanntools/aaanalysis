"""Script for getting color dict"""
from aaanalysis import utils as ut

# Main function
def plot_get_cdict(name: str = "DICT_COLOR") -> dict:
    """
    Returns color dictionaries specified for AAanalysis.

    Parameters
    ----------
    name : {'DICT_COLOR', 'DICT_CAT'}, default='DICT_COLOR'
        The name of the AAanalysis color dictionary.

         - ``DICT_COLOR``: Dictionary with default colors for plots.
         - ``DICT_CAT``: Dictionary with default colors for scale categories.

    Returns
    -------
    dict
       AAanalysis color dictionary.

    See Also
    --------
    * Our `Plotting Prelude <plotting_prelude.html>`_.

    Examples
    --------
    .. include:: examples/plot_get_cdict.rst
    """
    list_names = [ut.STR_DICT_COLOR, ut.STR_DICT_CAT]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    if name == ut.STR_DICT_COLOR:
        return ut.DICT_COLOR
    else:
        return ut.DICT_COLOR_CAT
