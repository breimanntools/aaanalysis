"""Script for getting color dict"""
from aaanalysis import utils as ut

# Main function
def plot_get_cdict(name: str = "DICT_COLOR") -> dict:
    """
    Returns color dictionarie specified for AAanalysis.

    Parameters
    ----------
    name
        The name of the AAanalysis color dictionary.

         - ``DICT_COLOR``: Dictionary with default colors for plots.
         - ``DICT_CAT``: Dictionary with default colors for scale categories.

    Returns
    -------
    dict
       AAanalysis color dictionary.

    Examples
    --------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> dict_color = aa.plot_get_cdict(name="DICT_COLOR")
        >>> data = {"Keys": list(dict_color.keys()), 'Values': [1] * len(dict_color) }
        >>> aa.plot_settings(weight_bold=False)
        >>> ax = sns.barplot(data=data, x="Values", y="Keys", palette=dict_color, legend=False)
        >>> ax.xaxis.set_visible(False)
        >>> sns.despine()
        >>> plt.tight_layout()
        >>> plt.show()

    See Also
    --------
    - Our `Plotting Prelude <plotting_prelude.html>`_.
    """
    list_names = [ut.STR_DICT_COLOR, ut.STR_DICT_CAT]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    if name == ut.STR_DICT_COLOR:
        return ut.DICT_COLOR
    else:
        return ut.DICT_COLOR_CAT
