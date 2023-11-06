"""
Plotting utility function to obtain AAanalysis color maps.
"""
from typing import Union, List, Tuple
import seaborn as sns
from aaanalysis import utils as ut


# Helper functions
# Get color maps
def _get_cpp_cmap(n_colors=100, facecolor_dark=False):
    """Generate a diverging color map for CPP feature values."""
    ut.check_number_range(name="n_colors", val=n_colors, min_val=2, just_int=True)
    n = 5
    cmap = sns.color_palette(palette="RdBu_r", n_colors=n_colors + n * 2)
    cmap_low, cmap_high = cmap[0:int((n_colors + n * 2) / 2)], cmap[int((n_colors + n * 2) / 2):]
    c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
    add_to_end = 1  # Must be added to keep list size consistent
    cmap = cmap_low[0:-n] + c_middle + cmap_high[n+add_to_end:]
    return cmap


def _get_shap_cmap(n_colors=100, facecolor_dark=True):
    """Generate a diverging color map for feature values."""
    n = 20 # TODO check if 5 is better for CPP-SHAP heatmap
    cmap_low = sns.light_palette(ut.COLOR_SHAP_NEG, input="hex", reverse=True, n_colors=int(n_colors/2)+n)
    cmap_high = sns.light_palette(ut.COLOR_SHAP_POS, input="hex", n_colors=int(n_colors/2)+n)
    c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
    add_to_end = (n_colors+1)%2 # Must be added to keep list size consistent
    cmap = cmap_low[0:-n] + c_middle + cmap_high[n+add_to_end:]
    return cmap


# TODO check if needed later
def _get_cmap_with_gap(n_colors=100, pct_gap=10, pct_center=None,
                       color_pos=None, color_neg=None, color_center=None, input="hex"):
    """Generate a custom color map with a gap.

    """
    n_gap = int(n_colors*pct_gap/2)
    cmap_pos = sns.light_palette(color_pos, input=input, n_colors=int(n_colors/2)+n_gap)
    cmap_neg = sns.light_palette(color_neg, input=input, reverse=True, n_colors=int(n_colors/2)+n_gap)
    color_center = [cmap_neg[-1]] if color_center is None else color_center
    color_center = [color_center] if type(color_center) is str else color_center
    if pct_center is None:
        cmap = cmap_neg[0:-n_gap] + color_center + cmap_pos[n_gap:]
    else:
        n_center = int(n_colors * pct_center)
        n_gap += int(n_center/2)
        cmap = cmap_neg[0:-n_gap] + color_center * n_center + cmap_pos[n_gap:]
    return cmap


# II Main function
def plot_get_cmap(name: str = "CPP",
                  n_colors: int = 101,
                  facecolor_dark: bool = False
                  ) -> Union[List[Tuple[float, float, float]], List[str]]:
    """
    Returns color map specified for AAanalysis.

    Parameters
    ----------
    name
        The name of the AAanalysis color palettes.

         - ``CPP``: Continuous color map for CPP plots.
         - ``SHAP``: Continuous color map for CPP-SHP plots.

    n_colors
        Number of colors. Must be at least 3.
    facecolor_dark
        Whether central color in is black (if ``True``) or white.

    Returns
    -------
    list
        List with colors given as RGB tuples.

    Examples
    --------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> colors = aa.plot_get_cmap(name="CPP", n_colors=3)
        >>> data = {'Classes': ['Class A', 'Class B', 'Class C',], 'Values': [14, 23, 33]}
        >>> aa.plot_settings()
        >>> sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes", legend=False)
        >>> plt.show()

    See Also
    --------
    - `Plotting Prelude <plotting_prelude.html>`_.
    - `Matplotlib color names <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
    - :func:`seaborn.color_palette` function to generate a color palette in seaborn.
    - :func:`seaborn.light_palette function` to generate a lighter color palettes.
    - The `SHAP <shap:mod:shap>`_ package.
    """
    # Check input
    list_names = [ut.STR_CMAP_CPP, ut.STR_CMAP_SHAP]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    ut.check_bool(name="facecolor_dark", val=facecolor_dark)

    # Get color maps
    if name == ut.STR_CMAP_SHAP:
        ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
        return _get_shap_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    else:
        ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
        return _get_cpp_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
