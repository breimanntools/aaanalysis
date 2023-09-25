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

def _get_tab_color(n_colors=None):
    """Get default color lists for up to 9 categories """
    # Base lists
    list_colors_3_to_4 = ["tab:gray", "tab:blue", "tab:red", "tab:orange"]
    list_colors_5_to_6 = ["tab:blue", "tab:cyan", "tab:gray","tab:red",
                          "tab:orange", "tab:brown"]
    list_colors_8_to_9 = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                          "tab:gray", "gold", "tab:cyan", "tab:brown",
                          "tab:purple"]
    # Two classes
    if n_colors == 2:
        return ["tab:blue", "tab:red"]
    # Control/base + 2-3 classes
    elif n_colors in [3, 4]:
        return list_colors_3_to_4[0:n_colors]
    # 5-7 classes (gray in middle as visual "breather")
    elif n_colors in [5, 6]:
        return list_colors_5_to_6[0:n_colors]
    elif n_colors == 7:
        return ["tab:blue", "tab:cyan", "tab:purple", "tab:gray",
                "tab:red", "tab:orange", "tab:brown"]
    # 8-9 classes (colors from scale categories)
    elif n_colors in [8, 9]:
        return list_colors_8_to_9[0:n_colors]

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
    Returns color maps specified for AAanalysis.

    Parameters
    ----------
    name
        The name of the AAanalysis color palettes.

         - ``CPP``: Continuous color map for CPP plots.
         - ``SHAP``: Continuous color map for CPP-SHP plots.
         - ``CAT``: Color list for appealing visualization of categories.

    n_colors
        Number of colors in the color map. Must be >=2 for 'CPP' and 'SHAP' and 2-9 for 'CAT'.
    facecolor_dark
        Whether central color in 'CPP' and 'SHAP' is black (if ``True``) or white.

    Returns
    -------
    list
        List with colors given as RGB tuples (for 'CPP' and 'SHAP') or matplotlib color names (for 'CAT').

    Examples
    --------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> colors = aa.plot_get_cmap(name="CAT", n_colors=4)
        >>> data = {'Classes': ['Class A', 'Class B', 'Class C', "Class D"], 'Values': [23, 27, 43, 38]}
        >>> aa.plot_settings(no_ticks_x=True, font_scale=1.2)
        >>> sns.barplot(x='Classes', y='Values', data=data, palette=colors)
        >>> plt.show()

    See Also
    --------
    * Example notebooks in `Plotting Prelude <plotting_prelude.html>`_.
    * :func:`seaborn.color_palette` function to generate a color palette in seaborn.
    * :func:`seaborn.light_palette function` to generate a lighter color palettes.
    * `Matplotlib color names <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
    """
    # Check input
    list_names = [ut.STR_CMAP_CPP, ut.STR_CMAP_SHAP, ut.STR_CMAP_CAT]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    ut.check_bool(name="facecolor_dark", val=facecolor_dark)

    # Get color maps
    if name == ut.STR_CMAP_SHAP:
        ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
        return _get_shap_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    elif name == ut.STR_CMAP_CPP:
        ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
        return _get_cpp_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    elif name == ut.STR_CMAP_CAT:
        ut.check_number_range(name="n_colors", val=n_colors, min_val=2, max_val=9, just_int=True)
        return _get_tab_color(n_colors=n_colors)
