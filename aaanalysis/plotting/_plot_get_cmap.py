"""
This is a script for the frontend of the plotting utility function to obtain AAanalysis colormaps.
The backend is in general utility module to provide function to remaining AAanalysis modules.
"""
from typing import Union, List, Tuple, Optional
from aaanalysis import utils as ut


# II Main function
def plot_get_cmap(name: str = "CPP",
                  n_colors: int = 101,
                  facecolor_dark: bool = False,
                  ) -> Union[List[Tuple[float, float, float]], List[str]]:
    """
    Get colormaps specified for AAanalysis.

    Parameters
    ----------
    name : {'CPP', 'SHAP'}, default='CPP'
        The name of the AAanalysis color palettes.

         - ``CPP``: Continuous colormap for CPP plots.
         - ``SHAP``: Continuous colormap for CPP-SHP plots.

    n_colors : int, default=101
        Number of colors. Must be at least 3.
    facecolor_dark : bool, optional
        Whether central color in is black (if ``True``) or white (if ``False``).

    Returns
    -------
    cmap
        List with colors given as RGB tuples.

    See Also
    --------
    * `Plotting Prelude <plotting_prelude.html>`_.
    * `Matplotlib color names <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
    * :func:`seaborn.color_palette` function to generate a color palette in seaborn.
    * :func:`seaborn.light_palette function` to generate a lighter color palettes.
    * The `SHAP <shap:mod:shap>`_ package.

    Examples
    --------
    .. include:: examples/plot_get_cmap.rst
    """
    # Check input
    list_names = [ut.STR_CMAP_CPP, ut.STR_CMAP_SHAP]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
    ut.check_bool(name="facecolor_dark", val=facecolor_dark)
    # Get colormaps
    cmap = ut.plot_get_cmap_(cmap=name, n_colors=n_colors, facecolor_dark=facecolor_dark)
    return cmap

