"""
This is a script for frontend of plotting utility function to obtain AAanalysis color list.
The backend is in general utility module to provide function to remaining AAanalysis modules.
"""
from typing import List
from aaanalysis import utils as ut


# II Main function
def plot_get_clist(n_colors: int = 3) -> List[str]:
    """
    Get a manually curated list of 2 to 9 colors or 'husl' palette for more than 9 colors.

    This functions returns one of eight different color lists optimized for appealing visualization of categories.
    If more than 9 colors are selected, :func:`seaborn.color_palette` with 'husl' palette will be used.

    Parameters
    ----------
    n_colors : int, default=3
        Number of colors. Must be greater 2.

    Returns
    -------
    list
        Color list given as matplotlib color names.

    See Also
    --------
    * The example notebooks in `Plotting Prelude <plotting_prelude.html>`_.
    * `Matplotlib color names <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
    * :func:`seaborn.color_palette` function to generate a color palette in seaborn.

    Examples
    --------
    .. include:: examples/plot_get_clist.rst
    """
    # Check input
    ut.check_number_range(name="n_colors", val=n_colors, min_val=2, just_int=True)
    # Base lists
    colors = ut.plot_get_clist_(n_colors=n_colors)
    return colors


