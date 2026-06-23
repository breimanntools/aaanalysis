"""
This is a script for frontend of plotting utility function to obtain AAanalysis color list.
The backend is in general utility module to provide function to remaining AAanalysis modules.
"""
from typing import List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from aaanalysis import utils as ut


# I Helper Functions
def _valid_cmaps(kind="categorical"):
    """Valid cmap names for a kind: Matplotlib registry + seaborn qualitative generators (+ house maps for diverging)."""
    names = set(plt.colormaps()) | {"husl", "hls"}
    if kind == "diverging":
        names |= {ut.STR_CMAP_CPP, ut.STR_CMAP_SHAP}
    return names


def check_match_cmap_kind(cmap=None, kind="categorical"):
    """Check that cmap is a valid colormap name for the given kind."""
    if cmap is None:
        return None
    ut.check_str(name="cmap", val=cmap)
    if cmap in _valid_cmaps(kind=kind):
        return None
    if kind == "diverging":
        str_valid = "a Matplotlib colormap name, 'husl', 'hls', 'CPP', or 'SHAP'"
    elif cmap in [ut.STR_CMAP_CPP, ut.STR_CMAP_SHAP]:
        str_valid = (f"a Matplotlib colormap name, 'husl', or 'hls' "
                     f"('{cmap}' is only valid for kind='diverging')")
    else:
        str_valid = "a Matplotlib colormap name, 'husl', or 'hls'"
    raise ValueError(f"'cmap' ('{cmap}') should be {str_valid} for kind='{kind}'.")


# II Main Functions
def plot_get_clist(n_colors: int = 3,
                   kind: str = "categorical",
                   cmap: Optional[str] = None,
                   facecolor_dark: bool = False,
                   ) -> Union[List[str], List[Tuple[float, float, float]]]:
    """
    Get a list of ``n_colors`` colors for a categorical, continuous, or diverging palette.

    This is the single entry point for quickly obtaining a color list of any size and
    type. Following Matplotlib's colormap taxonomy, ``kind`` selects the palette family
    and the optional ``cmap`` selects the concrete palette within it:

     - ``categorical`` (qualitative): maximally *distinct* colors for discrete classes.
       Hand-curated for 2 to 9 colors, the ``'husl'`` palette for 10 to 20.
     - ``continuous`` (sequential or qualitative): ``n_colors`` sampled from any named
       palette. An ordered map such as ``'viridis'`` yields a perceptual ramp (encodes
       magnitude); a qualitative map such as ``'husl'`` yields distinct hues.
     - ``diverging``: two hues from a neutral center, for signed/centered data. The house
       ``'CPP'`` / ``'SHAP'`` maps or any Matplotlib diverging map (e.g. ``'coolwarm'``).

    For ``categorical`` / ``continuous`` the colors are produced via
    :func:`seaborn.color_palette`, so any Matplotlib colormap name plus seaborn's
    ``'husl'`` / ``'hls'`` generators are accepted as ``cmap``; see the seaborn palette
    documentation for the full set. For a full pre-sized diverging colormap (101 points
    by default) use :func:`plot_get_cmap` instead.

    .. versionadded:: 0.1.2

    Parameters
    ----------
    n_colors : int, default=3
        Number of colors. Must be at least 2 (``categorical``, at most 20) or 3
        (``continuous`` / ``diverging``).
    kind : {'categorical', 'continuous', 'diverging'}, default='categorical'
        Palette family to draw colors from.
    cmap : str, optional
        Name of the concrete palette. If ``None``, defaults to the curated list
        (``categorical``), ``'husl'`` (``continuous``), or ``'CPP'`` (``diverging``).
        Accepts any Matplotlib colormap name and seaborn's ``'husl'`` / ``'hls'``;
        ``'CPP'`` and ``'SHAP'`` are valid only for ``kind='diverging'``.
    facecolor_dark : bool, default=False
        Whether the central color of a diverging map is black (if ``True``) or white
        (if ``False``). Only applies to ``kind='diverging'``; ignored otherwise.

    Returns
    -------
    colors : list
        List of ``n_colors`` colors. Matplotlib color-name strings for the curated
        categorical list, otherwise RGB tuples.

    See Also
    --------
    * :func:`plot_get_cmap` for the pre-sized diverging ``CPP`` / ``SHAP`` colormaps.
    * :func:`plot_get_cdict` for the named category-to-color dictionaries.
    * The example notebooks in `Plotting Prelude <plotting_prelude.html>`_.
    * `Matplotlib colormaps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
    * :func:`seaborn.color_palette` function to generate a color palette in seaborn.

    Examples
    --------
    .. include:: examples/plot_get_clist.rst
    """
    # Check input
    list_kinds = ["categorical", "continuous", "diverging"]
    ut.check_str_options(name="kind", val=kind, list_str_options=list_kinds)
    ut.check_bool(name="facecolor_dark", val=facecolor_dark)
    check_match_cmap_kind(cmap=cmap, kind=kind)
    # Get color list per kind
    if kind == "categorical":
        ut.check_number_range(name="n_colors", val=n_colors, min_val=2, max_val=20, just_int=True)
        colors = ut.plot_get_clist_(n_colors=n_colors, cmap=cmap)
    elif kind == "continuous":
        ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
        colors = ut.plot_get_clist_(n_colors=n_colors, cmap=cmap or "husl")
    else:
        ut.check_number_range(name="n_colors", val=n_colors, min_val=3, just_int=True)
        colors = ut.plot_get_cmap_(cmap=cmap or ut.STR_CMAP_CPP, n_colors=n_colors,
                                   facecolor_dark=facecolor_dark)
    return colors
