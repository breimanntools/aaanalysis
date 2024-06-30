"""
This is a script for frontend of the setting plot legend.
The backend is in general utility module to provide function to remaining AAanalysis modules.
"""
from typing import Optional, List, Dict, Union, Tuple
from matplotlib import pyplot as plt
from aaanalysis import utils as ut

# I Helper functions


# II Main function
def plot_legend(ax: Optional[plt.Axes] = None,
                # Categories and colors
                dict_color: Dict[str, str] = None,
                list_cat: Optional[List[str]] = None,
                labels: Optional[List[str]] = None,
                # Position and Layout
                loc: Union[str, int] = "upper left",
                loc_out: bool = False,
                frameon: bool = False,
                y: Optional[Union[int, float]] = None,
                x: Optional[Union[int, float]] = None,
                n_cols: int = 3,
                labelspacing: Union[int, float] = 0.2,
                columnspacing: Union[int, float] = 1.0,
                handletextpad: Union[int, float] = 0.8,
                handlelength: Union[int, float] = 2.0,
                # Font and Style
                fontsize: Optional[Union[int, float]] = None,
                fontsize_title: Optional[Union[int, float]] = None,
                weight_font: str = "normal",
                weight_title: str = "normal",
                # Marker, Lines, and Area
                marker: Optional[Union[str, int, list]] = None,
                marker_size: Union[int, float, List[Union[int, float]]] = 10,
                lw: Union[int, float] = 0,
                linestyle: Optional[Union[str, list]] = None,
                edgecolor: Optional[str] = None,
                hatch: Optional[Union[str, List[str]]] = None,
                hatchcolor: str = "white",
                # Title
                title: Optional[str] = None,
                title_align_left: bool = True,
                # Additional arguments
                keep_legend: bool = False,
                **kwargs
                ) -> Union[plt.Axes, Tuple[List, List[str]]]:
    """
    Set an independently customizable plot legend.

    Legends can be flexibly adjusted based categories and colors provided in ``dict_color`` dictionary.
    This functions comprises the most convenient settings for ``func:`matplotlib.pyplot.legend``.

    Parameters
    ----------
    ax : plt.Axes, optional
        The axes to attach the legend to. If not provided, the current axes will be used.
    dict_color : dict, optional
        A dictionary mapping categories to colors.
    list_cat : list of str, optional
        List of categories to include in the legend (keys of ``dict_color``).
    labels : list of str, optional
        Legend labels corresponding to given categories.
    loc : int or str
        Location for the legend.
    loc_out : bool, default=False
        If ``True``, sets automatically ``x=0`` and ``y=-0.25`` if they are ``None``.
    frameon : bool, default=False
        If ``True``, a figure background patch (frame) will be drawn.
    y : int or float, optional
        The y-coordinate for the legend's anchor point.
    x : int or float, optional
        The x-coordinate for the legend's anchor point.
    n_cols : int, default=1
        Number of columns in the legend, at least 1.
    labelspacing : int or float, default=0.2
        Vertical spacing between legend items.
    columnspacing : int or float, default=1.0
        Horizontal spacing between legend columns.
    handletextpad : int or float, default=0.8
        Horizontal spacing between legend handle (marker) and label.
    handlelength : int or float, default=2.0
        Length of legend handle.
    fontsize : int or float, optional
        Font size of the legend text.
    fontsize_title : inf or float, optional
        Font size of the legend title.
    weight_font : str, default='normal'
        Weight of the font.
    weight_title : str, default='normal'
        Font weight for the legend title.
    marker : str, int, or list, optional
        Handle marker for legend items. Lines ('-') only visible if ``lw>0``.
    marker_size : int, float, or list, optional
        Marker size of legend items.
    lw : int or float, default=0
        Line width for legend items. If negative, corners are rounded.
    linestyle : str or list, optional
        Style of line. Only applied to lines (``marker='-'``).
    edgecolor : str, optional
        Edge color of legend items. Not applicable to lines.
    hatch : str or list, optional
        Filling pattern for default marker. Only applicable when ``marker=None``.
    hatchcolor : str, default='white'
        Hatch color of legend items. Only applicable when ``marker=None``.
    title : str, optional
        Legend title.
    title_align_left : bool, default=True
        Whether to align the title to the left.
    keep_legend: bool, default=False
        If ``True``, keep existing legend (must be within plot) and add a new one.
    **kwargs
        Further key word arguments for :attr:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    ax : plt.Axes
        The axes object on which legend is applied to.

    Notes
    -----
    Markers can be None (default), lines ('-') or one of the `matplotlib markers
    <https://matplotlib.org/stable/api/markers_api.html>`_.

    See Also
    --------
    * More examples in `Plotting Prelude <plotting_prelude.html>`_.
    * `Linestyles of markers <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_.
    * `Hatches <https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html>`_, which are filling patterns.
    * :class:`matplotlib.lines.Line2D` for available marker shapes and line properties.
    * :class:`matplotlib.axes.Axes`, which is the core object in matplotlib.
    * :func:`matplotlib.pyplot.gca` to get the current Axes instance.

    Examples
    --------
    .. include:: examples/plot_legend.rst
    """
    # Check input
    ut.check_ax(ax=ax, accept_none=True)
    if ax is None:
        ax = plt.gca()
    ut.check_dict(name="dict_color", val=dict_color, accept_none=False)
    ut.check_bool(name="title_align_left", val=title_align_left)
    ut.check_bool(name="loc_out", val=loc_out)
    ut.check_bool(name="frameon", val=frameon)
    ut.check_number_range(name="n_cols", val=n_cols, min_val=1, accept_none=True, just_int=True)
    ut.check_number_val(name="x", val=x, accept_none=True, just_int=False)
    ut.check_number_val(name="y", val=y, accept_none=True, just_int=False)
    ut.check_number_val(name="lw", val=lw, accept_none=True, just_int=False)
    args_non_neg = {"labelspacing": labelspacing, "columnspacing": columnspacing,
                    "handletextpad": handletextpad, "handlelength": handlelength,
                    "fontsize": fontsize, "fontsize_legend": fontsize_title}
    for key in args_non_neg:
        ut.check_number_range(name=key, val=args_non_neg[key], min_val=0, accept_none=True, just_int=False)
    ut.check_bool(name="add_legend", val=keep_legend, accept_none=False)
    # Create new legend
    ax = ut.plot_legend_(ax=ax, dict_color=dict_color, list_cat=list_cat, labels=labels,
                         loc=loc, loc_out=loc_out, y=y, x=x, n_cols=n_cols,
                         labelspacing=labelspacing, columnspacing=columnspacing,
                         handletextpad=handletextpad, handlelength=handlelength,
                         fontsize=fontsize, fontsize_title=fontsize_title,
                         weight_font=weight_font, weight_title=weight_title,
                         marker=marker, marker_size=marker_size, lw=lw, linestyle=linestyle, edgecolor=edgecolor,
                         hatch=hatch, hatchcolor=hatchcolor, title=title, title_align_left=title_align_left,
                         frameon=frameon, keep_legend=keep_legend, **kwargs)
    return ax
