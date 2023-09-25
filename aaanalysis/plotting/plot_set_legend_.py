"""
This is a script for setting plot legend.
"""
from typing import Optional, List, Dict, Union, Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from aaanalysis import utils as ut

# I Helper functions
# Checking functions
def check_cats(list_cat=None, dict_color=None, labels=None):
    """"""
    ut.check_dict(name="dict_color", val=dict_color, accept_none=False)
    if labels is not None:
        if list_cat is not None:
            if len(list_cat) != len(labels):
                raise ValueError(f"Length of 'list_cat' ({len(list_cat)}) and 'labels' ({len(labels)}) must match")
        elif len(dict_color) != len(labels):
            raise ValueError(f"Length of 'dict_color' ({len(dict_color)}) and 'labels' ({len(labels)}) must match")
    if list_cat is None:
        list_cat = list(dict_color.keys())
    else:
        raise ValueError("'list_cat' and 'dict_color' should not be None")
    return list_cat

# Helper functions
def _create_marker(color, category, shape, lw, edgecolor):
    if shape is None:
        args = {'facecolor': color, 'label': category, 'lw': lw}
        if edgecolor:
            args['edgecolor'] = edgecolor
        return mpl.patches.Patch(**args)
    return plt.Line2D([0], [0], marker=shape, color='w', markerfacecolor=color, markersize=10, label=category)



# II Main function
# TODO check, interface, doc, test
def plot_set_legend(ax: Optional[plt.Axes] = None,
                    handles: Optional[List] = None,
                    dict_color: Optional[Dict[str, str]] = None,
                    list_cat: Optional[List[str]] = None,
                    labels: Optional[List[str]] = None,
                    y: Optional[int] = None,
                    x: Optional[int] = None,
                    ncol: int = 3,
                    fontsize: Optional[int] = None,
                    weight: str = "normal",
                    lw: float = 0,
                    edgecolor: Optional[str] = None,
                    return_handles: bool = False,
                    loc: str = "upper left",
                    labelspacing: float = 0.2,
                    columnspacing: int = 1,
                    title: Optional[str] = None,
                    fontsize_legend: Optional[int] = None,
                    title_align_left: bool = True,
                    fontsize_weight: str = "normal",
                    shape: Optional[str] = None,
                    **kwargs) -> Union[plt.Axes, Tuple[List, List[str]]]:
    """
    Set a customizable legend for a plot.

    Parameters
    ----------
    ax
        The axes to attach the legend to. If not provided, the current axes will be used.
    handles
        Handles for legend items. If not provided, they will be generated based on `dict_color` and `list_cat`.
    dict_color
        A dictionary mapping categories to colors.
    list_cat
        List of categories to include in the legend.
    labels
        Labels for legend items.
    y
        The y-coordinate for the legend's anchor point.
    x
        The x-coordinate for the legend's anchor point.
    ncol
        Number of columns in the legend.
    fontsize
        Font size for the legend text.
    weight
        Weight of the font.
    lw
        Line width for legend items.
    edgecolor
        Edge color for legend items.
    return_handles
        Whether to return handles and labels. If `True`, function returns `handles, labels` instead of the axes.
    loc
        Location for the legend.
    labelspacing
        Vertical spacing between legend items.
    columnspacing
        Horizontal spacing between legend columns.
    title
        Title for the legend.
    fontsize_legend
        Font size for the legend title.
    title_align_left
        Whether to align the title to the left.
    fontsize_weight
        Font weight for the legend title.
    shape
        Marker shape for legend items. Refer to `matplotlib.lines.Line2D` for available shapes.

    Returns
    -------
    ax
        The axes with the legend applied. If `return_handles=True`, it returns handles and labels instead.

    Examples
    --------
    >>> import aaanalysis as aa
    >>> aa.plot_set_legend(ax=ax, dict_color={'Cat1': 'red', 'Cat2': 'blue'}, shape='o')

    See Also
    --------
    matplotlib.pyplot.legend
        For additional details on how the 'loc' parameter can be customized.
    matplotlib.lines.Line2D
        For details on the different types of marker shapes ('shape' parameter).
    """

    # Check input
    if ax is None:
        ax = plt.gca()
    list_cat = check_cats(list_cat=list_cat, dict_color=dict_color, labels=labels)
    args_float = {"y": y, "x": x, "lw": lw, "labelspacing": labelspacing,
                  "columnspacing": columnspacing}
    for key in args_float:
        ut.check_number_val(name=key, val=args_float[key], just_int=False)
    ut.check_number_range(name="ncol", val=ncol, min_val=1, just_int=True, accept_none=True)
    ut.check_bool(name="return_handles", val=return_handles)
    ut.check_bool(name="title_align_left", val=title_align_left)

    # TODO check other args
    # Generate legend items if not provided
    if not handles and dict_color and list_cat:
        handles = [_create_marker(dict_color[cat], cat, shape, lw, edgecolor) for cat in list_cat]

    # Return handles and labels if required
    if return_handles:
        return handles, labels if labels else list_cat
    # Set up legend properties
    labels = labels or list_cat
    args = dict(prop={"weight": weight, "size": fontsize}, **kwargs)
    if fontsize_legend:
        args["title_fontproperties"] = {"weight": fontsize_weight, "size": fontsize_legend}
    # Create the legend
    legend = ax.legend(handles=handles, labels=labels, bbox_to_anchor=(x, y), ncol=ncol, loc=loc,
                       labelspacing=labelspacing, columnspacing=columnspacing, borderpad=0, **args, title=title)
    # Align title if needed
    if title_align_left:
        legend._legend_box.align = "left"
    return ax
