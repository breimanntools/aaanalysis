"""
This is a script for setting plot legend.
"""
from typing import Optional, List, Dict, Union, Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from aaanalysis import utils as ut
import matplotlib.lines as mlines


# I Helper functions
# Checking functios
def check_dict_colors(dict_color=None, list_cat=None, labels=None):
    """"""
    ut.check_dict(name="dict_color", val=dict_color, accept_none=False)
    # If labels are provided
    if labels:
        # If list_cat is provided, check its length against labels
        if list_cat:
            if len(list_cat) != len(labels):
                raise ValueError(f"Length of 'list_cat' ({len(list_cat)}) and 'labels' ({len(labels)}) must match")
        # If list_cat isn't provided, check the length of dict_color against labels
        elif len(dict_color) != len(labels):
            raise ValueError(f"Length of 'dict_color' ({len(dict_color)}) and 'labels' ({len(labels)}) must match")
    # Default list_cat to the keys of dict_color if not provided
    if not list_cat:
        list_cat = list(dict_color.keys())
    return list_cat


def check_x_y(x=None, y=None):
    """"""
    ut.check_number_val(name="x", val=x, accept_none=True, just_int=False)
    ut.check_number_val(name="y", val=y, accept_none=True, just_int=False)
    # Check if matching
    if x is None and y is None:
        return
    if x is not None and y is not None:
        return
    raise ValueError(f"'x' ({x}) and 'y' ({y}) must either both be None or both given.")



def _check_validity(item, valid_items, item_name, list_cat):
    """General check validity function."""
    if item is None:
        return [item] * len(list_cat)
    item_is_list = isinstance(item, list)
    if not item_is_list:
        if item not in valid_items:
            raise ValueError(f"'{item_name}' ('{item}') must be one of the following: {valid_items}")
        else:
            return [item] * len(list_cat)
    else:
        wrong_items = [x for x in item if x not in valid_items]
        if len(wrong_items) != 0:
            raise ValueError(
                f"'{item_name}' contains wrong items ('{wrong_items}')! Must be one of the following: {valid_items}")
        if len(item) != len(list_cat):
            raise ValueError(f"Length of '{item_name}' ({item}) and categories ({list_cat}) must match!")
        else:
            return item


def check_marker(marker=None, list_cat=None, lw=0):
    """Check validity of marker."""
    all_markers = list(mlines.Line2D.markers.keys())
    all_markers.append("-")  # Allow '-' as a valid marker for a line
    if marker == "-" and lw <= 0:
        raise ValueError(f"If marker is '{marker}', 'lw' ({lw}) must be > 0.")
    return _check_validity(marker, all_markers, 'marker', list_cat)


def check_linestyle(marker_linestyle=None, list_cat=None):
    """Check validity of linestyle."""
    _lines = ['-', '--', '-.', ':']
    _names = ["solid", "dashed", "dashed-doted", "dotted"]
    dict_names_lines = dict(zip(_names, _lines))
    # Names to line styles
    if isinstance(marker_linestyle, str) and marker_linestyle in _names:
        marker_linestyle = dict_names_lines[marker_linestyle]
    elif isinstance(marker_linestyle, list):
        marker_linestyle = [dict_names_lines.get(x, x) for x in marker_linestyle]
    valid_linestyles = dict(zip(_lines, _names))
    return _check_validity(marker_linestyle, valid_linestyles, 'marker_linestyle', list_cat)


# Helper function
def _create_marker(color, category, marker, marker_size, lw, edgecolor, linestyle='-'):
    """Create custom marker based on input."""
    if marker == "-":  # If marker is '-', treat it as a line
        return plt.Line2D([0, 1], [0, 1], color=color, linestyle=linestyle, lw=lw, label=category)

    if marker is None:
        args = {'facecolor': color, 'label': category, 'lw': lw}
        if edgecolor:
            args['edgecolor'] = edgecolor
        return mpl.patches.Patch(**args)

    return plt.Line2D([0], [0], marker=marker, color='w', linestyle=linestyle, markerfacecolor=color,
                      markersize=marker_size, label=category)

# II Main function
def plot_set_legend(ax: Optional[plt.Axes] = None,
                    remove_legend: bool = True,
                    handles: Optional[List] = None,
                    return_handles: bool = False,
                    # Color and Categories
                    dict_color: Optional[Dict[str, str]] = None,
                    list_cat: Optional[List[str]] = None,
                    labels: Optional[List[str]] = None,
                    # Position and Layout
                    loc: str = "upper left",
                    loc_out: bool = False,
                    y: Optional[Union[int, float]] = None,
                    x: Optional[Union[int, float]] = None,
                    ncol: int = 3,
                    labelspacing: Union[int, float] = 0.2,
                    columnspacing: Union[int, float] = 1.0,
                    handletextpad: Union[int, float] = 0.8,
                    handlelength: Union[int, float] = 2,
                    # Font and Style
                    fontsize: Optional[Union[int, float]] = None,
                    fontsize_title: Optional[Union[int, float]] = None,
                    weight: str = "normal",
                    fontsize_weight: str = "normal",
                    # Line, Marker, and Area
                    lw: Union[int, float] = 0,
                    edgecolor: Optional[str] = None,
                    marker: Optional[Union[str, int, list]] = None,
                    marker_size: Union[int, float] = 10,
                    marker_linestyle: Union[str, list] = "-",
                    # Title
                    title: Optional[str] = None,
                    title_align_left: bool = True,
                    **kwargs
                    ) -> Union[plt.Axes, Tuple[List, List[str]]]:
    """
    Sets a customizable legend for a plot.

    Legends can be flexbily adjusted based on ``handles`` or categories and colors provided in ``dict_color``.
    This functions comprises the most convinient settings for ``func:`matplotlib.pyplot.legend``.

    Parameters
    ----------
    ax
        The axes to attach the legend to. If not provided, the current axes will be used.
    remove_legend:
        Remove legend of given or current axes.
    handles
        Handles for legend items. If not provided, they will be generated based on ``dict_color`` and ``list_cat``.
    return_handles
        Whether to return handles and labels. If ``True``, function returns ``handles``, ``labels`` instead of the axes.

    dict_color
        A dictionary mapping categories to colors.
    list_cat
        List of categories to include in the legend (keys of ``dict_color``).
    labels
        Labels for legend items corresponding to given categories.
    loc
        Location for the legend.
    loc_out
        If ``True``, sets automatically ``x=0`` and ``y=-0.25`` if they are ``None``.
    y
        The y-coordinate for the legend's anchor point.
    x
        The x-coordinate for the legend's anchor point.
    ncol
        Number of columns in the legend, at least 1.
    labelspacing
        Vertical spacing between legend items.
    columnspacing
        Horizontal spacing between legend columns.
    handletextpad
        Horizontal spacing bewtween legend handle (marker) and label.
    handlelength
        Length of legend handle (marker).
    fontsize
        Font size for the legend text.
    fontsize_title
        Font size for the legend title.
    weight
        Weight of the font.
    fontsize_weight
        Font weight for the legend title.
    lw
        Line width for legend items.
    edgecolor
        Edge color for legend items.
    marker
        Marker shape for legend items. '-' is added to the :mod:`matplotlib` default options to use lines as markers.
    marker_size
        Marker size for legend items.
    marker_linestyle
        Marker linestyle of legend items.
    title
        Title for the legend.
    title_align_left
        Whether to align the title to the left.
    **kwargs
        Furhter key word arguments for :func:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    ax
        The axes with the legend applied. If ``return_handles=True``, it returns ``handles`` and ``labels`` instead.

    Examples
    --------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> data = {'Classes': ['A', 'B', 'C'], 'Values': [23, 27, 43]}
        >>> colors = aa.plot_get_clist()
        >>> aa.plot_settings()
        >>> sns.barplot(x='Classes', y='Values', data=data, palette=colors)
        >>> sns.despine()
        >>> dict_color = dict(zip(["Class A", "Class B", "Class C"], colors))
        >>> aa.plot_set_legend(dict_color=dict_color, ncol=3, x=0, y=1.1, handletextpad=0.4)
        >>> plt.tight_layout()
        >>> plt.show()

    See Also
    --------
    * `Matplotlib markers <https://matplotlib.org/stable/api/markers_api.html>`.
    * `Linestyles of markers <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`
    * :func:`matplotlib.lines.Line2D` for available marker shapes and line properties.
    * :func:`matplotlib.axes.Axes`, which is the core object in matplotlib.
    * :func:`matplotlib.pyplot.gca` to get the current Axes instance.
    """
    # Check basic input
    ut.check_ax(ax=ax, accept_none=True)
    if ax is None:
        ax = plt.gca()

    ut.check_bool(name="remove_current_legend", val=remove_legend)
    ut.check_bool(name="return_handles", val=return_handles)
    ut.check_bool(name="title_align_left", val=title_align_left)
    ut.check_bool(name="loc_out", val=loc_out)

    ut.check_number_range(name="ncol", val=ncol, min_val=1, accept_none=True, just_int=True)
    args_float = {"labelspacing": labelspacing, "columnspacing": columnspacing,
                  "handletextpad": handletextpad, "handlelength": handlelength,
                  "fontsize": fontsize, "fontsize_legend": fontsize_title,
                  "lw": lw, "marker_size": marker_size}
    for key in args_float:
        ut.check_number_val(name=key, val=args_float[key], accept_none=True, just_int=False)
    check_x_y(x=x, y=y)

    # Set y an x if legend should be outside
    if loc_out:
        x = x if x is not None else 0
        y = y if y is not None else -0.25

    # Universal Legend Arguments
    args = dict(loc=loc,
                ncol=ncol,
                fontsize=fontsize,
                labelspacing=labelspacing,
                columnspacing=columnspacing,
                handletextpad=handletextpad,
                handlelength=handlelength,
                borderpad=0, #  Fractional whitespace inside the legend border.
                title=title,
                edgecolor=edgecolor,
                prop={"weight": weight, "size": fontsize})
    args.update(kwargs)
    if fontsize_title:
        args["title_fontproperties"] = {"weight": fontsize_weight, "size": fontsize_title}
    if x is not None and y is not None:
        args["bbox_to_anchor"] = (x, y)

    # Use default plt.legend if no dict_color provided
    if dict_color is None:
        plt.legend(handles=handles, labels=labels, **args)
        return ax

    # Check if legend exists
    if remove_legend:
        if ax.get_legend() is not None and len(ax.get_legend().get_lines()) > 0:
            ax.legend_.remove()

    # Adjust and check dict_colors and all arguments depending on it
    list_cat = check_dict_colors(list_cat=list_cat, dict_color=dict_color, labels=labels)
    marker = check_marker(marker=marker, list_cat=list_cat, lw=lw)
    marker_linestyle = check_linestyle(marker_linestyle=marker_linestyle, list_cat=list_cat)

    # Generate legend items if not provided using dict_cat and list_cat
    if not handles and dict_color and list_cat:
        handles = [_create_marker(dict_color[cat], cat, marker[i], marker_size, lw, edgecolor, marker_linestyle[i])
                   for i, cat in enumerate(list_cat)]
    labels = list_cat if labels is None else labels
    # Create the legend
    legend = ax.legend(handles=handles, labels=labels, **args)

    # Align title if needed
    if title_align_left:
        legend._legend_box.align = "left"

    if return_handles:
        return handles, labels if labels else list_cat
    else:
        return ax
