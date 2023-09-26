"""
This is a script for setting plot legend.
"""
from typing import Optional, List, Dict, Union, Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from aaanalysis import utils as ut
import matplotlib.lines as mlines


# I Helper functions
def _items_grouped(lst):
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:  # Same item as before, continue
            continue
        if lst[i] in lst[:i]:  # Different item but already appeared earlier in the list
            return False
    return True


def _check_validity(item, valid_items, item_name, list_cat):
    """General check validity function."""
    if item is None:
        return [item] * len(list_cat)
    item_is_list = isinstance(item, list)
    if not item_is_list:
        if item not in valid_items:
            raise ValueError(f"'{item_name}' ('{item}') must be one of following: {valid_items}")
        else:
            return [item] * len(list_cat)
    else:
        wrong_items = [x for x in item if x not in valid_items]
        if len(wrong_items) != 0:
            raise ValueError(
                f"'{item_name}' contains wrong items ('{wrong_items}')! Must be one of following: {valid_items}")
        if len(item) != len(list_cat):
            raise ValueError(f"Length must match of '{item_name}' ({item}) and categories ({list_cat}).")
        else:
            return item

# Checking functios
def check_list_cat(dict_color=None, list_cat=None):
    """"""
    if list_cat is None:
        return list(dict_color.keys())
    # Check if all elements of list_cat are keys in dict_colors
    if not all(elem in dict_color for elem in list_cat):
        missing_keys = [elem for elem in list_cat if elem not in dict_color]
        raise ValueError(f"The following keys in 'list_cat' are not in 'dict_colors': {', '.join(missing_keys)}")
    # Check if lengths match
    if len(dict_color) != len(list_cat):
        raise ValueError(f"Length must match between 'list_cat' ({len(list_cat)}) and 'dict_colors' ({len(dict_color)}).")


def check_labels(list_cat=None, labels=None):
    """"""
    if labels is None:
        return list_cat
    # If list_cat is provided, check its length against labels
    if len(list_cat) != len(labels):
        raise ValueError(f"Length must match of 'labels' ({len(labels)}) and categories ({len(list_cat)}).")


def check_marker(marker=None, list_cat=None, lw=0):
    """Check validity of marker."""
    all_markers = list(mlines.Line2D.markers.keys())
    all_markers.append("-")  # Allow '-' as a valid marker for a line
    if marker == "-" and lw <= 0:
        raise ValueError(f"If marker is '{marker}', 'lw' ({lw}) must be > 0.")
    return _check_validity(marker, all_markers, 'marker', list_cat)

def check_marker_size(marker_size=None, list_cat=None):
    """"""
    if isinstance(marker_size, (int, float)):
        ut.check_number_val(name='marker_size', val=marker_size, accept_none=True, just_int=False)
        marker_size = [marker_size] * len(list_cat)
        return marker_size
    elif marker_size is None:
        marker_size = [marker_size] * len(list_cat)
        return marker_size
    elif isinstance(marker_size, list):
        for i in marker_size:
            ut.check_number_val(name='marker_size', val=i, accept_none=True, just_int=False)
        return marker_size
    elif isinstance(marker_size, list) and len(marker_size) != len(list_cat):
        raise ValueError(f"Length must match of 'marker_size' (marker_size) and categories ({list_cat}).")
    else:
        raise ValueError(f"'marker_size' has wrong data type: {type(marker_size)}")


def check_linestyle(marker_linestyle=None, list_cat=None):
    """Check validity of linestyle."""
    _lines = ['-', '--', '-.', ':', ]
    _names = ["solid", "dashed", "dashed-doted", "dotted"]
    dict_names_lines = dict(zip(_names, _lines))
    # Names to line styles
    if isinstance(marker_linestyle, str) and marker_linestyle in _names:
        marker_linestyle = dict_names_lines[marker_linestyle]
    elif isinstance(marker_linestyle, list):
        marker_linestyle = [dict_names_lines.get(x, x) for x in marker_linestyle]
    valid_linestyles = dict(zip(_lines, _names))
    return _check_validity(marker_linestyle, valid_linestyles, 'marker_linestyle', list_cat)


def check_hatches(marker=None, hatch=None, list_cat=None):
    """"""

    valid_hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    # Check if hatch can be chosen
    if hatch is None:
        return [None] * len(list_cat)
    if marker is not None:
        raise ValueError(f"'marker' ('{marker}') must be None if 'hatch' ('{hatch}') is given.")
    # Check if hatch is valid
    if isinstance(hatch, str):
        if hatch not in valid_hatches:
            raise ValueError(f"'hatch' ('{hatch}') must be one of following: {valid_hatches}")
        return [hatch] * len(list_cat)
    # Check if hatch match with categories
    if isinstance(hatch, list):
        wrong_hatch = [x for x in hatch if x not in valid_hatches]
        if len(wrong_hatch) != 0:
            raise ValueError(
                f"'hatch' contains wrong values ('{wrong_hatch}')! Must be one of following: {valid_hatches}")
        if len(hatch) != len(list_cat):
            raise ValueError(f"Length must match of 'hatch' ({hatch}) and categories ({list_cat}).")
        else:
            return hatch

# Helper function
def _create_marker(color, label, marker, marker_size, lw, edgecolor, linestyle, hatch):
    """Create custom marker based on input."""
    if marker is None:
        args = {'facecolor': color,
                'label': label,
                'lw': lw,
                'hatch': hatch,
                "edgecolor": edgecolor}
        return mpl.patches.Patch(**args)
    # If marker is '-', treat it as a line
    if marker == "-":
        if lw <= 0:
            raise ValueError("'lw' should not be greater 0 if 'marker' is a line ('-').")
        return plt.Line2D([0, 1], [0, 1], color=color, linestyle=linestyle, lw=lw, label=label)


    return plt.Line2D(xdata=[0], ydata=[0],
                      marker=marker,
                      color=edgecolor,
                      markerfacecolor=color,
                      markersize=marker_size,
                      label=label,
                      lw=0,
                      markeredgewidth=lw)


# II Main function
def plot_set_legend(ax: Optional[plt.Axes] = None,
                    # Categories and colors
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
                    marker_size: Union[int, float, List[Union[int, float]]] = 10,
                    marker_linestyle: Union[str, list] = "-",
                    hatch: Optional[str] = None,
                    # Title
                    title: str = None,
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
        Marker for legend items. '-' is added to the
    marker_size
        Marker size for legend items.
    marker_linestyle
        Marker linestyle of legend items.
    hatch
        Filling pattern for default marker.
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
    * `Hatches <https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html>`_ filling patterns.
    * :func:`matplotlib.lines.Line2D` for available marker shapes and line properties.
    * :func:`matplotlib.axes.Axes`, which is the core object in matplotlib.
    * :func:`matplotlib.pyplot.gca` to get the current Axes instance.
    """
    # Check input
    ut.check_ax(ax=ax, accept_none=True)
    if ax is None:
        ax = plt.gca()

    ut.check_dict(name="dict_color", val=dict_color, accept_none=False)
    list_cat = check_list_cat(dict_color=dict_color, list_cat=list_cat)
    labels = check_labels(list_cat=list_cat, labels=labels)
    hatch = check_hatches(marker=marker, hatch=hatch, list_cat=list_cat) # Must be before check_marker !

    marker = check_marker(marker=marker, list_cat=list_cat, lw=lw)
    marker_size = check_marker_size(marker_size, list_cat=list_cat)
    marker_linestyle = check_linestyle(marker_linestyle=marker_linestyle, list_cat=list_cat)

    ut.check_bool(name="title_align_left", val=title_align_left)
    ut.check_bool(name="loc_out", val=loc_out)

    ut.check_number_range(name="ncol", val=ncol, min_val=1, accept_none=True, just_int=True)
    args_float = {"x": x, "y": y,
                  "labelspacing": labelspacing, "columnspacing": columnspacing,
                  "handletextpad": handletextpad, "handlelength": handlelength,
                  "fontsize": fontsize, "fontsize_legend": fontsize_title,
                  "lw": lw}
    for key in args_float:
        ut.check_number_val(name=key, val=args_float[key], accept_none=True, just_int=False)

    # Remove existing legend
    if ax.get_legend() is not None and len(ax.get_legend().get_lines()) > 0:
        ax.legend_.remove()

    # Update legend arguments
    edgecolor = "white" if edgecolor is None else edgecolor
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

    # Adjust x and y
    if loc_out:
        # Set values under legend
        x = 0 if x is None else x
        y = -0.25 if y is None else y

    if x is not None or y is not None:
        # Set default bbox values
        x = 1 if x is None else x
        y = 1 if y is None else y
        args["bbox_to_anchor"] = (x, y)

    # Generate legend items
    handles = [_create_marker(dict_color[cat], labels[i], marker[i], marker_size[i], lw, edgecolor, marker_linestyle[i], hatch[i])
               for i, cat in enumerate(list_cat)]
    # Create the legend
    legend = ax.legend(handles=handles, labels=labels, **args)
    # Align title if needed
    if title_align_left:
        legend._legend_box.align = "left"
    return ax
