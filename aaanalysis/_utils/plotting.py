"""
This is a script for the backend of the plotting module functions used by other AAanalysis modules.
"""
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import warnings

from .check_type import check_number_range


# I Helper function
def _create_marker(color, label, marker, marker_size, lw, edgecolor, linestyle, hatch, hatchcolor):
    """Create custom marker based on input."""
    # Default marker (matching to plot)
    if marker is None:
        return mpl.patches.Patch(facecolor=color,
                                 label=label,
                                 lw=lw,
                                 hatch=hatch,
                                 edgecolor=hatchcolor)
    # If marker is '-', treat it as a line
    if marker == "-":
         return plt.Line2D(xdata=[0, 1], ydata=[0, 1],
                           color=color,
                           linestyle=linestyle,
                           lw=lw,
                           label=label)
    # Creates marker element without line (lw=0)
    return plt.Line2D(xdata=[0], ydata=[0],
                      marker=marker,
                      label=label,
                      markerfacecolor=color,
                      color=edgecolor,
                      markersize=marker_size,
                      lw=0,
                      markeredgewidth=lw)


# Check functions
def _marker_has(marker, val=None):
    if isinstance(marker, str):
        return marker == val
    elif marker is None:
        return False
    elif isinstance(marker, list):
        return any([x == val for x in marker])
    else:
        raise ValueError(f"'marker' ({marker}) is wrong")


def _marker_has_no(marker, val=None):
    if isinstance(marker, str):
        return marker != val
    elif marker is None:
        return False
    elif isinstance(marker, list):
        return any([x != val for x in marker])
    else:
        raise ValueError(f"'marker' ({marker}) is wrong")


# Checking functions for list inputs
def _check_list_cat(dict_color=None, list_cat=None):
    """Ensure items in list_cat are keys in dict_color and match in length."""
    if not list_cat:
        return list(dict_color.keys())
    if not all(elem in dict_color for elem in list_cat):
        missing_keys = [elem for elem in list_cat if elem not in dict_color]
        raise ValueError(f"The following keys in 'list_cat' are not in 'dict_colors': {', '.join(missing_keys)}")
    if len(dict_color) < len(list_cat):
        raise ValueError(
            f"'dict_colors' (n={len(dict_color)}) must contain >= elements than 'list_cat' (n={len(list_cat)}).")
    return list_cat


def _check_labels(list_cat=None, labels=None):
    """Validate labels and match their length to list_cat."""
    if labels is None:
        labels = list_cat
    if len(list_cat) != len(labels):
        raise ValueError(f"Length must match of 'labels' ({len(labels)}) and categories ({len(list_cat)}).")
    return labels


# Checking functions for inputs that can be list or single values (redundancy accepted for better user communication)
def _check_hatches(marker=None, hatch=None, list_cat=None):
    """Check validity of list_hatche."""
    valid_hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    # Check if hatch is valid
    if isinstance(hatch, str):
        if hatch not in valid_hatches:
            raise ValueError(f"'hatch' ('{hatch}') must be one of following: {valid_hatches}")
    if isinstance(hatch, list):
        wrong_hatch = [x for x in hatch if x not in valid_hatches]
        if len(wrong_hatch) != 0:
            raise ValueError(
                f"'hatch' contains wrong values ('{wrong_hatch}')! Should be one of: {valid_hatches}")
        if len(hatch) != len(list_cat):
            raise ValueError(f"Length must match of 'hatch' ({hatch}) and categories ({list_cat}).")  # Check if hatch can be chosen
    # Warn for parameter conflicts
    if _marker_has_no(marker, val=None) and hatch:
        warnings.warn(f"'hatch' can only be applied to the default marker, set 'marker=None'.", UserWarning)
    # Create hatch list
    list_hatch = [hatch] * len(list_cat) if not isinstance(hatch, list) else hatch
    return list_hatch


def _check_marker(marker=None, list_cat=None, lw=0):
    """Check validity of markers"""
    # Add '-' for line and None for default marker
    valid_markers = [None, "-"] + list(mlines.Line2D.markers.keys())
    # Check if marker is valid
    if not isinstance(marker, list) and marker not in valid_markers:
        raise ValueError(f"'marker' ('{marker}') must be one of following: {valid_markers}")
    if isinstance(marker, list):
        wrong_markers = [x for x in marker if x not in valid_markers]
        if len(wrong_markers) != 0:
            raise ValueError(f"'marker' contains wrong values  ('{wrong_markers}'). Should be one of: {valid_markers}")
        if len(marker) != len(list_cat):
            raise ValueError(f"Length must match of 'marker' ({marker}) and categories ({list_cat}).")
    # Warn for parameter conflicts
    if _marker_has(marker, val="-") and lw <= 0:
        warnings.warn(f"Marker lines ('-') are only shown if 'lw' ({lw}) is > 0.", UserWarning)
    # Create marker list
    list_marker = [marker] * len(list_cat) if not isinstance(marker, list) else marker
    return list_marker


def _check_marker_size(marker_size=10, list_cat=None):
    """Check size of markers"""
    # Check if marker_size is valid
    if isinstance(marker_size, (int, float)):
        check_number_range(name='marker_size', val=marker_size, min_val=0, accept_none=True, just_int=False)
    elif isinstance(marker_size, list):
        for i in marker_size:
            check_number_range(name='marker_size', val=i, min_val=0, accept_none=True, just_int=False)
    elif isinstance(marker_size, list) and len(marker_size) != len(list_cat):
        raise ValueError(f"Length must match of 'marker_size' (marker_size) and categories ({list_cat}).")
    else:
        raise ValueError(f"'marker_size' has wrong data type: {type(marker_size)}")
    # Create marker_size list
    list_marker_size = [marker_size] * len(list_cat) if not isinstance(marker_size, list) else marker_size
    return list_marker_size


def _check_linestyle(linestyle=None, list_cat=None, marker=None):
    """Check validity of linestyle."""
    _lines = ['-', '--', '-.', ':', ]
    _names = ["solid", "dashed", "dashed-doted", "dotted"]
    valid_mls = _lines + _names
    # Check if marker_linestyle is valid
    if isinstance(linestyle, list):
        wrong_mls = [x for x in linestyle if x not in valid_mls]
        if len(wrong_mls) != 0:
            raise ValueError(
                f"'marker_linestyle' contains wrong values ('{wrong_mls}')! Should be one of: {valid_mls}")
        if len(linestyle) != len(list_cat):
            raise ValueError(f"Length must match of 'marker_linestyle' ({linestyle}) and categories ({list_cat}).")
    # Check if marker_linestyle is conflicting with other settings
    if isinstance(linestyle, str):
        if linestyle not in valid_mls:
            raise ValueError(f"'marker_linestyle' ('{linestyle}') must be one of following: {_lines},"
                             f" or corresponding names: {_names} ")
    # Warn for parameter conflicts
    if linestyle is not None and _marker_has_no(marker, val="-"):
        warnings.warn(f"'linestyle' ({linestyle}) is only applicable to marker lines ('-'), not to '{marker}'.", UserWarning)
    # Create list_marker_linestyle list
    list_marker_linestyle = [linestyle] * len(list_cat) if not isinstance(linestyle, list) else linestyle
    return list_marker_linestyle


# II Main Functions
# DEV: General function for plot_gcfs
def plot_gco(option='font.size', show_options=False):
    """Get current option from plotting context"""
    current_context = sns.plotting_context()
    if show_options:
        print(current_context)
    try:
        option_value = current_context[option]  # Typically font_size
    except KeyError:
        options = list(current_context.keys())
        raise ValueError(f"Option not valid, select from the following: {options}")
    return option_value


# DEV: plot_get_cdict and plot_get_cmap are implemented in main utils
# Remaining backend plotting functions
def plot_get_clist_(n_colors=3):
    """Get manually curated list of 2 to 9 colors."""
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
    else:
        return sns.color_palette(palette="husl", n_colors=n_colors)


def plot_legend_(ax=None, dict_color=None, list_cat=None, labels=None,
                 loc="upper left", loc_out=False, y=None, x=None, n_cols=3,
                 labelspacing=0.2, columnspacing=1.0, handletextpad=0.8, handlelength=2.0,
                 fontsize=None, fontsize_title=None, weight_font="normal", weight_title="normal",
                 marker=None, marker_size=10, lw=0, linestyle=None, edgecolor=None,
                 hatch=None, hatchcolor="white", title=None, title_align_left=True,
                 frameon=False, keep_legend=False, **kwargs):
    """Sets an independently customizable plot legend"""
    # Check input
    if ax is None:
        ax = plt.gca()
    list_cat = _check_list_cat(dict_color=dict_color, list_cat=list_cat)
    labels = _check_labels(list_cat=list_cat, labels=labels)
    marker = _check_marker(marker=marker, list_cat=list_cat, lw=lw)
    hatch = _check_hatches(marker=marker, hatch=hatch, list_cat=list_cat)
    linestyle = _check_linestyle(linestyle=linestyle, list_cat=list_cat, marker=marker)
    marker_size = _check_marker_size(marker_size=marker_size, list_cat=list_cat)
    # Save or remove existing legend
    old_legend = ax.get_legend() if keep_legend else None
    if ax.get_legend() is not None and len(ax.get_legend().get_lines()) > 0:
        ax.legend_.remove()
    # Update legend arguments
    args = dict(loc=loc, ncol=n_cols, fontsize=fontsize, labelspacing=labelspacing, columnspacing=columnspacing,
                handletextpad=handletextpad, handlelength=handlelength, borderpad=0, title=title,
                edgecolor=edgecolor, prop={"weight": weight_font, "size": fontsize}, frameon=frameon)
    args.update(kwargs)
    if fontsize_title:
        args["title_fontproperties"] = {"weight": weight_title, "size": fontsize_title}
    else:
        args["title_fontproperties"] = {"weight": weight_title}
    if loc_out:
        x, y = x or 0, y or -0.25
    if x or y:
        args["bbox_to_anchor"] = (x or 0, y or 1)
    # Create handles and legend
    handles = [_create_marker(dict_color[cat], labels[i], marker[i], marker_size[i],
                              lw, edgecolor, linestyle[i], hatch[i], hatchcolor)
               for i, cat in enumerate(list_cat)]
    # Create new legend
    legend = ax.legend(handles=handles, labels=labels, **args)
    if title_align_left:
        legend._legend_box.align = "left"
    # Add the legend as an artist (must be inside plot)
    if keep_legend and old_legend:
        ax.add_artist(old_legend)
    return ax
