"""
Plotting utility functions for AAanalysis to create publication ready figures. Can
be used for any Python project independently of AAanalysis.
"""
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import aaanalysis.utils as ut
from typing import List, Union, Tuple
import warnings

LIST_FONTS = ['Arial', 'Avant Garde',
              'Bitstream Vera Sans', 'Computer Modern Sans Serif',
              'DejaVu Sans', 'Geneva',
              'Helvetica', 'Lucid',
              'Lucida Grande', 'Verdana']


# I Helper functions
# Check plot_settings
def check_font(font="Arial"):
    """"""
    if font not in LIST_FONTS:
        error_message = f"'font' ({font}) not in recommended fonts: {LIST_FONTS}. Set font manually by:" \
                        f"\n\tplt.rcParams['font.sans-serif'] = '{font}'"
        raise ValueError(error_message)


def check_fig_format(fig_format="pdf"):
    """"""
    list_fig_formats = ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps',
                        'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', 'webp']
    ut.check_str(name="fig_format", val=fig_format)
    if fig_format not in list_fig_formats:
        raise ValueError(f"'fig_format' should be one of following: {list_fig_formats}")


def check_grid_axis(grid_axis="y"):
    list_grid_axis = ["y", "x", "both"]
    if grid_axis not in list_grid_axis:
        raise ValueError(f"'grid_axis' ({grid_axis}) should be one of following: {list_grid_axis}")


# Check plot_set_legend
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


# Get color maps
def _get_cpp_cmap(n_colors=100, facecolor_dark=False):
    """Generate a diverging color map for CPP feature values."""
    ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=2, just_int=True)
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


# II Main functions
# Plotting colors
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
        ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=3, just_int=True)
        return _get_shap_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    elif name == ut.STR_CMAP_CPP:
        ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=3, just_int=True)
        return _get_cpp_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    elif name == ut.STR_CMAP_CAT:
        ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=2, max_val=9, just_int=True)
        return _get_tab_color(n_colors=n_colors)


def plot_get_cdict(name: str = "DICT_COLOR") -> dict:
    """
    Returns color dictionaries specified for AAanalysis.

    Parameters
    ----------
    name
        The name of the AAanalysis color dictionary.

         - ``DICT_COLOR``: Dictionary with default colors for plots.
         - ``DICT_CAT``: Dictionary with default colors for scale categories.

    Returns
    -------
    dict
       AAanalysis color dictionary.

    Examples
    --------
    >>> import aaanalysis as aa
    >>> dict_color = aa.plot_get_cdict(name="DICT_COLOR")

    """
    list_names = [ut.STR_DICT_COLOR, ut.STR_DICT_CAT]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    if name == ut.STR_DICT_COLOR:
        return ut.DICT_COLOR
    else:
        return ut.DICT_COLOR_CAT


# Plotting adjuster
def plot_settings(font_scale: float = 1,
                  font: str = "Arial",
                  fig_format: str = "pdf",
                  weight_bold: bool = True,
                  adjust_only_font: bool = False,
                  adjust_further_elements: bool = True,
                  grid: bool = False,
                  grid_axis: str = "y",
                  no_ticks: bool = False,
                  short_ticks: bool = False,
                  no_ticks_x: bool = False,
                  short_ticks_x: bool = False,
                  no_ticks_y: bool = False,
                  short_ticks_y: bool = False,
                  show_options: bool = False) -> None:
    """
    Configure general settings for plot visualization with various customization options.

    This function modifies the global settings of :mod:`matplotlib` and :mod:`seaborn` libraries.
    PDFs are embedded such that they can be edited using image editing software.

    Parameters
    ----------
    font_scale
       Scaling factor to scale the size of font elements. Consistent with :func:`seaborn.set_context`.
    font
       Name of text font. Common options are 'Arial', 'Verdana', 'Helvetica', or 'DejaVu Sans' (Matplotlib default).
    fig_format
       Specifies the file format for saving plots. Most backends support png, pdf, ps, eps and svg.
    weight_bold
       If ``True``, font and line elements are bold.
    adjust_only_font
       If ``True``, only the font style will be adjusted, leaving other elements unchanged.
    adjust_further_elements
       If ``True``, makes additional visual and layout adjustments to the plot (errorbars, legend).
    grid
       If ``True``, display the grid in plots.
    grid_axis
       Choose the axis ('y', 'x', 'both') to apply the grid to.
    no_ticks
       If ``True``, remove all tick marks on both x and y axes.
    short_ticks
       If ``True``, display short tick marks on both x and y axes. Is ignored if ``no_ticks=True``.
    no_ticks_x
       If ``True``, remove tick marks on the x-axis.
    short_ticks_x
       If ``True``, display short tick marks on the x-axis. Is ignored if ``no_ticks=True``.
    no_ticks_y
       If ``True``, remove tick marks on the y-axis.
    short_ticks_y
       If ``True``, display short tick marks on the y-axis. Is ignored if ``no_ticks=True``.
    show_options
       If ``True``, show all plot runtime configurations of matplotlib.

    Examples
    --------
    Create default seaborn plot:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
        >>> sns.barplot(x='Classes', y='Values', data=data)
        >>> sns.despine()
        >>> plt.title("Seaborn default")
        >>> plt.tight_layout()
        >>> plt.show()

    Adjust polts with AAanalysis:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns
        >>> import aaanalysis as aa
        >>> data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
        >>> colors = aa.plot_get_cmap(name="CAT", n_colors=3)
        >>> aa.plot_settings()
        >>> sns.barplot(x='Classes', y='Values', data=data, palette=colors)
        >>> sns.despine()
        >>> plt.title("Adjusted")
        >>> plt.tight_layout()
        >>> plt.show()

    See Also
    --------
    * :func:`seaborn.set_context`, where ``font_scale`` is utilized.
    * :data:`matplotlib.rcParams`, which manages the global settings in :mod:`matplotlib`.
    """
    # Check input
    ut.check_non_negative_number(name="font_scale", val=font_scale, min_val=0, just_int=False)
    check_font(font=font)
    check_fig_format(fig_format=fig_format)
    check_grid_axis(grid_axis=grid_axis)
    args_bool = {"weight_bold": weight_bold, "adjust_only_font": adjust_only_font,
                 "adjust_further_elements": adjust_further_elements, "grid": grid,
                 "short_ticks": short_ticks, "short_ticks_x": short_ticks_x, "short_ticks_y": short_ticks_y,
                 "no_ticks": no_ticks, "no_ticks_y": no_ticks_y, "no_ticks_x": no_ticks_x,
                 "show_options": show_options,}
    for key in args_bool:
        ut.check_bool(name=key, val=args_bool[key])

    # Warning
    if no_ticks and any([short_ticks, short_ticks_x, short_ticks_y]):
        warnings.warn("`no_ticks` is set to True, so 'short_ticks' parameters will be ignored.")
    if no_ticks_x and short_ticks_x:
        warnings.warn("`no_ticks_x` is set to True, so 'short_ticks_x' will be ignored.")
    if no_ticks_y and short_ticks_y:
        warnings.warn("`no_ticks_y` is set to True, so 'short_ticks_y' will be ignored.")

    # Print all plot settings/runtime configurations of matplotlib
    if show_options:
        print(plt.rcParams.keys)

    # Set embedded fonts in PDF
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["pdf.fonttype"] = 42

    # Change only font style
    if adjust_only_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = font
        return

    # Apply all changes
    sns.set_context("talk", font_scale=font_scale)
    # Font settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font
    font_settings = {'family': 'sans-serif', "weight": "bold"} if weight_bold else {'family': 'sans-serif'}
    mpl.rc('font', **font_settings)
    # Grid
    plt.rcParams["axes.grid.axis"] = grid_axis
    plt.rcParams["axes.grid"] = grid
    # Adjust weight of text and lines
    if weight_bold:
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
    else:
        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["xtick.minor.width"] = 0.6
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["ytick.minor.width"] = 0.6
    # Handle tick options (short are default matplotlib options, otherwise from seaborn)
    if short_ticks or short_ticks_x:
        plt.rcParams["xtick.major.size"] = 3.5
        plt.rcParams["xtick.minor.size"] = 2
    if short_ticks or short_ticks_y:
        plt.rcParams["ytick.major.size"] = 3.5
        plt.rcParams["ytick.minor.size"] = 2
    if no_ticks or no_ticks_x:
        plt.rcParams["xtick.major.size"] = 0
        plt.rcParams["xtick.minor.size"] = 0
    if no_ticks or no_ticks_y:
        plt.rcParams["ytick.major.size"] = 0
        plt.rcParams["ytick.minor.size"] = 0
    # Handle figure format
    if fig_format == "pdf":
        mpl.rcParams['pdf.fonttype'] = 42
    elif "svg" in fig_format:
        mpl.rcParams['svg.fonttype'] = 'none'
    # Additional adjustments
    if adjust_further_elements:
        # Error bars
        plt.rcParams["errorbar.capsize"] = 10
        # Legend
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.fontsize"] = "medium"
        plt.rcParams["legend.loc"] = 'upper right'


def plot_gcfs():
    """Get current font size, which is set by :func:`plot_settings` function."""
    # Get the current plotting context
    current_context = sns.plotting_context()
    font_size = current_context['font.size']
    return font_size

# TODO check, interface, doc, test
def plot_set_legend(ax=None, handles=None, dict_color=None, list_cat=None, labels=None, y=-0.2, x=0.5, ncol=3,
                    fontsize=11, weight="normal", lw=0, edgecolor=None, return_handles=False, loc="upper left",
                    labelspacing=0.2, columnspacing=1, title=None, fontsize_legend=None, title_align_left=True,
                    fontsize_weight="normal", shape=None, **kwargs):
    """
    Set a customizable legend for a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, default=None
        The axes to attach the legend to.
    handles : list, default=None
        Handles for legend items.
    dict_color : dict, default=None
        A dictionary mapping categories to colors.
    list_cat : list, default=None
        List of categories to include in the legend.
    labels : list, default=None
        Labels for legend items.
    y : float, default=-0.2
        The y-coordinate for the legend's anchor point.
    x : float, default=0.5
        The x-coordinate for the legend's anchor point.
    ncol : int, default=3
        Number of columns in the legend.
    fontsize : int, default=11
        Font size for the legend text.
    weight : str, default='normal'
        Weight of the font.
    lw : float, default=0
        Line width for legend items.
    edgecolor : color, default=None
        Edge color for legend items.
    return_handles : bool, default=False
        Whether to return handles and labels.
    loc : str, default='upper left'
        Location for the legend.
    labelspacing : float, default=0.2
        Vertical spacing between legend items.
    columnspacing : int, default=1
        Horizontal spacing between legend columns.
    title : str, default=None
        Title for the legend.
    fontsize_legend : int, default=None
        Font size for the legend title.
    title_align_left : bool, default=True
        Whether to align the title to the left.
    fontsize_weight : str, default='normal'
        Font weight for the legend title.
    shape : str, default=None
        Marker shape for legend items.
    **kwargs : dict
        Additional arguments passed directly to ax.legend() for finer control.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the legend applied.

    See Also
    --------
    matplotlib.pyplot.legend : For additional details on how the 'loc' parameter can be customized.
    matplotlib.lines.Line2D : For additional details on the different types of marker shapes ('shape' parameter).

    Examples
    --------
    >>> import aaanalysis as aa
    >>> aa.plot_set_legend(ax=ax, dict_color={'Cat1': 'red', 'Cat2': 'blue'}, shape='o')
    """
    # Check input
    if ax is None:
        ax = plt.gca()
    list_cat = check_cats(list_cat=list_cat, dict_color=dict_color, labels=labels)
    args_float = {"y": y, "x": x, "lw": lw, "labelspacing": labelspacing,
                  "columnspacing": columnspacing}
    for key in args_float:
        ut.check_float(name=key, val=args_float[key])
    ut.check_non_negative_number(name="ncol", val=ncol, min_val=1, just_int=True, accept_none=False)
    ut.check_non_negative_number(name="ncol", val=ncol, min_val=0, just_int=False, accept_none=True)
    ut.check_bool(name="return_handles", val=return_handles)
    ut.check_bool(name="title_align_left", val=title_align_left)

    # TODO check other args
    # Prepare the legend handles
    dict_leg = {cat: dict_color[cat] for cat in list_cat}
    # Generate function for legend markers based on provided shape
    if shape is None:
        if edgecolor is None:
            f = lambda l, c: mpl.patches.Patch(facecolor=l, label=c, lw=lw, edgecolor=l)
        else:
            f = lambda l, c: mpl.patches.Patch(facecolor=l, label=c, lw=lw, edgecolor=edgecolor)
    else:
        f = lambda l, c: plt.Line2D([0], [0], marker=shape, color='w', markerfacecolor=l, markersize=10, label=c)
    # Create handles if not provided
    handles = [f(l, c) for c, l in dict_leg.items()] if handles is None else handles
    # Return handles and labels if required
    if return_handles:
        return handles, labels
    # Prepare labels and args
    if labels is None:
        labels = list(dict_leg.keys())
    args = dict(prop={"weight": weight, "size": fontsize}, **kwargs)
    if fontsize_legend is not None:
        args["title_fontproperties"] = {"weight": fontsize_weight, "size": fontsize_legend}
    # Create the legend
    legend = ax.legend(handles=handles, labels=labels, bbox_to_anchor=(x, y), ncol=ncol, loc=loc,
                       labelspacing=labelspacing, columnspacing=columnspacing, borderpad=0, **args, title=title)
    # Align the title if required
    if title_align_left:
        legend._legend_box.align = "left"
    return ax
