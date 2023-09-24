"""
Plotting utility functions for AAanalysis to create publication ready figures. Can
be used for any Python project independently of AAanalysis.
"""
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import aaanalysis.utils as ut
from typing import List, Union, Tuple

STR_CMAP_CPP = "CPP"
STR_CMAP_SHAP = "SHAP"
STR_CMAP_TAB = "TAB"
STR_DICT_COLOR = "DICT_COLOR"
STR_DICT_CAT = "DICT_CAT"

LIST_FONTS = ['Arial', 'Avant Garde',
              'Bitstream Vera Sans', 'Computer Modern Sans Serif',
              'DejaVu Sans', 'Geneva',
              'Helvetica', 'Lucid',
              'Lucida Grande', 'Verdana']


# Helper functions
# Check plot_settings
def check_font_style(font="Arial"):
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
    ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=2)
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


# Default plotting function
def plot_get_cmap(name: str = "CPP",
                  n_colors: int = 100,
                  facecolor_dark: bool = False
                  ) -> Union[List[Tuple[float, float, float]], List[str]]:
    """
    Returns color maps specified for AAanalysis.

    Parameters
    ----------
    name
        Name of the AAanalysis color palettes:
         - 'CPP': Continuous color map for CPP plots (with gap at center).
         - 'SHAP': Continuous color map for CPP-SHP plots (with gap at center).
         - 'TAB': List of Tableau (tab) colors for appealing visualization of categories.
    n_colors
        Number of colors in the color map. Must be >=2 for 'CPP' and 'SHAP' and 2-9 for 'TAB'.
    facecolor_dark
        Whether to use a dark face color for 'CPP' and 'SHAP'.

    Returns
    -------
    list
        List with colors given as RGB tuples (for 'CPP' and 'SHAP') or matplotlib color names (for 'TAB').

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> import aaanalysis as aa
    >>> colors = plot_get_cmap(name="TAB", n_colors=3)
    >>> df_seq = aa.load_dataset(name="SEQ_AMYLO", n=100)
    >>> data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 45, 17]}
    >>> sns.barplot(x='Classes', y='Values', data=data, palette=colors)
    >>> plt.show()

    See Also
    --------
    * Example notebooks in `Plotting Prelude <plotting_prelude.html>`_.
    * sns.color_palette function to generate a color palette in seaborn.
    * sns.light_palette function to generate a lighter color palettes.
    * `Matplotlib color names <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
    """
    # Check input
    list_names = [STR_CMAP_CPP, STR_CMAP_SHAP, STR_CMAP_TAB]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    ut.check_bool(name="facecolor_dark", val=facecolor_dark)

    # Get color maps
    if name == STR_CMAP_SHAP:
        ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=3)
        return _get_shap_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    elif name == STR_CMAP_CPP:
        ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=3)
        return _get_cpp_cmap(n_colors=n_colors, facecolor_dark=facecolor_dark)
    elif name == STR_CMAP_TAB:
        ut.check_non_negative_number(name="n_colors", val=n_colors, min_val=2, max_val=9)
        return _get_tab_color(n_colors=n_colors)


def plot_get_cdict(name: str = "DICT_COLOR") -> dict:
    """
    Returns color dictionaries specified for AAanalysis.

    Parameters
    ----------
    name
        Name of the AAanalysis color dictionary:
         - 'DICT_COLOR': Dictionary with default colors for plots.
         - 'DICT_CAT': Dictionary with default colors for scale categories.

    Returns
    -------
    dict
       Specific AAanalysis color dictionary.

    Examples
    --------
    >>> import aaanalysis as aa
    >>> dict_color = aa.plot_get_cdict(name="DICT_COLOR")

    """
    list_names = [STR_DICT_COLOR, STR_DICT_CAT]
    if name not in list_names:
        raise ValueError(f"'name' must be one of following: {list_names}")
    if name == STR_DICT_COLOR:
        return ut.DICT_COLOR
    else:
        return ut.DICT_COLOR_CAT

# TODO check, interface, doc, test
def plot_settings(fig_format="pdf", verbose=False, grid=False, grid_axis="y",
                  font_scale=0.7, font="Arial",
                  change_size=True, weight_bold=True, adjust_elements=True,
                  short_ticks=False, no_ticks=False,
                  no_ticks_y=False, short_ticks_y=False, no_ticks_x=False, short_ticks_x=False):
    """
    Configure general settings for plot visualization with various customization options.

    Parameters
    ----------
    fig_format : str, default='pdf'
        Specifies the file format for saving the plot.
    verbose : bool, default=False
        If True, enables verbose output.
    grid : bool, default=False
        If True, makes the grid visible.
    grid_axis : str, default='y'
        Choose the axis ('y', 'x', 'both') to apply the grid to.
    font_scale : float, default=0.7
        Sets the scale for font sizes in the plot.
    font : str, default='Arial'
        Name of sans-serif font (e.g., 'Arial', 'Verdana', 'Helvetica', 'DejaVu Sans')
    change_size : bool, default=True
        If True, adjusts the size of plot elements.
    weight_bold : bool, default=True
        If True, text elements appear in bold.
    adjust_elements : bool, default=True
        If True, makes additional visual and layout adjustments to the plot.
    short_ticks : bool, default=False
        If True, uses short tick marks.
    no_ticks : bool, default=False
        If True, removes all tick marks.
    no_ticks_y : bool, default=False
        If True, removes tick marks on the y-axis.
    short_ticks_y : bool, default=False
        If True, uses short tick marks on the y-axis.
    no_ticks_x : bool, default=False
        If True, removes tick marks on the x-axis.
    short_ticks_x : bool, default=False
        If True, uses short tick marks on the x-axis.

    Notes
    -----
    This function modifies the global settings of Matplotlib and Seaborn libraries.

    Examples
    --------
    >>> import aaanalysis as aa
    >>> aa.plot_settings(fig_format="pdf", font_scale=1.0, weight_bold=False)
    """
    # Check input
    check_fig_format(fig_format=fig_format)
    check_font_style(font=font)
    check_grid_axis(grid_axis=grid_axis)
    args_bool = {"verbose": verbose, "grid": grid, "change_size": change_size, "weight_bold": weight_bold,
                 "adjust_elements": adjust_elements,
                 "short_ticks": short_ticks, "no_ticks": no_ticks, "no_ticks_y": no_ticks_y,
                 "short_ticks_y": short_ticks_y, "no_ticks_x": no_ticks_x, "short_ticks_x": short_ticks_x}
    for key in args_bool:
        ut.check_bool(name=key, val=args_bool[key])
    ut.check_non_negative_number(name="font_scale", val=font_scale, min_val=0, just_int=False)

    # Set embedded fonts in PDF
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["pdf.fonttype"] = 42
    if verbose:
        print(plt.rcParams.keys)    # Print all plot settings that can be modified in general
    if not change_size:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = font
        mpl.rc('font', **{'family': font})
        return
    sns.set_context("talk", font_scale=font_scale)  # Font settings https://matplotlib.org/3.1.1/tutorials/text/text_props.html
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font
    if weight_bold:
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
    else:
        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["xtick.minor.width"] = 0.6
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["ytick.minor.width"] = 0.6
    if short_ticks:
        plt.rcParams["xtick.major.size"] = 3.5
        plt.rcParams["xtick.minor.size"] = 2
        plt.rcParams["ytick.major.size"] = 3.5
        plt.rcParams["ytick.minor.size"] = 2
    if short_ticks_x:
        plt.rcParams["xtick.major.size"] = 3.5
        plt.rcParams["xtick.minor.size"] = 2
    if short_ticks_y:
        plt.rcParams["ytick.major.size"] = 3.5
        plt.rcParams["ytick.minor.size"] = 2
    if no_ticks:
        plt.rcParams["xtick.major.size"] = 0
        plt.rcParams["xtick.minor.size"] = 0
        plt.rcParams["ytick.major.size"] = 0
        plt.rcParams["ytick.minor.size"] = 0
    if no_ticks_x:
        plt.rcParams["xtick.major.size"] = 0
        plt.rcParams["xtick.minor.size"] = 0
    if no_ticks_y:
        plt.rcParams["ytick.major.size"] = 0
        plt.rcParams["ytick.minor.size"] = 0

    plt.rcParams["axes.labelsize"] = 17 #13.5
    plt.rcParams["axes.titlesize"] = 16.5 #15
    if fig_format == "pdf":
        mpl.rcParams['pdf.fonttype'] = 42
    elif "svg" in fig_format:
        mpl.rcParams['svg.fonttype'] = 'none'
    font = {'family': font, "weight": "bold"} if weight_bold else {"family": font}
    mpl.rc('font', **font)
    if adjust_elements:
        # Error bars
        plt.rcParams["errorbar.capsize"] = 10   # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
        # Grid
        plt.rcParams["axes.grid.axis"] = grid_axis  # 'y', 'x', 'both'
        plt.rcParams["axes.grid"] = grid
        # Legend
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.fontsize"] = "medium" #"x-small"
        plt.rcParams["legend.loc"] = 'upper right'  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html


def plot_gcfs():
    """Get current font size, which is set by ``plot_settings`` function"""
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
