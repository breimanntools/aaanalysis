#! /usr/bin/python3
"""
Default plotting functions
"""
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import aaanalysis.utils as ut



LIST_AA_COLOR_PALETTES = ["FEAT", "SHAP", "GGPLOT"]
LIST_AA_COLOR_DICTS = ["DICT_SCALE_CAT", "DICT_COLOR"]
LIST_AA_COLORS = LIST_AA_COLOR_PALETTES + LIST_AA_COLOR_DICTS

LIST_FONTS = ['Arial', 'Avant Garde', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'DejaVu Sans',
              'Geneva', 'Helvetica', 'Lucid', 'Lucida Grande', 'Verdana']


# Helper functions
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
def _get_shap_cmap(n_colors=100, facecolor_dark=True):
    """Generate a diverging color map for feature values."""
    n = 20
    cmap_low = sns.light_palette(ut.COLOR_SHAP_NEG, input="hex", reverse=True, n_colors=int(n_colors/2)+n)
    cmap_high = sns.light_palette(ut.COLOR_SHAP_POS, input="hex", n_colors=int(n_colors/2)+n)
    c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
    cmap = cmap_low[0:-n] + c_middle + cmap_high[n:]
    return cmap


def _get_feat_cmap(n_colors=100, facecolor_dark=False):
    """Generate a diverging color map for feature values."""
    n = 5
    cmap = sns.color_palette("RdBu_r", n_colors=n_colors + n * 2)
    cmap_low, cmap_high = cmap[0:int((n_colors + n * 2) / 2)], cmap[int((n_colors + n * 2) / 2):]
    c_middle = [(0, 0, 0)] if facecolor_dark else [cmap_low[-1]]
    cmap = cmap_low[0:-n] + c_middle + cmap_high[n:]
    return cmap


def _get_ggplot_cmap(n_colors=100):
    """Generate a circular GGplot color palette."""
    cmap = sns.color_palette("husl", n_colors)
    return cmap


def _get_default_colors(name=None, n_colors=100, facecolor_dark=True):
    """Retrieve default color maps based on palette name."""
    args = dict(n_colors=n_colors, facecolor_dark=facecolor_dark)
    if name == "SHAP":
        return _get_shap_cmap(**args)
    elif name == "FEAT":
        return _get_feat_cmap(**args)
    elif name == "GGPLOT":
        return _get_ggplot_cmap(n_colors=n_colors)


def _get_cmap_with_gap(n_colors=100, color_pos=None, color_neg=None, color_center=None, pct_gap=10, pct_center=None,
                       input="hex"):
    """Generate a custom color map with a gap."""
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
def plot_get_cmap(name=None, n_colors=100, facecolor_dark=False,
                  color_pos=None, color_neg=None, color_center=None,
                  input="hex", pct_gap=10, pct_center=None):
    """
    Retrieve color maps or color dictionaries specified for AAanalysis.

    Parameters
    ----------
    name : str, optional
        The name of the color palette to use in AAanalysis. Options include:
         - 'SHAP', 'FEAT', 'GGPLOT': Return color maps for SHAP plots, CPP feature maps/heatmaps,
            and datagrouping as in GGplot, respectively.
         - 'DICT_COLOR', 'DICT_SCALE_CAT': Return default color dictionaries for plots (e.g., bars in CPPPlot.profile)
            and scale categories (e.g., CPPPlot.heatmap), respectively.
    n_colors : int, default=100
        Number of colors in the color map.
    facecolor_dark : bool, default=False
        Whether to use a dark face color for 'SHAP' and 'FEAT'.
    color_pos : str, optional
        Hex code for the positive color.
    color_neg : str, optional
        Hex code for the negative color.
    color_center : str or list, optional
        Hex code or list for the center color.
    input : str, {'rgb', 'hls', 'husl', 'xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.
    pct_gap : int, default=10
        Percentage size of the gap between color ranges.
    pct_center : float, optional
        Percentage size of the center color in the map.

    Returns
    -------
    cmap : list or dict
        If 'name' parameter is 'SHAP', 'FEAT', or 'GGPLOT', a list of colors specified for AAanalysis will be returned.
        If 'name' parameter is None, a list of colors based on provided colors

    See Also
    --------
    sns.color_palette : Function to generate a color palette in seaborn.
    sns.light_palette : Function to generate a lighter color palette in seaborn.
    """
    # TODO check color dict name
    if name in LIST_AA_COLOR_PALETTES:
        cmap = _get_default_colors(name=name, n_colors=n_colors, facecolor_dark=facecolor_dark)
        return cmap
    cmap = _get_cmap_with_gap(n_colors=n_colors, color_pos=color_pos, color_neg=color_neg,
                              color_center=color_center, pct_gap=pct_gap, pct_center=pct_center,
                              input=input)
    return cmap


def plot_get_cdict(name=None):
    """
    Retrieve color dictionaries specified for AAanalysis.

    Parameters
    ----------
    name : str, {'DICT_COLOR', 'DICT_SCALE_CAT'}
        The name of default color dictionaries for plots (e.g., bars in CPPPlot.profile)
        and scale categories (e.g., CPPPlot.heatmap), respectively.

    Returns
    -------
    cmap :  dict
       Specific AAanalysis color dictionary.
    """
    # TODO check color dict name
    color_dict = ut.DICT_COLOR if name == "DICT_COLORS" else ut.DICT_COLOR_CAT
    return color_dict


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
    """Get current font size, which is set by ut.plot_settings function"""
    # Get the current plotting context
    current_context = sns.plotting_context()
    font_size = current_context['font.size']
    return font_size


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
