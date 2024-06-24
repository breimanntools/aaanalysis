"""
Plotting utility functions for AAanalysis to create publication ready figures. Can
be used for any Python project independently of AAanalysis.
"""
from typing import Union
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import aaanalysis.utils as ut
import warnings


LIST_FONTS = ['Arial', 'Courier New', 'DejaVu Sans', 'Times New Roman', 'Verdana']


# I Helper functions
# Check plot_settings
def check_font(font="Arial"):
    if font not in LIST_FONTS:
        error_message = f"'font' ({font}) not in recommended fonts: {LIST_FONTS}. Set font manually by:" \
                        f"\n\tplt.rcParams['font.sans-serif'] = '{font}'"
        raise ValueError(error_message)


# Helper function
def set_tick_size(axis=None, major_size=None, minor_size=None):
    """Set tick size of the given axis."""
    plt.rcParams[f"{axis}tick.major.size"] = major_size
    plt.rcParams[f"{axis}tick.minor.size"] = minor_size


# II Main functions
def plot_settings(font_scale: Union[int, float] = 1,
                  font: str = "Arial",
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
                  show_options: bool = False
                  ) -> None:
    """
    Configure general plot settings.

    This function modifies the global settings of :mod:`matplotlib` and :mod:`seaborn` libraries.
    It adjusts font embedding for vector formats like PDF and SVG, ensuring compatibility and
    editability across various viewers and editing software.

    Parameters
    ----------
    font_scale : int or float, default=1
       Scaling factor to scale the size of font elements. Consistent with :func:`seaborn.set_context`.
    font : {'Arial', 'Courier New', 'DejaVu Sans', 'Times New Roman', 'Verdana'}, default='Arial'
       Name of text font. Common options are 'Arial' or 'DejaVu Sans' (Matplotlib default).
    weight_bold : bool, default=True
       If ``True``, font and line elements are bold.
    adjust_only_font : bool, default=False
       If ``True``, only the font style will be adjusted, leaving other elements unchanged.
    adjust_further_elements : bool, default=True
       If ``True``, makes additional visual and layout adjustments to the plot (errorbars, legend).
    grid : bool, default=False
       If ``True``, display the grid in plots.
    grid_axis : {'y', 'x', 'both'}, default='y'
       Choose the axis ('y', 'x', 'both') to apply the grid to.
    no_ticks : bool, default=False
       If ``True``, remove all tick marks on both x and y axes.
    short_ticks : bool, default=False
       If ``True``, display short tick marks on both x and y axes. Is ignored if ``no_ticks=True``.
    no_ticks_x : bool, default=False
       If ``True``, remove tick marks on the x-axis.
    short_ticks_x : bool, default=False
       If ``True``, display short tick marks on the x-axis. Is ignored if ``no_ticks=True``.
    no_ticks_y : bool, default=False
       If ``True``, remove tick marks on the y-axis.
    short_ticks_y : bool, default=False
       If ``True``, display short tick marks on the y-axis. Is ignored if ``no_ticks=True``.
    show_options : bool, default=False
       If ``True``, show all plot runtime configurations of matplotlib.

    Notes
    -----
    * ``grid_axis`` work only for axis with numerical values.

    See Also
    --------
    * More examples in `Plotting Prelude <plotting_prelude.html>`_.
    * :func:`seaborn.set_context`, where ``font_scale`` is utilized.
    * :data:`matplotlib.rcParams`, which manages the global settings in :mod:`matplotlib`.

    Examples
    --------
    .. include:: examples/plot_settings.rst
    """
    # Check input
    ut.check_number_range(name="font_scale", val=font_scale, min_val=0, just_int=False)
    check_font(font=font)
    ut.check_str_options(name="grid_axis", val=grid_axis,
                         list_str_options=["y", "x", "both"])
    args_bool = {"weight_bold": weight_bold, "adjust_only_font": adjust_only_font,
                 "adjust_further_elements": adjust_further_elements, "grid": grid,
                 "short_ticks": short_ticks, "short_ticks_x": short_ticks_x, "short_ticks_y": short_ticks_y,
                 "no_ticks": no_ticks, "no_ticks_y": no_ticks_y, "no_ticks_x": no_ticks_x,
                 "show_options": show_options,}
    for key in args_bool:
        ut.check_bool(name=key, val=args_bool[key])

    # Warning
    if no_ticks and any([short_ticks, short_ticks_x, short_ticks_y]):
        warnings.warn("`no_ticks` is set to True, so 'short_ticks' parameters will be ignored.", UserWarning)
    if no_ticks_x and short_ticks_x:
        warnings.warn("`no_ticks_x` is set to True, so 'short_ticks_x' will be ignored.", UserWarning)
    if no_ticks_y and short_ticks_y:
        warnings.warn("`no_ticks_y` is set to True, so 'short_ticks_y' will be ignored.", UserWarning)

    # Print all plot settings/runtime configurations of matplotlib
    if show_options:
        ut.print_out(plt.rcParams.keys)

    # Set all values to matplotlib default
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Change only font style
    if adjust_only_font:
        mpl.rcParams['pdf.fonttype'] = 42  # Set embedded fonts in PDF via TrueType
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['svg.fonttype'] = 'none'
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = font
        return

    # Apply all changes
    sns.set_context("talk", font_scale=font_scale)

    # Handle storing of vectorized figure format
    mpl.rcParams['pdf.fonttype'] = 42  # Set embedded fonts in PDF via TrueType
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    # Font settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font
    _weight = "bold" if weight_bold else "normal"
    font_settings = {'family': 'sans-serif', "weight": _weight}
    mpl.rc('font', **font_settings)

    # Grid
    plt.rcParams["axes.grid.axis"] = grid_axis
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.linewidth"] = 1 if weight_bold else 0.8
    plt.rcParams["axes.axisbelow"] = True

    # Adjust weight of text and lines
    if weight_bold:
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
    else:
        plt.rcParams["axes.labelweight"] = "normal"
        plt.rcParams["axes.titleweight"] = "normal"
        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["xtick.minor.width"] = 0.6
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["ytick.minor.width"] = 0.6

    # Tick sizes
    default_major_size = 6
    default_minor_size = 4
    short_major_size = 3.5
    short_minor_size = 2
    no_major_size = 0
    no_minor_size = 0
    # Set x ticks
    if no_ticks or no_ticks_x:
        set_tick_size(axis="x", major_size=no_major_size, minor_size=no_minor_size)
    elif short_ticks or short_ticks_x:
        set_tick_size(axis="x", major_size=short_major_size, minor_size=short_minor_size)
    else:
        set_tick_size(axis="x", major_size=default_major_size, minor_size=default_minor_size)

    # Set y ticks
    if no_ticks or no_ticks_y:
        set_tick_size(axis="y", major_size=no_major_size, minor_size=no_minor_size)
    elif short_ticks or short_ticks_y:
        set_tick_size(axis="y", major_size=short_major_size, minor_size=short_minor_size)
    else:
        set_tick_size(axis="y", major_size=default_major_size, minor_size=default_minor_size)

    # Additional adjustments
    if adjust_further_elements:
        # Error bars
        plt.rcParams["errorbar.capsize"] = 10
        # Legend
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.fontsize"] = "medium"
        plt.rcParams["legend.loc"] = 'upper right'
