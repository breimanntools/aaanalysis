"""
This is a script for the backend of the plotting module functions used by other AAanalysis modules.
"""
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt


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


# II Main Functions
# DEV: General function for plot_gcfs
def plot_gco(option='font.size', show_options=False):
    """Get current option from plotting context"""
    current_context = sns.plotting_context()
    if show_options:
        print(current_context)
    option_value = current_context[option]  # Typically font_size
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
                loc="upper left", loc_out=False, y=None, x=None, ncol=3,
                labelspacing=0.2, columnspacing=1.0, handletextpad=0.8, handlelength=2.0,
                fontsize=None, fontsize_title=None, weight_font="normal", weight_title="normal",
                marker=None, marker_size=10, lw=0, linestyle=None, edgecolor=None,
                hatch=None, hatchcolor="white", title=None, title_align_left=True,
                **kwargs):
    """Sets an independently customizable plot legend"""
    # Remove existing legend
    if ax.get_legend() is not None and len(ax.get_legend().get_lines()) > 0:
        ax.legend_.remove()

    # Update legend arguments
    args = dict(loc=loc, ncol=ncol, fontsize=fontsize, labelspacing=labelspacing, columnspacing=columnspacing,
                handletextpad=handletextpad, handlelength=handlelength, borderpad=0, title=title,
                edgecolor=edgecolor, prop={"weight": weight_font, "size": fontsize})
    args.update(kwargs)

    if fontsize_title:
        args["title_fontproperties"] = {"weight": weight_title, "size": fontsize_title}

    if loc_out:
        x, y = x or 0, y or -0.25
    if x or y:
        args["bbox_to_anchor"] = (x or 0, y or 1)

    # Create handles and legend
    handles = [_create_marker(dict_color[cat], labels[i], marker[i], marker_size[i],
                              lw, edgecolor, linestyle[i], hatch[i], hatchcolor)
               for i, cat in enumerate(list_cat)]

    legend = ax.legend(handles=handles, labels=labels, **args)
    if title_align_left:
        legend._legend_box.align = "left"
    return ax






