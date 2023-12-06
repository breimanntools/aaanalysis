"""
This is a script for internal plotting utility functions used in the backend.
"""
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Helper functions
def _get_color_map(labels, color):
    """Get color map"""
    unique_labels = sorted(set(labels))
    if isinstance(color, list) and len(color) != 1:
        if len(color) < len(unique_labels):
            color *= len(unique_labels)
        color_map = {label: color[i] for i, label in enumerate(unique_labels)}
    else:
        color = color[0] if isinstance(color, list) else color
        color_map = {label: color for label in unique_labels}
    return color_map, unique_labels


def _get_positions_lengths_colors(labels, color_map):
    """Get the positions """
    positions, lengths, colors = [], [], []
    current_label, start_pos = labels[0], 0
    for i, label in enumerate(labels + [None]):
        if label != current_label or i == len(labels):
            positions.append(start_pos)
            length = i - start_pos
            lengths.append(length)
            colors.append(color_map[current_label])
            start_pos = i
            current_label = label
    return positions, lengths, colors

def _get_xy_wh(ax=None, position=None, pos=None, bar_spacing=None, length=None, bar_width=None):
    """Get x and y position together with width and height/length"""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if position == 'bottom':
        y = y_min + bar_spacing
        return (pos, y), (length, bar_width)
    elif position == 'top':
        y = y_max - bar_spacing - bar_width
        return (pos, y), (length, bar_width)
    elif position == 'left':
        x =  x_min - bar_spacing - bar_width
        return (x, pos), (bar_width, length)
    elif position == 'right':
        x = x_max + bar_spacing
        return (x, pos), (bar_width, length)
    else:
        raise ValueError("Position should be 'left', 'right', 'top', or 'bottom'.")

def _get_xy_hava(position=None, xy=None, wh=None, label_spacing_factor=1.5):
    """Get x and y position together with horizontal alignment and vertical alignment"""
    bar_width = wh[1] if position in ['bottom', 'top'] else wh[0]
    if position == 'bottom':
        text_x = xy[0] + wh[0] / 2
        text_y = xy[1] + bar_width * label_spacing_factor
        ha, va = 'center', 'top'
    elif position == 'top':
        text_x = xy[0] + wh[0] / 2
        text_y = xy[1] + wh[1] - bar_width * (label_spacing_factor-0.25)
        ha, va = 'center', 'bottom'
    elif position == 'left':
        text_x = xy[0] - bar_width * (label_spacing_factor-1)
        text_y = xy[1] + wh[1] / 2
        ha, va = 'right', 'center'
    else:  # Assuming position == 'right'
        text_x = xy[0] + wh[0] + bar_width * (label_spacing_factor-1)
        text_y = xy[1] + wh[1] / 2
        ha, va = 'left', 'center'
    return text_x, text_y, ha, va


def _add_bar_labels(ax=None, bar_labels_align=None, position=None, bar_width=None,
                    labels=None, positions=None, lengths=None, bar_labels=None, bar_spacing=None,
                    label_spacing_factor=1.5):
    label_map = {label: bar_labels[i] for i, label in enumerate(sorted(set(labels)))}
    rotation = 0 if bar_labels_align == 'horizontal' else 90
    for pos, length in zip(positions, lengths):
        xy, wh = _get_xy_wh(ax=ax, position=position, pos=pos, bar_spacing=bar_spacing, length=length, bar_width=bar_width)
        text_x, text_y, ha, va = _get_xy_hava(position=position, xy=xy, wh=wh, label_spacing_factor=label_spacing_factor)
        ax.text(text_x, text_y, label_map[labels[int(pos + length / 2)]], ha=ha, va=va, rotation=rotation,
                transform=ax.transData, clip_on=False)


def _add_text_labels(ax=None, position=None, bar_width=None, bar_spacing=None, label_spacing_factor=1.5,
                     nx=None, ny=None, xtick_label_rotation=45, ytick_label_rotation=0):
    """Add text labels next to the bars."""
    # Obtain tick labels and locations based on the specified position
    if position in ['left', 'right']:
        tick_labels = [item.get_text() for item in ax.get_yticklabels()]
        tick_locs = ax.get_yticks()
        ax.yaxis.set_visible(False)  # Hide the original y-axis
    else:  # ['top', 'bottom']
        tick_labels = [item.get_text() for item in ax.get_xticklabels()]
        tick_locs = ax.get_xticks()
        ax.xaxis.set_visible(False)  # Hide the original x-axis
    label_spacing = bar_spacing * label_spacing_factor
    for idx, label in enumerate(tick_labels):
        if position == 'left':
            ax.text(-label_spacing - bar_width, tick_locs[idx], label,
                    ha='right', va='center', rotation=ytick_label_rotation)
        elif position == 'right':
            ax.text(nx + label_spacing + bar_width, tick_locs[idx], label,
                    ha='left', va='center', rotation=ytick_label_rotation)
        elif position == 'top':
            ax.text(tick_locs[idx], - label_spacing - bar_width, label,
                    ha='left', va='bottom', rotation=xtick_label_rotation)
        elif position == 'bottom':
            ax.text(tick_locs[idx], ny + label_spacing + bar_width, label,
                    ha='right', va='top', rotation=xtick_label_rotation)

# TODO use for CPP plots
# Main function
def plot_add_bars(ax=None, labels=None, colors='tab:gray', bar_position='left',
                  bar_spacing=0.05,  bar_width=0.1, bar_labels=None, label_spacing_factor=1.5,
                  bar_labels_align='horizontal', set_tick_labels=True,
                  xtick_label_rotation=45, ytick_label_rotation=0):
    """
    Add colored bars along a specified axis of the plot based on label grouping.

    Parameters:
        ax (matplotlib.axes._axes.Axes): The axes to which bars will be added.
        labels (list or array-like): Labels determining bar grouping and coloring.
        bar_position (str): The position to add the bars ('left', 'right', 'top', 'bottom').
        bar_spacing (float): Spacing between plot and added bars.
        label_spacing_factor (float): Spacing between plot and text label as factor of 'bar_spacing'.
            If 1, text is directly placed next to bar.
        colors (str or list): Either a matplotlib color string, or a list of colors for each unique label.
        bar_labels (list, optional): Labels for the bars.
        bar_labels_align (str): Text alignment for bar labels, either 'horizontal' or other valid matplotlib alignment.
        bar_width (float): The width of the bars, expressed in the units of the opposite axis's elements.
            Specifically, when adding a vertical bar (with position 'left' or 'right'), `bar_width` denotes
            the horizontal extent of the bar, interpreted in terms of the spacing between adjacent elements on the x-axis.
            Conversely, when adding a horizontal bar (with position 'top' or 'bottom'), `bar_width` represents the vertical
             extent, mapped onto the y-axis spacing.
        xtick_label_rotation (int): X-axis label rotation.
        ytick_label_rotation (int): Y-axis label rotation.
        set_tick_labels (bool): Whether to adjust tick labels.

    Note:
        This function adds colored bars in correspondence with the provided `labels` to visualize groupings in plots.
    """
    # Get the number of plotted items
    nx, ny = int(max(plt.xlim())), int(max(plt.ylim()))
    num_plotted_items = ny if bar_position in ['left', 'right'] else nx
    if not isinstance(labels, list):
        labels = list(labels)
        # Check if labels match the shape of the data
    if len(labels) != num_plotted_items:
        raise ValueError(f"Mismatch: The number of labels ({len(labels)}) must match to number of plotted items ({num_plotted_items}).")
    single_color = isinstance(colors, str) or (isinstance(colors, (list, tuple)) and len(set(colors)) == 1)
    color_map, _ = _get_color_map(labels, colors)
    positions, lengths, colors = _get_positions_lengths_colors(labels, color_map)
    args_pos = dict(position=bar_position, bar_width=bar_width, bar_spacing=bar_spacing)
    if bar_labels is not None:
        _add_bar_labels(ax=ax, bar_labels_align=bar_labels_align,
                        labels=labels, positions=positions, lengths=lengths,
                        bar_labels=bar_labels, label_spacing_factor=label_spacing_factor, **args_pos)
    # Adding bars
    args = dict(transform=ax.transData, clip_on=False)
    for pos, length, bar_color in zip(positions, lengths, colors):
        xy, wh = _get_xy_wh(ax=ax, pos=pos, length=length, **args_pos)
        # Add edgecolor if only one color is specified
        edgecolor = "white" if single_color else bar_color
        ax.add_patch(mpatches.Rectangle(xy=xy, width=wh[0], height=wh[1],
                                        facecolor=bar_color, edgecolor=edgecolor, linewidth=0.5,
                                        **args))
    if set_tick_labels:
        _add_text_labels(ax=ax, **args_pos, label_spacing_factor=label_spacing_factor,
                         xtick_label_rotation=xtick_label_rotation, ytick_label_rotation=ytick_label_rotation,
                         nx=nx, ny=ny)


def plot_gco(option='font.size', show_options=False):
    """Get current option from plotting context"""
    current_context = sns.plotting_context()
    if show_options:
        print(current_context)
    option_value = current_context[option]  # Typically font_size
    return option_value