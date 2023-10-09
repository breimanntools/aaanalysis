"""
This is a script for internal plotting utility functions used in the backend.
"""
import seaborn as sns
import matplotlib.patches as mpatches


# Helper functions
def _get_color_map(labels, color):
    unique_labels = sorted(set(labels))
    if isinstance(color, list):
        if len(color) != len(unique_labels):
            raise ValueError("If color is a list, it must have the same length as the number of unique labels.")
        color_map = {label: color[i] for i, label in enumerate(unique_labels)}
    else:
        color_map = {label: color for label in unique_labels}
    return color_map, unique_labels


def _get_positions_lengths_colors(labels, color_map):
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

def _get_xy_wh(ax=None, position=None, pos=None, barspacing=None, length=None, bar_width=None):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if position == 'bottom':
        y = y_min + barspacing
        return (pos, y), (length, bar_width)
    elif position == 'top':
        y = y_max - barspacing - bar_width
        return (pos, y), (length, bar_width)
    elif position == 'left':
        x =  x_min - barspacing - bar_width
        return (x, pos), (bar_width, length)
    elif position == 'right':
        x = x_max + barspacing
        return (x, pos), (bar_width, length)
    else:
        raise ValueError("Position should be 'left', 'right', 'top', or 'bottom'.")

def _get_xy_hava(position=None, xy=None, wh=None):
    bar_width = wh[1] if position in ['bottom', 'top'] else wh[0]
    if position == 'bottom':
        text_x = xy[0] + wh[0] / 2
        text_y = xy[1] + bar_width * 1.5
        ha, va = 'center', 'top'
    elif position == 'top':
        text_x = xy[0] + wh[0] / 2
        text_y = xy[1] + wh[1] - bar_width
        ha, va = 'center', 'bottom'
    elif position == 'left':
        text_x = xy[0] - bar_width
        text_y = xy[1] + wh[1] / 2
        ha, va = 'right', 'center'
    else:  # Assuming position == 'right'
        text_x = xy[0] + wh[0] + bar_width*0.5
        text_y = xy[1] + wh[1] / 2
        ha, va = 'left', 'center'
    return text_x, text_y, ha, va

def _add_bar_labels(ax=None, bar_labels_align=None, position=None, bar_width=None,
                    labels=None, positions=None, lengths=None, bar_labels=None, barspacing=None):
    label_map = {label: bar_labels[i] for i, label in enumerate(sorted(set(labels)))}
    rotation = 0 if bar_labels_align == 'horizontal' else 90
    for pos, length in zip(positions, lengths):
        xy, wh = _get_xy_wh(ax=ax, position=position, pos=pos, barspacing=barspacing, length=length, bar_width=bar_width)
        text_x, text_y, ha, va = _get_xy_hava(position=position, xy=xy, wh=wh)
        ax.text(text_x, text_y, label_map[labels[int(pos + length / 2)]], ha=ha, va=va, rotation=rotation,
                transform=ax.transData, clip_on=False)



# Main function
def plot_add_bars(ax, labels, position='left', bar_spacing=0.05, colors='tab:gray', bar_labels=None,
                  bar_labels_align='horizontal', bar_width=0.1):
    """
    Add colored bars along a specified axis of the plot based on label grouping.

    Parameters:
        ax (matplotlib.axes._axes.Axes): The axes to which bars will be added.
        labels (list or array-like): Labels determining bar grouping and coloring.
        position (str): The position to add the bars ('left', 'right', 'top', 'bottom').
        bar_spacing (float): Spacing between plot and added bars.
        colors (str or list): Either a matplotlib color string, or a list of colors for each unique label.
        bar_labels (list, optional): Labels for the bars.
        bar_labels_align (str): Text alignment for bar labels, either 'horizontal' or other valid matplotlib alignment.
        bar_width (float): Width of the bars.

    Note:
        This function adds colored bars in correspondence with the provided `labels` to visualize groupings in plots.
    """

    if not isinstance(labels, list):
        labels = list(labels)
    single_color = isinstance(colors, str) or (isinstance(colors, (list, tuple)) and len(colors) == 1)
    color_map, _ = _get_color_map(labels, colors)
    positions, lengths, colors = _get_positions_lengths_colors(labels, color_map)
    args_get = dict(position=position, bar_width=bar_width, barspacing=bar_spacing)
    if bar_labels is not None:
        _add_bar_labels(ax=ax, bar_labels_align=bar_labels_align,
                        labels=labels, positions=positions, lengths=lengths, bar_labels=bar_labels, **args_get)
    # Adding bars
    args = dict(transform=ax.transData, clip_on=False)
    for pos, length, bar_color in zip(positions, lengths, colors):
        xy, wh = _get_xy_wh(ax=ax, pos=pos, length=length, **args_get)
        # Add edgecolor if only one color is specified
        edgecolor = "white" if single_color else bar_color
        ax.add_patch(mpatches.Rectangle(xy=xy, width=wh[0], height=wh[1],
                                        facecolor=bar_color, edgecolor=edgecolor, linewidth=0.5,
                                        **args))


def plot_gco(option='font.size', show_options=False):
    """Get current option from plotting context"""
    current_context = sns.plotting_context()
    if show_options:
        print(current_context)
    option_value = current_context[option]  # Typically font_size
    return option_value