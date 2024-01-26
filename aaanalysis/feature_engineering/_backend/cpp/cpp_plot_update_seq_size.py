"""
This is a script for the backend of the CPPPlot.update_seq_size() method.
"""
import matplotlib.pyplot as plt
from ._utils_cpp_plot_positions import PlotPositions

# II Main Function
def update_seq_size_(ax=None, tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                     max_x_dist=0.1,
                     tmd_color="mediumspringgreen", jmd_color="blue",
                     tmd_seq_color="black", jmd_seq_color="white"):
    """Update the font size of the sequence characters to prevent overlap."""
    colors = [jmd_color] * len(jmd_n_seq) + [tmd_color] * len(tmd_seq) + [jmd_color] * len(jmd_c_seq)
    dict_seq_color = {tmd_color: tmd_seq_color, jmd_color: jmd_seq_color}
    # Get all x-axis tick labels
    labels = ax.xaxis.get_ticklabels(which="both")
    # Compute the positions of the labels and sort them
    f = lambda l: l.get_window_extent(ax.figure.canvas.get_renderer())
    tick_positions = [f(l).x0 for l in labels]
    sorted_tick_positions, sorted_labels = zip(*sorted(zip(tick_positions, labels), key=lambda t: t[0]))
    # Adjust font size to prevent overlap
    pp = PlotPositions()
    seq_size = pp.get_optimal_fontsize(ax=ax, labels=sorted_labels, max_x_dist=max_x_dist)
    lw = plt.gcf().get_size_inches()[0]/5
    for l, c in zip(sorted_labels, colors):
        l.set_fontsize(seq_size)
        l.set_bbox(dict(facecolor=c, edgecolor=c, zorder=0.1, alpha=1,
                        clip_on=False,
                        pad=0, linewidth=lw))
        l.set_color(dict_seq_color[c])
    return ax, seq_size


def update_tmd_jmd_labels(fig=None, seq_size=None, fontsize_tmd_jmd=None, weight_tmd_jmd="bold"):
    """Adjust size and position of TMD-JMD labels"""
    fs_labels = fontsize_tmd_jmd if fontsize_tmd_jmd else seq_size
    if fig is not None and len(fig.axes) > 1:
        ax2 = fig.axes[1]
        ax2.spines["bottom"].set_position(("outward", seq_size + 7))
        ax2.tick_params(axis='x', labelsize=fs_labels)
        for label in ax2.get_xticklabels():
            print(weight_tmd_jmd)
            label.set_weight(weight_tmd_jmd)
