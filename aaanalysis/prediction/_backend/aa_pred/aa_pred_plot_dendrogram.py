"""
This is a script for the backend of AAPredPlot.group_cluster(kind="dendrogram"): the sample
relation tree alone, drawn as a rectangular or circular (radial) dendrogram of the very same
linkage the clustermap kind uses, with the leaves colored by up to two per-sample annotations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.cluster import hierarchy

from .aa_pred_plot_linkage import sample_correlation_, sample_linkage_
from .aa_pred_plot_clustermap import resolve_label_colors_


# I Helper Functions
_TREE_COLOR = "black"
_TREE_LW = 1.0
_NAME_FONTSIZE = 9
# At most this many leaf names per layout before the names are thinned (every k-th is shown).
_MAX_NAMES_RECTANGULAR = 40
_MAX_NAMES_CIRCULAR = 60
# Rectangular annotation strip width relative to the tree panel.
_STRIP_RATIO = 0.04
# Circular geometry, relative to the tree radius: ring width, gap between rings / names.
_RING_WIDTH = 0.07
_RING_GAP = 0.015


def _build_tracks(labels=None, dict_color=None, legend_title="Class", labels_row=None,
                  dict_color_row=None, legend_title_row=None):
    """Ordered leaf annotations as (values, label->color, title); ``labels`` is innermost."""
    tracks = []
    if labels is not None:
        tracks.append((list(labels), resolve_label_colors_(list(labels), dict_color),
                       legend_title or "Class"))
    if labels_row is not None:
        tracks.append((list(labels_row), resolve_label_colors_(list(labels_row), dict_color_row),
                       legend_title_row or "Class"))
    return tracks


def _name_step(n, max_names):
    """Show every k-th leaf name so at most ``max_names`` names are drawn."""
    return max(1, int(np.ceil(n / max_names)))


def _make_grid(figsize, tracks, n_strips=0):
    """Figure on one flat grid (kept flat so ``plt.tight_layout`` can arrange it).

    Columns: ``n_leg = max(n_tracks, 1)`` equal main columns (the tree spans them all) plus
    ``n_strips`` thin annotation-strip columns. Rows: the main row plus, if there are tracks, a
    slim legend row holding one blank legend axes per track side by side (the last one also
    spans the strip columns). Returns ``(fig, gs, n_leg, legend_axes)``.
    """
    fig = plt.figure(figsize=figsize)
    # Spacing goes through the figure's subplot params: spacing set on the grid itself would mark
    # it as locally modified, which ``plt.tight_layout`` refuses to arrange.
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    n_leg = max(len(tracks), 1)
    width_ratios = [1 / n_leg] * n_leg + [_STRIP_RATIO] * n_strips
    if not tracks:
        gs = fig.add_gridspec(1, n_leg + n_strips, width_ratios=width_ratios)
        return fig, gs, n_leg, []
    # Legend height in inches: title line plus one line per class of the longest legend.
    max_classes = max(len(dict_color) for _, dict_color, _ in tracks)
    legend_h = 0.45 + 0.24 * max_classes
    main_h = max(figsize[1] - legend_h, 1.0)
    gs = fig.add_gridspec(2, n_leg + n_strips, width_ratios=width_ratios,
                          height_ratios=[main_h, legend_h])
    legend_axes = []
    for j in range(n_leg):
        cell = gs[1, j:] if j == n_leg - 1 else gs[1, j]
        ax_leg = fig.add_subplot(cell)
        ax_leg.set_axis_off()
        legend_axes.append(ax_leg)
    return fig, gs, n_leg, legend_axes


def _draw_track_legends(legend_axes, tracks):
    """One bold-titled, left-aligned, frameless class legend per track in its legend axes."""
    for ax_leg, (_, dict_color, title) in zip(legend_axes, tracks):
        handles = [mpatches.Patch(color=dict_color[k], label=k) for k in dict_color]
        leg = ax_leg.legend(handles=handles, title=title, loc="upper left", alignment="left",
                            bbox_to_anchor=(0, 1), ncol=1, fontsize=10, title_fontsize=11,
                            frameon=False, borderaxespad=0)
        leg.get_title().set_fontweight("bold")


def _tree_coords(linkage):
    """Leaf order and link coordinates (scipy convention: leaf k at 5 + 10k, height = distance)."""
    dn = hierarchy.dendrogram(linkage, no_plot=True)
    return dn["leaves"], np.asarray(dn["icoord"]), np.asarray(dn["dcoord"])


def _plot_rectangular(fig, gs, n_leg, leaves, icoord, dcoord, names, tracks):
    """Tree panel (root left, leaves right), one color strip per track, names on the far right."""
    n = len(leaves)
    ax = fig.add_subplot(gs[0, :n_leg])
    segments = [np.column_stack([dc, ic]) for ic, dc in zip(icoord, dcoord)]
    ax.add_collection(LineCollection(segments, colors=_TREE_COLOR, linewidths=_TREE_LW))
    d_max = float(dcoord.max()) if dcoord.size else 0.0
    ax.set_xlim(d_max * 1.03 if d_max > 0 else 1.0, 0)
    # First leaf on top (as in the clustermap rows).
    ax.set_ylim(10 * n, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    # Hide (not clear) the y ticks: the strips share this y axis and set the leaf-name ticks.
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    pos = 5 + 10 * np.arange(n)
    ax_last = ax
    for j, (values, dict_color, _) in enumerate(tracks):
        ax_s = fig.add_subplot(gs[0, n_leg + j], sharey=ax)
        colors = [dict_color[values[i]] for i in leaves]
        ax_s.barh(y=pos, width=1, height=10, left=0, color=colors, linewidth=0)
        ax_s.set_xlim(0, 1)
        for spine in ax_s.spines.values():
            spine.set_visible(False)
        ax_s.set_xticks([])
        ax_s.tick_params(left=False, labelleft=False)
        ax_last = ax_s
    step = _name_step(n, _MAX_NAMES_RECTANGULAR)
    ax_last.yaxis.tick_right()
    ax_last.set_yticks(pos[::step])
    ax_last.set_yticklabels([names[i] for i in leaves][::step], fontsize=_NAME_FONTSIZE)
    ax_last.tick_params(axis="y", right=False, left=False, labelright=True, labelleft=False,
                        pad=2)
    ax.set_ylim(10 * n, 0)
    return ax


def _plot_circular(fig, gs, leaves, icoord, dcoord, names, tracks):
    """Radial tree (root at the center), one color ring per track, names around the outside."""
    n = len(leaves)
    ax = fig.add_subplot(gs[0, :], projection="polar")
    d_max = float(dcoord.max()) if dcoord.size else 0.0
    radius = d_max if d_max > 0 else 1.0

    def _theta(y):
        return 2 * np.pi * np.asarray(y, dtype=float) / (10 * n)

    def _r(d):
        return radius - np.asarray(d, dtype=float)

    segments = []
    for ic, dc in zip(icoord, dcoord):
        # scipy link: (x0, d0) -> (x0, d1) -> (x3, d1) -> (x3, d0); vertical legs become radial
        # lines, the horizontal crossbar an arc at the merge radius.
        t0, t3 = _theta(ic[0]), _theta(ic[3])
        segments.append(np.array([[t0, _r(dc[0])], [t0, _r(dc[1])]]))
        n_pts = max(2, int(np.ceil(abs(t3 - t0) / (2 * np.pi) * 360)))
        arc_t = np.linspace(t0, t3, n_pts)
        segments.append(np.column_stack([arc_t, np.full(n_pts, _r(dc[1]))]))
        segments.append(np.array([[t3, _r(dc[2])], [t3, _r(dc[3])]]))
    ax.add_collection(LineCollection(segments, colors=_TREE_COLOR, linewidths=_TREE_LW))
    theta_leaf = _theta(5 + 10 * np.arange(n))
    width = 2 * np.pi / n
    r_next = radius * (1 + _RING_GAP)
    for values, dict_color, _ in tracks:
        colors = [dict_color[values[i]] for i in leaves]
        ax.bar(theta_leaf, height=radius * _RING_WIDTH, width=width, bottom=r_next,
               color=colors, linewidth=0, align="center")
        r_next += radius * (_RING_WIDTH + _RING_GAP)
    r_text = r_next + radius * _RING_GAP
    step = _name_step(n, _MAX_NAMES_CIRCULAR)
    leaf_names = [names[i] for i in leaves]
    for k in range(0, n, step):
        deg = float(np.degrees(theta_leaf[k]))
        flip = 90 < deg < 270
        ax.text(theta_leaf[k], r_text, leaf_names[k], rotation=deg + 180 if flip else deg,
                rotation_mode="anchor", ha="right" if flip else "left", va="center",
                fontsize=_NAME_FONTSIZE, clip_on=False)
    ax.set_ylim(0, r_text)
    ax.set_axis_off()
    return ax


# II Main Functions
def plot_dendrogram_(data=None, names=None, labels=None, dict_color=None, legend_title="Class",
                     labels_row=None, dict_color_row=None, legend_title_row=None,
                     layout="rectangular", figsize=(7, 9), title=None):
    """Rectangular or circular sample dendrogram with colored leaves. Returns (fig, ax_tree)."""
    n = np.asarray(data).shape[0]
    if names is None:
        names = [str(i) for i in range(n)]
    names = [str(name) for name in names]
    # Same correlation + row linkage as the clustermap kind, so both draw an identical tree.
    corr_df = sample_correlation_(data=data, names=names)
    linkage = sample_linkage_(corr_df=corr_df, axis=0)
    leaves, icoord, dcoord = _tree_coords(linkage)
    tracks = _build_tracks(labels=labels, dict_color=dict_color, legend_title=legend_title,
                           labels_row=labels_row, dict_color_row=dict_color_row,
                           legend_title_row=legend_title_row)
    if layout == "circular":
        fig, gs, _, legend_axes = _make_grid(figsize=figsize, tracks=tracks, n_strips=0)
        ax = _plot_circular(fig, gs, leaves, icoord, dcoord, names, tracks)
    else:
        fig, gs, n_leg, legend_axes = _make_grid(figsize=figsize, tracks=tracks,
                                                 n_strips=len(tracks))
        ax = _plot_rectangular(fig, gs, n_leg, leaves, icoord, dcoord, names, tracks)
    _draw_track_legends(legend_axes, tracks)
    if title is not None:
        fig.suptitle(title)
    return fig, ax
