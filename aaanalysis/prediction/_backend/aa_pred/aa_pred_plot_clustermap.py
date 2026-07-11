"""
Backend for AAPredPlot.group_cluster(kind="clustermap"): cluster samples by explanation
similarity (Pearson correlation of their per-sample importance/SHAP vectors), with up to two
annotation sidebars (a column/top and a row/left class strip) and their titled legends.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

import aaanalysis.utils as ut


# I Helper Functions
def _resolve_label_colors(labels, colors):
    """Map each distinct label to a color (dict wins; else the house palette)."""
    order = list(dict.fromkeys(labels))
    if isinstance(colors, dict):
        missing = [g for g in order if g not in colors]
        if missing:
            raise ValueError(f"'colors' dict is missing colors for labels: {missing}")
        return colors
    palette = ut.plot_get_clist_(n_colors=max(len(order), 2))
    return {g: palette[i] for i, g in enumerate(order)}


def _titled_legend(fig, dict_color, title, anchor_y):
    """One left-aligned, bold-titled class legend along the bottom band; ncol = #classes."""
    handles = [mpatches.Patch(color=dict_color[k], label=k) for k in dict_color]
    leg = fig.legend(handles=handles, title=title, loc="lower left", alignment="left",
                     bbox_to_anchor=(0.28, anchor_y), ncol=len(dict_color), fontsize=11,
                     frameon=False, title_fontsize=12)
    leg.get_title().set_fontweight("bold")
    return leg


# II Main Functions
def plot_clustermap_(data=None, names=None, labels=None, colors=None, legend_title="Class",
                     labels_row=None, colors_row=None, legend_title_row=None,
                     cmap="GnBu", figsize=(11, 11), cbar_label="Pearson correlation (r)",
                     title=None):
    """Two-annotation correlation clustermap of per-sample vectors. Returns (fig, ax_heatmap)."""
    values = np.asarray(data, dtype=float)
    n = values.shape[0]
    if names is None:
        names = [str(i) for i in range(n)]
    # Sample x sample Pearson correlation of the explanation vectors. A sample whose
    # vector has zero variance (e.g. an all-zero SHAP row) yields NaN correlations, which
    # break the hierarchical linkage; treat those as uncorrelated (0) with self-corr 1.
    corr = np.corrcoef(values)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=list(names), columns=list(names))
    # Column (top) annotation from `labels`; row (left) annotation from `labels_row`. A single
    # annotation is mirrored onto both sidebars (the matrix is symmetric).
    col_dict = row_dict = None
    col_colors = row_colors = None
    if labels is not None:
        col_dict = _resolve_label_colors(list(labels), colors)
        col_colors = pd.Series([col_dict[lbl] for lbl in labels], index=list(names), name="")
    if labels_row is not None:
        row_dict = _resolve_label_colors(list(labels_row), colors_row)
        row_colors = pd.Series([row_dict[lbl] for lbl in labels_row], index=list(names), name="")
    elif labels is not None:
        row_colors = col_colors
    # Thin the tick labels so a dense sample set stays legible (at most ~25 names per axis).
    step = max(1, int(np.ceil(n / 25)))
    g = sns.clustermap(corr_df, cmap=cmap, vmin=-1, vmax=1, figsize=figsize,
                       col_colors=col_colors, row_colors=row_colors,
                       dendrogram_ratio=0.11, colors_ratio=0.015,
                       xticklabels=step, yticklabels=step,
                       cbar_pos=(0.05, 0.075, 0.17, 0.02),
                       cbar_kws=dict(orientation="horizontal", label=cbar_label))
    # Tighten margins to remove dead space; keep a slim bottom band for the colorbar and legends.
    g.gs.update(bottom=0.18, top=(0.95 if title is not None else 0.985), left=0.01, right=0.90)
    # Colorbar: horizontal, bottom-left, with an edge only along the bottom (as thick as the ticks).
    cb = g.ax_cbar
    cb.set_position([0.05, 0.075, 0.17, 0.02])
    cb.set_xlabel(cbar_label, fontsize=12)
    _tw = 1.8
    cb.tick_params(labelsize=11, width=_tw, length=5, color="black")
    for _s in cb.spines.values():
        _s.set_visible(False)
    cb.spines["bottom"].set_visible(True)
    cb.spines["bottom"].set_linewidth(_tw)
    cb.spines["bottom"].set_color("black")
    # Names close to the heatmap (small tick pad), no tick marks. Right (y) names stay horizontal;
    # bottom (x) names are turned 45 degrees (anchored at their right end) for readability.
    g.ax_heatmap.tick_params(left=False, bottom=False, right=False, top=False, pad=0.5)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=9, rotation=0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=9, rotation=45,
                                 ha="right", rotation_mode="anchor")
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    if title is not None:
        g.figure.suptitle(title)
    # Titled, left-aligned legends stacked at bottom-left. With two annotations the row (left)
    # legend sits above the column (top) legend, matching the sample-clustering appendix layout.
    if col_dict is not None and row_dict is not None:
        _titled_legend(g.figure, row_dict, legend_title_row or "Class", 0.055)
        _titled_legend(g.figure, col_dict, legend_title, 0.012)
    else:
        single = col_dict if col_dict is not None else row_dict
        if single is not None:
            single_title = legend_title if col_dict is not None else (legend_title_row or "Class")
            _titled_legend(g.figure, single, single_title, 0.03)
    return g.figure, g.ax_heatmap
