"""
Backend for AAPredPlot.predict(kind="clustermap"): cluster samples by explanation similarity
(Pearson correlation of their per-sample importance/SHAP vectors).
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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


def _annotation_legend(ax, dict_color, title, anchor_y):
    """Draw one class-color legend (a color swatch per label) for a sidebar annotation."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=dict_color[k]) for k in dict_color]
    return ax.legend(handles, list(dict_color), title=title, frameon=False,
                     bbox_to_anchor=(1.25, anchor_y), loc="upper left", fontsize=9)


# II Main Functions
def plot_clustermap_(data=None, names=None, labels=None, labels_row=None, colors=None,
                     colors_row=None, cmap="GnBu", figsize=(9, 9),
                     cbar_label="Pearson correlation (r)", title=None,
                     legend_title="Class", legend_title_row=None):
    """Correlation clustermap of per-sample importance vectors. Returns (fig, ax_heatmap)."""
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
    # Column (top) sidebar from `labels`; row (left) sidebar from `labels_row`. When
    # `labels_row` is None the column annotation is mirrored onto the rows, preserving the
    # single-annotation figure; a distinct `labels_row` gives the two-sidebar figure.
    col_colors = row_colors = None
    dict_color_col = dict_color_row = None
    if labels is not None:
        dict_color_col = _resolve_label_colors(list(labels), colors)
        col_colors = [dict_color_col[l] for l in labels]
    if labels_row is not None:
        dict_color_row = _resolve_label_colors(list(labels_row), colors_row)
        row_colors = [dict_color_row[l] for l in labels_row]
    elif labels is not None:
        row_colors = col_colors
    g = sns.clustermap(corr_df, cmap=cmap, vmin=-1, vmax=1,
                       row_colors=row_colors, col_colors=col_colors,
                       figsize=figsize, xticklabels=False,
                       cbar_kws=dict(label=cbar_label))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
    if title is not None:
        g.figure.suptitle(title)
    if dict_color_col is not None and dict_color_row is not None:
        # Two distinct annotations -> stacked legends (column annotation on top).
        leg_col = _annotation_legend(g.ax_heatmap, dict_color_col, legend_title, 1.0)
        g.ax_heatmap.add_artist(leg_col)
        _annotation_legend(g.ax_heatmap, dict_color_row, legend_title_row or legend_title, 0.6)
    else:
        dict_color = dict_color_col if dict_color_col is not None else dict_color_row
        if dict_color is not None:
            _annotation_legend(g.ax_heatmap, dict_color, legend_title, 1.0)
    return g.figure, g.ax_heatmap
