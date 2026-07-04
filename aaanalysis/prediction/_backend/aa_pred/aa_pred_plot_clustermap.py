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


# II Main Functions
def plot_clustermap_(data=None, names=None, labels=None, colors=None, cmap="GnBu",
                     figsize=(9, 9), cbar_label="Pearson correlation (r)", title=None):
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
    side_colors = None
    dict_color = None
    if labels is not None:
        dict_color = _resolve_label_colors(list(labels), colors)
        side_colors = [dict_color[l] for l in labels]
    g = sns.clustermap(corr_df, cmap=cmap, vmin=-1, vmax=1,
                       row_colors=side_colors, col_colors=side_colors,
                       figsize=figsize, xticklabels=False,
                       cbar_kws=dict(label=cbar_label))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
    if title is not None:
        g.figure.suptitle(title)
    if dict_color is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=dict_color[k]) for k in dict_color]
        g.ax_heatmap.legend(handles, list(dict_color), title="Class", frameon=False,
                            bbox_to_anchor=(1.25, 1.0), loc="upper left", fontsize=9)
    return g.figure, g.ax_heatmap
