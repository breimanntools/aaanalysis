"""
This is a script for the backend of the ShapModelPlot.clustermap() method.

Clusters samples by *explanation similarity*: the pairwise Pearson correlation of their
per-sample SHAP-value vectors, so proteins group by *why* the model scores them.
"""
import numpy as np
import pandas as pd
import seaborn as sns

import aaanalysis.utils as ut


# I Helper Functions
def comp_shap_correlation(shap_values=None, names=None):
    """Pairwise Pearson correlation of per-sample SHAP-value vectors.

    Rows of ``shap_values`` are samples and columns are features, so transposing and
    correlating yields an (n_samples, n_samples) matrix of explanation similarity.
    A sample whose SHAP vector has zero variance (e.g. an all-constant or all-zero
    vector) has an undefined correlation; this is reported clearly rather than left
    to surface as an opaque non-finite-distance error inside the clustering.
    """
    df_cor = pd.DataFrame(shap_values, index=names).T.corr()
    diag = pd.Series(np.diag(df_cor.to_numpy()), index=df_cor.index)
    bad = list(diag.index[diag.isna()])
    if len(bad) > 0:
        raise ValueError(f"Explanation similarity is undefined for samples with a constant "
                         f"(zero-variance) SHAP vector: {bad}. Remove or perturb these samples.")
    return df_cor


def _add_class_legend(grid=None, dict_color=None):
    """Add a class-color legend for the row/column sidebars (top-right, house style)."""
    fs = ut.plot_gco()
    list_cat = list(dict_color)
    ut.plot_legend_(ax=grid.ax_col_dendrogram, dict_color=dict_color, list_cat=list_cat,
                    labels=[str(c) for c in list_cat], loc="center left", x=1.05, y=0.5,
                    n_cols=1, title="Class", fontsize=fs, fontsize_title=fs)


# II Main Functions
def plot_shap_clustermap(df_cor=None, dict_color=None, row_colors=None, col_colors=None,
                         method="complete", figsize=(6, 6), cmap="GnBu",
                         vmin=-1, vmax=1, tick_labels=4, tree_linewidth=1.0,
                         cbar_label="Correlation (r)"):
    """Draw the SHAP-correlation clustermap and return the seaborn ``ClusterGrid``."""
    fs = ut.plot_gco()
    grid = sns.clustermap(df_cor, figsize=figsize, vmin=vmin, vmax=vmax, cmap=cmap,
                          method=method, row_colors=row_colors, col_colors=col_colors,
                          xticklabels=tick_labels, yticklabels=tick_labels,
                          tree_kws=dict(linewidth=tree_linewidth),
                          cbar_pos=(0.04, 0.85, 0.18, 0.03),
                          cbar_kws=dict(orientation="horizontal"))
    # Sample names are the tick labels; an axis title would be redundant/misleading
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(labelsize=fs)
    # Colorbar: left-align the label so it never overflows the figure's left edge
    grid.ax_cbar.set_xlabel(cbar_label, loc="left", fontsize=fs)
    grid.ax_cbar.tick_params(labelsize=fs)
    _add_class_legend(grid=grid, dict_color=dict_color)
    return grid
