"""
This is a script for the backend of the AAclustPlot object for all plotting functions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list

import aaanalysis.utils as ut
from ._utils_aaclust import _compute_medoids, _compute_centers


# I Helper Functions
# Computation helper functions
def _get_mean_rank(data):
    _df = data.copy()
    _df['BIC_rank'] = _df[ut.COL_BIC].rank(ascending=False)
    _df['CH_rank'] = _df[ut.COL_CH].rank(ascending=False)
    _df['SC_rank'] = _df[ut.COL_SC].rank(ascending=False)
    rank = _df[['BIC_rank', 'CH_rank', 'SC_rank']].mean(axis=1).round(2)
    return rank


def _get_components(X, model_class=None, n_components=2, model_kwargs=None):
    """
    Apply dimensionality reduction on X using the specified mode.

    Returns:
    df_components : DataFrame, shape (n_samples, n_components)
        The reduced components labeled "PC1", "PC2", etc.
    """
    model_kwargs["n_components"] = n_components
    # Initialize and fit the model
    model = model_class(**model_kwargs)
    model_name = model_class.__name__
    components = model.fit_transform(X)
    # Create DataFrame
    columns = [f"{model_name}{i + 1}" for i in range(n_components)]
    if isinstance(model, PCA):
        explained_var = 100 * model.explained_variance_ratio_
        columns = [f"{col.replace('PCA', 'PC')} ({explained_var[i]:.1f}%)" for i, col in enumerate(columns)]
    df_components = pd.DataFrame(components, columns=columns)
    return df_components, model


def _get_clustered_order(df, method='average'):
    linked = linkage(df.corr().fillna(0), method=method)
    new_order = leaves_list(linked)
    return df.iloc[:, new_order]


# II Main Functions
def plot_eval(df_eval=None, dict_xlims=None, figsize=None, colors=None):
    """Plot evaluation of AAclust clustering results"""
    df_eval[ut.COL_RANK] = _get_mean_rank(df_eval)
    df_eval = df_eval.sort_values(by=ut.COL_RANK, ascending=True)
    # Plotting
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=figsize)
    for i, col in enumerate(ut.COLS_EVAL_AACLUST):
        ax = axes[i]
        sns.barplot(ax=ax, data=df_eval, y=ut.COL_NAME, x=col, color=colors[i])
        # Customize subplots
        ax.set_ylabel("")
        ax.set_xlabel(col)
        # Adjust spines
        ax = ut.adjust_spine_to_middle(ax=ax)
        if i == 0:
            ax.set_title("Clustering", weight="bold")
        elif i == 2:
            ax.set_title("Quality measures", weight="bold")
        ax.tick_params(axis='y', which='both', left=False)
        if i != 0:
            ut.ticks_0(ax=ax)
    # Set xlims
    if dict_xlims is not None:
        for i in dict_xlims:
            axes[i].set_xlim(dict_xlims[i])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)
    return fig, axes


@ut.catch_runtime_warnings()
def plot_center_or_medoid(X=None, labels=None,
                          plot_centers=True, metric="correlation",
                          component_x=1, component_y=1,
                          model_class=None, model_kwargs=None,
                          ax=None, figsize=(7, 6),
                          dot_alpha=0.75, dot_size=100,
                          legend=True, palette=None):
    """Plot compressed (e.g., by PCA) clustering results with highlighting cluster centers or cluster medoids"""
    n_components = max(component_x, component_y)
    df_components, model = _get_components(X, model_class, n_components, model_kwargs)
    # Get centers or medoids and obtain compress them
    if plot_centers:
        X_ref, labels_ref = _compute_centers(X, labels=labels)
    else:
        X_ref, labels_ref, _ = _compute_medoids(X, labels=labels, metric=metric)
    X_ref_transformed = model.transform(X_ref)
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    n_clusters = len(set(labels))
    if palette is None:
        palette = sns.color_palette("husl", n_clusters)
    ax = sns.scatterplot(x=df_components.iloc[:, component_x - 1],
                         y=df_components.iloc[:, component_y - 1],
                         hue=labels, palette=palette, alpha=dot_alpha,
                         s=dot_size, ax=ax, legend=legend)
    # Highlight centers or medoids
    kwargs = dict(x=X_ref_transformed[:, component_x - 1],
                  y=X_ref_transformed[:, component_y - 1],
                  hue=labels_ref, palette=palette,
                  s=dot_size * 1.5, edgecolor="k", linewidth=1,
                  legend=False, ax=ax)
    if plot_centers:
        ax = sns.scatterplot(**kwargs, marker="X")
    else:
        ax = sns.scatterplot(**kwargs)
    plt.xlabel(df_components.columns[component_x - 1])
    plt.ylabel(df_components.columns[component_y - 1])
    if legend:
        ax.legend(title="clusters", bbox_to_anchor=(1, 0.95), loc='upper left')
    plt.tight_layout()
    return ax, df_components


def plot_correlation(df_corr=None, labels=None, labels_ref=None, cluster_x=True, method="average",
                     bar_position="left", bar_colors="gray",
                     bar_width_x=0.1, bar_spacing_x=0.1, bar_width_y=0.1, bar_spacing_y=0.1,
                     xtick_label_rotation=90, ytick_label_rotation=0,
                     vmin=-1, vmax=1, cmap="viridis", kwargs_heatmap=None):
    """Plots heatmap for clustering results with rows (y-axis) corresponding to scales and columns (x-axis) to clusters."""
    # Adjust order of df_corr
    if cluster_x:
        df_corr = _get_clustered_order(df_corr, method=method)
    # Plot heatmap
    _kwargs_heatmap = {"cmap": cmap, "vmin": vmin, "vmax": vmax,
                       "cbar_kws": {"label": "Pearson correlation"}}
    if kwargs_heatmap is not None:
        _kwargs_heatmap.update(kwargs_heatmap)
    ax = sns.heatmap(data=df_corr, **_kwargs_heatmap)
    # Adjust ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_label_rotation, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_label_rotation)
    # Customizing colorbart tick lines
    if 'cbar' not in _kwargs_heatmap:
        cbar = ax.collections[0].colorbar
        lw = ut.plot_gco(option="axes.linewidth")
        fs = ut.plot_gco(option="font.size")
        cbar.ax.tick_params(axis='y', width=lw, length=6, color='black', labelsize=fs-1)
    # Add bars for highlighting clustering
    labels_are_sorted = list(labels) == sorted(labels)
    y_labels = labels
    x_labels = labels if labels_ref is None else labels_ref
    if labels_are_sorted:
        x_labels = list(sorted(x_labels))
    if bar_position is not None:
        for pos in bar_position:
            _labels = x_labels if pos in ["top", "bottom"] else y_labels
            _bar_width = bar_width_x if pos in ["top", "bottom"] else bar_width_y
            _bar_spacing = bar_spacing_x if pos in ["top", "bottom"] else bar_spacing_y
            ut.plot_add_bars(ax=ax, labels=_labels, bar_position=pos, set_tick_labels=True,
                             bar_spacing=_bar_spacing, bar_width=_bar_width, colors=bar_colors,
                             xtick_label_rotation=xtick_label_rotation,
                             ytick_label_rotation=ytick_label_rotation)
    plt.tight_layout()
    return ax
