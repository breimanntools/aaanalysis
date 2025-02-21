"""
This is a script for the backend of the dPULearnPlot methods.
"""
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import aaanalysis.utils as ut


# I Helper Functions
def _plot_neg_homogeneity(ax=None, df_eval=None, col=None, colors=None, val_name="Value"):
    """Plot homogeneity among identified negatives"""
    sns.barplot(ax=ax, data=df_eval, y=ut.COL_NAME, x=col, color=colors[0])
    # Adding the numbers to the end of the bars
    for p in ax.patches:
        val = p.get_width()
        val = round(val, 3) if val < 1 else int(val)
        ax.text(p.get_width() * 0.97, p.get_y() + p.get_height() / 2, val, ha='right', va='center')
    # Customize subplots
    ax.set_ylabel("")
    ax.set_xlabel(val_name)
    # Adjust spines
    ax = ut.adjust_spine_to_middle(ax=ax)
    ut.ticks_0(ax=ax)
    ax.tick_params(axis='y', which='both', left=False)


def _plot_dist_dissimilarity(ax=None, df_eval=None, cols=None, colors=None, val_name="Value"):
    """Plot how strong the distributions of identified negatives and the other groups differ"""
    df_auc = df_eval[[ut.COL_NAME] + cols]
    df_auc = df_auc.melt(id_vars=ut.COL_NAME, value_vars=cols, var_name="Reference", value_name=val_name)
    dict_color = dict(zip(cols, colors[1:len(cols) + 1]))
    ax = sns.barplot(ax=ax, data=df_auc, y=ut.COL_NAME, x=val_name,
                     palette=dict_color, hue="Reference", legend=False, edgecolor="white")
    # Customize subplots
    ax.set_ylabel("")
    ax.set_xlabel(val_name)
    # Adjust spines
    ax = ut.adjust_spine_to_middle(ax=ax)
    ut.ticks_0(ax=ax)
    ax.tick_params(axis='y', which='both', left=False)
    return ax


# II Main Functions
def plot_eval(df_eval=None, figsize=None, dict_xlims=None, colors=None, legend=True, legend_y=-0.175):
    """Plot evaluation of AAclust clustering results"""
    cols_eval = [x for x in ut.COLS_EVAL_DPULEARN if x in list(df_eval)]
    cols_homogeneity = cols_eval[1:3]
    cols_auc = [c for c in cols_eval if "AUC" in c]
    cols_kld = [c for c in cols_eval if "KLD" in c]
    kld_in = len(cols_kld) != 0
    n_colss = 3 if not kld_in else 4
    fig, axes = plt.subplots(1, n_colss, sharey=True, figsize=figsize)
    # Show Homogeneity
    for i, col in enumerate(cols_homogeneity):
        ax = axes[i]
        _plot_neg_homogeneity(ax=ax, df_eval=df_eval, col=col, colors=colors, val_name=col.replace("_", " "))
        if i == 0:
            ax.set_title("Homogeneity", weight="bold", ha="left")
    # Plot distribution dissimilarity
    ax = _plot_dist_dissimilarity(ax=axes[2], df_eval=df_eval, cols=cols_auc, colors=colors,  val_name="avg AUC")
    if kld_in:
        _plot_dist_dissimilarity(ax=axes[3], df_eval=df_eval, cols=cols_kld, colors=colors, val_name="avg KLD")
        ax.set_title("Dissimilarity", weight="bold", ha="left")
    else:
        ax.set_title("Dissimilarity", weight="bold")
    if legend:
        labels = [x.split("_")[3].capitalize() for x in cols_auc]
        dict_color = dict(zip(labels, colors[1:len(labels) + 1]))
        ut.plot_legend_(ax=ax, dict_color=dict_color,
                        title="Reference datasets", y=legend_y, handletextpad=0.1)
    # Adjust plot
    if dict_xlims is not None:
        for i in dict_xlims:
            # Check that KLD axis are only set if valid
            if i <= n_colss:
                axes[i].set_xlim(dict_xlims[i])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)
    return fig, axes


def plot_pca(df_pu=None, labels=None, figsize=(6, 6),
             pc_x=1, pc_y=2, show_pos_mean_x=False, show_pos_mean_y=False,
             names=None, colors=None,
             legend=True, legend_y=-0.15, args_scatter=None):
    """Generates a PCA plot based on provided parameters."""
    plt.figure(figsize=figsize)
    cols_pc = [x for x in list(df_pu) if "abs" not in x and "PC" in x]
    label_x = [x for x in cols_pc if f'PC{pc_x}' in x][0]
    label_y = [y for y in cols_pc if f'PC{pc_y}' in y][0]
    # Map colors to labels
    dict_color = {label: color for label, color in zip(sorted(set(labels)), colors)}
    fs = ut.plot_gco(option="font.size")
    # Plotting
    for label in reversed(sorted((set(labels)))):
        subset = df_pu[labels == label]
        plt.scatter(subset[label_x], subset[label_y], color=dict_color[label], label=label, **args_scatter)
    # Handling mean lines for positive samples
    if show_pos_mean_x or show_pos_mean_y:
        pos_samples = df_pu[labels == 1]  # Assuming label '1' is for positive samples
        mean_x = pos_samples[label_x].mean() if show_pos_mean_x else None
        mean_y = pos_samples[label_y].mean() if show_pos_mean_y else None
        lw = ut.plot_gco(option="lines.linewidth")
        args = dict(ha='right', fontsize=fs, color=colors[1])
        if mean_x is not None:
            str_mean_x = fr"$\bar{{x}}_{{\text{{{label_x.split(' ')[0]}}}}}$"
            plt.axvline(mean_x, color='black', linestyle='--', linewidth=lw)
            plt.text(mean_x, plt.gca().get_ylim()[1], str_mean_x,  va="top", **args)

        if mean_y is not None:
            str_mean_y = fr"$\bar{{x}}_{{\text{{{label_y.split(' ')[0]}}}}}$"
            plt.axhline(mean_y, color='black', linestyle='--', linewidth=lw)
            plt.text(plt.gca().get_xlim()[1], mean_y, str_mean_y, va="bottom", **args)

    # Legend settings
    if legend:
        dict_color = {label: color for label, color in zip(names, colors)}
        ut.plot_legend_(ax=plt.gca(), dict_color=dict_color, title="Datasets",
                        n_cols=1, y=legend_y, handletextpad=0.4, fontsize=fs, fontsize_title=fs - 1, weight_title="bold")
    # Labels
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    sns.despine()
    plt.tight_layout()
    ax = plt.gca()
    return ax