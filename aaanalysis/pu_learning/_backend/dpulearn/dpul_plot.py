"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import aaanalysis.utils as ut


# I Helper Functions
def _plot_neg_homogeneity(ax=None, df_eval=None, col=None, colors=None, val_name="Value"):
    """"""
    sns.barplot(ax=ax, data=df_eval, y=ut.COL_NAME, x=col, color=colors[0])
    # Adding the numbers to the end of the bars
    for p in ax.patches:
        val = p.get_width()
        val = '{:.3f}'.format(val) if val < 1 else int(val)
        ax.text(p.get_width() * 0.97, p.get_y() + p.get_height() / 2, val, ha='right', va='center')
    # Customize subplots
    ax.set_ylabel("")
    ax.set_xlabel(val_name)
    # Adjust spines
    ax = ut.adjust_spines(ax=ax)
    ut.x_ticks_0(ax=ax)
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
    ax = ut.adjust_spines(ax=ax)
    ut.x_ticks_0(ax=ax)
    ax.tick_params(axis='y', which='both', left=False)
    return ax


# II Main Functions
def plot_eval(df_eval=None, figsize=None, colors=None, legend=True, legend_y=-0.175):
    """Plot evaluation of AAclust clustering results"""
    # Plotting
    cols_eval = [x for x in ut.COLS_EVAL_DPULEARN if x in list(df_eval)]
    cols_homogeneity = cols_eval[1:3]
    cols_auc = [c for c in cols_eval if "AUC" in c]
    cols_kld = [c for c in cols_eval if "KLD" in c]
    kld_in = len(cols_kld) != 0
    ncols = 3 if not kld_in else 4
    fig, axes = plt.subplots(1, ncols, sharey=True, figsize=figsize)
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
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)
    return fig, axes

"""
def _plot_pca(df_pred=None, filter_classes=None, x=None, y=None,  others=True, highlight_rel=True,
              figsize=(6, 6), highlight_mean=True, list_classes=None, dict_color=None):
    if dict_color is None:
        pass
    plt.figure(figsize=figsize)
    # Filtering
    x_min, x_max = df_pred[x].min(), df_pred[x].max()
    y_min, y_max = df_pred[y].min(), df_pred[y].max()
    df_pred = df_pred.copy()
    if filter_classes is not None:
        mask = [x in filter_classes for x in df_pred[ut.COL_CLASS]]
        df_pred = df_pred[mask]
    df_pred[ut.COL_CLASS] = [ut.CLASS_NONSUB_PRED if ut.CLASS_NONSUB_PRED in x else x for x in df_pred[ut.COL_CLASS]]
    # Plotting
    ut.plot_settings()
    if list_classes is None:
        list_classes = [] if not others else [ut.CLASS_OTHERS]
        list_classes.extend([ut.CLASS_NONSUB_PRED, ut.CLASS_NONSUB, ut.CLASS_SUBEXPERT])
    if not highlight_rel:
        dict_color[ut.CLASS_NONSUB_PRED] = ut.COLOR_OTHERS
    for c in list_classes:
        d = df_pred[df_pred[ut.COL_CLASS] == c].copy()
        ax = sns.scatterplot(data=d, x=x, y=y, color=dict_color[c],
                             legend=False, alpha=1)
        f_min = lambda i: i - abs(i*0.1)
        f_max = lambda i: i + abs(i*0.1)
        plt.ylim(f_min(y_min), f_max(y_max))
        plt.xlim(f_min(x_min), f_max(x_max))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            f = lambda x: [f'{i:.2f}' for i in x]
            ax.set_yticklabels(f(ax.get_yticks()), size=ut.LEGEND_FONTSIZE-2)
            ax.set_xticklabels(f(ax.get_xticks()), size=ut.LEGEND_FONTSIZE-2)
    if not highlight_rel:
        list_classes.remove(ut.CLASS_NONSUB_PRED)
    list_cat = [x for x in ut.LIST_CLASSES if x in list_classes]
    ax = ut.set_legend_handles_labels(ax=plt.gca(), dict_color=dict_color, list_cat=list_cat,
                                      ncol=2, fontsize=ut.LEGEND_FONTSIZE)
    sns.despine()
    plt.tight_layout()
    # Highlight mean values
    if highlight_mean:
        df = df_pred[df_pred[ut.COL_CLASS] == ut.CLASS_SUBEXPERT].copy()
        # TODO check for axis!
        mean_x = df[x].mean()
        mean_y = df[y].mean()
        color = dict_color[ut.CLASS_SUBEXPERT]
        plt.axhline(mean_y, color="black", linestyle="--", linewidth=1.75)
        plt.axvline(mean_x, color="black", linestyle="--", linewidth=1.75)
        x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]
        plt.text(mean_x, y_max, f" mean {x.split(' ')[0]}", va="top", ha="left", color=color)
        plt.text(x_max, mean_y, f"mean {y.split(' ')[0]}", va="bottom", ha="right", color=color)
    return plt.gcf()
"""

