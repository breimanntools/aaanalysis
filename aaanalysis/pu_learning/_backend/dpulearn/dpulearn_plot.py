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


# II Main Functions
def plot_eval(df_eval=None, dict_xlims=None, figsize=None, colors=None):
    """Plot evaluation of AAclust clustering results"""
    df_eval = df_eval.sort_values(by=ut.COL_RANK, ascending=True)
    # Plotting
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=figsize)
    for i, col in enumerate(ut.COLS_EVAL_AACLUST):
        ax = axes[i]
        sns.barplot(ax=ax, data=df_eval, y=df_eval.index, x=col, color=colors[i])
        # Customize subplots
        ax.set_ylabel("")
        ax.set_xlabel(col)
        # Adjust spines
        ax = _adjust_spines(ax=ax)
        # Manual xlims, if needed
        if dict_xlims and col in dict_xlims:
            ax.set_xlim(dict_xlims[col])
        if i == 0:
            ax.set_title("Clustering", weight="bold")
        elif i == 2:
            ax.set_title("Quality measures", weight="bold")
        ax.tick_params(axis='y', which='both', left=False)
        _x_ticks_0(ax=ax)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)
    return fig, axes


def _plot_pca(df_pred=None, filter_classes=None, x=None, y=None,  others=True, highlight_rel=True,
              figsize=(6, 6), highlight_mean=True, list_classes=None, dict_color=None):
    """"""
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


