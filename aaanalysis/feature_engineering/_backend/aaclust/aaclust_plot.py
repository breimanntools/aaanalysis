"""
This is a script for the backend of the AAclustPlot object for all plotting functions.
"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import aaanalysis.utils as ut
import matplotlib.ticker as mticker


# I Helper Functions
# Computation helper functions
def _get_mean_rank(data):
    """"""
    _df = data.copy()
    _df['BIC_rank'] = _df[ut.COL_BIC].rank(ascending=False)
    _df['CH_rank'] = _df[ut.COL_CH].rank(ascending=False)
    _df['SC_rank'] = _df[ut.COL_SC].rank(ascending=False)
    rank = _df[['BIC_rank', 'CH_rank', 'SC_rank']].mean(axis=1).round(2)
    return rank

def _get_components(data=None, model_class=None):
    """"""

# Plotting helper functions
def _adjust_spines(ax=None):
    """Adjust spines to be in middle if data range from <0 to >0"""
    min_val, max_val = ax.get_xlim()
    if max_val > 0 and min_val >= 0:
        sns.despine(ax=ax)
    else:
        sns.despine(ax=ax, left=True)
        current_lw = ax.spines['bottom'].get_linewidth()
        ax.axvline(0, color='black', linewidth=current_lw)
        val = max([abs(min_val), abs(max_val)])
        ax.set_xlim(-val, val)
    return ax


def _x_ticks_0(ax):
    """Apply custom formatting for x-axis ticks."""
    def custom_x_ticks(x, pos):
        """Format x-axis ticks."""
        return f'{x:.2f}' if x else f'{x:.0f}'
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_x_ticks))




# II Main Functions
def plot_eval(df_eval=None, dict_xlims=None, figsize=None, colors=None):
    """Plot evaluation of AAclust clustering results"""
    df_eval[ut.COL_RANK] = _get_mean_rank(df_eval)
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
              figsize=(6, 6), highlight_mean=True, list_classes=None):
    """"""
    plt.figure(figsize=figsize)
    dict_color = ut.DICT_COLORS.copy()
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
        mean_x = df[x].mean()
        mean_y = df[y].mean()
        color = dict_color[ut.CLASS_SUBEXPERT]
        plt.axhline(mean_y, color="black", linestyle="--", linewidth=1.75)
        plt.axvline(mean_x, color="black", linestyle="--", linewidth=1.75)
        x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]
        plt.text(mean_x, y_max, f" mean {x.split(' ')[0]}", va="top", ha="left", color=color)
        plt.text(x_max, mean_y, f"mean {y.split(' ')[0]}", va="bottom", ha="right", color=color)
    return plt.gcf()

def plot_correlation(df_corr=None):
    """"""
    ax = sns.heatmap(df_corr, cmap="viridis", vmin=-1, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
