"""
This is a script for the backend of the CPPPlot.eval() method.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import aaanalysis.utils as ut

BAR_WIDTH = 0.8
COLOR_BASE = "tab:gray"

# Helper functions
def _plot_n_features(ax=None, df_eval=None, dict_color=None):
    """Plots stacked bar charts for multiple sets on a single Axes."""
    names = df_eval[ut.COL_NAME].to_list()
    list_n_feat_sets = [x[1] for x in df_eval[ut.COL_N_FEAT]]
    n_sets = len(list_n_feat_sets)
    categories = list(dict_color.keys())
    # Set the width of the bars
    for i, list_n_feat in enumerate(list_n_feat_sets):
        bottom_pos = 0  # Start at 0 for each new set
        for j, category in enumerate(categories):
            counts = list_n_feat[j]
            # Stack the bars for each category
            ax.barh(i, counts, left=bottom_pos, height=BAR_WIDTH*0.66, color=dict_color[category],
                    label=category if i == 0 else "")
            bottom_pos += counts  # Increase the left offset by the count just plotted
    ax.set_yticks(np.arange(n_sets))
    ax.set_yticklabels(names)
    ax.set_xlabel(ut.COL_N_FEAT)
    sns.despine()


def _plot_range_abs_auc(ax=None, df_eval=None):
    """Boxplot for abs AUC ranges"""
    names = df_eval[ut.COL_NAME].to_list()
    range_abs_auc = df_eval[ut.COL_RANGE_ABS_AUC].tolist()
    # Boxplot
    boxprops = dict(facecolor=COLOR_BASE, color='black')
    arg = dict(color='black')
    ax.boxplot(range_abs_auc, positions=np.arange(len(names)), vert=True,
               patch_artist=True, boxprops=boxprops,
               whiskerprops=arg, capprops=arg, medianprops=arg, showfliers=False)
    # Scatter plot for min and max values
    min_total = None
    max_total = None
    for i, quin in enumerate(range_abs_auc):
        # quin is a list with the 5-number summary: [min, 25%, 50%, 75%, max]
        min_val = quin[0]
        max_val = quin[-1]
        ax.scatter(min_val, i, color=COLOR_BASE, edgecolor="black", zorder=3, s=40)
        ax.scatter(max_val, i, color=COLOR_BASE, edgecolor="black", zorder=3, s=40)
        if min_total is None or min_val < min_total:
            min_total = min_val
        if max_total is None or max_val > max_total:
            max_total = max_val
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    x_adjust = (max_total - min_total)*0.1
    ax.set_xlim(min_total-x_adjust, max_total+x_adjust)
    ax.set_xticks([round(min_total, 2), round(max_total, 2)])
    ax.set_xticklabels([f'{min_total: .1f}', f'{max_total: .1f}'])
    ax.set_xlabel('abs AUC\n')
    sns.despine()


def _plot_avg_mean_dif(ax=None, df_eval=None):
    """Hued bar plot for average mean_dif"""
    dict_color = ut.plot_get_cdict_()
    color_pos = dict_color["FEAT_POS"]
    color_neg = dict_color["FEAT_NEG"]
    # Extract the tuples of mean differences
    list_mean_dif = df_eval[ut.COL_AVG_MEAN_DIF].tolist()
    names = df_eval[ut.COL_NAME].to_list()
    # Set the width of the bars
    bar_height = BAR_WIDTH/2
    # Plotting
    fs = ut.plot_gco()
    args_in = dict(va='center', ha='right', color="white", fontsize=fs-1)
    args_out = dict(va='center', ha='left', color="black", fontsize=fs-1)
    max_abs_dif = max([max([abs(x[0]), abs(x[1])]) for x in list_mean_dif])
    for i, (pos, neg) in enumerate(list_mean_dif):
        i_pos = i + bar_height / 2
        ax.barh(i_pos, pos, height=bar_height, color=color_pos)
        if pos > max_abs_dif/2:
            ax.text(pos, i_pos, f'{pos:.2f}', **args_in)
        else:
            ax.text(pos, i_pos, f'{pos:.2f}', **args_out)
        i_neg = i - bar_height / 2
        ax.barh(i_neg, abs(neg), height=bar_height, color=color_neg)
        if abs(neg) > max_abs_dif/2:
            ax.text(abs(neg), i_neg, f'{neg:.2f}', **args_in)
        else:
            ax.text(abs(neg), i_neg, f'{neg:.2f}', **args_out)
    # Setting the labels and title
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('mean dif\n(avg pos/neg)')


def _plot_n_clusters(ax=None, df_eval=None):
    """Plot number of clusters"""
    list_n_clusters = df_eval[ut.COL_N_CLUST].tolist()
    names = df_eval[ut.COL_NAME].to_list()
    # Plotting
    n_max = max(list_n_clusters)
    for i, n_cluster in enumerate(list_n_clusters):
        ax.barh(i, n_cluster, height=BAR_WIDTH*0.66, color=COLOR_BASE)
        if n_cluster > n_max/2:
            ax.text(n_cluster, i, n_cluster, va='center', ha='right', color="white")
        else:
            ax.text(n_cluster, i, n_cluster, va='center', ha='left', color="black")
    # Setting the labels and title
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('n_clusters')

def _plot_feat_per_cluster(ax=None, df_eval=None):
    """Plot average number of features per cluster"""
    list_avg_n_feat = df_eval[ut.COL_AVG_N_FEAT_PER_CLUST].tolist()
    list_std_n_feat = df_eval[ut.COL_STD_N_FEAT_PER_CLUST].tolist()
    names = df_eval[ut.COL_NAME].to_list()
    # Plotting
    for i, (avg_n_feat, std_n_feat) in enumerate(zip(list_avg_n_feat, list_std_n_feat)):
        ax.barh(i, avg_n_feat, height=BAR_WIDTH*0.66, color=COLOR_BASE)
        ax.errorbar(avg_n_feat, i, xerr=std_n_feat, ecolor="black", elinewidth=1,
                    capsize=5, capthick=1)
    # Setting the labels and title
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(0, x_max)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('n feat/clust\n(avg ± std)')

# II Main functions
def plot_eval(df_eval=None, figsize=(8, 5), dict_xlims=None, dict_color=None, legend=True, legend_y=-0.3, list_cat=None):
    """Plot evaluation of CPP feature sets"""
    # Reverse order to have first dataset on top
    _df_eval = df_eval.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 5, sharey=True, figsize=figsize)
    # Plotting individual components
    _plot_n_features(ax=axes[0], df_eval=_df_eval, dict_color=dict_color)
    _plot_range_abs_auc(ax=axes[1], df_eval=_df_eval)
    _plot_avg_mean_dif(ax=axes[2], df_eval=_df_eval)
    _plot_n_clusters(ax=axes[3], df_eval=_df_eval)
    _plot_feat_per_cluster(ax=axes[4], df_eval=_df_eval)
    # Set xlims
    if dict_xlims is not None:
        for i in dict_xlims:
            axes[i].set_xlim(dict_xlims[i])
    # Adding central titles
    x_min = axes[1].get_xlim()[0]
    axes[1].set_title('Discriminative Power', ha='left', va='center', x=x_min, weight="bold")
    axes[3].set_title('Redundancy', ha='left', va='center', weight="bold")
    # Set legend under plot
    n_feat_cat = np.array([x[1] for x in df_eval[ut.COL_N_FEAT]]).sum(axis=0)
    # Filter non-occurring categories
    _list_cat = [cat for cat, n in zip(dict_color, n_feat_cat) if n != 0]
    if list_cat is None:
        list_cat = _list_cat
    else:
        list_cat = [x for x in list_cat if x in _list_cat]
    plt.tight_layout()
    if legend:
        fs = ut.plot_gco()
        ut.plot_legend_(ax=axes[0], dict_color=dict_color, list_cat=list_cat, loc="upper left",
                        n_cols=np.ceil(len(list_cat) / 2), y=legend_y,
                        title="Scale category", fontsize=fs-1,
                        labelspacing=0.1, columnspacing=0.4, handletextpad=0.2, handlelength=2)
        plt.subplots_adjust(wspace=0.25, hspace=0, bottom=0.35)
    else:
        plt.subplots_adjust(wspace=0.25, hspace=0)
    return fig, axes
