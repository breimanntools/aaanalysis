"""
This is a script for the backend of the cpp_plot.ranking method
"""
import statistics
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import aaanalysis as aa
import aaanalysis.utils as ut
from ._utils_cpp_plot import add_part_seq, add_feature_title, get_color_dif
from ._utils_feature import get_feature_matrix_, get_positions_


# I Helper Functions
# Adjust df_feat
def adjust_df_feat(df_feat=None, col_dif=None, xlim_dif=None, feature_val_in_percent=True):
    """"""
    df_feat = df_feat.copy()
    if feature_val_in_percent:
        if max(df_feat[col_dif]) - min(df_feat[col_dif]) < 1:
            df_feat[col_dif] *= 100
    else:
        if max(df_feat[col_dif]) - min(df_feat[col_dif]) > 1:
            df_feat[col_dif] /= 100
            xlim_dif = (xlim_dif[0]/100, xlim_dif[1]/100)
    return df_feat, xlim_dif


# Add feature position
def _add_pos_bars(ax=None, x=None, y=None, color="tab:gray", height=0.5, width=1.0, alpha=1.0):
    """"""
    bar = mpl.patches.Rectangle((x, y), width=width, height=height, linewidth=0, color=color, zorder=4, clip_on=False,
                                alpha=alpha)
    ax.add_patch(bar)


def _add_position_bars(ax=None, df_feat=None):
    """"""
    for y, row in df_feat.iterrows():
        positions = [int(x) for x in row[ut.COL_POSITION].split(",")]
        is_segment = positions[-1] - positions[0] == len(positions) - 1
        # Add segment or pattern
        if is_segment:
            _add_pos_bars(ax=ax, x=positions[0] - 1, y=y - 0.25, width=len(positions))
        else:
            # Add pattern
            for pos in positions:
                _add_pos_bars(ax=ax, x=pos - 1, y=y - 0.25)


def _get_tmd_jmd_label(jmd_n_len=10, jmd_c_len=10, space=3):
    """"""
    jmd_len = jmd_c_len + jmd_n_len
    # Space factors should be between 1 and max-1
    total_space = space*2
    space_n_factor = max(min(int(round(jmd_n_len/jmd_len*total_space)), total_space-1), 1)
    space_c_factor = total_space - space_n_factor
    x_label = ""
    x_label += "JMD-N" + " " * space_n_factor if jmd_n_len > 0 else " " * (4 + space_c_factor)
    x_label += "TMD"
    x_label += " " * space_c_factor + "JMD-C" if jmd_c_len > 0 else " " * (4 + space_n_factor)
    return x_label


def plot_feature_position(ax=None, df=None, n=20, space=3, tmd_len=20, jmd_n_len=10, jmd_c_len=10, fontsize_label=None):
    """"""
    height = 0.01 * n
    plt.sca(ax)
    # Set y ticks
    plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    list_ticks = list(reversed(df[ut.COL_SUBCAT]))
    plt.yticks(range(0, n), list_ticks)
    # Set x ticks
    x_max = jmd_n_len + tmd_len + jmd_c_len
    plt.xticks(range(0, x_max))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    sns.despine(top=True, right=True, left=False, bottom=True)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    add_part_seq(ax=ax, y=len(df)-0.5, height=height, **args_len)
    add_part_seq(ax=ax, y=-0.5, height=len(df), alpha=0.075, **args_len)
    _add_position_bars(ax=ax, df_feat=df)
    # Adjust xlabel
    x_label = _get_tmd_jmd_label(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, space=space)
    plt.xlabel(x_label, size=fontsize_label)


# Add feature mean difference
def _get_std_dif(df_feat=None, df_parts=None, labels=None, in_percent=True, df_scales=None):
    """Compute pooled standard deviation between difference of mean(reference) and mean(test) as follows:

        SDpooled = √((n1-1)*SD1*SD1 + (n2-1)*SD2*SD2)/(n1+n2-2)

        where,
        SD1 = Standard Deviation for group 1
        SD2 = Standard Deviation for group 2
        n1 =  Sample Size for group 1
        n2 =  Sample Size for group 2
    """
    X = get_feature_matrix_(features=df_feat[ut.COL_FEATURE].to_list(), df_parts=df_parts, df_scales=df_scales)
    mask_test = [bool(x) for x in labels]
    mask_ref = [not bool(x) for x in labels]
    list_pooled_std = []
    for i in range(len(df_feat)):
        f_std = lambda x: statistics.stdev(x)
        vals1 = X[:, i][mask_ref]
        vals2 = X[:, i][mask_test]
        sd1, sd2 = f_std(vals1), f_std(vals2)
        n1, n2 = len(vals1), len(vals2)
        pooled_std = math.sqrt((sd1*sd1 * (n1-1) + (sd2*sd2 * (n2-1))))/(n1+n2+2)
        factor = 100 if in_percent else 1
        list_pooled_std.append(round(pooled_std, 3) * factor)
    return list_pooled_std


def _add_annotation_extreme_val(sub_fig=None, max_neg_val=-2, min_pos_val=2, text_size=9):
    """"""
    args = dict(va='center_baseline', size=text_size, xytext=(0.5, 0), textcoords='offset points',
                color="white")
    for p in sub_fig.patches:
        val = round(p.get_width(), 1)
        if val <= max_neg_val:
            x = max_neg_val + 1
            ha = "left"
            sub_fig.annotate(f"{int(val)}%", (x, p.get_y() + p.get_height() / 2), ha=ha, **args)
        if val >= min_pos_val:
            x = min_pos_val - 1
            ha = "right"
            sub_fig.annotate(f"{int(val)}%", (x, p.get_y() + p.get_height()/2), ha=ha, **args)


def plot_feature_mean_dif(ax=None, df=None, df_parts=None, df_scales=None, labels=None, x_dif=None, n=20, xlim=(-22, 22), error_bar=True,
                          fontsize_annotation=8, in_percent=True, capsize=2):
    """"""
    plt.sca(ax)
    colors = [get_color_dif(mean_dif=x) for x in df[x_dif]]
    df["hue"] = colors     # Adjust to use palette for sns.barplot after v0.14.0
    args = dict(hue="hue", palette={x: x for x in colors}, legend=False)
    sub_fig = sns.barplot(ax=ax, data=df, y="feature", x=x_dif, **args)
    if error_bar:
        pooled_std = _get_std_dif(df_feat=df, labels=labels, in_percent=in_percent, df_parts=df_parts,
                                  df_scales=df_scales)
        plt.errorbar(x=df[x_dif], y=range(0, n), xerr=pooled_std,
                     fmt='none', c='tab:gray', capsize=capsize)
    sns.despine(top=True, right=True, left=False, bottom=False)
    _add_annotation_extreme_val(sub_fig=sub_fig, text_size=fontsize_annotation,
                                max_neg_val=xlim[0], min_pos_val=xlim[1])
    # Add values for importance
    plt.axvline(x=0, color='gray', linestyle='-')
    plt.yticks(range(0, n), list(df[ut.COL_SUBCAT]))
    plt.xlim(xlim)
    x_tick_max = int(xlim[1]/5)*5 if not (0 < xlim[1] < 1) else int(xlim[1]*100/5)*5/100
    x_tick_min = int(xlim[0]/5)*5 if not (0 < abs(xlim[0]) < 1) else int(xlim[0]*100/5)*5/100
    plt.xticks(ticks=[x_tick_min, 0, x_tick_max], labels=[str(x_tick_min), "0", str(x_tick_max)])
    plt.ylabel("")


# Add feature importance
def _add_annotation_right(sub_fig=None, an_in_val=2, max_val=10.0, text_size=8):
    """"""
    args = dict(va='center_baseline', size=text_size, xytext=(0.5, 0), textcoords='offset points')
    args_left = dict(ha='left', **args)
    args_right = dict(ha="right", **args, color="white")
    for p in sub_fig.patches:
        val = round(p.get_width(), 1)
        if val < an_in_val:
            sub_fig.annotate(f"{round(val, 1)}%", (val, p.get_y() + p.get_height()/2), **args_left)
        else:
            x = max_val if val > max_val else val
            sub_fig.annotate(f"{round(val, 1)}%", (x, p.get_y() + p.get_height()/2), **args_right)


def plot_feature_rank(ax=None, df=None, n=20, xlim=(0, 8),
                      fontsize_annotation=8, col_rank=ut.COL_FEAT_IMPORT,
                      shap_plot=False):
    """"""
    df = df.copy()
    plt.sca(ax)
    if shap_plot:
        colors = [ut.COLOR_SHAP_NEG if v < 0 else ut.COLOR_SHAP_POS for v in df[col_rank]]
    else:
        colors = ["tab:gray"] * len(df)
    df[col_rank] = abs(df[col_rank]).round(1)
    df["hue"] = colors     # Adjust to use palette for sns.barplot after v0.14.0
    args = dict(hue="hue", palette={x: x for x in colors}, legend=False)
    sub_fig = sns.barplot(ax=ax, data=df, y="feature", x=col_rank, **args)
    plt.yticks(range(0, n), list(df[ut.COL_SUBCAT]))
    plt.ylabel("")
    x_max = df[col_rank].max()
    if xlim[1] < x_max:
        xlim = (xlim[0], x_max)
    plt.xlim(xlim)
    _add_annotation_right(sub_fig=sub_fig, text_size=fontsize_annotation, an_in_val=x_max/2, max_val=xlim[1])
    str_sum = f"Σ={round(df[col_rank].sum(), 1)}%"
    args = dict(ha="right", size=fontsize_annotation)
    x = xlim[1] * 1.2
    plt.text(x, n-2.5, str_sum, weight="normal", **args)
    if shap_plot:
        plt.text(x, n-1.5, "negative", weight="bold", color=ut.COLOR_SHAP_NEG, va="top", **args)
        plt.text(x, n-1.5, "positive", weight="bold", color=ut.COLOR_SHAP_POS, va="bottom", **args)


# II Main Functions
# TODO adjust to work with flexible TMD/JMD length choice
def plot_ranking(figsize=(7, 5), df_feat=None, top_n=25, df_scales=None, labels=None,
                 df_parts=None, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                 name_test="test", name_ref="ref",
                 fontsize_titles=11,
                 fontsize_labels=11,
                 fontsize_annotations=11,
                 feature_val_in_percent=True,
                 error_bar=True, shap_plot=False,
                 capsize=2, tmd_jmd_space=2,
                 col_rank=ut.COL_FEAT_IMPORT, xlim_dif=(-17.5, 17.5),
                 col_dif=ut.COL_MEAN_DIF, xlim_rank=(0, 8)):
    """"""
    df_feat = df_feat.copy()
    # Adjust df_feat
    df_feat = df_feat.copy().reset_index(drop=True).head(top_n)
    df_feat, xlim_dif = adjust_df_feat(df_feat=df_feat, col_dif=col_dif, xlim_dif=xlim_dif,
                                       feature_val_in_percent=feature_val_in_percent)
    df_feat[ut.COL_POSITION] = get_positions_(features=df_feat[ut.COL_FEATURE],
                                              tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    # Plotting (three subplots)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)
    # 1. Plot feature positions
    plot_feature_position(ax=axes[0], df=df_feat, n=top_n, space=tmd_jmd_space,
                          tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                          fontsize_label=fontsize_labels)
    y = -top_n/25   # Empirically optimized location
    add_feature_title(y=y, fontsize_title=fontsize_titles)
    # 2. Barplot mean difference
    plot_feature_mean_dif(ax=axes[1], df=df_feat, df_parts=df_parts, df_scales=df_scales, labels=labels,
                          n=top_n, x_dif=col_dif, xlim=xlim_dif,
                          error_bar=error_bar, in_percent=feature_val_in_percent,
                          capsize=capsize,
                          fontsize_annotation=fontsize_annotations)
    sns.despine(ax=axes[1], top=True, right=True, left=True, bottom=False)
    plt.title(f"Mean difference\nof feature value", size=fontsize_titles, weight="bold")
    label_mean_dif = f"{name_test} - {name_ref}"
    label_mean_dif += " [%]" if feature_val_in_percent else ""
    plt.xlabel(label_mean_dif, size=fontsize_labels)
    # 3. Barplot importance
    plot_feature_rank(ax=axes[2], df=df_feat, n=top_n, xlim=xlim_rank,
                      col_rank=col_rank, shap_plot=shap_plot,
                      fontsize_annotation=fontsize_annotations)
    plt.title(f"{ut.LABEL_FEAT_RANKING}\n(top {top_n} features)", size=fontsize_titles, ha="center", weight="bold")
    label_ranking = ut.LABEL_FEAT_IMPACT if shap_plot else ut.LABEL_FEAT_IMPORT
    plt.xlabel(label_ranking, size=fontsize_labels)
    # Adjust axis
    for i in range(3):
        axes[i].tick_params(which='major', axis="both", labelsize=fontsize_labels)
        if i > 0:
            axes[i].tick_params(which='major', axis="y", length=0, labelsize=0)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    fig = plt.gcf()
    ax = plt.gca()
    return fig, ax