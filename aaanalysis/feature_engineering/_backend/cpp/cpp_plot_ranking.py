"""
This is a script for the backend of the CPPPlot.ranking() method.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import aaanalysis.utils as ut
from .utils_feature import get_positions_


# I Helper Functions
# Adjust df_feat
def _adjust_df_feat(df_feat=None, col_dif=None):
    """Adjusts feature values in `df_feat` based on percentage scaling and sets the limits for difference columns."""
    df_feat = df_feat.copy()
    if max(df_feat[col_dif]) - min(df_feat[col_dif]) <= 2:
        df_feat[col_dif] *= 100
    return df_feat


# 1. Subplot: Feature position
def _add_pos_bars(ax=None, x=None, y=None, color="tab:gray", height=0.5, width=1.0, alpha=1.0):
    """Adds a rectangle (bar) to the axis `ax` at specified `x`, `y` position with given dimensions and style."""
    bar = mpl.patches.Rectangle((x, y), width=width, height=height, linewidth=0, color=color, zorder=4, clip_on=False,
                                alpha=alpha)
    ax.add_patch(bar)


def _add_position_bars(ax=None, df_feat=None):
    """Plots position bars for each feature in `df_feat` on the given axis `ax`."""
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


def _add_part_seq(ax=None, jmd_n_len=10, jmd_c_len=10, tmd_len=20, y=-0.75, height=0.2,
                 tmd_color="mediumspringgreen", jmd_color="blue", alpha=1.0, start=0.0):
    """Add colored box for sequence parts in figure"""
    list_color = [jmd_color, tmd_color, jmd_color]
    list_length = [jmd_n_len, tmd_len, jmd_c_len]
    # Add jmd_n
    for length, color in zip(list_length, list_color):
        bar = mpl.patches.Rectangle((start, y), width=length, height=height, linewidth=0, color=color, zorder=4,
                                    clip_on=False, alpha=alpha)
        start += length
        ax.add_patch(bar)


def _get_tmd_jmd_label(jmd_n_len=10, jmd_c_len=10, space=3):
    """Generates a label for TMD and JMD regions based on their lengths and spacing."""
    jmd_len = jmd_c_len + jmd_n_len
    name_tmd = ut.options["name_tmd"]
    name_jmd_n = ut.options["name_jmd_n"]
    name_jmd_c = ut.options["name_jmd_c"]
    if jmd_len == 0:
        return name_tmd
    # Space factors should be between 1 and max-1
    total_space = space*2
    space_n_factor = max(min(int(round(jmd_n_len / jmd_len * total_space)), total_space - 1), 1)
    space_c_factor = total_space - space_n_factor
    x_label = ""
    x_label += name_jmd_n + " " * space_n_factor if jmd_n_len > 0 else " " * (4 + space_c_factor)
    x_label += name_tmd
    x_label += " " * space_c_factor + name_jmd_c if jmd_c_len > 0 else " " * (4 + space_n_factor)
    return x_label


def plot_feature_position(ax=None, df=None, n=20, space=3, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                          tmd_color="mediumspringgreen", jmd_color="blue", fontsize_label=None, tmd_jmd_alpha=0.075):
    """Plots the feature positions for a given DataFrame `df` on the axis `ax` with specified formatting parameters."""
    fig_height = plt.gcf().get_size_inches()[1]
    height = min([0.01 * n, 0.2]) * 5/fig_height
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
    args_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    # Add sequence part under plot
    _add_part_seq(ax=ax, y=len(df)-0.5, height=height, **args_len, **args_color)
    # Add sequence area in plot
    _add_part_seq(ax=ax, y=-0.5, height=len(df), alpha=tmd_jmd_alpha, **args_len, **args_color)
    _add_position_bars(ax=ax, df_feat=df)
    # Adjust xlabel
    x_label = _get_tmd_jmd_label(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, space=space)
    plt.xlabel(x_label, size=fontsize_label)


# 2. Subplot: Feature mean difference
def _add_annotation_extreme_val(sub_fig=None, max_neg_val=-2, min_pos_val=2, text_size=9):
    """Adds annotations for extreme values in a subplot `sub_fig`, defining thresholds for maximum negative and minimum positive values."""
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


def plot_feature_mean_dif(ax=None, df=None, col_dif=None, n=20, xlim=(-22, 22), fontsize_annotation=8):
    """Plots the mean difference of features in `df` on the axis `ax`, with custom range `xlim` and annotation font size."""
    plt.sca(ax)
    colors = [ut.get_color_dif(mean_dif=x) for x in df[col_dif]]
    df["hue"] = colors     # Adjust to use palette for sns.barplot after v0.14.0
    args = dict(hue="hue", palette={x: x for x in colors}, legend=False)
    sub_fig = sns.barplot(ax=ax, data=df, y="feature", x=col_dif, **args)
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


# 3. Subplot: Ranking based on absolute AUC, feature importance or feature impact
def _add_annotation_right(sub_fig=None, an_in_val=2, max_val=10.0, text_size=8):
    """Adds right-aligned annotations for feature importance in a subplot `sub_fig`, considering an inner value threshold and maximum value."""
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


def plot_feature_rank(ax=None, df_feat=None, n=20, xlim=(0, 4),
                      fontsize_annotation=8, col_imp=ut.COL_FEAT_IMPORT,
                      shap_plot=False, rank_info_xy=None):
    """Plots the feature ranking based on `df` on the axis `ax`, adjusting for SHAP values if `shap_plot` is True."""
    df_feat = df_feat.copy()
    plt.sca(ax)
    if shap_plot:
        colors = [ut.COLOR_SHAP_NEG if v < 0 else ut.COLOR_SHAP_POS for v in df_feat[col_imp]]
    else:
        colors = ["tab:gray"] * len(df_feat)
    df_feat[col_imp] = abs(df_feat[col_imp]).round(1)
    df_feat["hue"] = colors     # Adjust to use palette for sns.barplot after v0.14.0
    args = dict(hue="hue", palette={x: x for x in colors}, legend=False)
    sub_fig = sns.barplot(ax=ax, data=df_feat, y="feature", x=col_imp, **args)
    plt.yticks(range(0, n), list(df_feat[ut.COL_SUBCAT]))
    plt.ylabel("")

    # Set x-axis limits
    xlim = plt.xlim() if xlim is None else xlim
    x_max = max(df_feat[col_imp].max(), xlim[1])
    xlim = (xlim[0], x_max)
    plt.xlim(xlim)

    _add_annotation_right(sub_fig=sub_fig, text_size=fontsize_annotation, an_in_val=x_max/2, max_val=xlim[1])
    # Add legend
    str_sum = f"Σ={round(df_feat[col_imp].sum(), 1)}%"
    args = dict(ha="right", size=fontsize_annotation)
    rank_info_xy_default = (xlim[1]*1.1, n-2.5)
    _rank_info_xy = ut.adjust_tuple_elements(tuple_in=rank_info_xy,
                                            tuple_default=rank_info_xy_default)
    x, y = _rank_info_xy
    plt.text(x, y, str_sum, weight="normal", **args)
    if shap_plot:
        plt.text(x, y+1, "negative", weight="bold", color=ut.COLOR_SHAP_NEG, va="top", **args)
        plt.text(x, y+1, "positive", weight="bold", color=ut.COLOR_SHAP_POS, va="bottom", **args)


# II Main Functions
def plot_ranking(df_feat=None,
                 n_top=15, rank=False,
                 col_dif=None, col_imp=None,
                 shap_plot=False,
                 figsize=(7, 5),
                 tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                 tmd_color="mediumspringgreen", jmd_color="blue",
                 tmd_jmd_alpha=0.075,
                 name_test="test", name_ref="ref",
                 fontsize_titles=11,
                 fontsize_labels=11,
                 fontsize_annotations=11,
                 tmd_jmd_space=2,
                 xlim_dif=(-17.5, 17.5),
                 xlim_rank=(0, 4),
                 rank_info_xy=None):
    """Plot ranking of feature DataFrame"""
    # Adjust df_feat
    if rank:
        df_feat = df_feat.sort_values(by=col_imp, key=lambda x: abs(x), ascending=False)
    df_feat = df_feat.head(n_top).copy().reset_index(drop=True)
    df_feat = _adjust_df_feat(df_feat=df_feat, col_dif=col_dif)
    df_feat[ut.COL_POSITION] = get_positions_(features=df_feat[ut.COL_FEATURE],
                                              tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    # Plotting (three subplots)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)
    # 1. Plot feature positions
    plot_feature_position(ax=axes[0], df=df_feat, n=n_top, space=tmd_jmd_space,
                          tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                          tmd_color=tmd_color, jmd_color=jmd_color, tmd_jmd_alpha=tmd_jmd_alpha,
                          fontsize_label=fontsize_labels)
    plt.title(ut.LABEL_FEAT_POS, x=0, weight="bold", fontsize=fontsize_titles)
    # 2. Barplot mean difference
    plot_feature_mean_dif(ax=axes[1], df=df_feat,
                          n=n_top, col_dif=col_dif, xlim=xlim_dif,
                          fontsize_annotation=fontsize_annotations)
    sns.despine(ax=axes[1], top=True, right=True, left=True, bottom=False)
    axes[1].set_title(f"Mean difference\nof feature value",
                      size=fontsize_titles, weight="bold")
    label_mean_dif = f"{name_test} - {name_ref} [%]"
    axes[1].set_xlabel(label_mean_dif, size=fontsize_labels)
    # 3. Barplot importance
    plot_feature_rank(ax=axes[2], df_feat=df_feat, n=n_top, xlim=xlim_rank,
                      col_imp=col_imp, shap_plot=shap_plot, rank_info_xy=rank_info_xy,
                      fontsize_annotation=fontsize_annotations)
    axes[2].set_title(f"{ut.LABEL_FEAT_RANKING}\n(top {n_top} features)",
                      size=fontsize_titles, ha="center", weight="bold")
    label_ranking = ut.LABEL_FEAT_IMPACT if shap_plot else ut.LABEL_FEAT_IMPORT
    axes[2].set_xlabel(label_ranking, size=fontsize_labels)
    # Adjust axis
    for i, ax in enumerate(axes):
        ax.tick_params(which='major', axis="both", labelsize=fontsize_labels)
        if i > 0:
            ax.tick_params(which='major', axis="y", length=0, labelsize=0)
    return fig, axes
