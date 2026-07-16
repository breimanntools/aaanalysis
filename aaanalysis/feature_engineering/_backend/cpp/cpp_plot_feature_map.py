"""
This is a script for the backend of the CPPPlot.feature_map method.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import aaanalysis.utils as ut

from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_positions import PlotPartPositions
from ._utils_cpp_plot_map import plot_heatmap_
from ._utils_cpp_plot import get_sorted_list_cat_


# The cumulative-importance bar strips (top per-position, right per-subcategory) are sized in GRID
# CELL units, not figure fractions, so they lock to a near-constant physical size like the cheat-sheet
# reference instead of ballooning on dense grids or vanishing on sparse ones. Each strip scales gently
# with the grid's perpendicular extent (1/6 of a cell per grid cell -- reproduces the cheat sheet:
# 28 subcat -> 4.7-cell top, 40 positions -> 6.7-cell right) but is CLAMPED to a [min, max] cell range.
# This cell-unit clamp supersedes the earlier absolute-inch column cap (the constant-cell sizer keeps
# the column at right_cells * cell_w inches, so a separate inch cap is no longer needed).
_IMP_BAR_SCALE = 1 / 6
_IMP_BAR_TOP_MIN_CELLS, _IMP_BAR_TOP_MAX_CELLS = 2.5, 5.5
_IMP_BAR_RIGHT_MIN_CELLS, _IMP_BAR_RIGHT_MAX_CELLS = 3.5, 7.5

# Length (points) of the top strip's y-tick mark, and the offset its value is drawn at. Both render on
# the INNER (left) side of the right-hand spine, so the value is offset far enough to clear the mark.
_IMP_BAR_TICK_LEN = 3
_IMP_BAR_TICK_PAD = _IMP_BAR_TICK_LEN + 1.5

# Absolute backstop (inches) on the right importance-bar column width. The cell-unit clamp above
# already bounds the column on the constant-cell sizer path (column = right_cells * cell_w). On the
# FIXED-figure path (no sizer: an explicit figsize with auto_font off / cell_size unset) the column
# would otherwise grow with the figure and the bars run edge-to-edge, so cap it here as well.
_MAX_IMP_COL_IN = 1.5


# I Helper Functions


# Add feature importance plot elements
def plot_feat_importance_bars_subcat(ax=None,
                                     df_feat=None,
                                     df_cat=None,
                                     col_cat=None,
                                     col_imp=None,
                                     shap_plot=False,
                                     annotation_th=None,
                                     label=None,
                                     fontsize_label=12,
                                     fontsize_annotations=11,
                                     fontsize_imp_bar=9,
                                     ha="left",
                                     position=(1, 0),
                                     multialignment="right",
                                     weight_annotation="bold"):
    """Display a vertical bar plot (y-axis) for feature importance/impact sorted by categories"""
    list_cat = get_sorted_list_cat_(df_cat=df_cat,
                                    list_cat=df_feat[col_cat].to_list(),
                                    col_cat=col_cat)
    if shap_plot:
        # Cumulative feature impact per subcategory, stacked in ONE direction (matching the
        # top per-position bars): one thin white-edged segment per contributing feature
        # (red=positive, blue=negative), stacked in the feature order (NOT grouped by sign),
        # so red and blue interleave as they occur rather than piling all negatives at the
        # tail. Index each row numerically (row i in list_cat order, every row reserves its
        # slot even with no impact) so bars stay aligned with the heatmap rows -- see
        # test_shap_impact_bar_aligns_with_correct_row.
        df = df_feat[[col_cat, col_imp]]
        totals = []
        for i, cat in enumerate(list_cat):
            vals = df.loc[df[col_cat] == cat, col_imp].values
            left = 0.0
            for v in vals:
                if v == 0:
                    continue
                color = ut.COLOR_SHAP_POS if v > 0 else ut.COLOR_SHAP_NEG
                ax.barh(i, abs(v), left=left, color=color,
                        edgecolor="white", linewidth=0.3, align="edge")
                left += abs(v)
            totals.append(left)
    else:
        # Get feature importance per scale class
        df_imp = df_feat[[col_cat, col_imp]].groupby(by=col_cat).sum()
        dict_imp = dict(zip(df_imp.index, df_imp[col_imp]))
        list_imp = [dict_imp[x] for x in list_cat]
        ax.barh(list_cat, list_imp, color=ut.COLOR_FEAT_IMP, edgecolor=None, align="edge")
    sns.despine(ax=ax, bottom=True, top=False, left=False)

    # Add label
    ax.set_xlabel(label, size=fontsize_label, weight="bold",
                  ha=ha, position=position, multialignment=multialignment)
    ax.xaxis.set_label_position("top")
    # Add annotations (only for non-signed cumulative importance bars). Draw the % label
    # INSIDE each bar, right-aligned at the bar tip in white, so high-impact features read
    # immediately with the label sitting on the bar rather than outside it. Only bars
    # at/above the threshold are annotated, so they are long enough to hold the label.
    if not shap_plot:
        # epsilon avoids 2.0000000000000004 -> 3; floor of 1 avoids a degenerate 0-width axis when
        # every importance is (near) zero.
        v_max = max(1, int(np.ceil(max(list_imp) - 1e-9)))
        annotation_th = v_max / 2 if annotation_th is None else annotation_th
        for i, val in enumerate(list_imp):
            if val >= annotation_th:
                ax.text(val, i + 0.45, f"{round(val, 1)}% ",
                        va="center", ha="right",
                        weight=weight_annotation,
                        color="white",
                        size=fontsize_imp_bar)

    # Adjust ticks
    ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    ax.tick_params(axis='x', which='both', labelsize=fontsize_annotations,
                   pad=0, length=3)
    for label in ax.get_yticklabels():
        label.set_visible(False)

    if shap_plot:
        ax.set_xlim(0, max(totals + [0]))
    else:
        ax.set_xlim(0, v_max)


def plot_feat_importance_bars_pos(ax=None,
                                  df_feat=None,
                                  df_cat=None,
                                  col_cat=None,
                                  col_imp=None,
                                  shap_plot=False,
                                  start=1,
                                  tmd_len=20,
                                  jmd_n_len=10,
                                  jmd_c_len=10):
    """Display a horizontal (x-axis) bar plot for feature importance/impact per position"""
    # Get feature importance per position class
    pp = PlotPartPositions(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat.copy(),
                           col_cat=col_cat, col_val=col_imp,
                           value_type="sum", normalize=False)
    # Plot bars
    if shap_plot:
        # Cumulative feature impact per position, stacked in one direction with one thin
        # white-edged segment per contributing feature (red=positive, blue=negative), stacked
        # in the subcategory order (NOT grouped by sign), so red and blue interleave as they
        # occur rather than piling all negatives at the top.
        totals = []
        for j in range(df_pos.shape[1]):
            vals = df_pos.iloc[:, j].values
            bottom = 0.0
            for v in vals:
                if v == 0:
                    continue
                color = ut.COLOR_SHAP_POS if v > 0 else ut.COLOR_SHAP_NEG
                ax.bar(j, abs(v), bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.3, align="edge")
                bottom += abs(v)
            totals.append(bottom)
        ax.set_ylim(0, max(totals + [0]))
    else:
        list_imp = list(df_pos.sum())
        x_ticks = list(range(0, len(list_imp)))
        ax.bar(x_ticks, list_imp,
               color=ut.COLOR_FEAT_IMP, edgecolor=None, align="edge")
        # epsilon avoids 2.0000000000000004 -> 3; floor of 1 avoids a degenerate 0-height axis when
        # every position sums to (near) zero.
        v_max = max(1, int(np.ceil(max(list_imp) - 1e-9)))
        ax.set_ylim(0, v_max)
    # Keep the y-axis (spine) on the RIGHT, directly next to the "Cumulative feature importance"
    # label -- the tick mark and its value are placed on the spine's inner side by plot_feature_map.
    sns.despine(ax=ax, bottom=False, top=True, left=True, right=False)
    # Adjust ticks
    ax.set_xticks([])
    ax.tick_params(axis='y', length=_IMP_BAR_TICK_LEN, pad=1)


def add_feat_importance_map(ax=None, df_feat=None, df_cat=None,
                            col_cat=None, col_imp=None,
                            shap_plot=False,
                            imp_ths=(0.2, 0.5, 1),
                            imp_marker_size=(3, 5.5, 8),
                            start=None, args_len=None):
    """Overlay feature importance symbols on the heatmap based on the importance values."""
    th1, th2, th3 = imp_ths
    ms1, ms2, ms3 = imp_marker_size
    pp = PlotPartPositions(**args_len, start=start)
    df_pos = pp.get_df_pos(df_feat=df_feat.copy(), df_cat=df_cat,
                           col_cat=col_cat, col_val=col_imp,
                           value_type="sum", normalize=False)
    _df = pd.melt(df_pos.reset_index(), id_vars="index")
    _df.columns = [ut.COL_SUBCAT, ut.COL_POSITION, col_imp]
    _list_sub_cat = _df[ut.COL_SUBCAT].unique()
    for i, sub_cat in enumerate(_list_sub_cat):
        _dff = _df[_df[ut.COL_SUBCAT] == sub_cat]
        for pos, val in enumerate(_dff[col_imp]):
            _symbol = "■"
            color = "black"
            # In SHAP mode the impact is signed; markers encode its magnitude (abs)
            val = abs(val) if shap_plot else val
            size = ms3 if val >= th3 else (ms2 if val >= th2 else ms1)
            _args_symbol = dict(ha="center", va="center", color=color, size=size)
            if val >= th1:
                ax.text(pos + 0.5, i + 0.5, _symbol, **_args_symbol)


def add_feat_importance_legend(ax=None,
                               legend_imp_xy=None,
                               imp_ths=(0.2, 0.5, 1),
                               label=None,
                               fontsize_title=None,
                               fontsize_annotations=None):
    """Add a custom legend indicating the meaning of feature importance symbols."""
    # Define the sizes for the legend markers
    list_labels = [f"  >{float(x)}%" for x in imp_ths]
    list_imp_marker_sizes = [3, 5, 6]

    # Create the legend handles manually
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label=label,
                   markersize=size, markerfacecolor='black', linewidth=0)
        for label, size in zip(list_labels, list_imp_marker_sizes)]

    # Create the second legend
    ax.legend(handles=legend_handles,
              title=label,
              loc='lower left',
              bbox_to_anchor=legend_imp_xy,
              frameon=False,
              title_fontsize=fontsize_title,
              fontsize=fontsize_annotations,
              labelspacing=0.25,
              columnspacing=0, handletextpad=0, handlelength=0,
              borderpad=0)


# II Main Functions
def plot_feature_map(df_feat=None, df_cat=None,
                     shap_plot=False,
                     col_cat="subcategory", col_val="mean_dif", col_imp="feat_importance",
                     name_test="TEST", name_ref="REF",
                     figsize=(8, 8),
                     add_imp_bar_top=True,
                     imp_bar_th=None,
                     imp_bar_label_type="long",
                     imp_ths=(0.2, 0.5, 1), imp_marker_sizes=(3, 5, 8),
                     start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                     tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                     tmd_color="mediumspringgreen", jmd_color="blue",
                     tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None,
                     fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                     fontsize_titles=11, fontsize_labels=12,
                     fontsize_annotations=11,
                     fontsize_imp_bar=9,
                     add_xticks_pos=False,
                     grid_linewidth=0.01, grid_linecolor=None,
                     border_linewidth=2,
                     facecolor_dark=False, vmin=None, vmax=None,
                     cmap=None, cmap_n_colors=101,
                     cbar_pct=True, cbar_kws=None, cbar_xywh=(0.5, None, 0.2, None),
                     dict_color=None, legend_kws=None, legend_xy=(-0.1, -0.01),
                     legend_imp_xy=(1.25, 0),
                     xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                     seq_char_fill=False, optimize_labels=False, size_grid=False):
    """Create a comprehensive feature map with a heatmap, feature importance bars, and custom legends."""
    # Get fontsize
    pe = PlotElements()
    fs = ut.plot_gco()
    fs_titles = fs-1 if fontsize_titles is None else fontsize_titles
    fs_labels = fs if fontsize_labels is None else fontsize_labels
    fs_annotations = fs-1 if fontsize_annotations is None else fontsize_annotations
    fs_imp_bar = fs-3 if fontsize_imp_bar is None else fontsize_imp_bar

    # Group arguments
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_fs = dict(seq_size=seq_size,
                   fontsize_labels=fs_labels,
                   fontsize_tmd_jmd=fontsize_tmd_jmd)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)

    # Plot. In SHAP mode, the sample-level impact is shown EITHER as stacked bars
    # (with the heatmap showing the mean difference) OR, when col_val is a feature
    # impact column, directly in the heatmap cells with the bars switched off.
    bars_off = shap_plot and ut.COL_FEAT_IMPACT in col_val
    if bars_off:
        # Heatmap-only layout: impact lives in the cells, no cumulative-impact bars
        # (plain layout like CPPPlot.heatmap, so the frontend's tight_layout is compatible)
        fig, ax_hm = plt.subplots(figsize=figsize)
        ax_bt = ax_br = ax_empty = None
    else:
        # Size the bar strips in grid-cell units and clamp to a [min, max] cell range so they lock
        # to a near-constant physical size (the sizer holds each cell fixed, so a strip of K cells is
        # K*cell inches regardless of figure size). Right strip scales with the number of positions,
        # top strip with the number of subcategory rows. This replaces the earlier figure-ratio
        # column with an absolute-inch cap -- the clamp already bounds the column in cell units.
        n_cols_grid = tmd_len + jmd_n_len + jmd_c_len
        n_rows_grid = max(1, int(df_feat[col_cat].nunique()))
        right_cells = float(np.clip(n_cols_grid * _IMP_BAR_SCALE,
                                    _IMP_BAR_RIGHT_MIN_CELLS, _IMP_BAR_RIGHT_MAX_CELLS))
        top_cells = float(np.clip(n_rows_grid * _IMP_BAR_SCALE,
                                  _IMP_BAR_TOP_MIN_CELLS, _IMP_BAR_TOP_MAX_CELLS))
        # On the fixed-figure path (no constant-cell sizer) the column width tracks the figure, so
        # cap it at `_MAX_IMP_COL_IN` inches: solve fig_w * rc/(n_cols + rc) = _MAX_IMP_COL_IN for rc.
        # The sizer path is left to the cell-unit clamp (its column is already well under the cap).
        fig_w = figsize[0]
        if not size_grid and fig_w > _MAX_IMP_COL_IN and \
                fig_w * right_cells / (n_cols_grid + right_cells) > _MAX_IMP_COL_IN:
            right_cells = n_cols_grid * _MAX_IMP_COL_IN / (fig_w - _MAX_IMP_COL_IN)
        gridspc_kw = {'width_ratios': [n_cols_grid, right_cells], "wspace": 0, "hspace": 0}
        if add_imp_bar_top:
            gridspc_kw["height_ratios"] = [top_cells, n_rows_grid]
        # NOTE: no constrained/tight layout engine here. The frontend finalises the
        # layout with fig.tight_layout(); switching a constrained engine to tight
        # after the heatmap colorbar exists raises, and the engines also fight the
        # manually placed colorbar/legend axes. Spacing is controlled explicitly via
        # the gridspec ratios (wspace/hspace=0 glue the bars to the heatmap).
        fig, axes = plt.subplots(figsize=figsize,
                                 nrows=2 if add_imp_bar_top else 1,
                                 ncols=2,
                                 gridspec_kw=gridspc_kw)
        if add_imp_bar_top:
            ax_hm, ax_bt, ax_br, ax_empty = axes[1, 0], axes[0, 0], axes[1, 1], axes[0, 1]
            ax_hm.sharex(ax_bt)
        else:
            ax_hm, ax_br = axes[0], axes[1]
            ax_bt = ax_empty = None
        ax_hm.sharey(ax_br)
    # Set colorbar and legend arguments (diverging SHAP colormap for signed sample-level impact)
    if shap_plot and ut.COL_FEAT_IMPACT in col_val:
        cmap = ut.STR_CMAP_SHAP if cmap is None else cmap
        label_cbar = ut.LABEL_CBAR_FEAT_IMPACT_CUM
    else:
        cmap = ut.STR_CMAP_CPP if cmap is None else cmap
        label_cbar = f"Feature value\n{name_test} - {name_ref}"
    _cbar_kws, cbar_ax = pe.adjust_cbar_kws(fig=fig,
                                            cbar_kws=cbar_kws,
                                            cbar_xywh=cbar_xywh,
                                            label=label_cbar,
                                            fontsize_labels=fs_labels)

    n_cat = len(set(df_feat[ut.COL_CAT]))
    _legend_kws = pe.adjust_cat_legend_kws(legend_kws=legend_kws,
                                           n_cat=n_cat,
                                           legend_xy=legend_xy,
                                           fontsize_labels=fs_labels)
    # Plot feat importance/impact bars (unless impact is shown in the heatmap)
    if not bars_off:
        imp_word = "impact" if shap_plot else "importance"
        _narrow = (tmd_len + jmd_n_len + jmd_c_len) < 10
        show_only_max = add_imp_bar_top != "long"
        args_ticks_0 = dict(show_zero=False, show_only_max=show_only_max, precision=1)

        # Plot the top per-position bars first; their cumulative-importance max drives the corner
        # layout.
        top_ymax = None
        if add_imp_bar_top:
            plot_feat_importance_bars_pos(ax=ax_bt,
                                          df_feat=df_feat.copy(),
                                          df_cat=df_cat.copy(),
                                          col_imp=col_imp,
                                          col_cat=col_cat,
                                          shap_plot=shap_plot,
                                          start=start,
                                          **args_len)
            top_ymax = (float(ax_bt.get_ylim()[1]) if shap_plot
                        else max([p.get_height() for p in ax_bt.patches] + [0.0]))
        # The subcategory x-tick is dropped on the shortest strips (top_ymax <= 1.5; the values are
        # already printed on the bars) and shown above that.
        # For SHAP impact the right per-subcategory bars carry no on-bar value labels, so keep the
        # subcategory importance axis (its only scale reference); for plain importance drop it on the
        # shortest strips where the values are already printed on the bars.
        show_subcat_xtick = add_imp_bar_top and (shap_plot or top_ymax > 1.5)

        if add_imp_bar_top:
            label_imp_bar = f"Cumulative\nfeature\n{imp_word}"
        elif _narrow:
            # Narrow grid: stack the label so it stays over the (narrow) bars instead of running
            # left into the shortened "Feature" title (same text as the top-bars label).
            label_imp_bar = f"Cumulative\nfeature\n{imp_word}"
        else:
            label_imp_bar = f"Cumulative feature\n{imp_word}"
        fs_label_bar = fs_titles - 1 if add_imp_bar_top else fs_titles
        ha_bar = "left" if add_imp_bar_top else "right"
        position_bar = (0.1, 0) if add_imp_bar_top else (1, 0)
        multialignment_bar = "center" if add_imp_bar_top else "right"
        if add_imp_bar_top and imp_bar_label_type != "long":
            label_imp_bar = "FI" if imp_bar_label_type == "short" else None
            position_bar = (0.5, 0)
        plot_feat_importance_bars_subcat(ax=ax_br,
                                         df_feat=df_feat.copy(),
                                         df_cat=df_cat.copy(),
                                         col_imp=col_imp,
                                         col_cat=col_cat,
                                         shap_plot=shap_plot,
                                         label=label_imp_bar,
                                         annotation_th=imp_bar_th,
                                         fontsize_label=fs_label_bar,
                                         fontsize_imp_bar=fs_imp_bar,
                                         fontsize_annotations=fontsize_annotations,
                                         ha=ha_bar,
                                         position=position_bar,
                                         multialignment=multialignment_bar,
                                         weight_annotation="bold")
        # Subcategory x-axis: dropped on the shortest strips (values are printed on the bars); shown
        # at the right otherwise, and for the no-top (heatmap) layout. Label stays in its normal spot.
        if add_imp_bar_top and not show_subcat_xtick:
            ax_br.set_xticks([])
        else:
            ut.ticks_0(ax_br, **args_ticks_0)

        if add_imp_bar_top:
            # The y-axis spine stays on the RIGHT, directly next to the "Cumulative feature
            # importance" label, but the tick MARK and its VALUE both render on the spine's INNER
            # (left) side. Drawing them outward -- matplotlib's default for a right-hand axis --
            # runs the value straight through that label, which is the overlap this corner exists to
            # avoid; the strip's height does not reliably buy enough clearance, so the inner side is
            # unconditional rather than a fallback for short strips.
            # Place the number AT the actual strip top. For plain importance the axis top is an
            # integer (whole-percent cumulative); for SHAP impact it is the raw fractional max, so
            # format it to one decimal instead of ceiling it (which would float above the strip).
            ax_bt.yaxis.set_ticks_position("right")
            y_top = float(ax_bt.get_ylim()[1])
            v_label = f"{y_top:.1f}" if shap_plot else f"{int(round(y_top))}"
            ax_bt.set_yticks([y_top])
            ax_bt.tick_params(axis="y", labelleft=False, labelright=False,
                              length=_IMP_BAR_TICK_LEN, direction="in")
            ax_bt.annotate(v_label, xy=(1.0, y_top), xycoords=ax_bt.get_yaxis_transform(),
                           xytext=(-_IMP_BAR_TICK_PAD, 0), textcoords="offset points",
                           ha="right", va="center", size=fontsize_annotations,
                           annotation_clip=False)

    # Plot heatmap
    plot_heatmap_(df_feat=df_feat.copy(), df_cat=df_cat,
                  col_cat=col_cat, col_val=col_val,
                  ax=ax_hm, figsize=figsize, fill=seq_char_fill,
                  start=start, **args_len, **args_seq,
                  **args_part_color, **args_seq_color,
                  **args_fs, weight_tmd_jmd=weight_tmd_jmd,
                  add_xticks_pos=add_xticks_pos,
                  grid_linewidth=grid_linewidth, grid_linecolor=grid_linecolor,
                  border_linewidth=border_linewidth,
                  facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                  cmap=cmap, cmap_n_colors=cmap_n_colors,
                  cbar_ax=cbar_ax, cbar_pct=cbar_pct, cbar_kws=_cbar_kws,
                  dict_color=dict_color, legend_kws=_legend_kws,
                  optimize_labels=optimize_labels,
                  **args_xtick)
    # Add feature position title
    sns.despine(ax=ax_hm, top=True, left=True, right=True, bottom=True)
    n_positions = tmd_len + jmd_n_len + jmd_c_len
    if add_imp_bar_top and not bars_off:
        ax_hm.set_title("Scale (subcategory)  ", x=0, ha="right", weight="bold", fontsize=fs_titles)
    else:
        # On an ultra-narrow grid (< 10 columns) the padded "... + Positions" title runs into the
        # cumulative-importance label; drop the padding and the "+ Positions" so the two clear.
        _title = "Feature\nScale (subcategory)" if n_positions < 10 else ut.LABEL_FEAT_POS
        ax_hm.set_title(_title, x=0, weight="bold", fontsize=fs_titles)
    # Add feature importance map
    add_feat_importance_map(df_feat=df_feat, df_cat=df_cat,
                            col_cat=col_cat, col_imp=col_imp,
                            shap_plot=shap_plot,
                            imp_ths=imp_ths,
                            imp_marker_size=imp_marker_sizes,
                            ax=ax_hm, start=start, args_len=args_len)

    legend_imp_xy_default = (1.25, 0)
    _legend_imp_xy = ut.adjust_tuple_elements(tuple_in=legend_imp_xy,
                                              tuple_default=legend_imp_xy_default)
    label_feat_imp = "Feature impact" if shap_plot else "Feature importance"
    add_feat_importance_legend(ax=cbar_ax,
                               imp_ths=imp_ths,
                               legend_imp_xy=_legend_imp_xy,
                               label=label_feat_imp,
                               fontsize_title=fs_labels,
                               fontsize_annotations=fs_annotations)
    if add_imp_bar_top and not bars_off:
        ax_empty.axis("off")
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.sca(ax_hm)
    ax = ax_hm
    return fig, ax
