"""
This is a script for the backend of the CPPPlot.heatmap() method.
"""
import matplotlib.pyplot as plt

import aaanalysis.utils as ut
from ._utils_cpp_plot_elements import PlotElements
from ._utils_cpp_plot_map import plot_heatmap_


# II Main Functions
def plot_heatmap(df_feat=None, df_cat=None,
                 shap_plot=False,
                 col_cat="subcategory", col_val="mean_dif",
                 normalize=False,
                 name_test="TEST", name_ref="REF",
                 figsize=(8, 8),
                 start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                 tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                 tmd_color="mediumspringgreen", jmd_color="blue",
                 tmd_seq_color="black", jmd_seq_color="white",
                 seq_size=None, fontsize_tmd_jmd=None, weight_tmd_jmd="normal",
                 fontsize_labels=None,
                 add_xticks_pos=False,
                 grid_linewidth=0.01, grid_linecolor=None,
                 border_linewidth=2,
                 facecolor_dark=None, vmin=None, vmax=None,
                 cmap=None, cmap_n_colors=None,
                 cbar_pct=True, cbar_kws=None,
                 dict_color=None, legend_kws=None,
                 xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                 ytick_size=None):
    """Plot heatmap of feature values"""
    # Set fontsize
    pe = PlotElements()
    fs_labels, _ = pe.adjust_fontsize(fontsize_labels=fontsize_labels)

    # Group arguments
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    args_fs = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd, fontsize_labels=fontsize_labels)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)

    # Set SHAP arguments
    if facecolor_dark is None:
        facecolor_dark = shap_plot

    if shap_plot:
        cmap = "SHAP"
        label_cbar = ut.LABEL_CBAR_FEAT_IMPACT_CUM
    else:
        label_cbar = f"Feature value\n{name_test} - {name_ref}"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Set color bar and legend arguments
    _cbar_kws, cbar_ax = pe.adjust_cbar_kws(fig=fig,
                                            cbar_kws=cbar_kws,
                                            cbar_xywh=None,
                                            label=label_cbar,
                                            fontsize_labels=fs_labels)

    # Set cat legend arguments
    n_cat = len(set(df_feat[ut.COL_CAT]))
    _legend_kws = pe.adjust_cat_legend_kws(legend_kws=legend_kws,
                                           n_cat=n_cat,
                                           legend_xy=None,
                                           fontsize_labels=fs_labels)

    # Plot heatmap
    ax = plot_heatmap_(df_feat=df_feat, df_cat=df_cat,
                       col_cat=col_cat, col_val=col_val, normalize=normalize,
                       ax=ax, figsize=figsize,
                       start=start, **args_len, **args_seq,
                       **args_part_color, **args_seq_color,
                       **args_fs, weight_tmd_jmd=weight_tmd_jmd,
                       add_xticks_pos=add_xticks_pos,
                       grid_linecolor=grid_linecolor, grid_linewidth=grid_linewidth,
                       border_linewidth=border_linewidth,
                       facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                       cmap=cmap, cmap_n_colors=cmap_n_colors,
                       cbar_ax=cbar_ax, cbar_pct=cbar_pct, cbar_kws=_cbar_kws,
                       dict_color=dict_color, legend_kws=_legend_kws,
                       **args_xtick, ytick_size=ytick_size)
    return ax
