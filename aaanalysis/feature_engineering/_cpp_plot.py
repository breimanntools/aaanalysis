"""
This is a script for ...
"""
from typing import Optional, Dict, Union, List, Tuple, Type
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
import matplotlib as mpl

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.check_feature import (check_split_kws,
                                     check_parts_len, check_match_features_seq_parts,
                                     check_df_parts, check_match_df_parts_features, check_match_df_parts_list_parts,
                                     check_df_scales, check_match_df_scales_features,
                                     check_df_cat, check_match_df_cat_features,
                                     check_match_df_parts_df_scales, check_match_df_scales_df_cat)
from ._backend.cpp.utils_cpp_plot import get_optimal_fontsize

from ._backend.cpp.cpp_plot_feature import plot_feature
from ._backend.cpp.cpp_plot_ranking import plot_ranking
from ._backend.cpp.cpp_plot_profile import plot_profile
from ._backend.cpp.cpp_plot_heatmap import plot_heatmap
from ._backend.cpp.cpp_plot_feature_map import plot_feature_map


# TODO simplify checks & interface (end-to-end check with tests & docu)
# TODO plot_functions test & refactor (end-to-end)
# TODO remove decorators for redundant signatures (not compiled at reading time -> IDE docuemntation problems)

# I Helper Functions
def check_value_type(value_type=None, count_in=True):
    """Check if value type is valid"""
    list_value_type = ["count", "sum", "mean"]
    if count_in:
        list_value_type.append("count")
    if value_type not in list_value_type:
        raise ValueError(f"'value_type' ('{value_type}') should be on of following: {list_value_type}")


# Check for plotting methods
def check_args_size(seq_size=None, fontsize_tmd_jmd=None):
    """Check if sequence size parameters match"""
    ut.check_number_range(name="seq_size", val=seq_size, min_val=0, accept_none=True, just_int=False)
    ut.check_number_range(name="fontsize_tmd_jmd", val=fontsize_tmd_jmd, min_val=0, accept_none=True, just_int=False)
    args_size = dict(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
    return args_size


def check_args_xtick(xtick_size=None, xtick_width=None, xtick_length=None):
    """Check if x tick parameters non-negative float"""
    args = dict(accept_none=True, just_int=False, min_val=0)
    ut.check_number_range(name="xtick_size", val=xtick_size, **args)
    ut.check_number_range(name="xtick_width", val=xtick_width, **args)
    ut.check_number_range(name="xtick_length", val=xtick_length, **args)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    return args_xtick


def check_args_ytick(ytick_size=None, ytick_width=None, ytick_length=None):
    """Check if y tick parameters non-negative float"""
    args = dict(accept_none=True, just_int=False, min_val=1)
    ut.check_number_range(name="ytick_size", val=ytick_size, **args)
    ut.check_number_range(name="ytick_width", val=ytick_width, **args)
    ut.check_number_range(name="ytick_length", val=ytick_length, **args)
    args_ytick = dict(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
    return args_ytick


def check_part_color(tmd_color=None, jmd_color=None):
    """Check if part colors valid"""
    ut.check_color(name="tmd_color", val=tmd_color)
    ut.check_color(name="jmd_color", val=jmd_color)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    return args_part_color


def check_seq_color(tmd_seq_color=None, jmd_seq_color=None):
    """Check sequence colors"""
    ut.check_color(name="tmd_seq_color", val=tmd_seq_color)
    ut.check_color(name="jmd_seq_color", val=jmd_seq_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    return args_seq_color


def check_figsize(figsize=None):
    """"""
    ut.check_tuple(name="figsize", val=figsize, n=2)
    ut.check_number_range(name="figsize:width", val=figsize[0], min_val=1, just_int=False)
    ut.check_number_range(name="figsize:height", val=figsize[1], min_val=1, just_int=False)


def check_dict_color(dict_color=None, df_cat=None):
    """Check if color dictionary is matching to DataFrame with categories"""
    list_cats = list(sorted(set(df_cat[ut.COL_CAT])))
    if dict_color is None:
        dict_color = ut.DICT_COLOR_CAT
    if not isinstance(dict_color, dict):
        raise ValueError(f"'dict_color' should be a dictionary with colors for: {list_cats}")
    list_cat_not_in_dict_cat = [x for x in list_cats if x not in dict_color]
    if len(list_cat_not_in_dict_cat) > 0:
        error = f"'dict_color' not complete! Following categories are missing from 'df_cat': {list_cat_not_in_dict_cat}"
        raise ValueError(error)
    for key in dict_color:
        color = dict_color[key]
        ut.check_color(name=key, val=color)
    return dict_color


# Check barplot and profile
def check_grid_axis(grid_axis=None):
    """"""
    list_valid = ["x", 'y', 'both', None]
    if grid_axis not in list_valid:
        raise ValueError(f"'grid_axis' ('{grid_axis}') not valid. Chose from following: {list_valid}")


# Check stat plot
def check_ylabel_fontweight(ylabel_fontweight=None, accept_none=True):
    """"""
    if accept_none and ylabel_fontweight is None:
        return
    name = "ylabel_fontweight"
    list_weights = ['light', 'medium', 'bold']
    if type(ylabel_fontweight) in [float, int]:
        ut.check_number_range(name=name, val=ylabel_fontweight, min_val=0, max_val=1000, just_int=False)
    elif isinstance(ylabel_fontweight, str):
        if ylabel_fontweight not in list_weights:
            error = f"'{name}' ({ylabel_fontweight}) should be one of following: {list_weights}"
            raise ValueError(error)
    else:
        error = f"'{name}' ({ylabel_fontweight}) should be either numeric value in range 0-1000" \
                f"\n\tor one of following: {list_weights}"
        raise ValueError(error)


def check_names_to_show(df_seq=None, names_to_show=None):
    """"""
    if names_to_show is None:
        return []
    names_to_show = ut.check_list_like(name="name_to_show", val=names_to_show, accept_str=True)
    list_names = df_seq[ut.COL_NAME].to_list()
    missing_names = [x for x in names_to_show if x not in list_names]
    if len(missing_names) > 0:
        raise ValueError(f"Following names from 'names_to_show' are not in '{ut.COL_NAME}' "
                         f"column of 'df_seq': {missing_names}")


# Plotting functions
# TODO simplify interface (delete old profile)
# TODO add importance plot for heatmap
# TODO merge (grid_axis=None -> grid=False,)
# TODO normalize=True/False (always normalize for positions)

# II Main Functions
class CPPPlot:
    """
    Plot CPP results at global or individual sample level as ranking plot, profile, or map (heatmap, feature map).

    """
    def __init__(self,
                 df_scales: Optional[pd.DataFrame] = None,
                 df_cat: Optional[pd.DataFrame] = None,
                 jmd_n_len: int = 10,
                 jmd_c_len: int = 10,
                 verbose: bool = True,
                 accept_gaps: bool = False,
                 ):
        """
        Parameters
        ----------
        df_scales
            DataFrame with amino acid scales. Default from :meth:`load_scales` with 'name'='scales_cat'.
        df_cat
            DataFrame with default categories for physicochemical amino acid scales.
            Default from :meth:`load_categories`
        jmd_n_len
            Length of JMD-N (>=0).
        jmd_c_len
            Length of JMD-C (>=0).
        accept_gaps
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        verbose
            If ``True``, verbose outputs are enabled. Global 'verbose' setting is used if ``None``.
        """
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        check_df_scales(df_scales=df_scales)
        check_df_cat(df_cat=df_cat)
        check_parts_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, accept_none_len=True)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        df_scales, df_cat = check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        # General settings
        self._verbose = ut.check_verbose(verbose)
        self._accept_gaps = accept_gaps
        # Set data parameters
        self._df_cat = df_cat
        self._df_scales = df_scales
        # Set consistent length of JMD_N, JMD_C, TMD flanking amino acids (TMD-E)
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len
        # Axes dict for plotting
        self.ax_seq = None

    # Plotting methods for single feature
    def feature(self,
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (5.6, 4.8),
                feature=str,
                df_seq=None,
                labels=None,
                names_to_show=None,
                show_seq=False,
                name_test="TEST",
                name_ref="REF",
                color_test="tab:green",
                color_ref="tab:gray",
                fontsize_mean_dif=15,
                fontsize_name_test=13,
                fontsize_name_ref=13,
                fontsize_names_to_show=11,
                histplot=False,
                alpha_hist=0.1,
                alpha_dif=0.2,
                ) -> plt.Axes:
        """Plot distributions of feature values for test and reference datasets highlighting their mean difference.

        Parameters
        ----------

        Returns
        -------
        ax
            Pre-defined Axes object to plot on. If `None`, a new Axes object is created.
        figsize : tuple, default=(5.6, 4.8)
            Figure size (width, height) in inches.
        """

        # Check input
        # TODO check input, add docstring, typing
        #feature = ut.check_list_like(name="feature", val=feature, convert=True, accept_str=True)
        check_names_to_show(df_seq=df_seq, names_to_show=names_to_show)
        # Plot feature
        ax = plot_feature(ax=ax, figsize=figsize,
                          feature=feature, df_scales=self._df_scales, accept_gaps=self._accept_gaps,
                          df_seq=df_seq, labels=labels,
                          names_to_show=names_to_show, show_seq=show_seq,
                          name_test=name_test, name_ref=name_ref,
                          color_test=color_test, color_ref=color_ref,
                          fontsize_mean_dif=fontsize_mean_dif,
                          fontsize_name_test=fontsize_name_test,
                          fontsize_name_ref=fontsize_name_ref,
                          fontsize_names_to_show=fontsize_names_to_show,
                          histplot=histplot, alpha_hist=alpha_hist, alpha_dif=alpha_dif)
        return ax

    # Plotting methods for group and single level
    def ranking(self,
                ax=None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
                df_feat=None,
                df_parts=None,
                tmd_len=20,
                labels=None,
                top_n=15,
                name_test="TEST",
                name_ref="REF",
                error_bar=False,
                shap_plot=False,
                fontsize_titles=11,
                fontsize_labels=11,
                fontsize_annotations=11,
                feature_val_in_percent=True,
                capsize=2,
                tmd_jmd_space=2,
                col_dif=ut.COL_MEAN_DIF,
                col_rank=ut.COL_FEAT_IMPORT,
                xlim_dif=(-17.5, 17.5),
                xlim_rank=(0, 5)
                ):
        """"""
        # Check input
        # TODO check input, add docstring, typing
        ut.check_bool(name="shap_plot", val=shap_plot)
        if error_bar:
            if labels is None:
                raise ValueError("'labels' should not be None if 'error_bar' is True")
            if df_parts is None:
                raise ValueError("'df_parts' should not be None if 'error_bar' is True")
        # Plot ranking
        fig, ax = plot_ranking(figsize=figsize, df_feat=df_feat, top_n=top_n, df_parts=df_parts,
                               tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                               df_scales=self._df_scales, labels=labels,
                               name_test=name_test, name_ref=name_ref,
                               error_bar=error_bar, shap_plot=shap_plot,
                               fontsize_titles=fontsize_titles,
                               fontsize_labels=fontsize_labels,
                               fontsize_annotations=fontsize_annotations,
                               feature_val_in_percent=feature_val_in_percent,
                               capsize=capsize, tmd_jmd_space=tmd_jmd_space,
                               col_dif=col_dif, col_rank=col_rank, xlim_dif=xlim_dif, xlim_rank=xlim_rank)
        return fig, ax

    def profile(self,
                ax=None,
                figsize=(7, 5),

                df_feat=None,

                shap_plot=False,

                col_value="feat_importance",
                value_type="sum",
                normalize=True,

                dict_color=None,

                edge_color="none",
                bar_width=0.75,

                tmd_len=20,
                start=1,
                jmd_n_seq=None,
                tmd_seq=None,
                jmd_c_seq=None,
                tmd_color="mediumspringgreen",
                jmd_color="blue",
                tmd_seq_color="black",
                jmd_seq_color="white",
                seq_size=None,
                fontsize_tmd_jmd=None,

                xtick_size=11.0,
                xtick_width=2.0,
                xtick_length=5.0,

                ytick_size=None,
                ytick_width=None,
                ytick_length=5.0,
                ylim=None,

                xticks_pos=False,
                add_jmd_tmd=True,
                highlight_tmd_area=True,
                highlight_alpha=0.15,

                grid_axis=None,

                add_legend_cat=False,
                legend_kws=None):
        """
        Plot CPP profile for given features from 'df_feat'.

        Parameters
        ----------
        df_feat : class:`pandas.DataFrame`, optional, default=None
            Dataframe containing the features to be plotted. If ``None``, default features from the instance will be used.

        col_value : str, default='mean_dif'
            Column name in df_feat which contains the values to be plotted.
        value_type : str, default='count'
            Type of value. Available options are specified by the `check_value_type` function.
        normalize : bool, default=False
            If True, the feature values will be normalized.
        figsize : tuple, default=(7, 5)
            Size of the plot.
        dict_color : dict, optional
            Dictionary mapping categories to colors.
        edge_color : str, default='none'
            Color of the edges of the bars.
        bar_width : float, default=0.75
            Width of the bars.
        add_jmd_tmd : bool, default=True
            If True, adds JMD and TMD lines/annotations to the plot.
        tmd_len : int, default=20
            Length of the TMD.
        start : int, default=1
            Start position.
        jmd_n_seq : str, optional
            JMD N-terminal sequence.
        tmd_seq : str, optional
            TMD sequence.
        jmd_c_seq : str, optional
            JMD C-terminal sequence.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMD.
        tmd_seq_color : str, default='black'
            Color for TMD sequence.
        jmd_seq_color : str, default='white'
            Color for JMD sequence.
        seq_size : float, optional
            Font size for sequence annotations.
        fontsize_tmd_jmd : float, optional
            Font size for TMD and JMD annotations.
        xtick_size : float, default=11.0
            Size for x-tick labels.
        xtick_width : float, default=2.0
            Width of the x-ticks.
        xtick_length : float, default=5.0
            Length of the x-ticks.
        xticks_pos : bool, default=False
            If True, x-tick positions are adjusted based on given sequences.
        ytick_size : float, optional
            Size for y-tick labels.
        ytick_width : float, default=2.0
            Width of the y-ticks.
        ytick_length : float, default=5.0
            Length of the y-ticks.
        ylim : tuple, optional
            Y-axis limits.
        highlight_tmd_area : bool, default=True
            If True, highlights the TMD area on the plot.
        highlight_alpha : float, default=0.15
            Alpha value for TMD area highlighting.
        grid_axis : str, default='both'
            Axis on which the grid is drawn if not None. Options: None, 'both', 'x', 'y'.
        add_legend_cat : bool, default=True
            If True, a legend is added for categories.
        legend_kws : dict, optional
            Keyword arguments for the legend.
        shap_plot : bool, default=False
            If True, SHAP (SHapley Additive exPlanations) plot is generated.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        """
        # Group arguments
        # TODO CHECK
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                                jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
        args_size = check_args_size(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)

        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)

        # Checking input
        # Args checked by Matplotlib: title, legend_kws
        # Args checked by internal plotting functions: ylim
        ut.check_number_range(name="bar_width", val=bar_width, min_val=0, just_int=False)
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        ut.check_number_range(name="tmd_area_alpha", val=highlight_alpha, min_val=0, max_val=1, just_int=False)
        ut.check_bool(name="add_jmd_tmd", val=add_jmd_tmd)
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_color(name="edge_color", val=edge_color, accept_none=True)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_df(df=df_feat, name="df_feat", cols_requiered=col_value, cols_nan_check=col_value)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_value_type(value_type=value_type, count_in=True)
        check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self._df_cat)
        check_grid_axis(grid_axis=grid_axis)    # TODO replace against check from valid strings in utils

        # Plot profile
        ax = plot_profile(figsize=figsize, ax=ax, df_feat=df_feat, df_cat=self._df_cat,
                          col_value=col_value, value_type=value_type, normalize=normalize,
                          dict_color=dict_color,
                          edge_color=edge_color, bar_width=bar_width,
                          add_jmd_tmd=add_jmd_tmd,
                          start=start, **args_len, **args_seq,
                          tmd_color=tmd_color, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                          jmd_seq_color=jmd_seq_color,
                          seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd,
                          xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length,
                          xticks_pos=xticks_pos, ytick_size=ytick_size, ytick_width=ytick_width,
                          ytick_length=ytick_length, ylim=ylim,
                          highlight_tmd_area=highlight_tmd_area, highlight_alpha=highlight_alpha,
                          grid_axis=grid_axis, add_legend_cat=add_legend_cat,
                          legend_kws=legend_kws, shap_plot=shap_plot)
        return ax

    def heatmap(self,
                df_feat=None,
                y="subcategory",
                col_value="mean_dif",
                value_type="mean",
                normalize=False,
                figsize=(8, 8),
                dict_color=None,

                vmin=None,
                vmax=None,
                grid_on=True,
                cmap="RdBu_r",
                cmap_n_colors=None,
                cbar_kws=None,
                facecolor_dark=False,

                add_jmd_tmd=True,
                tmd_len=20,
                start=1,
                jmd_n_seq=None,
                tmd_seq=None,
                jmd_c_seq=None,
                linecolor=None,
                tmd_color="mediumspringgreen",
                jmd_color="blue",
                tmd_seq_color="black",
                jmd_seq_color="white",
                seq_size=None,
                fontsize_tmd_jmd=None,
                xticks_pos=False,
                xtick_size=11.0,
                xtick_width=2.0,
                xtick_length=5.0,
                ytick_size=None,

                add_legend_cat=True,
                legend_kws=None,
                cbar_pct=True,
                ):
        """
        Plot CPP heatmap of the selected value column with scale information (y-axis) versus sequence position (x-axis).

        This is a wrapper function for :func:`seaborn.heatmap`, designed to highlight differences between two sets
        of sequences at the positional level (e.g., amino acid level for protein sequences).

        Parameters
        ----------
        df_feat : :class:`~pandas.DataFrame`, shape (n_feature, n_feature_information)
            DataFrame containing unique identifiers, scale information, statistics, and positions for each feature.
        y : {'category', 'subcategory', 'scale_name'}, str, default='subcategory'
            Name of the column in the feature DataFrame representing scale information (shown on the y-axis).
        col_value : {'mean_dif', 'feat_impact', 'abs_auc', 'std_test', ...}, default='mean_dif'
            Name of the column in the feature DataFrame containing numerical values to display.
        value_type : {'mean', 'sum', 'std'}, default='mean'
            Method to aggregate numerical values from 'col_value'.
        normalize : {True, False, 'positions', 'positions_only'}, default=False
            Specifies normalization for numerical values in 'col_value':

            - False: Set value at all positions of a feature without further normalization.

            - True: Set value at all positions of a feature and normalize across all features.

            - 'positions': Value/number of positions set at each position of a feature and normalized across features.
              Recommended when aiming to emphasize features with fewer positions using 'col_value'='feat_impact' and 'value_type'='mean'.

        figsize : tuple(float, float), default=(10,7)
            Width and height of the figure in inches passed to :func:`matplotlib.pyplot.figure`.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise, inferred from data and other keyword arguments.
        cmap : matplotlib colormap name or object, or list of colors, default='seismic'
            Name of color map assigning data values to color space. If 'SHAP', colors from
            `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ will be used (recommended for feature impact).
        cmap_n_colors : int, optional
            Number of discrete steps in diverging or sequential color map.
        dict_color : dict, optional
            Map of colors for scale categories classifying scales shown on y-axis.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
        add_jmd_tmd : bool, default=True
            Whether to add colored bar under heatmap indicating sequence parts (JMD-N, TMD, JMD-C).
        tmd_len : int, >0
            Length of TMD to be depiceted.
        start : int, >=0
            Position label of first amino acid position (starting at N-terminus).
        tmd_seq : str, optional
            Sequence of TMD. 'tmd_len' is set to length of TMD if sequence for TMD, JMD-N and JMD-C are given.
            Recommended if feature impact or mean difference should be depicted for one sample.
        jmd_n_seq : str, optional
            Sequence of JMD_N. 'jmd_n_len' is set to length of JMD_N if sequence for TMD, JMD-N and JMD-C are given.
            Recommended if feature impact or mean difference should be depicted for one sample.
        jmd_c_seq : str, optional
            Sequence of JMD_C. 'jmd_c_len' is set to length of JMD_C if sequence for TMD, JMD-N and JMD-C are given.
            Recommended if feature impact or mean difference should be depicted for one sample.
        tmd_color : str, default='mediumspringgreen'
            Color of TMD bar.
        jmd_color : str, default='blue'
            Color of JMD-N and JMD-C bar.
        tmd_seq_color : str, default='black'
            Color of TMD sequence.
        jmd_seq_color : str, default='white'
            Color of JMD-N and JMD-C sequence.
        seq_size : float, optional
            Font size of all sequence parts in points. If ``None``, optimized automatically.
        fontsize_tmd_jmd : float, optional
            Font size of 'TMD', 'JMD-N' and 'JMD-C'  label in points. If ``None``, optimized automatically.
        xtick_size : float, default=11.0
            Size of x ticks in points. Passed as 'size' argument to :meth:`matplotlib.axes.Axes.set_xticklabels`.
        xtick_width : float, default=2.0
            Widht of x ticks in points. Passed as 'width' argument to :meth:`matplotlib.axes.Axes.tick_params`.
        xtick_length : float, default=5.0,
            Length of x ticks in points. Passed as 'length' argument to :meth:`matplotlib.axes.Axes.tick_params`.
        ytick_size : float, optional
            Size of scale information as y ticks in points. Passed to :meth:`matplotlib.axes.Axes.tick_params`.
            If ``None``, optimized automatically.
        add_legend_cat : bool, default=True,
            Whether to add legend for categories under plot and classification of scales at y-axis.
        legend_kws : dict, optional
            Keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`
        kwargs : other keyword arguments
            All other keyword arguments passed to :meth:`matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        ax : matplotlib Axes
        Axes object containing the heatmap.

        Warnings
        --------
        * 'cmap_n_colors' is effective only if 'vmin' and 'vmax' align with the data.
        * 'tmd_seq_color' and 'jmd_seq_color' are applicable only when 'tmd_seq', 'jmd_n_seq', and 'jmd_c_seq' are provided.

        See Also
        --------
        seaborn.heatmap
            Plotting heatmap using seaborn.
            See `Seaborn documentation <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_ for more details.

        Examples
        --------

        Plot CPP feature heatmap:

        .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> import aaanalysis as aa
            >>> sf = aa.SequenceFeature()
            >>> df_seq = aa.load_dataset(name='SEQ_DISULFIDE', min_len=100)
            >>> labels = list(df_seq["label"])
            >>> df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10)
            >>> #split_kws = sf.get_split_kws(n_split_min=1, n_split_max=3, split_types=["Segment", "PeriodicPattern"])
            >>> #df_scales = aa.load_scales(unclassified_in=False).sample(n=10, axis=1)
            >>> #cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales)
            >>> #df_feat = cpp.run(labels=labels)
            >>> #cpp.plot_heatmap(df_feat=df_feat)
            >>> #plt.tight_layout()

        """
        # Group arguments
        args_size = check_args_size(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
        # TODO CHECK
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                                tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)

        # Checking input
        # Args checked by Matplotlib: title, cmap, cbar_kws, legend_kws
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        ut.check_number_range(name="ytick_size", val=ytick_size, accept_none=True, just_int=False, min_val=1)
        ut.check_number_range(name="cmap_n_colors", val=cmap_n_colors, min_val=1, accept_none=True, just_int=True)
        ut.check_bool(name="add_jmd_tmd", val=add_jmd_tmd)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_df(df=df_feat, name="df_feat", cols_requiered=col_value, cols_nan_check=col_value)
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat)
        check_value_type(value_type=value_type, count_in=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self._df_cat)

        # Get df positions
        ax = plot_heatmap(df_feat=df_feat, df_cat=self._df_cat, col_cat=y, col_value=col_value, value_type=value_type,
                          normalize=normalize, figsize=figsize,
                          dict_color=dict_color, vmin=vmin, vmax=vmax, grid_on=grid_on,
                          cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws,
                          facecolor_dark=facecolor_dark, add_jmd_tmd=add_jmd_tmd,
                          start=start, *+args_len, **args_seq,
                          tmd_color=tmd_color, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                          jmd_seq_color=jmd_seq_color,
                          seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd,
                          xticks_pos=xticks_pos, xtick_size=xtick_size, xtick_width=xtick_width,
                          xtick_length=xtick_length, ytick_size=ytick_size,
                          add_legend_cat=add_legend_cat, legend_kws=legend_kws, cbar_pct=cbar_pct,
                          linecolor=linecolor)
        return ax

    # Plotting method for only group level
    def feature_map(self,
                    df_feat=None,
                    y="subcategory",
                    col_value="mean_dif",
                    value_type="mean",
                    normalize=False,
                    figsize=(8, 8),
                    dict_color=None,

                    vmin=None,
                    vmax=None,
                    grid_on=True,
                    cmap="RdBu_r",
                    cmap_n_colors=None,
                    cbar_kws=None,
                    facecolor_dark=False,

                    tmd_len=20,
                    start=1,
                    linecolor=None,
                    tmd_color="mediumspringgreen",
                    jmd_color="blue",
                    tmd_seq_color="black",
                    jmd_seq_color="white",
                    seq_size=None,
                    fontsize_tmd_jmd=None,
                    xtick_size=11.0,
                    xtick_width=2.0,
                    xtick_length=5.0,
                    ytick_size=None,

                    add_legend_cat=True,
                    legend_kws=None,
                    cbar_pct=True,
                    ):
        # TODO CHECK
        # TODO cbar & feature importance y location depend on n features.
        # TODO bar label not in
        # TODO TMD size dep on size of plot (change)
        # Group arguments
        args_size = check_args_size(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        # Checking input
        # Args checked by Matplotlib: title, cmap, cbar_kws, legend_kws
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len)
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        ut.check_number_range(name="ytick_size", val=ytick_size, accept_none=True, just_int=False, min_val=1)
        ut.check_number_range(name="cmap_n_colors", val=cmap_n_colors, min_val=1, accept_none=True, just_int=True)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_df(df=df_feat, name="df_feat", cols_requiered=col_value, cols_nan_check=col_value)
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat)
        check_value_type(value_type=value_type, count_in=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self._df_cat)
        # Get df positions
        ax = plot_feature_map(df_feat=df_feat, df_cat=self._df_cat, y=y, col_value=col_value, value_type=value_type,
                              normalize=normalize, figsize=figsize,
                              dict_color=dict_color, vmin=vmin, vmax=vmax, grid_on=grid_on,
                              cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws,
                              facecolor_dark=facecolor_dark,
                              start=start, **args_len,
                              tmd_color=tmd_color, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                              jmd_seq_color=jmd_seq_color,
                              seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd,
                              xtick_size=xtick_size, xtick_width=xtick_width,
                              xtick_length=xtick_length, ytick_size=ytick_size,
                              legend_kws=legend_kws, cbar_pct=cbar_pct, linecolor=linecolor)
        return ax

    def update_seq_size(self):
        """"""
        # TODO legend changes slightly if sequnece length altered (e.g. PTPRM_MOUSE vs A4_HUMAN)
        # TODO look for more extreme example and text
        f = lambda l: l.get_window_extent(ax.figure.canvas.get_renderer())
        ax = self.ax_seq
        labels = ax.xaxis.get_ticklabels(which="both")
        tick_positions = [f(l).x0 for l in labels]
        sorted_tick_positions, sorted_labels = zip(*sorted(zip(tick_positions, labels), key=lambda t: t[0]))
        # Adjust font size to prevent overlap
        seq_size = get_optimal_fontsize(ax, sorted_labels)
        for l in sorted_labels:
            l.set_fontsize(seq_size)
