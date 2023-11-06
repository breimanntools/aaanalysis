"""
This is a script for ...
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.cpp.cpp_plot import CPPPlots, get_optimal_fontsize


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe

# TODO simplify checks & interface (end-to-end check with tests & docu)
# TODO plot_functions test & refactor (end-to-end)
# TODO use decorator for redudant signatures (https://github.com/scverse/scanpy/blob/master/scanpy/preprocessing/_qc.py)

# I Helper Functions
def check_value_type(val_type=None, count_in=True):
    """Check if value type is valid"""
    list_value_type = ["mean", "sum", "std"]
    if count_in:
        list_value_type.append("count")
    if val_type not in list_value_type:
        raise ValueError(f"'val_type' ('{val_type}') should be on of following: {list_value_type}")


def check_normalize(normalize=True):
    """Check normalize parameter"""
    if not (type(normalize) == bool or normalize in ["positions", "positions_only"]):
        raise ValueError(f"'normalize' ('{normalize}') should be bool or, if normalized for positions, 'positions'.")
    normalize_for_positions = False if type(normalize) is bool else "positions" in normalize
    normalize = normalize if type(normalize) is bool else "positions" == normalize
    return normalize, normalize_for_positions


# Check for plotting methods
def check_args_size(seq_size=None, tmd_jmd_fontsize=None):
    """Check if sequence size parameters match"""
    ut.check_number_range(name="seq_size", val=seq_size, min_val=0, accept_none=True, just_int=False)
    ut.check_number_range(name="tmd_jmd_fontsize", val=tmd_jmd_fontsize, min_val=0, accept_none=True, just_int=False)
    args_size = dict(seq_size=seq_size, tmd_jmd_fontsize=tmd_jmd_fontsize)
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
        dict_color = ut.DICT_COLOR
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


def check_parameters(func=None, name_called_func=None, e=None):
    """Check parameters string from error message of third party packages"""
    list_arg_str = ["property ", "attribute ", "argument ", "parameter "]
    str_error = ""
    for arg_str in list_arg_str:
        if arg_str in str(e):
            error_arg = str(e).split(arg_str)[1]
            str_error += "Error due to {} parameter. ".format(error_arg)
            break
    args = [x for x in inspect.getfullargspec(func).args if x != "self"]
    str_error += "Arguments are allowed from {} and as follows: {}".format(name_called_func, args)
    return str_error


# Check heatmap plotting
def check_vmin_vmax(vmin=None, vmax=None):
    """Check if number of cmap colors is valid with given value range"""
    ut.check_number_val(name="vmin", val=vmin, accept_none=True, just_int=False)
    ut.check_number_val(name="vmax", val=vmax, accept_none=True, just_int=False)
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(f"'vmin' ({vmin}) < 'vmax' ({vmax}) not fulfilled.")


# Check barplot and profile
def check_grid_axis(grid_axis=None):
    """"""
    list_valid = ["x", 'y', 'both']
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


# Plotting functions
def _get_df_pos(df_feat=None, df_cat=None, y="subcategory", val_col="mean_dif",
                value_type="mean", normalize=False,
                tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Helper method for plotting"""
    normalize, normalize_for_pos = check_normalize(normalize=normalize)
    cpp_plot = CPPPlots(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    df_pos = cpp_plot.get_df_pos(df=df_feat.copy(), y=y, value_type=value_type, val_col=val_col,
                                 normalize=normalize,
                                 normalize_for_pos=normalize_for_pos)
    # Sort according to given categories
    list_cat = list(df_cat[y].drop_duplicates())
    list_col = list(df_pos.T)
    sorted_col = [x for x in list_cat if x in list_col]
    df_pos = df_pos.T[sorted_col].T
    return df_pos


def _add_importance_map(ax=None, df_feat=None, df_cat=None, start=None, args_len=None, y=None):
    """"""
    _df_pos = _get_df_pos(df_feat=df_feat, df_cat=df_cat, y=y, val_col=ut.COL_FEAT_IMPORTANCE,
                          value_type="sum", normalize="positions_only", start=start, **args_len)
    _df = pd.melt(_df_pos.reset_index(), id_vars="index")
    _df.columns = [ut.COL_SUBCAT, "position", ut.COL_FEAT_IMPORTANCE]
    _list_sub_cat = _df[ut.COL_SUBCAT].unique()
    for i, sub_cat in enumerate(_list_sub_cat):
        _dff = _df[_df[ut.COL_SUBCAT] == sub_cat]
        for pos, val in enumerate(_dff[ut.COL_FEAT_IMPORTANCE]):
            _symbol = "■"  # "•"
            color = "black"
            size = 12 if val >= 1 else (8 if val >= 0.5 else 4)
            _args_symbol = dict(ha="center", va="center", color=color, size=size)
            if val >= 0.2:
                ax.text(pos + 0.5, i + 0.5, _symbol, **_args_symbol)


def _set_size_to_optimized_value(seq_size=None, tmd_jmd_fontsize=None, opt_size=None):
    """Set sizes to given value if None"""
    if tmd_jmd_fontsize is None:
        tmd_jmd_fontsize = opt_size
    args_size = dict(seq_size=seq_size, tmd_jmd_fontsize=tmd_jmd_fontsize)
    return args_size


# TODO simplify interface (delete old profile)
# TODO add importance plot for heatmap
# TODO add ranking

# II Main Functions
class CPPPlot:
    """
    Plot CPP results at global or individual sample level as ranking plot, profile, or map (heatmap, feature map).

    Parameters
    ----------
    accept_gaps : bool, default = False
        Whether to accept missing values by enabling omitting for computations (if True).
    jmd_n_len : int, >=0, default = 10
        Length of JMD-N.
    jmd_c_len : int, >=0, default = 10
        Length of JMD-C.
    ext_len : int, >=0, default = 4
        Length of TMD-extending part (starting from C and N terminal part of TMD).
        Conditions: ext_len < jmd_m_len and ext_len < jmd_c_len.
    verbose : bool, default = True
        Whether to print progress information about the algorithm (if True).

    """
    def __init__(self,
                 df_cat=None,
                 accept_gaps=False,
                 jmd_n_len=10,
                 jmd_c_len=10,
                 ext_len=4,
                 verbose=True
                 ):
        # Load default scales if not specified

        ut.check_bool(name="verbose", val=verbose)
        if df_cat is None:
            df_cat = aa.load_scales(name=ut.COL_SCALE_ID)
            self.df_cat = df_cat
        self._verbose = verbose
        self._accept_gaps = accept_gaps
        # Set consistent length of JMD_N, JMD_C, TMD flanking amino acids (TMD-E)
        self.jmd_n_len = jmd_n_len
        self.jmd_c_len = jmd_c_len
        self.ext_len = ext_len
        # Axes dict for plotting
        self.ax_seq = None

    # Plotting methods
    def ranking(self):
        """"""
        pass

    def profile(self, df_feat=None, y="category", val_col="mean_dif", val_type="count", normalize=False,
                figsize=(7, 5), title=None, title_kws=None,
                dict_color=None, edge_color="none", bar_width=0.75,
                add_jmd_tmd=True, tmd_len=20, start=1,
                jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                seq_size=None, tmd_jmd_fontsize=None,
                xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, xticks_pos=False,
                ytick_size=None, ytick_width=2.0, ytick_length=5.0, ylim=None,
                highlight_tmd_area=True, highlight_alpha=0.15,
                grid=False, grid_axis="both",
                add_legend_cat=True, legend_kws=None,
                shap_plot=False,
                **kwargs):
        """
        Plot feature profile for given features from 'df_feat'.

        Parameters
        ----------
        df_feat : class:`pandas.DataFrame`, optional, default=None
            Dataframe containing the features to be plotted. If None, default features from the instance will be used.
        y : str, default='category'
            Column name in df_feat which contains the categories for grouping.
        val_col : str, default='mean_dif'
            Column name in df_feat which contains the values to be plotted.
        val_type : str, default='count'
            Type of value. Available options are specified by the `check_value_type` function.
        normalize : bool, default=False
            If True, the feature values will be normalized.
        figsize : tuple, default=(7, 5)
            Size of the plot.
        title : str, optional
            Title of the plot.
        title_kws : dict, optional
            Keyword arguments to customize the title appearance.
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
        tmd_jmd_fontsize : float, optional
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
        grid : bool, default=False
            If True, a grid is added to the plot.
        grid_axis : str, default='both'
            Axis on which the grid is drawn. Options: 'both', 'x', 'y'.
        add_legend_cat : bool, default=True
            If True, a legend is added for categories.
        legend_kws : dict, optional
            Keyword arguments for the legend.
        shap_plot : bool, default=False
            If True, SHAP (SHapley Additive exPlanations) plot is generated.
        **kwargs : dict
            Other keyword arguments passed to internal functions or plotting libraries.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        """
        # Group arguments
        args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
        args_size = check_args_size(seq_size=seq_size, tmd_jmd_fontsize=tmd_jmd_fontsize)
        # TODO CHECK
        args_len = ut.check_args_len(tmd_len=tmd_len, jmd_n_len=self.jmd_n_len, jmd_c_len=self.jmd_c_len,
                                     jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
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
        ut.check_bool(name="grid", val=grid)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_color(name="edge_color", val=edge_color, accept_none=True)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)

        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=val_col, col_type=[float, int])
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_value_type(val_type=val_type, count_in=True)
        check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self.df_cat)
        check_grid_axis(grid_axis=grid_axis)
        # Get df positions
        df_feat = self.add_positions(df_feat=df_feat, tmd_len=args_len["tmd_len"], start=start)
        df_pos = _get_df_pos(df_feat=df_feat, df_cat=self.df_cat, y=y, val_col=val_col,
                             value_type=val_type, normalize=normalize, start=start, **args_len)
        # Plotting
        cpp_plot = CPPPlots(**args_len, start=start)
        try:
            ax = cpp_plot.profile(df_pos=df_pos, figsize=figsize, ylim=ylim,
                                  dict_color=dict_color, edge_color=edge_color, bar_width=bar_width,
                                  add_legend=add_legend_cat, legend_kws=legend_kws, shap_plot=shap_plot,
                                  **args_xtick, **kwargs)
        except AttributeError as e:
            error_message = check_parameters(func=self.profile, name_called_func="pd.DataFrame.plot", e=e)
            raise AttributeError(error_message)
        cpp_plot.set_title(title=title, title_kws=title_kws)

        # Autosize tmd sequence & annotation
        opt_size = cpp_plot.optimize_label_size(ax=ax, df_pos=df_pos, label_term=False)
        # Set default ylabel
        ylabel = "Feature impact" if shap_plot else f"Feature count (-/+ {val_col})"
        ax.set_ylabel(ylabel, size=opt_size)
        # Adjust y ticks
        ytick_size = opt_size if ytick_size is None else ytick_size
        plt.yticks(size=ytick_size)
        plt.tick_params(axis="y", color="black", width=ytick_width, length=ytick_length, bottom=False)
        sns.despine(top=True, right=True)
        # Add grid
        if grid:
            ax.set_axisbelow(True)  # Grid behind datasets
            ax.grid(which="major", axis=grid_axis, linestyle="-")
        # Add tmd area
        if highlight_tmd_area:
            cpp_plot.highlight_tmd_area(ax=ax, x_shift=-0.5, tmd_color=tmd_color, alpha=highlight_alpha)
        # Add tmd_jmd sequence if sequence is given
        if type(tmd_seq) == str:
            ax = cpp_plot.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                          xticks_pos=xticks_pos, heatmap=False, x_shift=0,
                                          xtick_size=xtick_size)  # Add tmd_jmd bar
            self.ax_seq = ax
        elif add_jmd_tmd:
            size = opt_size if tmd_jmd_fontsize is None else tmd_jmd_fontsize
            cpp_plot.add_tmd_jmd_bar(ax=ax, x_shift=-0.5, **args_part_color, add_white_bar=False)
            cpp_plot.add_tmd_jmd_xticks(ax=ax, x_shift=0, **args_xtick)
            cpp_plot.add_tmd_jmd_text(ax=ax, x_shift=-0.5, tmd_jmd_fontsize=size)

        # Set current axis to main axis object depending on tmd sequence given or not
        plt.yticks(size=ytick_size)
        plt.tick_params(axis="y", color="black", width=ytick_width, length=ytick_length, bottom=False)
        plt.sca(plt.gcf().axes[0])
        ax = plt.gca()
        return ax

    def heatmap(self, df_feat=None, y="subcategory", val_col="mean_dif", val_type="mean", normalize=False,
                figsize=(8, 5), title=None, title_kws=None,
                vmin=None, vmax=None, grid_on=True,
                cmap="RdBu_r", cmap_n_colors=None, dict_color=None, cbar_kws=None, facecolor_dark=False,
                add_jmd_tmd=True, tmd_len=20, start=1,
                jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                seq_size=None, tmd_jmd_fontsize=None,
                xticks_pos=False, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, ytick_size=None,
                add_legend_cat=True, legend_kws=None,
                add_importance_map=False, cbar_pct=False, **kwargs):
        """
        Plot a featuremap of the selected value column with scale information (y-axis) versus sequence position (x-axis).

        This is a wrapper function for :func:`seaborn.heatmap`, designed to highlight differences between two sets
        of sequences at the positional level (e.g., amino acid level for protein sequences).

        Parameters
        ----------
        df_feat : :class:`~pandas.DataFrame`, shape (n_feature, n_feature_information)
            DataFrame containing unique identifiers, scale information, statistics, and positions for each feature.
        y : {'category', 'subcategory', 'scale_name'}, str, default = 'subcategory'
            Name of the column in the feature DataFrame representing scale information (shown on the y-axis).
        val_col : {'mean_dif', 'feat_impact', 'abs_auc', 'std_test', ...}, str, default = 'mean_dif'
            Name of the column in the feature DataFrame containing numerical values to display.
        val_type : {'mean', 'sum', 'std'}, str, default = 'mean'
            Method to aggregate numerical values from 'val_col'.
        normalize : {True, False, 'positions', 'positions_only'}, bool/str, default = False
            Specifies normalization for numerical values in 'val_col':

            - False: Set value at all positions of a feature without further normalization.

            - True: Set value at all positions of a feature and normalize across all features.

            - 'positions': Value/number of positions set at each position of a feature and normalized across features.
              Recommended when aiming to emphasize features with fewer positions using 'val_col'='feat_impact' and 'value_type'='mean'.

        figsize : tuple(float, float), default = (10,7)
            Width and height of the figure in inches passed to :func:`matplotlib.pyplot.figure`.
        title : str, optional
            Title of figure used by :func:`matplotlib.pyplot.title`.
        title_kws : dict, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.title`.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise, inferred from data and other keyword arguments.
        cmap : matplotlib colormap name or object, or list of colors, default = 'seismic'
            Name of color map assigning data values to color space. If 'SHAP', colors from
            `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ will be used (recommended for feature impact).
        cmap_n_colors : int, optional
            Number of discrete steps in diverging or sequential color map.
        dict_color : dict, optional
            Map of colors for scale categories classifying scales shown on y-axis.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
        add_jmd_tmd : bool, default = True
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
        tmd_color : str, default = 'mediumspringgreen'
            Color of TMD bar.
        jmd_color : str, default = 'blue'
            Color of JMD-N and JMD-C bar.
        tmd_seq_color : str, default = 'black'
            Color of TMD sequence.
        jmd_seq_color : str, default = 'white'
            Color of JMD-N and JMD-C sequence.
        seq_size : float, optional
            Font size of all sequence parts in points. If None, optimized automatically.
        tmd_jmd_fontsize : float, optional
            Font size of 'TMD', 'JMD-N' and 'JMD-C'  label in points. If None, optimized automatically.
        xtick_size : float, default = 11.0
            Size of x ticks in points. Passed as 'size' argument to :meth:`matplotlib.axes.Axes.set_xticklabels`.
        xtick_width : float, default = 2.0
            Widht of x ticks in points. Passed as 'width' argument to :meth:`matplotlib.axes.Axes.tick_params`.
        xtick_length : float, default = 5.0,
            Length of x ticks in points. Passed as 'length' argument to :meth:`matplotlib.axes.Axes.tick_params`.
        ytick_size : float, optional
            Size of scale information as y ticks in points. Passed to :meth:`matplotlib.axes.Axes.tick_params`.
            If None, optimized automatically.
        add_legend_cat : bool, default = True,
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
        - 'cmap_n_colors' is effective only if 'vmin' and 'vmax' align with the data.

        - 'tmd_seq_color' and 'jmd_seq_color' are applicable only when 'tmd_seq', 'jmd_n_seq', and 'jmd_c_seq' are provided.

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
        args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
        args_size = check_args_size(seq_size=seq_size, tmd_jmd_fontsize=tmd_jmd_fontsize)
        # TODO CHECK
        args_len = ut.check_args_len(tmd_len=tmd_len, jmd_n_len=self.jmd_n_len, jmd_c_len=self.jmd_c_len, **args_seq)
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
        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=val_col, col_type=[float, int])
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self.df_cat)
        check_value_type(val_type=val_type, count_in=False)
        check_vmin_vmax(vmin=vmin, vmax=vmax)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self.df_cat)

        # Get df positions
        df_feat = self.add_positions(df_feat=df_feat, tmd_len=args_len["tmd_len"], start=start)
        df_pos = _get_df_pos(df_feat=df_feat, df_cat=self.df_cat, y=y, val_col=val_col,
                             value_type=val_type, normalize=normalize, start=start, **args_len)
        # Plotting
        cpp_plot = CPPPlots(**args_len, start=start)
        cpp_plot.set_figsize(figsize=figsize)   # figsize is not used as argument in seaborn (but in pandas)
        try:
            linecolor = "gray" if facecolor_dark else "black"
            if "linecolor" in kwargs:
                linecolor = kwargs["linecolor"]
            else:
                kwargs["linecolor"] = linecolor
            ax = cpp_plot.heatmap(df_pos=df_pos, vmin=vmin, vmax=vmax, grid_on=grid_on,
                                  cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws,
                                  x_shift=0.5, ytick_size=ytick_size, facecolor_dark=facecolor_dark,
                                  cbar_pct=cbar_pct, **args_xtick, **kwargs)
            ax.axvline(self.jmd_n_len, color=linecolor, linestyle="-", linewidth=1.5)
            ax.axvline(x=self.jmd_n_len + args_len["tmd_len"], color=linecolor, linestyle="-", linewidth=1.5)

        except AttributeError as e:
            error_message = check_parameters(func=self.heatmap, name_called_func="sns.heatmap", e=e)
            raise AttributeError(error_message)
        cpp_plot.set_title(title=title, title_kws=title_kws)
        # Autosize tmd sequence & annotation
        opt_size = cpp_plot.optimize_label_size(ax=ax, df_pos=df_pos)
        # Add importance map
        if add_importance_map:
            _add_importance_map(ax=ax, df_feat=df_feat, df_cat=self.df_cat,
                                start=start, args_len=args_len, y=y)
        # Add scale classification
        if add_legend_cat:
            ax = cpp_plot.add_legend_cat(ax=ax, df_pos=df_pos, df_cat=self.df_cat, y=y, dict_color=dict_color,
                                         legend_kws=legend_kws)
        # Add tmd_jmd sequence if sequence is given
        if isinstance(tmd_seq, str):
            ax = cpp_plot.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                          xticks_pos=xticks_pos,
                                          x_shift=0.5, xtick_size=xtick_size)
            self.ax_seq = ax
        # Add tmd_jmd bar
        elif add_jmd_tmd:
            size = opt_size if tmd_jmd_fontsize is None else tmd_jmd_fontsize
            cpp_plot.add_tmd_jmd_bar(ax=ax, **args_part_color)
            cpp_plot.add_tmd_jmd_xticks(ax=ax, x_shift=0.5, **args_xtick)
            cpp_plot.add_tmd_jmd_text(ax=ax, x_shift=0, tmd_jmd_fontsize=size)
        # Set current axis to main axis object depending on tmd sequence given or not
        plt.sca(plt.gcf().axes[0])
        ax = plt.gca()
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
