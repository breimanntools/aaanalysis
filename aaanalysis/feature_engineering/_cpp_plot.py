"""
This is a script for the frontend of the CPPPlot class.
"""
from typing import Optional, Dict, Union, List, Tuple, Type, Literal
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import aaanalysis.utils as ut

from ._backend.check_feature import (check_split_kws,
                                     check_parts_len,
                                     check_match_features_seq_parts,
                                     check_match_df_parts_features,
                                     check_match_df_parts_list_parts,
                                     check_df_scales,
                                     check_match_df_scales_features,
                                     check_df_cat,
                                     check_match_df_cat_features,
                                     check_match_df_parts_df_scales,
                                     check_match_df_seq_jmd_len,
                                     check_match_df_scales_df_cat)
from ._backend.check_cpp_plot import (check_args_xtick,
                                      check_args_ytick,
                                      check_part_color,
                                      check_seq_color,
                                      check_match_dict_color_df)

from ._backend.cpp.cpp_plot_eval import plot_eval
from ._backend.cpp.cpp_plot_feature import plot_feature
from ._backend.cpp.cpp_plot_ranking import plot_ranking
from ._backend.cpp.cpp_plot_profile import plot_profile
from ._backend.cpp.cpp_plot_heatmap import plot_heatmap
from ._backend.cpp.cpp_plot_feature_map import plot_feature_map
from ._backend.cpp.cpp_plot_update_seq_size import get_tmd_jmd_seq, update_seq_size_, update_tmd_jmd_labels


# I Helper Functions
# Checks for eval plot
def check_match_dict_color_list_cat(dict_color=None, list_cat=None):
    """Check if all categories from list_cat are in dict_color"""
    if list_cat is None:
        return dict_color # Skip test
    # Check list_cat for duplicates
    if len(list_cat) != len(set(list_cat)):
        raise ValueError(f"'list_cat' {list_cat} should not contain duplicates.")

    missing_cat = [x for x in list_cat if x not in dict_color]
    if len(missing_cat) > 0:
        raise ValueError(f"Following categories from 'list_cat' miss a color in 'dict_color': {missing_cat}")
    return {cat: dict_color[cat] for cat in list_cat}


def check_df_eval(df_eval):
    """Check if columns in df_eval have valid values"""
    if len(df_eval) <= 1:
        raise ValueError("'df_eval' should contain at least two features sets")
    # Check quality measures
    list_n_feat = df_eval[ut.COL_N_FEAT].values[0]
    range_abs_auc = df_eval[ut.COL_RANGE_ABS_AUC].values[0]
    avg_mean_dif = df_eval[ut.COL_RANGE_ABS_AUC].values[0]
    n_clusters = df_eval[ut.COL_N_CLUST].values[0]
    avg_n_feat_per_clust = df_eval[ut.COL_AVG_N_FEAT_PER_CLUST].values[0]
    std_n_feat_per_clust = df_eval[ut.COL_STD_N_FEAT_PER_CLUST].values[0]
    ut.check_number_val(name=f"{ut.COL_N_FEAT}: n_features", val=list_n_feat[0], just_int=True)
    _ = ut.check_list_like(name=f"{ut.COL_N_FEAT}: list_n_feat", val=list_n_feat[1], accept_str=False)
    _ = ut.check_list_like(name=ut.COL_RANGE_ABS_AUC, val=range_abs_auc, accept_str=False)
    ut.check_number_val(name=f"{ut.COL_AVG_MEAN_DIF} (pos)", val=avg_mean_dif[0], just_int=False)
    ut.check_number_val(name=f"{ut.COL_AVG_MEAN_DIF} (neg)", val=avg_mean_dif[1], just_int=False)
    ut.check_number_range(name=ut.COL_N_CLUST, val=n_clusters, min_val=1, just_int=True)
    ut.check_number_range(name=ut.COL_AVG_N_FEAT_PER_CLUST, val=avg_n_feat_per_clust, just_int=False)
    ut.check_number_range(name=ut.COL_STD_N_FEAT_PER_CLUST, val=std_n_feat_per_clust, just_int=False)


def check_match_df_eval_list_cat(df_eval=None, list_cat=None):
    """Check if number of features per category in df_eval and list_cat match"""
    names = df_eval[ut.COL_NAME].to_list()
    list_n_feat_sets = [x[1] for x in df_eval[ut.COL_N_FEAT]]
    for list_n_feat, name in zip(list_n_feat_sets, names):
        if len(list_n_feat) != len(list_cat):
            raise ValueError(f"Number of features per category in '{name}' does not match with 'list_cat' {list_cat}")


# Checks for feature plot
def check_match_df_seq_names_to_show(df_seq=None, names_to_show=None):
    """Check if """
    if names_to_show is None:
        return # Skip check
    if ut.COL_NAME not in df_seq:
        raise ValueError(f"'df_seq' must contain '{ut.COL_NAME}' column if 'names_to_show' ({names_to_show}) is not none")
    list_names = df_seq[ut.COL_NAME].to_list()
    missing_names = [x for x in names_to_show if x not in list_names]
    if len(missing_names) > 0:
        raise ValueError(f"Following names from 'names_to_show' are not in '{ut.COL_NAME}' "
                         f"column of 'df_seq': {missing_names}")


# Checks for main CPP plots (ranking, profile, maps)
def check_cmap_for_heatmap(cmap=None):
    """Check if cmap is valid or 'SHAP'"""
    if cmap in [ut.STR_CMAP_SHAP, ut.STR_CMAP_CPP]:
        return None     # Skip test
    ut.check_cmap(name="cmap", val=cmap, accept_none=True)


def check_col_dif(col_dif=None, shap_plot=False):
    """Check if col_dif is string and set default"""
    ut.check_str(name="col_dif", val=col_dif, accept_none=False)
    if col_dif is None:
        col_dif = ut.COL_MEAN_DIF
    if not shap_plot:
        if col_dif != ut.COL_MEAN_DIF:
            raise ValueError(f"'col_dif' ('{col_dif}') must be '{ut.COL_MEAN_DIF}'")
    else:
        if ut.COL_MEAN_DIF not in col_dif:
            raise ValueError(f"If 'shap_plot=True', 'col_dif' ('{col_dif}') must follow '{ut.COL_MEAN_DIF}_'name''")


def check_col_imp(col_imp=None, shap_plot=False):
    """Check if col_imp is string and set default"""
    ut.check_str(name="col_imp", val=col_imp, accept_none=True)
    if col_imp is None:
        col_imp = ut.COL_FEAT_IMPACT if shap_plot else ut.COL_FEAT_IMPORT
    if not shap_plot:
        if ut.COL_FEAT_IMPORT not in col_imp:
            raise ValueError(f"'col_imp' ('{col_imp}') must be '{ut.COL_FEAT_IMPORT} or follow '{ut.COL_FEAT_IMPORT}_'name'")
    else:
        if ut.COL_FEAT_IMPACT not in col_imp:
            raise ValueError(f"If 'shap_plot=True', 'col_imp' ('{col_imp}') must follow '{ut.COL_FEAT_IMPACT}_'name''")
    return col_imp


def check_col_val(col_val=None, shap_plot=False, sample_mean_dif=False):
    """Check if col_val is valid"""
    list_valid_col_val = [ut.COL_MEAN_DIF, ut.COL_ABS_MEAN_DIF, ut.COL_ABS_AUC, ut.COL_FEAT_IMPORT]
    str_error_shap = (f"If 'shap_plot=True', 'col_val' ('{col_val}') must follow '{ut.COL_FEAT_IMPACT}_'name'' or "
                      f"'{ut.COL_MEAN_DIF}_name'")
    str_add = f"Should be one of: {list_valid_col_val}" if not shap_plot else str_error_shap
    ut.check_str(name="col_val", val=col_val, accept_none=False, str_add=str_add)
    if not shap_plot:
        if sample_mean_dif:
            if ut.COL_MEAN_DIF not in col_val and col_val not in list_valid_col_val:
                raise ValueError(f"'col_val' ('{col_val}') must follow {ut.COL_MEAN_DIF}_name' or "
                                 f"should be one of: {list_valid_col_val}")
        elif col_val not in list_valid_col_val:
            raise ValueError(f"'col_val' ('{col_val}') should be one of: {list_valid_col_val}")
    else:
        if ut.COL_FEAT_IMPACT not in col_val and ut.COL_MEAN_DIF not in col_val:
            raise ValueError(str_error_shap)
    return col_val


def check_match_shap_plot_add_legend_cat(shap_plot=False, add_legend_cat=False):
    """Check if not both are True"""
    if shap_plot and add_legend_cat:
        raise ValueError(f"'shap_plot' ({shap_plot}) and 'add_legend_cat' ({add_legend_cat}) can not be both True.")


def check_imp_tuples(name="imp_th", imp_tuples=None):
    """Check if legend importance thresholds are valid"""
    ut.check_tuple(name=name, val=imp_tuples, n=3,
                   accept_none=False, check_number=True,
                   accept_none_number=False, str_add=f"Or should not contain None values: {imp_tuples}")
    th1, th2, th3 = imp_tuples
    if th1 >= th2 or th2 >= th3:
        raise ValueError(f"'{name}' ({imp_tuples}) should contain 3 numbers in ascending order")
    if th1 <= 0:
        raise ValueError(f"Minium of '{name}' ({imp_tuples[0]}) should be > 0")


# Check update_seq_size
def check_match_ax_seq_len(ax=None, jmd_n_len=10, jmd_c_len=10):
    """Check if ax matches with requiered length"""
    labels = ax.xaxis.get_ticklabels(which="both")
    f = lambda l: l.get_window_extent(ax.figure.canvas.get_renderer())
    tick_positions = [f(l).x0 for l in labels]
    _, sorted_labels = zip(*sorted(zip(tick_positions, labels), key=lambda t: t[0]))
    tmd_jmd_seq = "".join([x.get_text() for x in sorted_labels])
    min_len = jmd_n_len + 1 + jmd_c_len
    if len(tmd_jmd_seq) <= min_len:
        raise ValueError(f"TMD-JMD sequence from 'ax' is shorter than minimum ({min_len}; jmd_n: {jmd_n_len}, tmd:>1, jmd_c: {jmd_c_len}."
                         f"\n Sequence (len: {len(tmd_jmd_seq)}) retrieved from 'ax' is: '{tmd_jmd_seq}'")


# II Main Functions
class CPPPlot:
    """
    Plotting class for :class:`CPP` (Comparative Physicochemical Profiling) results [Breimann25a]_.

    This class supports multiple plot types for group or sample-level analysis, including ranking plots,
    profiles, heatmaps, and feature maps.

    """
    def __init__(self,
                 df_scales: Optional[pd.DataFrame] = None,
                 df_cat: Optional[pd.DataFrame] = None,
                 jmd_n_len: int = 10,
                 jmd_c_len: int = 10,
                 accept_gaps: bool = False,
                 verbose: bool = True
                 ):
        """
        Parameters
        ----------
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len: int, default=10
            Length of JMD-C (>=0).
        accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        verbose: bool, default=True
            If ``True``, verbose outputs are enabled.

        Notes
        -----
        Several methods provide the ``shap_plot`` parameter, which allows to specify whether a plot visualizes
        the results of the group-level CPP analysis or the sample-level CPP-SHAP analysis (if ``shap_plot=True``).

        - **CPP Analysis**: Group-level analysis of the most discriminant features between a test and a reference group.
          The overall results are visualized by the :meth:`CPPPlot.feature_map`, revealing the characteristic
          physicochemical signature of the test group.
        - **CPP-SHAP Analysis**: Sample-level analysis of the CPP feature impact with single-residue resolution.

        The methods showing the CPP and CPP-SHAP analysis results are as follows:

        - :meth:`CPPPlot.ranking`: the 'CPP/-SHAP ranking plot' shows the top n ranked features, their feature value differences,
          and feature importance/impact.
        - :meth:`CPPPlot.profile`: the 'CPP/-SHAP profile' shows the cumulative feature importance/impact per residue position.
        - :meth:`CPPPlot.heatmap`: the 'CPP/-SHAP heatmap' shows the feature value difference or feature impact per
          residue position (x-axis) and scale subcategory (y-axis).

        See Also
        --------
        * :class:`CPP`: the respective computation class for the **CPP Analysis**.
        * :class:`ShapModel`: the class combining CPP with the `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_
          explainable Artificial Intelligence (AI) framework.
        * `Anatomy of a figure <https://matplotlib.org/stable/gallery/showcase/anatomy.html>`_ matplotlib guide on
          figure elements.

        Examples
        --------
        .. include:: examples/cpp_plot.rst
        """
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        verbose = ut.check_verbose(verbose)
        check_df_scales(df_scales=df_scales)
        check_df_cat(df_cat=df_cat)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        check_parts_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, accept_none_tmd_len=True)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        df_scales, df_cat = check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        # General settings
        self._verbose = verbose
        self._accept_gaps = accept_gaps
        # Set data parameters
        self._df_cat = df_cat
        self._df_scales = df_scales
        # Set consistent length of JMD-N and JMD-C
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len

    @staticmethod
    def eval(df_eval: pd.DataFrame = None,
             figsize: Tuple[int or float, int or float] = (6, 4),
             dict_xlims: Optional[dict] = None,
             legend: bool = True,
             legend_y: float = -0.3,
             dict_color: Optional[dict] = None,
             list_cat: Optional[List[str]] = None,
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot evaluation output of CPP comparing multiple sets of identified feature sets.

        Evaluation measures are categorized into two groups:

        * **Discriminative Power** measures ('range_ABS_AUC' and 'avg_MEAN_DIF'), which
          assess the effectiveness of the feature set in distinguishing between the test and reference datasets.
        * **Redundancy** measures ('n_clusters', 'avg_n_feat_per_clust', and 'std_n_feat_per_clust'), which
          evaluate the internal redundancy of a feature set using Pearson correlation-based clustering.

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_feature_sets, n_metrics)
            DataFrame with evaluation measures for sets of identified features. Each `row` corresponds to a specific
            feature set. Requiered 'columns' are:

            - 'name': Name of the feature set.
            - 'n_features': Number of features per scale category given as list.
            - 'range_ABS_AUC': Quintile range of absolute AUC among all features (min, 25%, median, 75%, max).
            - 'avg_MEAN_DIF': Two mean differences averaged across all features (for positive and negative 'mean_dif')
            - 'n_clusters': Optimal number of clusters.
            - 'avg_n_feat_per_clust': Average number of features per cluster.
            - 'std_n_feat_per_clust': Standard deviation of feature number per cluster

        figsize : tuple, default=(6, 4)
            Figure dimensions (width, height) in inches.
        dict_xlims : dict, optional
            A dictionary containing x-axis limits for subplots. Keys should be the subplot axis number ({0, 1, 2, 4})
            and values should be tuple specifying (``xmin``, ``xmax``). If ``None``, x-axis limits are auto-scaled.
        legend : bool, default=True
            If ``True``, scale category legend is set under number of features measures.
        legend_y : float, default=-0.3
            Legend position regarding the plot y-axis applied if ``legend=True``.
        dict_color : dict, optional
            Color dictionary of scale categories for legend. Default from :meth:`plot_get_cdict` with ``name='DICT_CAT'``.
        list_cat : list of str, optional
            List of scale categories for which feature numbers are shown.

        Returns
        -------
        fig : plt.Figure
            Figure object for evaluation plot
        axes : array of plt.Axes
            Array of Axes objects, each representing a subplot within the figure.

        Notes
        -----
        * Altering ``figsize`` height could result in unappropriated legend spacing. This can be adjusted by the
          ``legend_y`` parameter together with using the :func:`matplotlib.pyplot.subplots_adjust` function,
          here used with (wspace=0.25, hspace=0, bottom=0.35) parameter settings.

        See Also
        --------
        * :meth:`CPP.eval`: the respective computaton method.
        * :func:`comp_auc_adjusted`.

        Examples
        --------
        .. include:: examples/cpp_plot_eval.rst
        """
        # Check input
        cols_requiered = [ut.COL_NAME, ut.COL_N_FEAT,
                          ut.COL_RANGE_ABS_AUC, ut.COL_AVG_MEAN_DIF,
                          ut.COL_N_CLUST, ut.COL_AVG_N_FEAT_PER_CLUST, ut.COL_STD_N_FEAT_PER_CLUST]
        ut.check_df(name="df_eval", df=df_eval, cols_requiered=cols_requiered, accept_none=False, accept_nan=False)
        check_df_eval(df_eval=df_eval)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_dict_xlims(dict_xlims=dict_xlims, n_ax=5)
        ut.check_bool(name="legend", val=legend)
        ut.check_number_val(name="legend_y", val=legend_y)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        list_cat = ut.check_list_like(name="list_cat", val=list_cat, accept_none=True, accept_str=True,
                                      check_all_str_or_convertible=True)
        if dict_color is None:
            dict_color = ut.plot_get_cdict_(name="DICT_CAT")
        if list_cat is None:
            list_cat = ut.LIST_CAT
        check_match_df_eval_list_cat(df_eval=df_eval, list_cat=list_cat)
        dict_color = check_match_dict_color_list_cat(dict_color=dict_color, list_cat=list_cat)
        # Plotting
        fig, axes = plot_eval(df_eval=df_eval, figsize=figsize, dict_xlims=dict_xlims,
                              legend=legend, legend_y=legend_y,
                              dict_color=dict_color, list_cat=list_cat)
        return fig, axes

    # Plotting method for single feature
    def feature(self,
                feature: str = None,
                df_seq: pd.DataFrame = None,
                labels: ut.ArrayLike1D = None,
                label_test: int = 1,
                label_ref: int = 0, 
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (5.6, 4.8),
                names_to_show: Optional[Union[List[str], str]] = None,
                name_test: str = "TEST",
                name_ref: str = "REF",
                color_test: str = "tab:green",
                color_ref: str = "tab:gray",
                show_seq: bool = False,
                histplot: bool = False,
                fontsize_mean_dif: Union[int, float, None] = 15,
                fontsize_name_test: Union[int, float, None] = 13,
                fontsize_name_ref: Union[int, float, None] = 13,
                fontsize_names_to_show: Union[int, float, None] = 11,
                alpha_hist: int or float = 0.1,
                alpha_dif: int or float = 0.2,
                ) -> plt.Axes:
        """
        Plot distributions of CPP feature values for test and reference datasets highlighting their mean difference.

        Introduced in [Breimann24a]_, a CPP feature is defined as a **Part-Split-Scale** combination. For a sample,
        a feature value is computed in three steps:

            1. **Part Selection**: Identify a specific sequence part.
            2. **Part-Splitting**: Divide the selected part into subsequences, creating a **Part-Split** combination.
            3. **Scale Value Assignment**: For each amino acid in the **Part-Split** subsequence,
               assign its corresponding scale value and calculate the average, which is termed the feature value.

        Parameters
        ----------
        feature : str
            Name of the feature for which test and reference set distributions and difference should be plotted.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence information
            in a distinct **Position-based**, **Part-based**, **Sequence-based**, or **Sequence-TMD-based** format.
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        ax : plt.Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new Axes object is created.
        figsize : tuple, default=(5.6, 4.8)
            Figure dimensions (width, height) in inches.
        names_to_show : list of str, optional
            Names of specific samples from ``df_seq`` to highlight on plot. 'name' column must be given in ``df_seq``
            if ``names_to_show`` is not ``None``.
        name_test : str, default="TEST"
            Name for the test dataset.
        name_ref : str, default="REF"
            Name for the reference dataset.
        color_test : str, default="tab:green"
            Color for the test dataset.
        color_ref : str, default="tab:gray"
            Color for the reference dataset.
        show_seq : bool, default=False
            If ``True``, show sequence of samples selected via ``names_to_show``.
        histplot : bool, default=False
            If ``True``, plot a histogram. If ``False``, plot a kernel density estimate (KDE) plot.
        fontsize_mean_dif : int or float, default=15
            Font size (>0) for displayed mean difference text.
        fontsize_name_test : int or float, default=13
            Font size (>0) for the name of the test dataset.
        fontsize_name_ref : int or float, default=13
            Font size (>0) for the name of the reference dataset.
        fontsize_names_to_show : int or float, default=11
            Font size (>0) for the names selected via ``names_to_show``.
        alpha_hist : int or float, default=0.1
            The transparency alpha value [0-1] for the histogram distributions.
        alpha_dif : int or float, default=0.2
            The transparency alpha value [0-1] for the mean difference area.

        Returns
        -------
        ax : plt.Axes
            CPP feature plot axes object.

        See Also
        --------
        * :class:`SequenceFeature` for details on CPP feature concept.
        * :meth:`SequenceFeature.get_df_parts` for details on formats of ``df_seq``.
        * The internally used :func:`seaborn.histplot` and :func:`seaborn.kdeplot` functions.

        Examples
        --------
        .. include:: examples/cpp_plot_feature.rst
        """
        # Check input
        ut.check_features(features=feature, list_scales=list(self._df_scales))
        ut.check_df_seq(df_seq=df_seq, accept_none=False)
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref],
                                 len_requiered=len(df_seq), allow_other_vals=False)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        names_to_show = ut.check_list_like(name="name_to_show", val=names_to_show, accept_str=True, accept_none=True)
        ut.check_str(name="name_test", val=name_test)
        ut.check_str(name="name_ref", val=name_ref)
        ut.check_color(name="color_test", val=color_test)
        ut.check_color(name="color_ref", val=color_ref)
        ut.check_bool(name="show_seq", val=show_seq)
        ut.check_bool(name="histplot", val=histplot)
        args_fs = ut.check_fontsize_args(fontsize_mean_dif=fontsize_mean_dif,
                                         fontsize_name_test=fontsize_name_test,
                                         fontsize_name_ref=fontsize_name_ref,
                                         fontsize_names_to_show=fontsize_names_to_show)
        ut.check_number_range(name="alpha_hist", val=alpha_hist, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="alpha_dif", val=alpha_dif, accept_none=False, min_val=0, max_val=1, just_int=False)
        check_match_df_seq_names_to_show(df_seq=df_seq, names_to_show=names_to_show)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        df_seq = check_match_df_seq_jmd_len(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)

        # Plot feature
        ax = plot_feature(ax=ax, figsize=figsize,
                          feature=feature, df_scales=self._df_scales, accept_gaps=self._accept_gaps,
                          df_seq=df_seq, labels=labels, label_test=label_test, label_ref=label_ref,
                          jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                          names_to_show=names_to_show, show_seq=show_seq,
                          name_test=name_test, name_ref=name_ref,
                          color_test=color_test, color_ref=color_ref,
                          **args_fs,
                          histplot=histplot, alpha_hist=alpha_hist, alpha_dif=alpha_dif)
        return ax

    # Plotting methods for multiple features (group and sample level)
    def ranking(self,
                df_feat: pd.DataFrame = None,
                shap_plot: bool = False,
                col_dif: str = "mean_dif",
                col_imp: str = "feat_importance",
                rank: bool = True,
                n_top: int = 15,
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
                tmd_len: int = 20,
                tmd_jmd_space: int = 2,
                tmd_color: str = "mediumspringgreen",
                jmd_color: str = "blue",
                tmd_jmd_alpha: Union[int, float] = 0.075,
                name_test: str = "TEST",
                name_ref: str = "REF",
                fontsize_titles: Union[int, float, None] = 12,
                fontsize_labels: Union[int, float, None] = 12,
                fontsize_annotations: Union[int, float, None] = 11,
                xlim_dif: Union[Tuple[Union[int, float], Union[int, float]], None] = (-17.5, 17.5),
                xlim_rank: Optional[Tuple[Union[int, float], Union[int, float]]] = (0, 4),
                rank_info_xy: Optional[Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]] = None,
                ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot CPP/-SHAP feature ranking based on feature importance or sample-specif feature impact.

        Introduced in [Breimann25a]_, this method visualizes the most important features for discriminating between
        the test and the reference dataset groups. At sample level, the feature impact derived from SHAP values
        of a specific sample can be used for ranking if ``shap_plot=True`` and 'feature_impact' column in ``df_feat``.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Must also include feature importance (`feat_importance`) or impact (``feat_impact_'name'``) columns.
        shap_plot : bool, default=False
            Set the analysis type: **CPP Analysis** (if ``False``) for group-level or
            **CPP-SHAP Analysis** for sample-level (or subgroup-level) results:

            **CPP Analysis**

            - ``col_dif``: Displays the group-level difference of feature values, with the `mean_dif` column selected by default.
            - ``col_imp``: Refers to the group-level `feat_importance` column (shown in gray) used for feature ranking.

            **CPP-SHAP Analysis**

            - ``col_dif``: Allows the selection of sample-specific differences against the reference group
              from a `mean_dif_'name'` column.
            - ``col_imp``: Enables the selection of specific feature impacts from a `feat_impact_'name'` column for
              an individual sample, where positive (red) and negative (blue) feature impacts are visualized in the ranking.

        col_dif : str, default='mean_dif'
            Column name in ``df_feat`` for differences in feature values. Must match with the ``shap_plot`` setting.
        col_imp : str, default='feat_importance'
            Column name in ``df_feat`` for feature importance/impact values. Must match with the ``shap_plot`` setting.
        rank : bool, default=True
            If ``True``, features will be ranked in descending order of ``col_imp`` values.
        n_top : int, default=15
            The number of top features to display. Should be 1 < ``n_top`` <= ``n_features``.
        figsize : tuple, default=(7, 5)
            Figure dimensions (width, height) in inches.
        tmd_len : int, default=20
            Length of TMD to be depicted (>0).
        tmd_jmd_space : int, default=2
            The space between TMD and JMD labels (>0) in the feature position subplot.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        tmd_jmd_alpha : int or float, default=0.075
            The transparency alpha value [0-1] of the TMD-JMD area in the feature position subplot.
        name_test : str, default="TEST"
            Name of the test dataset to show in the mean difference subplot.
        name_ref : str, default="REF"
            Name of reference dataset to show in the mean difference subplot.
        fontsize_titles : int or float, default=12
            Font size of the titles.
        fontsize_labels : int or float , default=12
            Font size of plot labels.
        fontsize_annotations : int or float, default=11
            Font size of annotations.
        xlim_dif : tuple, default=(-17.5, 17.5)
            x-axis limits for the mean difference subplot.
        xlim_rank : tuple, default=(0, 4)
            x-axis limits for the ranking subplot. If ``None``, determined automatically.
        rank_info_xy : tuple, optional
            Position (x-axis, y-axis) in ranking subplot for showing additional information (optimized if ``None``):

            - When ``shap_plot=False``: Displays sum of feature importance.
            - When ``shap_plot=True``: Show the sum of the absolute feature impact and the SHAP legend.

        Returns
        -------
        fig : plt.Figure
            The Figure object for the ranking plot.
        axes : array of plt.Axes
            Array of Axes objects, each representing a subplot within the figure.

        Notes
        -----
        * Features are shown as ordered in ``df_feat``. A ranking in descending order based one the following
          columns is recommended:

          - ``feat_importance``: when feature importance is in ``df_feat`` and ``shap_plot=False``.
          - ``feat_impact_'name'``: when sample-specific feature impact is in ``df_feat`` and ``shap_plot=True``.

        See Also
        --------
        * :meth:`CPP.run` for details on CPP statistical measures of the ``df_feat`` DataFrame.
        * :class:`SequenceFeature` for definition of sequence ``Parts``.
        * :meth:`CPPPlot.feature` for visualization of mean differences for specific features.

        Examples
        --------
        .. include:: examples/cpp_plot_ranking.rst
        """
        # Check input
        ut.check_bool(name="shap_plot", val=shap_plot)
        check_col_dif(col_dif=col_dif, shap_plot=shap_plot)
        col_imp = check_col_imp(col_imp=col_imp, shap_plot=shap_plot)
        df_feat = ut.check_df_feat(df_feat=df_feat, shap_plot=shap_plot, cols_requiered=[col_imp, col_dif])
        ut.check_number_range(name="n_top", val=n_top, min_val=2, max_val=len(df_feat), just_int=True)
        ut.check_bool(name="rank", val=rank, accept_none=False)
        ut.check_figsize(figsize=figsize, accept_none=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE],
                                       tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_number_range(name="tmd_jmd_space", val=tmd_jmd_space, min_val=1, just_int=True, accept_none=False)
        ut.check_color(name="tmd_color", val=tmd_color)
        ut.check_color(name="jmd_color", val=jmd_color)
        ut.check_number_range(name="tmd_jmd_alpha", val=tmd_jmd_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_str(name="name_test", val=name_test)
        ut.check_str(name="name_ref", val=name_ref)
        args = dict(min_val=0, exclusive_limits=True, accept_none=True, just_int=False)
        ut.check_number_range(name="fontsize_titles", val=fontsize_titles, **args)
        ut.check_number_range(name="fontsize_labels", val=fontsize_labels, **args)
        ut.check_number_range(name="fontsize_annotations", val=fontsize_annotations, **args)
        ut.check_lim(name="xlim_dif", val=xlim_dif)
        ut.check_lim(name="xlim_rank", val=xlim_rank)
        ut.check_tuple(name="rank_info_xy", val=rank_info_xy, n=2,
                       accept_none=True, check_number=True)

        # DEV: No match check for features and tmd (check_match_features_seq_parts) necessary
        # Plot ranking
        fig, axes = plot_ranking(df_feat=df_feat.copy(),
                                 n_top=n_top, rank=rank,
                                 col_dif=col_dif,
                                 col_imp=col_imp,
                                 shap_plot=shap_plot,
                                 figsize=figsize,
                                 tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                 tmd_color=tmd_color, jmd_color=jmd_color,
                                 tmd_jmd_alpha=tmd_jmd_alpha,
                                 name_test=name_test, name_ref=name_ref,
                                 fontsize_titles=fontsize_titles,
                                 fontsize_labels=fontsize_labels,
                                 fontsize_annotations=fontsize_annotations,
                                 tmd_jmd_space=tmd_jmd_space,
                                 xlim_dif=xlim_dif, xlim_rank=xlim_rank,
                                 rank_info_xy=rank_info_xy)

        # Adjust plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fig.tight_layout()
            plt.subplots_adjust(left=0.25, wspace=0.15)
        return fig, axes

    def profile(self,
                # Data and Plot Type
                df_feat: pd.DataFrame = None,
                shap_plot: bool = False,
                col_imp: Union[str, None] = "feat_importance",
                normalize: bool = True,
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),

                # Appearance of Parts (TMD-JMD)
                start: int = 1,
                tmd_len: int = 20,
                tmd_seq: Optional[str] = None,
                jmd_n_seq: Optional[str] = None,
                jmd_c_seq: Optional[str] = None,
                tmd_color: str = "mediumspringgreen",
                jmd_color: str = "blue",
                tmd_seq_color: str = "black",
                jmd_seq_color: str = "white",
                seq_size: Union[int, float] = None,
                fontsize_tmd_jmd: Union[int, float] = None,
                weight_tmd_jmd: Literal['normal', 'bold'] = "normal",
                add_xticks_pos: bool = False,
                highlight_tmd_area: bool = True,
                highlight_alpha: float = 0.15,

                # Legend, Axis, and Grid Configurations
                add_legend_cat: bool = False,
                dict_color: Optional[dict] = None,
                legend_kws: Optional[dict] = None,
                bar_width: Union[int, float] = 0.75,
                edge_color: Optional[str] = None,
                grid_axis: Optional[Literal['x', 'y', 'both']] = None,
                ylim: Optional[Tuple[float, float]] = None,
                xtick_size: Union[int, float] = 11.0,
                xtick_width: Union[int, float] = 2.0,
                xtick_length: Union[int, float] = 5.0,
                ytick_size: Optional[Union[int, float]] = None,
                ytick_width: Optional[Union[int, float]] = None,
                ytick_length: Union[int, float] = 5.0,
                ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot CPP/-SHAP profile showing feature importance/impact per residue position.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Must also include either ``feat_importance`` or ``feat_impact`` column.
        shap_plot : bool, default=False
            Set the analysis type: **CPP Analysis** (if ``False``) for group-level or
            **CPP-SHAP Analysis** for sample-level (or subgroup-level) results:

             **CPP Analysis**

             - ``col_imp``: Refers to the group-level `feat_importance` column (shown in gray),
               depicted by gray bars for each residue position.

             **CPP-SHAP Analysis**

             - ``col_imp``: Enables the selection of specific feature impacts from a `feat_impact_'name'` column for
               an individual sample, where positive (red) and negative (blue) feature impacts are visualized by +/- bars.

        col_imp : str or None, default='feat_importance'
            Column name in ``df_feat`` for feature importance/impact values. Must match with the ``shap_plot`` setting.
            If ``None``, the number of features per residue position will be shown.
        normalize : bool, default=True
            If ``True``, normalizes aggregated numerical values to a total of 100%.
        ax : plt.Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new Axes object is created.
        figsize : tuple, default=(7, 5)
            Figure dimensions (width, height) in inches.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD to be depicted (>0). Must match with all feature from ``df_feat``.
        tmd_seq : str, optional
            TMD sequence for specific sample.
        jmd_n_seq : str, optional
            JMD N-terminal sequence for specific sample. Length must match with 'jmd_n_len' attribute.
        jmd_c_seq : str, optional
            JMD C-terminal sequence for specific sample. Length must match with 'jmd_c_len' attribute.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        tmd_seq_color : str, default='black'
            Color for TMD sequence.
        jmd_seq_color : str, default='white'
            Color for JMD sequence.
        seq_size : int or float, optional
            Font size (>=0) for sequence characters. If ``None``, optimized automatically.
        fontsize_tmd_jmd : int or float, optional
            Font size (>=0) for the part labels: 'JMD-N', 'TMD', 'JMD-C'. If ``None``, optimized automatically.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for the part labels: 'JMD-N', 'TMD', 'JMD-C'.
        add_xticks_pos : bool, default=False
            If ``True``, include x-tick positions when TMD-JMD sequence is given.
        highlight_tmd_area : bool, default=True
            If ``True``, highlights the TMD area on the plot.
        highlight_alpha : float, default=0.15
            The transparency alpha value [0-1] for TMD area highlighting.
        add_legend_cat : bool, default=False
            If ``True``, the scale categories are indicated as stacked bars and a legend is added. If ``True``,
            ensure that ``shap_plot=False``.
        dict_color : dict, optional
            Color dictionary of scale categories for legend. Default from :meth:`plot_get_cdict` with ``name='DICT_CAT'``.
        legend_kws : dict, optional
            Keyword arguments for the legend passed to :meth:`plot_legend`.
        bar_width : int or float, default=0.75
            Width of the bars.
        edge_color : str, optional
            Color of the bar edges.
        grid_axis : {'x', 'y', 'both', None}, default=None
            Axis on which the grid is drawn if not ``None``.
        ylim : tuple, optional
            Y-axis limits. If ``None``, y-axis limits are set automatically.
        xtick_size : int or float, default=11.0
            Size of x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=5.0
            Length of the x-ticks (>0).
        ytick_size : int or float, optional
            Size of y-tick labels (>0).
        ytick_width : int or float, default=2.0
            Width of the y-ticks (>0).
        ytick_length : int or float, default=5.0
            Length of the y-ticks (>0).

        Returns
        -------
        fig : plt.Figure
            The Figure object for the CPP profile plot.
        ax : plt.Axes
            CPP profile plot axes object.

        Notes
        -----
        * ``tmd_seq_color`` and ``jmd_seq_color`` are applicable only when ``tmd_seq``, ``jmd_n_seq``,
          and ``jmd_c_seq`` are provided.

        Warnings
        --------
        * If ``ylim`` does not match with minimum and/or maximum of aggregate numerical values across all residue
          position, a ``UserWarning`` is raised and ``ylim`` will be adjusted automatically.

        Examples
        --------
        .. include:: examples/cpp_plot_profile.rst
        """
        # Check primary input
        ut.check_bool(name="shap_plot", val=shap_plot)
        if col_imp is None:
            df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat)
        else:
            col_imp = check_col_imp(col_imp=col_imp, shap_plot=shap_plot)
            df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat, shap_plot=shap_plot, cols_requiered=[col_imp])
        ut.check_bool(name="normalize", val=normalize)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)

        # Check specific TMD-JMD input
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                             jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                             check_jmd_seq_len_consistent=True)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        args_fs = ut.check_fontsize_args(seq_size=seq_size,
                                         fontsize_tmd_jmd=fontsize_tmd_jmd)
        ut.check_str_options(name="weight_tmd_jmd", val=weight_tmd_jmd,
                             list_str_options=["normal", "bold"])
        ut.check_bool(name="add_xticks_pos", val=add_xticks_pos)
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_number_range(name="tmd_area_alpha", val=highlight_alpha, min_val=0, max_val=1, just_int=False)
        check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE],
                                       tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                       tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        check_match_shap_plot_add_legend_cat(shap_plot=shap_plot, add_legend_cat=add_legend_cat)

        # Check plot styling input
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        dict_color = check_match_dict_color_df(dict_color=dict_color, df=df_feat)
        ut.check_number_range(name="bar_width", val=bar_width, min_val=0, just_int=False)
        ut.check_color(name="edge_color", val=edge_color, accept_none=True)
        ut.check_str_options(name="grid_axis", val=grid_axis, accept_none=True,
                             list_str_options=["y", "x", "both"])
        ut.check_lim(name="ylim", val=ylim, accept_none=True)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_ytick = check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)

        # Plot profile
        fig, ax = plot_profile(df_feat=df_feat, df_cat=self._df_cat, shap_plot=shap_plot,
                               col_imp=col_imp, normalize=normalize,
                               figsize=figsize, ax=ax,
                               start=start, **args_len, **args_seq,
                               **args_part_color, **args_seq_color,
                               **args_fs, weight_tmd_jmd=weight_tmd_jmd,
                               add_xticks_pos=add_xticks_pos,
                               highlight_tmd_area=highlight_tmd_area, highlight_alpha=highlight_alpha,
                               add_legend_cat=add_legend_cat, dict_color=dict_color, legend_kws=legend_kws,
                               bar_width=bar_width, edge_color=edge_color,
                               grid_axis=grid_axis, ylim=ylim, **args_xtick, **args_ytick)

        # Adjust plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.tight_layout()
        if tmd_seq is not None and seq_size is None:
            ax, seq_size = update_seq_size_(ax=ax, **args_seq, **args_part_color, **args_seq_color)
            if self._verbose:
                ut.print_out(f"Optimized sequence character fontsize is: {seq_size}")
        return fig, ax

    def heatmap(self,
                # Data and Plot Type
                df_feat: pd.DataFrame = None,
                shap_plot: bool = False,
                col_cat: Literal['category', 'subcategory', 'scale_name'] = "subcategory",
                col_val: str = "mean_dif",
                name_test: str = "TEST",
                name_ref: str = "REF",
                figsize: Tuple[Union[int, float], Union[int, float]] = (8, 8),

                # Appearance of Parts (TMD-JMD)
                start: int = 1,
                tmd_len: int = 20,
                tmd_seq: Optional[str] = None,
                jmd_n_seq: Optional[str] = None,
                jmd_c_seq: Optional[str] = None,
                tmd_color: str = "mediumspringgreen",
                jmd_color: str = "blue",
                tmd_seq_color: str = "black",
                jmd_seq_color: str = "white",
                seq_size: Optional[Union[int, float]] = None,
                fontsize_tmd_jmd: Optional[Union[int, float]] = None,
                weight_tmd_jmd: Literal['normal', 'bold'] = "normal",
                fontsize_labels: Union[int, float] = 12,
                add_xticks_pos: bool = False,

                # Legend, Axis, and Grid Configurations
                grid_linewidth: Union[int, float] = 0.01,
                grid_linecolor: Optional[str] = None,
                border_linewidth: Union[int, float] = 2,
                facecolor_dark: Optional[bool] = None,
                vmin: Optional[Union[int, float]] = None,
                vmax: Optional[Union[int, float]] = None,
                cmap: Optional[str] = None,
                cmap_n_colors: int = 101,
                cbar_pct: bool = True,
                cbar_kws: Optional[dict] = None,
                cbar_xywh: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = (0.7, None, 0.2, None),
                dict_color: Optional[dict] = None,
                legend_kws: Optional[dict] = None,
                legend_xy: Tuple[Optional[float], Optional[float]] = (-0.1, -0.01),
                xtick_size: Union[int, float] = 11.0,
                xtick_width: Union[int, float] = 2.0,
                xtick_length: Union[int, float] = 5.0,
                ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a CPP/-SHAP heatmap showing the feature value mean difference/feature impact
        per scale subcategory (y-axis) and residue position (x-axis).

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_feature, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Can also include feature impact (``feat_impact``) column.
        shap_plot : bool, default=False
            Set the analysis type: **CPP Analysis** (if ``False``) for group-level or
            **CPP-SHAP Analysis** for sample-level (or subgroup-level) results:

             **CPP Analysis**

            - ``col_val``: Displays typically the difference of feature values, either at group-level when the `mean_dif`
              column is selected or at sample-level (group-level) when a `mean_dif_'name'` column is provided.

            **CPP-SHAP Analysis**

            - ``col_val``: Enables typically the selection of specific feature impacts from a `feat_impact_'name'` column
              for an individual sample, where positive (red) and negative (blue) feature impacts are indicated.

        col_cat : {'category', 'subcategory', 'scale_name'}, default='subcategory'
            Column name in ``df_feat`` for scale classification (y-axis).
        col_val : {'mean_dif', 'abs_mean_dif', 'abs_auc', 'feat_importance', ``mean_dif_'name'``, ``feat_impact_'name'``}, default='mean_dif'
            Column name in ``df_feat`` for numerical values to display. Must match with the ``shap_plot`` setting.
        name_test : str, default="TEST"
            Name for the test dataset.
        name_ref : str, default="REF"
            Name for the reference dataset.
        figsize : tuple, default=(8, 8)
            Figure dimensions (width, height) in inches.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD to be depicted (>0). Must match with all feature from ``df_feat``.
        tmd_seq : str, optional
            TMD sequence for specific sample.
        jmd_n_seq : str, optional
            JMD N-terminal sequence for specific sample. Length must match with 'jmd_n_len' attribute.
        jmd_c_seq : str, optional
            JMD C-terminal sequence for specific sample. Length must match with 'jmd_c_len' attribute.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        tmd_seq_color : str, default='black'
            Color for TMD sequence.
        jmd_seq_color : str, default='white'
            Color for JMD sequence.
        seq_size : int or float, optional
            Font size (>=0) for sequence characters. If ``None``, optimized automatically.
        fontsize_tmd_jmd : int or float, optional
            Font size (>=0) for the part labels: 'JMD-N', 'TMD', 'JMD-C'. If ``None``, optimized automatically.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for the part labels: 'JMD-N', 'TMD', 'JMD-C'.
        fontsize_labels : int or float, default=12
            Font size (>= 0) for figure labels. If ``None``, determined automatically.
        add_xticks_pos : bool, default=False
            If ``True``, include x-tick positions when TMD-JMD sequence is given.
        grid_linewidth : int or float, default=0.01
            Line width for the grid.
        grid_linecolor : str, optional
            Color for the grid lines. If ``None``, automatically determined based on ``facecolor_dark``.
        border_linewidth : int or float, default=2
            Line width for the TMD-JMD border.
        facecolor_dark : bool, optional
            Sets background of heatmap to black (if ``True``) or white. If ``None``, automatically determined from
            ``shap_plot`` setting. Affects grid cells for missing or near-zero data based on ``col_val``.
        vmin : int or float, optional
            Minimum ``col_val`` value setting the lower end of the colormap. If ``None``, determined automatically.
        vmax : int or float, optional
            Maximum ``col_val`` value setting the upper end of the colormap. If ``None``, determined automatically.
        cmap : matplotlib colormap name or object, optional
            Name of the colormap to use. If ``None``, automatically determined ``col_val`` data and 'shap_plot' setting.
        cmap_n_colors : int, default=101
            Number of discrete steps (>1) in diverging or sequential colormap.
        cbar_pct : bool, default=True
            If ``True``, colorbar is represented in percentage and the ``col_val`` values are converted
            accordingly by multiplying with 100 if necessary.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for colorbar passed to :meth:`matplotlib.figure.Figure.colorbar`.
        cbar_xywh : tuple, default=(0.7, None, 0.2, None)
            Colorbar position and size: x-axis (left), y-axis (bottom), width, height. Values are optimized if ``None``.
        dict_color : dict, optional
            Color dictionary of scale categories classifying scales shown on y-axis. Default from
            :meth:`plot_get_cdict` with ``name='DICT_CAT'``.
        legend_kws : dict, optional
            Keyword arguments for the legend passed to :meth:`plot_legend`.
        legend_xy : tuple, default=(-0.1, -0.01)
            Position for scale category legend: x- and y-axis coordinates. Values are set to default if ``None``.
        xtick_size : int or float, default=11.0
            Size of x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=5.0
            Length of the x-ticks (>0).

        Returns
        -------
        fig : plt.Figure
            The Figure object for the CPP heatmap.
        ax : plt.Axes
            Array of Axes objects for the CPP heatmap.

        Notes
        -----
        ``tmd_seq_color`` and ``jmd_seq_color`` are applicable only when ``tmd_seq``, ``jmd_n_seq``,
        and ``jmd_c_seq`` are provided.

        See Also
        --------
        * :meth:`CPP.run` for details on CPP statistical measures of the ``df_feat`` DataFrame.
        * :class:`SequenceFeature` for definition of sequence ``Parts``.
        * :meth:`CPPPlot.feature` for visualization of mean differences for specific features.
        * :meth:`seaborn.heatmap` for seaborn heatmap.
        * :meth:`matplotlib.figure.Figure.colorbar` for colorbar arguments.
        * `Matplotlib Colormaps <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_ for further ``cmap`` options.
        * :meth:`plot_legend` used for setting scale category legend.

        Examples
        --------
        .. include:: examples/cpp_plot_heatmap.rst
        """
        # Check primary input
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_str_options(name="col_cat", val=col_cat,
                             list_str_options=[ut.COL_CAT, ut.COL_SUBCAT, ut.COL_SCALE_NAME])
        col_val = check_col_val(col_val=col_val, shap_plot=shap_plot)
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat, shap_plot=shap_plot,
                                   cols_requiered=col_val, cols_nan_check=col_val)
        ut.check_str(name="name_test", val=name_test)
        ut.check_str(name="name_ref", val=name_ref)
        ut.check_figsize(figsize=figsize, accept_none=True)

        # Check specific TMD-JMD input
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                             jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                             check_jmd_seq_len_consistent=True)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        args_fs = ut.check_fontsize_args(seq_size=seq_size,
                                         fontsize_tmd_jmd=fontsize_tmd_jmd,
                                         fontsize_labels=fontsize_labels)
        ut.check_str_options(name="weight_tmd_jmd", val=weight_tmd_jmd,
                             list_str_options=["normal", "bold"])
        ut.check_bool(name="add_xticks_pos", val=add_xticks_pos)
        check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE],
                                       tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                       tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)

        # Check plot styling input
        ut.check_number_range(name="grid_linewidth", val=grid_linewidth, min_val=0, just_int=False)
        ut.check_color(name="grid_linecolor", val=grid_linecolor, accept_none=True)
        ut.check_number_range(name="border_linewidth", val=border_linewidth, min_val=0, just_int=False)
        ut.check_bool(name="facecolor_dark", val=facecolor_dark, accept_none=True)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        check_cmap_for_heatmap(cmap=cmap)
        ut.check_number_range(name="cmap_n_colors", val=cmap_n_colors, min_val=2, accept_none=True, just_int=True)
        ut.check_bool(name="cbar_pct", val=cbar_pct)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_tuple(name="cbar_xywh", val=cbar_xywh, n=4, accept_none=False,
                       check_number=True, accept_none_number=True)
        dict_color = check_match_dict_color_df(dict_color=dict_color, df=df_feat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_tuple(name="legend_xy", val=legend_xy, n=2, accept_none=False,
                       check_number=True, accept_none_number=True)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)

        # Plot heatmap
        fig, ax = plot_heatmap(df_feat=df_feat, df_cat=self._df_cat,
                               shap_plot=shap_plot,
                               col_cat=col_cat, col_val=col_val,
                               name_test=name_test, name_ref=name_ref,
                               figsize=figsize,
                               start=start, **args_len, **args_seq,
                               **args_part_color, **args_seq_color,
                               **args_fs, weight_tmd_jmd=weight_tmd_jmd,
                               add_xticks_pos=add_xticks_pos,
                               grid_linewidth=grid_linewidth, grid_linecolor=grid_linecolor,
                               border_linewidth=border_linewidth,
                               facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                               cmap=cmap, cmap_n_colors=cmap_n_colors,
                               cbar_pct=cbar_pct, cbar_kws=cbar_kws, cbar_xywh=cbar_xywh,
                               dict_color=dict_color, legend_kws=legend_kws, legend_xy=legend_xy,
                               **args_xtick)

        # Adjust plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fig.tight_layout()
        if tmd_seq is not None and seq_size is None:
            ax, seq_size = update_seq_size_(ax=ax, **args_seq, **args_part_color, **args_seq_color)
            if self._verbose:
                ut.print_out(f"Optimized sequence character fontsize is: {seq_size}")
        return fig, ax

    def feature_map(self,
                    # Data and Plot Type
                    df_feat: pd.DataFrame = None,
                    col_cat: Literal['category', 'subcategory', 'scale_name'] = "subcategory",
                    col_val: str = "mean_dif",
                    col_imp: str = "feat_importance",
                    name_test: str = "TEST",
                    name_ref: str = "REF",
                    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 8),

                    # Feature importance
                    add_imp_bar_top: bool = True,
                    imp_bar_th: Optional[Union[int, float]] = None,
                    imp_bar_label_type: Union[Literal['long', 'short'], None] = 'long',
                    imp_ths: Tuple[Optional[float], Optional[float], Optional[float]] = (0.2, 0.5, 1),
                    imp_marker_sizes: Tuple[Optional[float], Optional[float], Optional[float]] = (3, 5, 8),

                    # Appearance of Parts (TMD-JMD)
                    start: int = 1,
                    tmd_len: int = 20,
                    tmd_seq: Optional[str] = None,
                    jmd_n_seq: Optional[str] = None,
                    jmd_c_seq: Optional[str] = None,
                    tmd_color: str = "mediumspringgreen",
                    jmd_color: str = "blue",
                    tmd_seq_color: str = "black",
                    jmd_seq_color: str = "white",
                    seq_size: Optional[Union[int, float]] = None,
                    fontsize_tmd_jmd: Optional[Union[int, float]] = None,
                    weight_tmd_jmd: Literal['normal', 'bold'] = "normal",
                    fontsize_titles: Union[int, float] = 11,
                    fontsize_labels: Union[int, float] = 12,
                    fontsize_annotations: Union[int, float] = 11,
                    fontsize_imp_bar: Union[int, float] = 9,
                    add_xticks_pos: bool = False,

                    # Legend, Axis, and Grid Configurations
                    grid_linewidth: Union[int, float] = 0.01,
                    grid_linecolor: Optional[str] = None,
                    border_linewidth: Union[int, float] = 2,
                    facecolor_dark: bool = False,
                    vmin: Optional[Union[int, float]] = None,
                    vmax: Optional[Union[int, float]] = None,
                    cmap: Optional[str] = None,
                    cmap_n_colors: int = 101,
                    cbar_pct: bool = True,
                    cbar_kws: Optional[dict] = None,
                    cbar_xywh: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = (0.5, None, 0.2, None),
                    dict_color: Optional[dict] = None,
                    legend_kws: Optional[dict] = None,
                    legend_xy: Tuple[Optional[float], Optional[float]] = (-0.1, -0.01),
                    legend_imp_xy: Tuple[Optional[float], Optional[float]] = (1.25, 0),
                    xtick_size: Union[int, float] = 11.0,
                    xtick_width: Union[int, float] = 2.0,
                    xtick_length: Union[int, float] = 5.0,
                    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot CPP feature map showing feature value mean difference and feature importance
        per scale subcategory (y-axis) and residue position (x-axis).

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_feature, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Must also include a feature importance column (``col_imp``).
        col_cat : {'category', 'subcategory', 'scale_name'}, default='subcategory'
            Column name in ``df_feat`` for scale information (y-axis).
        col_val : {'mean_dif', 'abs_mean_dif', 'abs_auc'}, default='mean_dif'
            Column name in ``df_feat`` for numerical values to display.
        col_imp :  {``feat_importance``, ``feat_importance_'name'``}, default='feat_importance'
            Column name in ``df_feat`` for feature importance (group-, subgroup- or sample-level).
        name_test : str, default="TEST"
            Name for the test dataset.
        name_ref : str, default="REF"
            Name for the reference dataset.
        figsize : tuple, default=(8, 8)
            Figure dimensions (width, height) in inches.
        add_imp_bar_top : bool, default=True
            If ``True``, add bars for cumulative feature importance per position (top).
        imp_bar_th : int or float, optional
            Threshold for cumulative feature importance per scale (right bars). If ``None``, determined automatically.
        imp_bar_label_type : {'long', 'short', None} default='long'
            Label type for cumulative feature importance bar chart. If ``None``, no label is shown.
        imp_ths : tuple, default=(0.2, 0.5, 1)
            Three ascending thresholds for feature importance (scale- and position-specific).
        imp_marker_sizes : tuple, default=(3, 5, 8)
            Size of three feature importance markers defined by ``impd_th``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD to be depicted (>0). Must match with all feature from ``df_feat``.
        tmd_seq : str, optional
            TMD sequence for specific sample.
        jmd_n_seq : str, optional
            JMD N-terminal sequence for specific sample. Length must match with 'jmd_n_len' attribute.
        jmd_c_seq : str, optional
            JMD C-terminal sequence for specific sample. Length must match with 'jmd_c_len' attribute.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        tmd_seq_color : str, default='black'
            Color for TMD sequence.
        jmd_seq_color : str, default='white'
            Color for JMD sequence.
        seq_size : int or float, optional
            Font size (>=0) for sequence characters. If ``None``, optimized automatically.
        fontsize_tmd_jmd : int or float, optional
            Font size (>=0) for the part labels: 'JMD-N', 'TMD', 'JMD-C'. If ``None``, optimized automatically.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for the part labels: 'JMD-N', 'TMD', 'JMD-C'.
        fontsize_titles : int or float, default=11
            Font size (>= 0) for figure titles. If ``None``, determined automatically.
        fontsize_labels : int or float, default=12
            Font size (>= 0) for figure labels. If ``None``, determined automatically.
        fontsize_annotations : int or float, default=10
            Font size (>= 0) for figure annotations. If ``None``, determined automatically.
        fontsize_imp_bar : int or float, default=9
            Font size (>= 0) for feature importance in bars. If ``None``, determined automatically.
        add_xticks_pos : bool, default=False
            If ``True``, include x-tick positions when TMD-JMD sequence is given.
        grid_linewidth : int or float, default=0.01
            Line width for the grid.
        grid_linecolor : str, optional
            Color for the grid lines. If ``None``, automatically determined based on ``facecolor_dark``.
        border_linewidth : int or float, default=2
            Line width for the TMD-JMD border.
        facecolor_dark : bool, optional
            Sets background of heatmap to black (if ``True``) or white. If ``None``, automatically determined from
            ``shap_plot`` setting. Affects grid cells for missing or near-zero data based on ``col_val``.
        vmin : int or float, optional
            Minimum ``col_val`` value setting the lower end of the colormap. If ``None``, determined automatically.
        vmax : int or float, optional
            Maximum ``col_val`` value setting the upper end of the colormap. If ``None``, determined automatically.
        cmap : matplotlib colormap name or object, optional
            Name of the colormap to use. If ``None``, automatically determined ``col_val`` data and 'shap_plot' setting.
        cmap_n_colors : int, default=101
            Number of discrete steps (>1) in diverging or sequential colormap.
        cbar_pct : bool, default=True
            If ``True``, colorbar is represented in percentage and the ``col_val`` values are converted
            accordingly by multiplying with 100 if necessary.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for colorbar passed to :meth:`matplotlib.figure.Figure.colorbar`.
        cbar_xywh : tuple, default=(0.7, None, 0.2, None)
            Colorbar position and size: x-axis (left), y-axis (bottom), width, height. Values are optimized if ``None``.
        dict_color : dict, optional
            Color dictionary of scale categories classifying scales shown on y-axis. Default from
            :meth:`plot_get_cdict` with ``name='DICT_CAT'``.
        legend_kws : dict, optional
            Keyword arguments for the legend passed to :meth:`plot_legend`.
        legend_xy : tuple, default=(-0.1, -0.01)
            Position for scale category legend: x- and y-axis coordinates. Values are set to default if ``None``.
        legend_imp_xy : tuple, default=(1.25, 0)
            Position for feature importance legend: x- and y-axis coordinates (relative to cbar).
        xtick_size : int or float, default=11.0
            Size of x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=5.0
            Length of the x-ticks (>0).

        Returns
        -------
        fig : plt.Figure
            The Figure object for the CPP feature map.
        ax : plt.Axes
            Array of Axes objects for the CPP feature map.

        Notes
        -----
        ``tmd_seq_color`` and ``jmd_seq_color`` are applicable only when ``tmd_seq``, ``jmd_n_seq``,
        and ``jmd_c_seq`` are provided.

        See Also
        --------
        * :meth:`CPP.run` for details on CPP statistical measures of the ``df_feat`` DataFrame.
        * :class:`SequenceFeature` for definition of sequence ``Parts``.
        * :meth:`CPPPlot.feature` for visualization of mean differences for specific features.
        * :meth:`seaborn.heatmap` for seaborn heatmap.
        * :meth:`matplotlib.figure.Figure.colorbar` for colorbar arguments.
        * `Matplotlib Colormaps <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_ for further ``cmap`` options.
        * :meth:`plot_legend` used for setting scale category legend.

        Examples
        --------
        .. include:: examples/cpp_plot_feature_map.rst
        """
        # Check primary input
        ut.check_str_options(name="col_cat", val=col_cat,
                             list_str_options=[ut.COL_CAT, ut.COL_SUBCAT, ut.COL_SCALE_NAME])
        col_val = check_col_val(col_val=col_val, sample_mean_dif=True)
        col_imp = check_col_imp(col_imp=col_imp)
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat,
                                   cols_requiered=[col_val, col_imp],
                                   cols_nan_check=col_val)
        ut.check_str(name="name_test", val=name_test)
        ut.check_str(name="name_ref", val=name_ref)
        ut.check_figsize(figsize=figsize, accept_none=True)

        #  Check feature importance presentation input
        ut.check_bool(name="add_imp_bar_top", val=add_imp_bar_top)
        ut.check_number_range(name="imp_bar_th", val=imp_bar_th, accept_none=True, min_val=0, just_int=False)
        ut.check_str_options(name="imp_bar_label_type", val=imp_bar_label_type, accept_none=True,
                             list_str_options=["short", "long", None])
        check_imp_tuples(name="imp_ths", imp_tuples=imp_ths)
        check_imp_tuples(name="imp_marker_sizes", imp_tuples=imp_marker_sizes)

        # Check specific TMD-JMD input
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                             jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                             check_jmd_seq_len_consistent=True)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        args_fs = ut.check_fontsize_args(seq_size=seq_size,
                                         fontsize_tmd_jmd=fontsize_tmd_jmd,
                                         fontsize_titles=fontsize_titles,
                                         fontsize_labels=fontsize_labels,
                                         fontsize_annotations=fontsize_annotations,
                                         fontsize_imp_bar=fontsize_imp_bar)
        ut.check_str_options(name="weight_tmd_jmd", val=weight_tmd_jmd,
                             list_str_options=["normal", "bold"])
        ut.check_bool(name="add_xticks_pos", val=add_xticks_pos)
        check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE],
                                       tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                       tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        # Check plot styling input
        ut.check_number_range(name="grid_linewidth", val=grid_linewidth, min_val=0, just_int=False)
        ut.check_color(name="grid_linecolor", val=grid_linecolor, accept_none=True)
        ut.check_number_range(name="border_linewidth", val=border_linewidth, min_val=0, just_int=False)
        ut.check_bool(name="facecolor_dark", val=facecolor_dark, accept_none=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        check_cmap_for_heatmap(cmap=cmap)
        ut.check_number_range(name="cmap_n_colors", val=cmap_n_colors, min_val=2, accept_none=True, just_int=True)
        ut.check_bool(name="cbar_pct", val=cbar_pct)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_tuple(name="cbar_xywh", val=cbar_xywh, n=4, accept_none=False,
                       check_number=True, accept_none_number=True)
        dict_color = check_match_dict_color_df(dict_color=dict_color, df=df_feat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_tuple(name="legend_xy", val=legend_xy, n=2, accept_none=False,
                       check_number=True, accept_none_number=True)
        ut.check_tuple(name="legend_imp_xy", val=legend_imp_xy, n=2, accept_none=False,
                       check_number=True, accept_none_number=True)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)

        # Plot feature map
        fig, ax = plot_feature_map(df_feat=df_feat, df_cat=self._df_cat,
                                   col_cat=col_cat, col_val=col_val, col_imp=col_imp,
                                   name_test=name_test, name_ref=name_ref,
                                   figsize=figsize,
                                   add_imp_bar_top=add_imp_bar_top,
                                   imp_bar_th=imp_bar_th,
                                   imp_bar_label_type=imp_bar_label_type,
                                   imp_ths=imp_ths, imp_marker_sizes=imp_marker_sizes,
                                   start=start, **args_len, **args_seq,
                                   **args_part_color, **args_seq_color,
                                   **args_fs, weight_tmd_jmd=weight_tmd_jmd,
                                   add_xticks_pos=add_xticks_pos,
                                   grid_linewidth=grid_linewidth, grid_linecolor=grid_linecolor,
                                   border_linewidth=border_linewidth,
                                   facecolor_dark=facecolor_dark, vmin=vmin, vmax=vmax,
                                   cmap=cmap, cmap_n_colors=cmap_n_colors,
                                   cbar_pct=cbar_pct, cbar_kws=cbar_kws, cbar_xywh=cbar_xywh,
                                   dict_color=dict_color, legend_kws=legend_kws, legend_xy=legend_xy,
                                   legend_imp_xy=legend_imp_xy,
                                   **args_xtick)

        # Adjust plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fig.tight_layout()
            plt.subplots_adjust(right=0.95)
        if tmd_seq is not None and seq_size is None:
            ax, seq_size = update_seq_size_(ax=ax, **args_seq, **args_part_color, **args_seq_color)
            if self._verbose:
                ut.print_out(f"Optimized sequence character fontsize is: {seq_size}")
        return fig, ax

    def update_seq_size(self,
                        ax: plt.Axes = None,
                        fig: Optional[plt.Figure] = None,
                        max_x_dist: float = 0.1,
                        fontsize_tmd_jmd: Union[int, float] = None,
                        weight_tmd_jmd: Literal['normal', 'bold'] = 'normal',
                        tmd_color: str = "mediumspringgreen",
                        jmd_color: str = "blue",
                        tmd_seq_color: str = "black",
                        jmd_seq_color: str = "white",
                        ) -> plt.Axes:
        """
        Update the font size of the sequence characters to prevent overlap.

        This method adjusts the font size of TMD-JMD sequence characters based on their provided sequences
        to ensure that the labels are clearly legible and do not overlap in the plot.

        Parameters
        ---------
        ax : plt.Axes
            CPP plot axes object.
        fig : plt.Figure, optional
            CPP plot figure object. If given, ``fontsize_tmd_jmd`` will be automatically adjusted.
        max_x_dist : float, default=0.1
            Maximum allowed horizontal distance between sequence characters during font size optimization.
            A greater value reduces potential overlaps of sequence characters.
        fontsize_tmd_jmd : int or float, optional
            Font size (>=0) for the part labels: 'JMD-N', 'TMD', 'JMD-C'. If ``None``, optimized automatically.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for the part labels: 'JMD-N', 'TMD', 'JMD-C'.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        tmd_seq_color : str, default='black'
            Color for TMD sequence.
        jmd_seq_color : str, default='white'
            Color for JMD sequences.

        Returns
        -------
        ax : plt.Axes
            CPP plot axes object.

        Notes
        -----
        * Use :meth:`CPPPlot.update_seq_size` AFTER :func:`matplotlib.pyplot.tight_layout`.

        See Also
        --------
        * :meth:`CPPPlot.profile` and :meth:`CPPPlot.heatmap` methods, which also use the ``tmd_seq``, ``jmd_n_seq``,
          and ``jmd_c_seq`` parameters. :meth:`CPPPlot.update_seq_size` should be called after further plot
          modifications that alter the size of figure or x-axis.

        Examples
        --------
        .. include:: examples/cpp_plot_update_seq_size.rst
        """
        # Check input
        ax = ut.check_ax(ax=ax, accept_none=False, return_first=True)
        ut.check_fig(fig=fig, accept_none=True)
        ut.check_number_range(name="max_x_dist", val=max_x_dist, min_val=0, just_int=False)
        ut.check_number_range(name="fontsize_tmd_jmd", val=fontsize_tmd_jmd, min_val=0, accept_none=True, just_int=False)
        ut.check_str_options(name="weight_tmd_jmd", val=weight_tmd_jmd,
                             list_str_options=["normal", "bold"])
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=self._jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=self._jmd_c_len)
        check_match_ax_seq_len(ax=ax, jmd_c_len=jmd_c_len, jmd_n_len=jmd_n_len)
        # Adjust font size to prevent overlap
        jmd_n_seq, tmd_seq, jmd_c_seq = get_tmd_jmd_seq(ax=ax, jmd_c_len=jmd_c_len, jmd_n_len=jmd_n_len)
        args_len, args_seq = check_parts_len(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                             jmd_c_len=jmd_c_len, jmd_n_len=jmd_n_len)
        ax, seq_size = update_seq_size_(ax=ax, **args_seq, max_x_dist=max_x_dist, **args_part_color, **args_seq_color)
        update_tmd_jmd_labels(fig=fig, seq_size=seq_size,
                              fontsize_tmd_jmd=fontsize_tmd_jmd,
                              weight_tmd_jmd=weight_tmd_jmd)
        if self._verbose:
            ut.print_out(f"Optimized sequence character fontsize is: {seq_size}")
        return ax
