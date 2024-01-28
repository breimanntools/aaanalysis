"""
This is a script for the frontend of the CPPPlot class.
"""
from typing import Optional, Dict, Union, List, Tuple, Type, Literal
import pandas as pd
import matplotlib.pyplot as plt

import aaanalysis.utils as ut

from ._backend.check_feature import (check_split_kws,
                                     check_parts_len,
                                     check_match_features_seq_parts,
                                     check_df_parts,
                                     check_match_df_parts_features,
                                     check_match_df_parts_list_parts,
                                     check_df_scales,
                                     check_match_df_scales_features,
                                     check_df_cat,
                                     check_match_df_cat_features,
                                     check_match_df_parts_df_scales,
                                     check_match_df_seq_jmd_len,
                                     check_match_df_scales_df_cat)
from ._backend.check_cpp_plot import (check_value_type,
                                      check_y_categorical,
                                      check_args_xtick,
                                      check_args_ytick,
                                      check_args_size,
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

# TODO simplify checks & interface (end-to-end check with tests & docu)


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
    ut.check_list_like(name=f"{ut.COL_N_FEAT}: list_n_feat", val=list_n_feat[1], accept_str=False)
    ut.check_list_like(name=ut.COL_RANGE_ABS_AUC, val=range_abs_auc, accept_str=False)
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
def check_col_dif(col_dif=None, shap_plot=False):
    """Check if col_dif is string and set default"""
    ut.check_str(name="col_dif", val=col_dif, accept_none=False)
    if col_dif is None:
        col_dif = ut.COL_MEAN_DIF
    if not shap_plot:
        if col_dif != ut.COL_MEAN_DIF:
            raise ValueError(f"If 'shap_plot=False', 'col_dif' ('{col_dif}') must be '{ut.COL_MEAN_DIF}'")
    else:
        if ut.COL_MEAN_DIF not in col_dif:
            raise ValueError(f"If 'shap_plot=True', 'col_dif' ('{col_dif}') must follow '{ut.COL_MEAN_DIF}_'name''")


def check_col_imp(col_imp=None, shap_plot=False):
    """Check if col_imp is string and set default"""
    ut.check_str(name="col_imp", val=col_imp, accept_none=True)
    if col_imp is None:
        col_imp = ut.COL_FEAT_IMPACT if shap_plot else ut.COL_FEAT_IMPORT
    if not shap_plot:
        if col_imp != ut.COL_FEAT_IMPORT:
            raise ValueError(f"If 'shap_plot=False', 'col_imp' ('{col_imp}') must be '{ut.COL_FEAT_IMPORT}'")
    else:
        if ut.COL_FEAT_IMPACT not in col_imp:
            raise ValueError(f"If 'shap_plot=True', 'col_imp' ('{col_imp}') must follow '{ut.COL_FEAT_IMPACT}_'name''")
    return col_imp


def check_match_shap_plot_add_legend_cat(shap_plot=False, add_legend_cat=False):
    """Check if not both are True"""
    if shap_plot and add_legend_cat:
        raise ValueError(f"'shap_plot' ({shap_plot}) and 'add_legend_cat' ({add_legend_cat}) can not be both True.")


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
    Plotting class for ``CPP`` (Comparative Physicochemical Profiling).

    This plotting class visualizes the results of the result of the :class:`aaanalysis.CPP` class. As introduced in
    [Breimann24c]_, the CPP results can be visualized at global or individual sample level as ranking plot, profile,
    or map (heatmap, feature map).

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
        * :class:`ShapExplainer`: the class combining CPP with the `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_
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
             dict_xlims: Optional[Union[None, dict]] = None,
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

        Introduced in [Breimann24a]_, a CPP feature is defined as a ``Part-Split-Scale`` combination. For a sample,
        a feature value is computed in three steps:

            1. **Part Selection**: Identify a specific sequence part.
            2. **Part-Splitting**: Divide the selected part into segments, creating a 'Part-Split' combination.
            3. **Scale Value Assignment**: For each amino acid in the 'Part-Split' segment,
               assign its corresponding scale value and calculate the average, which is termed the feature value.

        Parameters
        ----------
        feature : str
            Name of the feature for which test and reference set distributions and difference should be plotted.
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence information
            in a distinct ``Position-based``, ``Part-based``, ``Sequence-based``, or ``Sequence-TMD-based`` format.
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
        * :meth:`SequenceFeature.get_df_parts` for details on format of ``df_seq``.
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
        args = dict(min_val=0, exclusive_limits=True, accept_none=True, just_int=False)
        ut.check_number_range(name="fontsize_mean_dif", val=fontsize_mean_dif, **args)
        ut.check_number_range(name="fontsize_name_test", val=fontsize_name_test, **args)
        ut.check_number_range(name="fontsize_name_ref", val=fontsize_name_ref, **args)
        ut.check_number_range(name="fontsize_names_to_show", val=fontsize_names_to_show, **args)
        ut.check_number_range(name="alpha_hist", val=alpha_hist, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="alpha_dif", val=alpha_dif, accept_none=False, min_val=0, max_val=1, just_int=False)
        check_match_df_seq_names_to_show(df_seq=df_seq, names_to_show=names_to_show)
        df_seq = check_match_df_seq_jmd_len(df_seq=df_seq, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len)
        # Plot feature
        ax = plot_feature(ax=ax, figsize=figsize,
                          feature=feature, df_scales=self._df_scales, accept_gaps=self._accept_gaps,
                          df_seq=df_seq, labels=labels, label_test=label_test, label_ref=label_ref,
                          jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                          names_to_show=names_to_show, show_seq=show_seq,
                          name_test=name_test, name_ref=name_ref,
                          color_test=color_test, color_ref=color_ref,
                          fontsize_mean_dif=fontsize_mean_dif,
                          fontsize_name_test=fontsize_name_test,
                          fontsize_name_ref=fontsize_name_ref,
                          fontsize_names_to_show=fontsize_names_to_show,
                          histplot=histplot, alpha_hist=alpha_hist, alpha_dif=alpha_dif)
        return ax

    # Plotting methods for multiple features (group and sample level)
    def ranking(self,
                df_feat: pd.DataFrame = None,
                n_top: int = 15,
                shap_plot: bool = False,
                col_dif: str = "mean_dif",
                col_imp: str = "feat_importance",
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
                tmd_len: int = 20,
                tmd_jmd_space: int = 2,
                tmd_color: str = "mediumspringgreen",
                jmd_color: str = "blue",
                tmd_jmd_alpha: Union[int, float] = 0.075,
                name_test: str = "TEST",
                name_ref: str = "REF",
                fontsize_titles: Union[int, float, None] = 10,
                fontsize_labels: Union[int, float, None] = 11,
                fontsize_annotations: Union[int, float, None] = 11,
                xlim_dif: Tuple[Union[int, float], Union[int, float]] = (-17.5, 17.5),
                xlim_rank: Tuple[Union[int, float], Union[int, float]] = (0, 5),
                x_rank_info: Optional[Union[int, float]] = None,
                ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot CPP/-SHAP feature ranking based on feature importance or sample-specif feature impact.

        Introduced in [Breimann24c]_, this method visualizes the most important features for discriminating between
        the test and the reference dataset groups. At sample level, the feature impact derived from SHAP values
        of a specific sample can be used for ranking if ``shap_plot=True`` and 'feature_impact' column in ``df_feat``.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Must also include feature importance (``feat_importance``) or impact (``feat_impact_'name'``) columns.
        n_top : int, default=15
            The number of top features to display. Should be 1 < ``n_top`` <= ``n_features``.
        shap_plot : bool, default=False
            If ``True``, the positive (red) and negative (blue) feature impact is shown in the ranking subplot.
        col_dif : str, default='mean_dif'
            Column name in ``df_feat`` for differences in feature values. Two scenarios are available:

            - **CPP Analysis**: By default, selects the difference between the test group and the reference group
              from the ``mean_dif`` column.
            - **CPP-SHAP Analysis**: When ``shap_plot=True``, enables the selection of sample- or group-specific
              differences against the reference group from a ``mean_dif_'name'`` column.

        col_imp : str, default='feat_importance'
            Column name in ``df_feat`` for feature importance/impact values. Two options are supported:

            - **CPP Analysis**: By default, uses the ``feat_importance`` column to show feature importance.
            - **CPP-SHAP Analysis**:  When ``shap_plot=True``, allows selection of specific feature impacts from a
             ``feat_impact_'name'`` column for samples or a group.

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
        fontsize_titles : int or float, default=10
            Font size for the titles.
        fontsize_labels : int or float , default=11
            Font size for labels.
        fontsize_annotations : int or float, default=11
            Font size for annotations.
        xlim_dif : tuple, default=(-17.5, 17.5)
            x-axis limits for the mean difference subplot.
        xlim_rank : tuple, default=(0, 5)
            x-axis limits for the ranking subplot.
        x_rank_info : int, optional
            x-axis position in the ranking subplot for the total feature importance
            when ``shap_plot=False`` or feature impact and SHAP legend otherwise.

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
        ut.check_figsize(figsize=figsize, accept_none=True)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len)
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
        ut.check_number_range(name="x_sum", val=x_rank_info, min_val=0, accept_none=True, just_int=False)
        # DEV: No match check for features and tmd (check_match_features_seq_parts) necessary
        # Plot ranking
        fig, axes = plot_ranking(df_feat=df_feat.copy(), n_top=n_top,
                                 col_dif=col_dif,
                                 col_imp=col_imp,
                                 shap_plot=shap_plot,
                                 figsize=figsize,
                                 tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                 tmd_color=tmd_color, jmd_color=jmd_color,
                                 tmd_jmd_alpha=tmd_jmd_alpha,
                                 name_test=name_test, name_ref=name_ref,
                                 fontsize_titles=fontsize_titles,
                                 fontsize_labels=fontsize_labels,
                                 fontsize_annotations=fontsize_annotations,
                                 tmd_jmd_space=tmd_jmd_space,
                                 xlim_dif=xlim_dif, xlim_rank=xlim_rank,
                                 x_rank_info=x_rank_info)
        return fig, axes


    def profile(self,
                df_feat: pd.DataFrame = None,
                shap_plot: bool = False,
                col_imp: Union[str, None] = "feat_importance",
                normalize: bool = True,
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
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
                add_xticks_pos: bool = False,
                highlight_tmd_area: bool = True,
                highlight_alpha: float = 0.15,
                add_legend_cat: bool = False,
                dict_color: Optional[dict] = None,
                legend_kws: Optional[dict] = None,
                bar_width: Union[int, float] = 0.75,
                edge_color: Optional[str] = None,
                grid_axis: Optional[Literal['x', 'y', 'both']] = None,
                ylim: Tuple[float, float] = None,
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
            If ``True``, the positive (red) and negative (blue) feature impact is shown by +/- bars.
        col_imp : str or None, default='feat_importance'
            Column name in ``df_feat`` for feature importance/impact values to be shown per residue position.
            Two options are supported:

            - **CPP Analysis**: By default, uses the ``feat_importance`` column to show feature importance.
            - **CPP-SHAP Analysis**:  When ``shap_plot=True``, allows selection of specific feature impacts from a
             ``feat_impact_'name'`` column for samples or a group.

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
            TMD sequence.
        jmd_n_seq : str, optional
            JMD N-terminal sequence.
        jmd_c_seq : str, optional
            JMD C-terminal sequence.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        tmd_seq_color : str, default='black'
            Color for TMD sequence.
        jmd_seq_color : str, default='white'
            Color for JMD sequence.
        seq_size : int or float, optional
            Font size for sequence annotations.
        fontsize_tmd_jmd : int or float, optional
            Font size for the part labels (JMD-N, TMD, JMD-C).
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
            Keyword arguments for the legend.
        bar_width : int or float, default=0.75
            Width of the bars.
        edge_color : str, optional
            Color of the bar edges.
        grid_axis : {'x', 'y', 'both', None}, default=None
            Axis on which the grid is drawn if not ``None``.
        ylim : tuple, optional
            Y-axis limits. If ``None``, y-axis limits are set automatically.
        xtick_size : int or float, default=11.0
            Size for x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=5.0
            Length of the x-ticks (>0).
        ytick_size : int or float, optional
            Size for y-tick labels (>0).
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
        # Check specific input
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                             jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                             check_jmd_seq_len_consistent=True)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        args_size = check_args_size(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
        ut.check_bool(name="add_xticks_pos", val=add_xticks_pos)
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_number_range(name="tmd_area_alpha", val=highlight_alpha, min_val=0, max_val=1, just_int=False)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        dict_color = check_match_dict_color_df(dict_color=dict_color, df=df_feat)
        check_match_features_seq_parts(features=df_feat["feature"],
                                       tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                       tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        check_match_shap_plot_add_legend_cat(shap_plot=shap_plot, add_legend_cat=add_legend_cat)
        # Check general plot styling input
        ut.check_number_range(name="bar_width", val=bar_width, min_val=0, just_int=False)
        ut.check_color(name="edge_color", val=edge_color, accept_none=True)
        ut.check_grid_axis(grid_axis=grid_axis)
        ut.check_lim(name="ylim", val=ylim, accept_none=True)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_ytick = check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        # Plot profile
        fig, ax = plot_profile(df_feat=df_feat, df_cat=self._df_cat, shap_plot=shap_plot,
                               col_imp=col_imp, normalize=normalize,
                               figsize=figsize, ax=ax,
                               start=start,
                               **args_len, **args_seq, **args_size,
                               **args_part_color, **args_seq_color,
                               add_xticks_pos=add_xticks_pos,
                               highlight_tmd_area=highlight_tmd_area, highlight_alpha=highlight_alpha,
                               add_legend_cat=add_legend_cat, dict_color=dict_color, legend_kws=legend_kws,
                               bar_width=bar_width, edge_color=edge_color,
                               grid_axis=grid_axis, ylim=ylim, **args_xtick, **args_ytick)
        plt.tight_layout()
        if tmd_seq is not None and seq_size is None:
            ax, seq_size = update_seq_size_(ax=ax, **args_seq, **args_part_color, **args_seq_color)
            if self._verbose:
                ut.print_out(f"Optimized sequence character fontsize is: {seq_size}")
        return fig, ax

    def heatmap(self,
                df_feat=None,
                shap_plot: bool = False,
                y="subcategory",
                col_value="mean_dif",
                value_type="mean",
                normalize=False,
                figsize=(8, 8),
                vmin=None,
                vmax=None,
                ax: Optional[plt.Axes] = None,
                grid_on=True,
                cmap="RdBu_r",
                cmap_n_colors=None,
                cbar_kws=None,
                facecolor_dark=False,

                add_jmd_tmd=True,  # Remove
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
                add_xticks_pos=False,  # TODO check if change
                cbar_pct=True,

                add_legend_cat: bool = True,
                dict_color: Optional[dict] = None,
                legend_kws: Optional[dict] = None,
                xtick_size: Union[int, float] = 11.0,
                xtick_width: Union[int, float] = 2.0,
                xtick_length: Union[int, float] = 5.0,
                ytick_size: Optional[Union[int, float]] = None,
                ytick_width: Optional[Union[int, float]] = None,
                ytick_length: Union[int, float] = 5.0,
                ):
        """
        Plot CPP/-SHAP heatmap showing feature value mean difference/feature impact per scale subcategory (y-axis)
        and residue position (x-axis).

        This is a wrapper function for :func:`seaborn.heatmap`, designed to highlight differences between two sets
        of sequences at the positional level (e.g., amino acid level for protein sequences).

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_feature, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Can also include feature impact (``feat_impact``) column.
        y : {'category', 'subcategory', 'scale_name'}, default='subcategory'
            Column name in ``df_feat`` representing scale information (shown on the y-axis).
        col_value : {'abs_auc', 'mean_dif', 'std_test', 'feat_importance', 'feat_impact', ...}, default='mean_dif'
            Column name in ``df_feat`` containing numerical values to display.
        value_type : {'mean', 'sum', 'std'}, default='mean'
            Method to aggregate numerical values from ``col_value``.
        normalize : {True, False, 'positions', 'positions_only'}, default=False
            Specifies normalization for numerical values in ``col_value``:

            - False: Set value at all positions of a feature without further normalization.
            - True: Set value at all positions of a feature and normalize across all features.
            - 'positions': Value/number of positions set at each position of a feature and normalized across features.
              Recommended when aiming to emphasize features with fewer positions using 'col_value'='feat_impact' and 'value_type'='mean'.

        figsize : tuple, default=(10,7)
            Figure dimensions (width, height) in inches.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise, inferred from data and other keyword arguments.
        cmap : matplotlib colormap name or object, or list of colors, default='seismic'
            Name of color map assigning data values to color space. If 'SHAP', colors from
            `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ will be used (recommended for feature impact).
        cmap_n_colors : int, optional
            Number of discrete steps in diverging or sequential color map.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
        add_jmd_tmd : bool, default=True
            Whether to add colored bar under heatmap indicating sequence parts (JMD-N, TMD, JMD-C).
        tmd_len : int, default=20
            Length of TMD to be depicted (>0). Must match with all feature from ``df_feat``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
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
        seq_size : int or float, optional
            Font size of all sequence parts in points. If ``None``, optimized automatically.
        fontsize_tmd_jmd : int or float, optional
            Font size for the part labels (JMD-N, TMD, JMD-C).
        add_legend_cat : bool, default=True
            If ``True``, a legend is added for the scale categories.
        dict_color : dict, optional
            Color dictionary of scale categories classifying scales shown on y-axis. Default from
            :meth:`plot_get_cdict` with ``name='DICT_CAT'``.
        legend_kws : dict, optional
            Keyword arguments for the legend.
        xtick_size : int or float, default=11.0
            Size for x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=5.0
            Length of the x-ticks (>0).
        ytick_size : int or float, optional
            Size for y-tick labels (>0).
        ytick_width : int or float, optional
            Width of the y-ticks (>0).
        ytick_length : int or float, default=5.0
            Length of the y-ticks (>0).

        Returns
        -------
        ax : plt.Axes
            CPP heatmap plot axes object.

        Notes
        -----
        * ``cmap_n_colors`` is effective only if ``vmin`` and ``vmax`` align with the data.
        * ``tmd_seq_color`` and ``jmd_seq_color`` are applicable only when ``tmd_seq``, ``jmd_n_seq``,
           and ``jmd_c_seq`` are provided.

        See Also
        --------
        * :meth:`seaborn.heatmap` method for seaborn heatmap.

        Examples
        --------
        .. include:: examples/cpp_plot_heatmap.rst
        """
        # Group arguments
        args_size = check_args_size(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
        # TODO CHECK
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len,
                                                tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        # Checking input
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat, shap_plot=shap_plot)

        # Args checked by Matplotlib: title, cmap, cbar_kws, legend_kws
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        ut.check_number_range(name="ytick_size", val=ytick_size, accept_none=True, just_int=False, min_val=1)
        ut.check_number_range(name="cmap_n_colors", val=cmap_n_colors, min_val=1, accept_none=True, just_int=True)
        ut.check_bool(name="add_jmd_tmd", val=add_jmd_tmd)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_df(df=df_feat, name="df_feat", cols_requiered=col_value, cols_nan_check=col_value)
        check_y_categorical(df=df_feat, y=y)
        check_value_type(value_type=value_type, count_in=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        ut.check_figsize(figsize=figsize)
        dict_color = check_match_dict_color_df(dict_color=dict_color, df=df_feat)
        # Get df positions
        ax = plot_heatmap(df_feat=df_feat, df_cat=self._df_cat, col_cat=y, col_value=col_value, value_type=value_type,
                          normalize=normalize, figsize=figsize,
                          dict_color=dict_color, vmin=vmin, vmax=vmax, grid_on=grid_on,
                          cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws,
                          facecolor_dark=facecolor_dark, add_jmd_tmd=add_jmd_tmd,
                          start=start, **args_len, **args_seq,
                          tmd_color=tmd_color, jmd_color=jmd_color, tmd_seq_color=tmd_seq_color,
                          jmd_seq_color=jmd_seq_color,
                          seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd,
                          add_xticks_pos=add_xticks_pos, xtick_size=xtick_size, xtick_width=xtick_width,
                          xtick_length=xtick_length, ytick_size=ytick_size,
                          add_legend_cat=add_legend_cat, legend_kws=legend_kws, cbar_pct=cbar_pct,
                          linecolor=linecolor)
        plt.tight_layout()
        return ax

    # Plotting method for only group level
    def feature_map(self,
                    df_feat=None,
                    y="subcategory",
                    col_value="mean_dif",
                    value_type="mean",
                    normalize=False,
                    figsize=(8, 8),
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
                    cbar_pct=True,
                    add_legend_cat: bool = True,
                    dict_color: Optional[dict] = None,
                    legend_kws: Optional[dict] = None,
                    xtick_size: Union[int, float] = 11.0,
                    xtick_width: Union[int, float] = 2.0,
                    xtick_length: Union[int, float] = 5.0,
                    ytick_size: Optional[Union[int, float]] = None,
                    ytick_width: Optional[Union[int, float]] = None,
                    ytick_length: Union[int, float] = 5.0,

                    ):
        """
        Plot CPP feature map showing feature value mean difference and feature importance per scale subcategory
        (y-axis) and residue position (x-axis).

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_feature, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
            Must also include feature imporatnce (``feat_importance``) column.
        y : {'category', 'subcategory', 'scale_name'}, default='subcategory'
            Column name in ``df_feat`` representing scale information (shown on the y-axis).
        col_value : {'abs_auc', 'mean_dif', 'std_test', 'feat_importance', 'feat_impact', ...}, default='mean_dif'
            Column name in ``df_feat`` containing numerical values to display.
        value_type : {'mean', 'sum', 'std'}, default='mean'
            Method to aggregate numerical values from ``col_value``.
        normalize : {True, False, 'positions', 'positions_only'}, default=False
            Specifies normalization for numerical values in ``col_value``:

            - False: Set value at all positions of a feature without further normalization.
            - True: Set value at all positions of a feature and normalize across all features.
            - 'positions': Value/number of positions set at each position of a feature and normalized across features.
              Recommended when aiming to emphasize features with fewer positions using 'col_value'='feat_impact' and 'value_type'='mean'.

        figsize : tuple, default=(10,7)
            Figure dimensions (width, height) in inches.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise, inferred from data and other keyword arguments.
        cmap : matplotlib colormap name or object, or list of colors, default='seismic'
            Name of color map assigning data values to color space. If 'SHAP', colors from
            `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ will be used (recommended for feature impact).
        cmap_n_colors : int, optional
            Number of discrete steps in diverging or sequential color map.
            Color dictionary of scale categories classifying scales shown on y-axis. Default from :meth:`plot_get_cdict`
            with ``name='DICT_CAT'``.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
        tmd_len : int, default=20
            Length of TMD to be depicted (>0). Must match with all feature from ``df_feat``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_color : str, default='mediumspringgreen'
            Color of TMD bar.
        jmd_color : str, default='blue'
            Color of JMD-N and JMD-C bar.
        tmd_seq_color : str, default='black'
            Color of TMD sequence.
        jmd_seq_color : str, default='white'
            Color of JMD-N and JMD-C sequence.
        seq_size : int or float, optional
            Font size of all sequence parts in points. If ``None``, optimized automatically.
        fontsize_tmd_jmd : float, optional
            Font size of 'TMD', 'JMD-N' and 'JMD-C'  label in points. If ``None``, optimized automatically.

        add_legend_cat : bool, default=True
            If ``True``, a legend is added for the scale categories.
        dict_color : dict, optional
            Color dictionary of scale categories classifying scales shown on y-axis. Default from
            :meth:`plot_get_cdict` with ``name='DICT_CAT'``.
        legend_kws : dict, optional
            Keyword arguments for the legend.
        xtick_size : int or float, default=11.0
            Size for x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=5.0
            Length of the x-ticks (>0).
        ytick_size : int or float, optional
            Size for y-tick labels (>0).
        ytick_width : int or float, optional
            Width of the y-ticks (>0).
        ytick_length : int or float, default=5.0
            Length of the y-ticks (>0).

        Returns
        -------
        ax : plt.Axes
            CPP feature map axes object.

        Notes
        -----
        * If plotting is slow, set ``seq_size`` manually to avoid fontsize optimization.

        Examples
        --------
        .. include:: examples/cpp_plot_feature_map.rst
        """
        # TODO CHECK
        # TODO cbar & feature importance y location depend on n features.
        # TODO bar label not in
        # TODO TMD size dep on size of plot (change)
        # Group arguments
        args_size = check_args_size(seq_size=seq_size, fontsize_tmd_jmd=fontsize_tmd_jmd)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        # Checking input
        # Args checked by Matplotlib: title, cmap, cbar_kws, legend_kws
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self._df_cat)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=self._jmd_n_len, jmd_c_len=self._jmd_c_len)
        ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
        ut.check_number_range(name="ytick_size", val=ytick_size, accept_none=True, just_int=False, min_val=1)
        ut.check_number_range(name="cmap_n_colors", val=cmap_n_colors, min_val=1, accept_none=True, just_int=True)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_df(df=df_feat, name="df_feat", cols_requiered=col_value, cols_nan_check=col_value)
        check_y_categorical(df=df_feat, y=y)
        check_value_type(value_type=value_type, count_in=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        ut.check_figsize(figsize=figsize)
        dict_color = check_match_dict_color_df(dict_color=dict_color, df=df_feat)
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
        plt.subplots_adjust(right=0.95)
        return ax

    def update_seq_size(self,
                        ax: plt.Axes = None,
                        fig: Optional[plt.Figure] = None,
                        max_x_dist: float = 0.1,
                        fontsize_tmd_jmd: Union[int, float] = None,
                        weight_tmd_jmd: Literal['normal', 'bold'] = 'bold',
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
            Font size for the part labels (JMD-N, TMD, JMD-C).
        weight_tmd_jmd : {'normal', 'bold'}, default='bold'
            Font weight for the part labels (JMD-N, TMD, JMD-C).
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
        ut.check_font_weight(name="weight_tmd_jmd", font_weight=weight_tmd_jmd, accept_none=False)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
        check_match_ax_seq_len(ax=ax, jmd_c_len=self._jmd_c_len, jmd_n_len=self._jmd_n_len)
        # Adjust font size to prevent overlap
        jmd_n_seq, tmd_seq, jmd_c_seq = get_tmd_jmd_seq(ax=ax, jmd_c_len=self._jmd_c_len, jmd_n_len=self._jmd_n_len)
        args_len, args_seq = check_parts_len(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,
                                             jmd_c_len=self._jmd_c_len, jmd_n_len=self._jmd_n_len)
        ax, seq_size = update_seq_size_(ax=ax, **args_seq, max_x_dist=max_x_dist, **args_part_color, **args_seq_color)
        update_tmd_jmd_labels(fig=fig, seq_size=seq_size,
                              fontsize_tmd_jmd=fontsize_tmd_jmd,
                              weight_tmd_jmd=weight_tmd_jmd)
        if self._verbose:
            ut.print_out(f"Optimized sequence character fontsize is: {seq_size}")
        return ax