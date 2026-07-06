"""
This is a script for the frontend of the dPULearnPlot class for plotting results for the identification of
reliable negatives from unlabeled data using dPULearn.
"""
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Dict, Union, List, Tuple, Type


import aaanalysis.utils as ut
from ._backend.dpulearn.dpul_plot import (plot_eval, plot_pca)


# I Helper Functions
def _adjust_plot_args(names=None, colors=None, args_scatter=None, default_names=None, default_colors=None):
    """Adjust names, colors, and args_scatter arguments"""
    # Set default names if not provided
    if names is None:
        names = default_names
    # Extract and adjust colors from args_scatter
    if args_scatter is not None and ("color" in args_scatter or "c" in args_scatter):
        if "color" in args_scatter:
            extracted_colors = args_scatter.pop('color', None)
        else:
            extracted_colors = args_scatter.pop("c", None)
        if isinstance(extracted_colors, str):
            colors = [extracted_colors] * len(names)
        elif extracted_colors is not None:
            colors = extracted_colors
    # Set default colors if not provided
    if colors is None:
        colors = default_colors
    # Validate the match between names and colors
    check_match_names_colors(names=names, colors=colors)
    # Adjust args_scatter with default and additional arguments
    _args_scatter = dict(linewidth=0.5, edgecolor="white")
    if args_scatter is not None:
        # Avoid aliases in args_scatter
        for arg in ["edgecolors", "linewidths"]:
            if arg in args_scatter:
                _args_scatter.pop(arg.replace("s", ""), None)
        _args_scatter.update(args_scatter)
    return names, colors, _args_scatter


# Check functions
def check_match_df_pu_labels(df_pu=None, labels=None) -> None:
    """Check length match df_pu and labels"""
    n_samples = len(df_pu)
    if n_samples != len(labels):
        raise ValueError(f"Number of samples from 'df_pu' (n={n_samples}) does not match with 'labels' (n={len(labels)})")


def check_match_names_colors(names=None, colors=None) -> None:
    """Check if length matches of names and colors"""
    if len(names) != len(colors):
        raise ValueError(f"Length of 'names' (n={len(names)}) and 'colors' (n={len(colors)}) does not match.")


def check_match_add_groups(list_df_add=None, names_add=None, colors_add=None, pc_x=None, pc_y=None) -> None:
    """Validate the projected extra groups and that each carries the two selected PC columns."""
    n_add = len(list_df_add)
    if len(names_add) != n_add:
        raise ValueError(f"Length of 'names_add' (n={len(names_add)}) does not match the number of "
                         f"projected groups in 'df_pu_add' (n={n_add}).")
    if len(colors_add) != n_add:
        raise ValueError(f"Length of 'colors_add' (n={len(colors_add)}) does not match the number of "
                         f"projected groups in 'df_pu_add' (n={n_add}).")
    for i, df_add in enumerate(list_df_add):
        for pc in [pc_x, pc_y]:
            if not any(f"PC{pc}" in str(c) and "abs" not in str(c) for c in df_add.columns):
                raise ValueError(f"'df_pu_add[{i}]' has no 'PC{pc}' value column; project the extra "
                                 f"group with 'dPULearn.project' so its columns match 'df_pu'.")


def _adjust_add_groups(df_pu_add=None, names_add=None, colors_add=None):
    """Normalize the optional projected extra groups to aligned lists (df, name, color)."""
    if df_pu_add is None:
        return None, None, None
    list_df_add = df_pu_add if isinstance(df_pu_add, list) else [df_pu_add]
    n_add = len(list_df_add)
    if names_add is None:
        names_add = ["Projected"] if n_add == 1 else [f"Projected {i + 1}" for i in range(n_add)]
    elif isinstance(names_add, str):
        names_add = [names_add]
    if colors_add is None:
        colors_add = [ut.COLOR_NEG] if n_add == 1 else ut.plot_get_clist_(n_colors=n_add)
    elif isinstance(colors_add, str):
        colors_add = [colors_add]
    return list_df_add, list(names_add), list(colors_add)


# II Main Functions
class dPULearnPlot:
    """
    Plotting class for :class:`dPULearn` (deterministic Positive-Unlabeled Learning) results [Breimann25]_.

    Visualizes ``dPULearn`` results in two ways: the reliable negatives it identifies, shown in a
    compressed Principal Component Analysis (PCA) feature space (:meth:`pca`), and an evaluation
    comparing multiple sets of identified negatives (:meth:`eval`).

    Every plotting method returns a ``(fig, ax)`` pair (a thin tuple subclass): unpack as
    ``fig, ax = ...``. For backward compatibility, the returned object also forwards attribute
    access to ``ax``, so legacy ``ax = ...; ax.set_title(...)`` keeps working.

    .. versionadded:: 0.1.2

    """
    def __init__(self):
        """
        See Also
        --------
        * :class:`dPULearn`: the logic class whose identified negatives this visualizes.
        """

    @staticmethod
    def eval(df_eval: pd.DataFrame,
             figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4),
             dict_xlims: Optional[dict] = None,
             legend: bool = True,
             legend_y: float = -0.175,
             colors: Optional[List[str]] = None,
             ) -> Tuple[Figure, Axes]:
        """
        Plot evaluation output of dPULearn comparing multiple sets of identified negatives.

        Evaluation measures can be grouped into 'Homogeneity' measures ('avg STD' and 'avg IQR') assessing
        the similarity within the sets of identified negatives, and 'Dissimilarity' measures ('avg AUC', 'avg KLD')
        assessing the dissimilarity between the identified negatives and the other reference groups including
        positive samples ('Pos'), unlabeled samples ('Unl'), and ground-truth negative samples ('Neg') if given.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_datasets, n_metrics)
            DataFrame with evaluation measures for sets of identified negatives. Each `row` corresponds to a specific
            dataset including identified negatives. required 'columns' are:

            - 'name': Name of datasets containing identified negatives (typically named by identification approach).
            - 'avg_STD': Average standard deviation (STD) assessing homogeneity of identified negatives.
            - 'avg_IQR': Average interquartile range (IQR) assessing homogeneity of identified negatives.
            - 'avg_abs_AUC_DATASET': Average absolute area under the curve (AUC), which assesses the similarity between the
              set of identified negatives and other datasets. 'DATASET' must be 'pos' (positive samples) and 'unl'
              (unlabeled samples), as well as, optionally, 'neg' (ground-truth negative samples).

            Optional columns include:

            - 'avg_KLD_DATASET': The average Kullback-Leibler Divergence (KLD), which measures the distribution alignment
              between the set of identified negatives and other datasets ('pos', 'unl', or 'neg').

        figsize : tuple, default=(6, 4)
            Figure dimensions (width, height) in inches.
        dict_xlims : dict, optional
            A dictionary containing x-axis limits for subplots. Keys should be the subplot axis number ({0, 1, 2, 3})
            and values should be tuple specifying (``xmin``, ``xmax``). If ``None``, x-axis limits are auto-scaled.
        legend : bool, default=True
            If ``True``, legend is set under dissimilarity measures.
        legend_y : float, default=-0.175
            Legend position regarding the plot y-axis applied if ``legend=True``.
        colors : list of str of length 4, optional
            List of colors for identified negatives and the following reference datasets:
            positive samples ('Pos'), unlabeled samples ('Unl'), and ground-truth negative samples ('Neg').

        Returns
        -------
        fig : Figure
            Figure object for evaluation plot
        ax : array of Axes
            Array of Axes objects, each representing a subplot within the figure. .

        Notes
        -----
        * Ground-truth negatives are only shown if provided by ``df_eval``.

        See Also
        --------
        * :meth:`dPULearn.eval`: the respective computation method.
        * :func:`comp_auc_adjusted` and :func:`comp_kld`.

        Examples
        --------
        .. include:: examples/dpul_plot_eval.rst
        """
        # Check input
        cols_required = [ut.COL_NAME, ut.COL_AVG_STD, ut.COL_AVG_IQR, ut.COL_AVG_ABS_AUC_POS, ut.COL_AVG_ABS_AUC_UNL]
        ut.check_df(name="df_eval", df=df_eval, cols_required=cols_required, accept_none=False, accept_nan=False)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_dict_xlims(dict_xlims=dict_xlims, n_ax=4)
        ut.check_bool(name="legend", val=legend)
        ut.check_number_val(name="legend_y", val=legend_y)
        ut.check_list_colors(name="colors", val=colors, accept_none=True, min_n=4)
        # Plotting
        if colors is None:
            colors = [ut.COLOR_REL_NEG, ut.COLOR_POS, ut.COLOR_UNL, ut.COLOR_NEG]
        fig, axes = plot_eval(df_eval=df_eval, colors=colors,
                              figsize=figsize, dict_xlims=dict_xlims,
                              legend=legend, legend_y=legend_y)
        return ut.FigAxResult(fig, axes)

    @staticmethod
    def pca(df_pu: pd.DataFrame,
            labels=None,
            figsize: Tuple[Union[int, float], Union[int, float]] = (5, 5),
            pc_x: int = 1,
            pc_y: int = 2,
            show_pos_mean_x: bool = True,
            show_pos_mean_y: bool = True,
            colors: Optional[List[str]] = None,
            names: Optional[List[str]] = None,
            df_pu_add: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
            names_add: Optional[Union[str, List[str]]] = None,
            colors_add: Optional[Union[str, List[str]]] = None,
            legend: bool = True,
            legend_y: float = -0.15,
            kwargs_scatterplot: Optional[dict] = None,
            ) -> Tuple[Figure, Axes]:
        """
        Principal component analysis (PCA) plot for set of identified negatives.

        This method visualizes the differences between the set of identified negatives (labeled by 0) and the
        positive (1) and the unlabeled (2) sample groups. The selected principal components (PCs) represent
        a lower-dimensional feature space. Optionally, the average PC value for the positive samples can be shown,
        which was used for ``PCA-based identification`` of negatives.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        df_pu : pd.DataFrame, shape (n_samples, pca_features)
            A DataFrame with the PCA-transformed features obtained from ``dPULearn.df_pu_``.
        figsize : tuple, default=(5, 5)
            Figure dimensions (width, height) in inches.
        labels : array-like, shape (n_samples,)
            Dataset labels of samples in ``df_pu``. Labels should contain 0 (identified negative) and 1 (positive).
            Unlabeled samples (2) can also be provided.
        pc_x : int, default=1
            Index of the principal component (PC) to show at the x-axis.
        pc_y : int, default=2
            Index of the principal component (PC) to show at the y-axis.
        show_pos_mean_x : bool, default=True
            If ``True``, the mean of the x-axis PC values across the positive sample group is shown on the plot.
        show_pos_mean_y : bool, default=True
            If ``True``, the mean of the y-axis PC values across the positive sample group is shown on the plot.
        colors : list of str, optional
            List of colors for identified negatives (0), positive samples (1), and unlabeled samples (2).
        names : list of str, optional
            List of dataset names for identified negatives, positive samples, and unlabeled samples.
        df_pu_add : pd.DataFrame or list of pd.DataFrame, optional
            One or more groups of held-out samples projected into the fitted PC space (the output of
            :meth:`dPULearn.project`), each overlaid as an additional group on top of the ``df_pu``
            samples. Each DataFrame must carry the selected ``PCi`` value columns (same names as
            ``df_pu``). If ``None`` (default), the plot is byte-identical to the three-group figure.
        names_add : str or list of str, optional
            Name(s) for the projected extra group(s) in ``df_pu_add`` (shown in the legend). Defaults
            to ``"Projected"`` (single group) or ``"Projected i"`` (multiple).
        colors_add : str or list of str, optional
            Color(s) for the projected extra group(s) in ``df_pu_add``. Defaults to the ground-truth
            negative color for a single group, or a distinct color list for multiple groups.
        legend : bool, default=True
            If ``True``, legend is set under dissimilarity measures.
        legend_y : float, default=-0.15
            Legend position regarding the plot y-axis applied if ``legend=True``.
        kwargs_scatterplot : dict, optional
            Dictionary with keyword arguments for adjusting scatter plot (:func:`matplotlib.pyplot.scatter`).

        Returns
        -------
        fig : Figure
            Figure object containing the plot.
        ax : Axes
            PCA plot axes object.

        Notes
        -----
        * Returned as a ``(fig, ax)`` pair (see :class:`dPULearnPlot` for the shared return contract).

        See Also
        --------
        * :class:`dPULearn` for details on the data structure of ``df_pu``.
        * :func:`matplotlib.pyplot.scatter` for scatter plot arguments.

        Examples
        --------
        .. include:: examples/dpul_plot_pca.rst
        """
        # Check input
        ut.check_df(name="df_pu", df=df_pu, cols_required=[ut.COL_SELECTION_VIA], accept_none=True, accept_nan=True)
        n_pc = len([x for x in list(df_pu) if "PC" in x and not "abs_dif" in x])
        if n_pc < 2:
            raise ValueError(f"'df_pu' should contain at least two PCs (n={n_pc}).")
        labels = ut.check_labels(labels=labels) # Pre-check if proper format
        vals_required = [0, 1] if 2 not in set(labels) else [0, 1, 2]
        labels = ut.check_labels(labels=labels, vals_required=vals_required, allow_other_vals=False)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="pc_x", val=pc_x, min_val=1, max_val=n_pc, just_int=True)
        ut.check_number_range(name="pc_y", val=pc_y, min_val=1, max_val=n_pc, just_int=True)
        ut.check_bool(name="show_pos_mean_x", val=show_pos_mean_x)
        ut.check_bool(name="show_pos_mean_y", val=show_pos_mean_y)
        ut.check_bool(name="legend", val=legend)
        ut.check_number_val(name="legend_y", val=legend_y)
        ut.check_list_colors(name="colors", val=colors, accept_none=True, min_n=2, max_n=3)
        names = ut.check_list_like(name="names", val=names, accept_none=True, check_all_str_or_convertible=True)
        ut.check_dict(name="kwargs_scatterplot", val=kwargs_scatterplot, accept_none=True)
        check_match_df_pu_labels(df_pu=df_pu, labels=labels)
        # Normalize and validate the optional projected extra groups (default None -> unchanged plot)
        list_df_add, names_add, colors_add = _adjust_add_groups(df_pu_add=df_pu_add, names_add=names_add,
                                                                colors_add=colors_add)
        if list_df_add is not None:
            for i, df_add in enumerate(list_df_add):
                ut.check_df(name=f"df_pu_add[{i}]", df=df_add, accept_none=False, accept_nan=True)
            ut.check_list_colors(name="colors_add", val=colors_add)
            check_match_add_groups(list_df_add=list_df_add, names_add=names_add, colors_add=colors_add,
                                   pc_x=pc_x, pc_y=pc_y)
        # Set defaults colors and names
        default_names =  ["Identified negatives", "Positives", "Unlabeled"]
        default_colors = [ut.COLOR_REL_NEG, ut.COLOR_POS, ut.COLOR_UNL]
        names, colors, _args_scatter = _adjust_plot_args(names=names, colors=colors,
                                                         args_scatter=kwargs_scatterplot,
                                                         default_names=default_names,
                                                         default_colors=default_colors)
        # Plotting
        try:
            ax = plot_pca(df_pu=df_pu, labels=labels,
                          figsize=figsize, pc_x=pc_x, pc_y=pc_y,
                          show_pos_mean_x=show_pos_mean_x, show_pos_mean_y=show_pos_mean_y,
                          names=names, colors=colors,
                          list_df_add=list_df_add, names_add=names_add, colors_add=colors_add,
                          legend=legend, legend_y=legend_y, args_scatter=_args_scatter)
        except Exception as e:
            str_error = f"Following error occurred due to plt.scatter() function: {e}"
            raise ValueError(str_error)
        return ut.FigAxResult(ax.get_figure(), ax)

