"""
This is a script for the frontend of the dPULearnPlot class for plotting results for the identification of
reliable negatives from unlabeled data using dPULearn.
"""
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, List, Tuple, Type


import aaanalysis.utils as ut
from ._backend.dpulearn.dpul_plot import (plot_eval, plot_pca)
# TODO visualize loadings as feature plot (long-term)


# I Helper Functions
def check_match_df_pu_labels(df_pu=None, labels=None):
    """Check length match df_pu and labels"""
    n_samples = len(df_pu)
    if n_samples != len(labels):
        raise ValueError(f"Number of samples from 'df_pu' (n={n_samples}) does not match with 'labels' (n={len(labels)})")


def check_match_names_colors(names=None, colors=None):
    """Check if length matches of names and colors"""
    if len(names) != len(colors):
        raise ValueError(f"Length of 'names' (n={len(names)}) and 'colors' (n={len(colors)}) does not match.")


# II Main Functions
class dPULearnPlot:
    """
    Plotting class for the deterministic Positive-Unlabeled (dPULearn) model introduced in [Breimann24c]_.
    """
    def __int__(self):
        pass

    @staticmethod
    def eval(df_eval: pd.DataFrame = None,
             figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4),
             legend: bool = True,
             legend_y: float = -0.175,
             colors: Optional[List[str]] = None,
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot evaluation output of dPULearn comparing multiple sets of identified negatives.

        Evaluation measures can be grouped into 'Homogeneity' measures ('avg STD' and 'avg IQR') assessing
        the similarity within the sets of identified negatives, and 'Dissimilarity' measures ('avg AUC', 'avg KLD')
        assessing the dissimilarity between the identified negatives and the other reference groups including
        positive samples ('Pos'), unlabeled samples ('Unl'), and ground-truth negative samples ('Neg') if given.

        Parameters
        ----------
        df_eval : DataFrame, shape (n_datasets, n_metrics)
            DataFrame with evaluation measures for sets of identified negatives. Each `row` corresponds to a specific
            dataset including identified negatives. Requiered 'columns' are:

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
            Width and height of the figure in inches.
        legend : bool, default=True
            If ``True``, legend is set under dissimilarity measures.
        legend_y : float, default=-0.175
            Legend position regarding the plot y-axis applied if ``legend=True``.
        colors : list of str, optional
            List of colors for identified negatives and the following reference datasets:
            positive samples ('Pos'), unlabeled samples ('Unl'), and ground-truth negative samples ('Neg').

        Returns
        -------
        fig : plt.Figure
            Figure object for evaluation plot
        axes : array of plt.Axes
            Array of Axes objects, each representing a subplot within the figure. .

        Notes
        -----
        * Ground-truth negatives are only shown if provided by ``df_eval``.

        See Also
        --------
        * :meth:`dPULearn.eval` for details on evaluation measures.
        * :func:`comp_auc_adjusted` and :func:`comp_kld`.

        Examples
        --------
        .. include:: examples/dpul_plot_eval.rst
        """
        # Check input
        cols_requiered = [ut.COL_NAME, ut.COL_AVG_STD, ut.COL_AVG_IQR, ut.COL_AVG_ABS_AUC_POS, ut.COL_AVG_ABS_AUC_UNL]
        ut.check_df(name="df_eval", df=df_eval, cols_requiered=cols_requiered, accept_none=False, accept_nan=False)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True)
        ut.check_bool(name="legend", val=legend)
        ut.check_number_val(name="legend_y", val=legend_y)
        ut.check_list_colors(name="colors", val=colors, accept_none=True, min_n=4)
        # Plotting
        if colors is None:
            colors = [ut.COLOR_REL_NEG, ut.COLOR_POS, ut.COLOR_UNL, ut.COLOR_NEG]
        fig, axes = plot_eval(df_eval=df_eval, colors=colors, figsize=figsize, legend=legend, legend_y=legend_y)
        return fig, axes

    @staticmethod
    def pca(df_pu: pd.DataFrame = None,
            labels=None,
            figsize: Tuple[Union[int, float], Union[int, float]] = (6, 6),
            pc_x : int = 1,
            pc_y : int = 2,
            show_pos_mean_x=True,
            show_pos_mean_y=True,
            colors: Optional[List[str]] = None,
            names: Optional[List[str]] = None,
            legend : bool = True,
            legend_y : float = -0.175,
            args_scatter : Optional[dict] = None,
            ) -> plt.Axes:
        """
        Principal component analysis (PCA) plot for set of identified negatives.

        This method visualizes the differences between the set of identified negatives (labeled by 0) and the
        positive (1) and the unlabeled (2) sample groups. The selected principal components (PCs) represent
        a lower-dimensional feature space. Optionally, the average PC value for the positive samples can be shown,
        which was used for ``PCA-based identification`` of negatives.

        Parameters
        ----------
        df_pu : pd.DataFrame, shape (n_samples, pca_features)
            A DataFrame with the PCA-transformed features obtained from ``dPULearn.df_pu_``.
        figsize : tuple, default=(6, 6)
            Width and height of the figure in inches.
        labels : array-like, shape (n_samples,)
            Dataset labels of samples in ``df_pu``. Labels should contain 0 (identified negative) and 1 (positive).
            Unlabeled samples (2) can also be provided.
        pc_x : str, default='PC1'
            The name of the principal component (PC) to show at the x-axis.
        pc_y : str, default='PC2'
            The name of the principal component (PC) to show at the y-axis.
        show_pos_mean_x : bool, default False
            If ``True``, the mean of the x-axis PC values across the positive sample group is shown on the plot.
        show_pos_mean_y : bool, default False
            If ``True``, the mean of the y-axis PC values across the positive sample group is shown on the plot.
        colors : list of str, optional
            List of colors for identified negatives (0), positive samples (1), and unlabeled samples (2).
        names : list of str, optional
            List of dataset names for identified negatives, positive samples, and unlabeled samples.
        legend : bool, default=True
            If ``True``, legend is set under dissimilarity measures.
        legend_y : float, default=-0.175
            Legend position regarding the plot y-axis applied if ``legend=True``.
        args_scatter : dict, optional
            Dictionary with kwargs for adjusting scatter plot.

        Returns
        -------
        ax : plt.Axes
            PCA plot axes object.

        See Also
        -------
        * :class:`dPULearn` for details on the data structure of ``df_pu``.
        * :func:`matplotlib.pyplot.scatter` for scatter plot arguments.
        """
        # Check input
        ut.check_df(name="df_pu", df=df_pu, cols_requiered=[ut.COL_SELECTION_VIA], accept_none=True, accept_nan=True)
        n_pc = len([x for x in list(df_pu) if "PC" in x and not "abs_dif" in x])
        if n_pc < 2:
            raise ValueError(f"'df_pu' should contain at least two PCs (n={n_pc}).")
        ut.check_labels(labels=labels) # Pre-check if proper format
        vals_requiered = [0, 1] if 2 not in set(labels) else [0, 1, 2]
        ut.check_labels(labels=labels, vals_requiered=vals_requiered, allow_other_vals=False)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True)
        ut.check_number_range(name="pc_x", val=pc_x, min_val=1, max_val=n_pc, just_int=True)
        ut.check_number_range(name="pc_y", val=pc_y, min_val=1, max_val=n_pc, just_int=True)
        ut.check_bool(name="show_pos_mean_x", val=show_pos_mean_x)
        ut.check_bool(name="show_pos_mean_y", val=show_pos_mean_y)
        ut.check_bool(name="legend", val=legend)
        ut.check_number_val(name="legend_y", val=legend_y)
        ut.check_dict(name="args_scatter", val=args_scatter, accept_none=True)
        ut.check_list_colors(name="colors", val=colors, accept_none=True, min_n=2, max_n=3)
        names = ut.check_list_like(name="names", val=names, accept_none=True, check_all_str_or_convertible=True)
        check_match_df_pu_labels(df_pu=df_pu, labels=labels)
        # Set defaults colors and names
        if names is None:
            names = ["Identified negatives", "Positives", "Unlabeled"]
        if colors is None:
            colors = [ut.COLOR_REL_NEG, ut.COLOR_POS, ut.COLOR_UNL]
        check_match_names_colors(names=names, colors=colors)
        # Adjust args_scatter
        _args_scatter = dict(linewidth=0.5, edgecolor="white")
        if args_scatter is not None:
            _args_scatter.update(args_scatter)
        # Plotting
        ax = plot_pca(df_pu=df_pu, labels=labels,
                      figsize=figsize, pc_x=pc_x, pc_y=pc_y,
                      show_pos_mean_x=show_pos_mean_x, show_pos_mean_y=show_pos_mean_y,
                      names=names, colors=colors,
                      legend=legend, legend_y=legend_y, args_scatter=_args_scatter)
        return ax

