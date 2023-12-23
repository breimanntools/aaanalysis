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
from ._backend.dpulearn.dpul_plot import (plot_eval)
# TODO visualize loadings as feature plot (long-term)


# I Helper Functions

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
            An array of Axes objects, each representing a subplot within the figure. .

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
    def components(df_pu=None, x="PC1", y="PC2", show_mean_x=False, show_mean_y=False):
        """Plot PC map for PC analysis """
        # TODO finish program, refactor, check, test, examples


