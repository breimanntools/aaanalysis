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
            datas set including identified negatives. Requiered 'columns' are:

            - 'name': Name of datasets containing identified negatives (typically named by identification approach).
            - 'avg_std': Average standard deviation (STD) assessing homogeneity of identified negatives.
            - 'avg_iqr': Average interquartile range (IQR) assessing homogeneity of identified negatives.
            - 'avg_abs_auc_DATASET': Average absolute area under the curve (AUC) assessing the similarity between the
              set of identified negatives with other groups (positives, unlabeled, ground-truth negatives).
            - 'avg_kld_DATASET': Average Kullback-Leibler Divergence (KLD) assessing the distribution alignment
              between the set of identified negatives and the other groups.

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
        axes : plt.Axes
            Axes object(s) containing evaluation subplots.

        Notes
        -----
        * Ground-truth negatives are only shown if provided by ``df_eval``.

        See Also
        --------
        * :meth:`dPULearn.eval` for details on evaluation measures.
        * :func:`comp_auc_adjusted` and :func:`comp_kld`.
        """
        # TODO finish check, test, examples
        # Check input
        cols_requiered = ut.COLS_EVAL_DPULEARN_SIMILARITY + [x for x in ut.COLS_EVAL_DPULEARN_DISSIMILARITY
                                                             if "KLD" not in x]
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


