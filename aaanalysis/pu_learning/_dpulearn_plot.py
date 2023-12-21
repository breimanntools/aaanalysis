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
    """"""
    def __int__(self):
        pass

    @staticmethod
    def eval(df_eval: ut.ArrayLike2D,
             names: Optional[List[str]] = None,
             dict_xlims: Optional[Union[None, dict]] = None,
             figsize: Tuple[Union[int, float], Union[int, float]] = (7, 6),
             colors: List[str] = None,
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot eval output of dPULearn (n per component)

        Parameters
        ----------

        Returns
        -------

        """
        # Check input
        """
        ut.check_array_like(name="data", val=data_eval)
        ut.check_list_like(name="names", val=names, accept_none=True)
        df_eval = check_match_data_names(data=data_eval, names=names)
        check_dict_xlims(dict_xlims=dict_xlims)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True)
        """
        # Plotting
        colors = ut.plot_get_clist_(n_colors=4) if colors is None else colors
        fig, axes = plot_eval(df_eval=df_eval, colors=colors)
        return fig, axes

    @staticmethod
    def components(df_pu=None, x="PC1", y="PC2", show_mean_x=False, show_mean_y=False):
        """Plot PC map for PC analysis """

