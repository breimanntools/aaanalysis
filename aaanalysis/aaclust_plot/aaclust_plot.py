"""
This is a script for the plotting class of AAclust.
"""
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Dict, Union, List, Tuple, Type
from sklearn.base import  TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

import aaanalysis as aa
import aaanalysis.utils as ut

# I Helper Functions
def _get_rank(data):
    """"""
    _df = data.copy()
    _df['BIC_rank'] = _df['BIC'].rank(ascending=False)
    _df['CH_rank'] = _df['CH'].rank(ascending=False)
    _df['SC_rank'] = _df['SC'].rank(ascending=False)
    return _df[['BIC_rank', 'CH_rank', 'SC_rank']].mean(axis=1).round(2)

# TODO add check functions finish other methods, testing, compression

# II Main Functions
class AAclustPlot:
    """Plot results of AAclust analysis.

    Dimensionality reduction is performed for visualization using decomposition models such as
    Principal Component Analysis (PCA).

    Parameters
    ----------
    model_class
        A decomposition model class with ``n_components`` parameter.
    model_kwargs
        Keyword arguments to pass to the selected decomposition model.

    See Also
    --------
    * Scikit-learn `decomposition model classes <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition>`_.

    """
    def __init__(self,
                 model_class: Type[TransformerMixin] = PCA,
                 model_kwargs: Optional[Dict] = None):
        # Model parameters
        model_class = ut.check_mode_class(model_class=model_class)
        model_kwargs = ut.check_model_kwargs(model_class=model_class, model_kwargs=model_kwargs,
                                             param_to_check="n_components")
        self.model_class = model_class
        self.model_kwargs = model_kwargs

    @staticmethod
    def eval(data : ut.ArrayLike2D,
             names : Optional[List[str]] = None,
             dict_xlims : Optional[Union[None, dict]] = None,
             figsize : Optional[Tuple[int, int]] = (7, 6)):
        """Plot eval output of n_clusters, BIC, CH, SC"""
        columns = ["n_clusters", "BIC", "CH", "SC"]
        colors = aa.plot_get_clist(n_colors=4)

        # Check input
        data = ut.check_array_like(name="data", val=data)
        n_samples, n_features = data.shape
        if n_features != 4:
            raise ValueError(f"'data' should contain the following four columns: {columns}")
        if names is None:
            names = [f"Model {i}" for i in range(1, n_samples+1)]
        data = pd.DataFrame(data, columns=columns, index=names)
        data["rank"] = _get_rank(data)
        data = data.sort_values(by="rank", ascending=True)
        # Plotting
        fig, axes = plt.subplots(1, 4, sharey=True, figsize=figsize)
        for i, col in enumerate(columns):
            ax = axes[i]
            sns.barplot(ax=ax, data=data, y=data.index, x=col, color=colors[i])
            # Customize subplots
            ax.set_ylabel("")
            ax.set_xlabel(col)
            ax.axvline(0, color='black') #, linewidth=aa.plot_gcfs("axes.linewidth"))
            if dict_xlims and col in dict_xlims:
                ax.set_xlim(dict_xlims[col])
            if i == 0:
                ax.set_title("Number of clusters", weight="bold")
            elif i == 2:
                ax.set_title("Quality measures", weight="bold")
            sns.despine(ax=ax, left=True)
            ax.tick_params(axis='y', which='both', left=False)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0)


    def center(self, data):
        """PCA plot of clustering with centers highlighted"""

    def medoids(self, data):
        """PCA plot of clustering with medoids highlighted"""

    @staticmethod
    def correlation(df_corr=None):
        """Heatmap for correlation"""
