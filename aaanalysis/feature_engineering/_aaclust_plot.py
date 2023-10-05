"""
This is a script for the plotting class of AAclust.
"""
from sklearn.decomposition import PCA
from typing import Optional, Dict, Union, List, Tuple, Type
from sklearn.base import  TransformerMixin

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.aaclust_plot.aaclust_plot_eval import plot_eval


# I Helper Functions
def _get_components(data=None, model_class=None):
    """"""

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
        # Plotting
        fig, axes = plot_eval()


    def center(self, data):
        """PCA plot of clustering with centers highlighted"""

    def medoids(self, data):
        """PCA plot of clustering with medoids highlighted"""

    @staticmethod
    def correlation(df_corr=None):
        """Heatmap for correlation"""
