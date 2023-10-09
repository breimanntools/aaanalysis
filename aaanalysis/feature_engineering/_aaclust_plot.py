"""
This is a script for the frontend of the AAclustPlot class, used for plotting of the AAclust results.
"""
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Dict, Union, List, Tuple, Type
from sklearn.base import  TransformerMixin
import matplotlib.pyplot as plt

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.aaclust.aaclust_plot import plot_eval, plot_center_or_medoid, plot_correlation


# I Helper Functions
def check_match_data_names(data=None, names=None):
    """"""
    n_samples = len(data)
    if names is not None:
        if len(names) != n_samples:
            raise ValueError(f"n_samples does not match for 'data' ({n_samples}) and 'names' ({len(names)}).")
    else:
        names = [f"Set {i}" for i in range(1, n_samples + 1)]
    if not isinstance(data, pd.DataFrame):
        data = ut.check_array_like(name=data, val=data)
        n_samples, n_features = data.shape
        # Check matching number of features
        if n_features != 4:
            raise ValueError(f"'data' should contain the following four columns: {ut.COLS_EVAL_AACLUST}")
        df_eval = pd.DataFrame(data, columns=ut.COLS_EVAL_AACLUST, index=names)
    else:
        df_eval = data
        # Check data for missing columns
        missing_cols = [x for x in ut.COLS_EVAL_AACLUST if x not in list(df_eval)]
        if len(missing_cols) > 0:
            raise ValueError(f"'data' must contain the following columns: {missing_cols}")
        df_eval.index = names
    return df_eval


def check_dict_xlims(dict_xlims=None):
    """"""
    if dict_xlims is None:
        return
    ut.check_dict(name="dict_xlims", val=dict_xlims)
    wrong_keys = [x for x in list(dict_xlims) if x not in ut.COLS_EVAL_AACLUST]
    if len(wrong_keys) > 0:
        raise ValueError(f"'dict_xlims' should not contain the following keys: {wrong_keys}")
    for key in dict_xlims:
        if len(dict_xlims[key]) != 2:
            raise ValueError("'dict_xlims' values should be tuple with two numbers.")
        xmin, xmax = dict_xlims[key]
        ut.check_number_val(name="dict_xlims:min", val=xmin, just_int=False, accept_none=False)
        ut.check_number_val(name="dict_xlims:max", val=xmax, just_int=False, accept_none=False)
        if xmin >= xmax:
            raise ValueError(f"'dict_xlims:min' ({xmin}) should be < 'dict_xlims:max' ({xmax}) for '{key}'.")



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
        model_kwargs = ut.check_model_kwargs(model_class=model_class,
                                             model_kwargs=model_kwargs,
                                             param_to_check="n_components")
        self.model_class = model_class
        self.model_kwargs = model_kwargs

    @staticmethod
    def eval(data_eval: ut.ArrayLike2D,
             names: Optional[List[str]] = None,
             dict_xlims: Optional[Union[None, dict]] = None,
             figsize: Optional[Tuple[int, int]] = (7, 6)
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Evaluates and plots ``n_clusters`` and clustering metrics ``BIC``, ``CH``, and ``SC`` for the provided data.

        The clustering evaluation metrics (BIC, CH, and SC) are ranked by the average of their independent rankings.

        Parameters
        ----------
        data_eval : `array-like, shape (n_samples, n_features)`
            Evaluation matrix or DataFrame. `Rows` correspond to scale sets and `columns` to the following
            four evaluation measures:

            - ``n_clusters``: Number of clusters.
            - ``BIC``: Bayesian Information Criterion.
            - ``CH``: Calinski-Harabasz Index.
            - ``SC``: Silhouette Coefficient.

        names
            Names of scale sets from ``data``. If None, names are internally generated as 'Set 1', 'Set 2' etc.
        dict_xlims
            A dictionary containing x-axis limits (``xmin``, ``xmax``) for selected evaluation measure metric subplots.
            Keys should be names of the ``evaluation measures`` (e.g., 'BIC'). If None, x-axis are auto-scaled.
        figsize
            Width and height of the figure in inches.

        Returns
        -------
        fig
            Figure object containing the plots.
        axes
            Axes object(s) containing four subplots.

        Notes
        -----
        - The data is ranked in ascending order of the average ranking of the scale sets.

        See Also
        --------
        * :meth:`AAclust.eval` for details on evaluation measures.
        """
        # Check input
        ut.check_array_like(name="data", val=data_eval)
        ut.check_list_like(name="names", val=names, accept_none=True)
        df_eval = check_match_data_names(data=data_eval, names=names)
        check_dict_xlims(dict_xlims=dict_xlims)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True)
        # Plotting
        colors = aa.plot_get_clist(n_colors=4)
        fig, axes = plot_eval(df_eval=df_eval,
                              dict_xlims=dict_xlims,
                              figsize=figsize,
                              colors=colors)
        return fig, axes


    def center(self,
               X: ut.ArrayLike2D,
               labels: ut.ArrayLike1D = None,
               figsize: Optional[Tuple[int, int]] = (7, 6),
               dot_alpha: Optional[float] = 0.75,
               dot_size: Optional[int] = 100,
               component_x : Optional[int] = 1,
               component_y : Optional[int] = 2,
               ) -> pd.DataFrame:
        """PCA plot of clustering with centers highlighted"""
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="component_x", val=component_x, accept_none=False, min_val=1, just_int=True)
        ut.check_number_range(name="component_y", val=component_y, accept_none=False, min_val=1, just_int=True)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True)
        ut.check_number_range(name="dot_alpha", val=dot_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="dot_size", val=dot_size, accept_none=False, min_val=1, just_int=True)
        # Create plot
        df_components = plot_center_or_medoid(X, labels=labels, plot_centers=True,
                                              component_x=component_x, component_y=component_y,
                                              model_class=self.model_class, model_kwargs=self.model_kwargs,
                                              figsize=figsize, dot_size=dot_size, dot_alpha=dot_alpha)
        return df_components

    def medoids(self,
                X: ut.ArrayLike2D,
                labels: ut.ArrayLike1D = None,
                figsize: Optional[Tuple[int, int]] = (7, 6),
                dot_alpha: Optional[float] = 0.75,
                dot_size: Optional[int] = 100,
                component_x : Optional[int] = 1,
                component_y : Optional[int] = 2,
                metric: Optional[str] = "euclidean",
                ) -> pd.DataFrame:
        """PCA plot of clustering with medoids highlighted"""
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="component_x", val=component_x, accept_none=False, min_val=1, just_int=True)
        ut.check_number_range(name="component_y", val=component_y, accept_none=False, min_val=1, just_int=True)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True)
        ut.check_number_range(name="dot_alpha", val=dot_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="dot_size", val=dot_size, accept_none=False, min_val=1, just_int=True)
        ut.check_metric(metric=metric)
        # Create plot
        df_components = plot_center_or_medoid(X, labels=labels, plot_centers=False, component_x=component_x,
                                              component_y=component_y, metric=metric, model_class=self.model_class,
                                              model_kwargs=self.model_kwargs, figsize=figsize, dot_size=dot_size,
                                              dot_alpha=dot_alpha)

        return df_components

    @staticmethod
    def correlation(df_corr=None):
        """Heatmap for correlation"""
