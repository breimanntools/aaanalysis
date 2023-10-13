"""
This is a script for the frontend of the AAclustPlot class, used for plotting of the AAclust results.
"""
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Dict, Union, List, Tuple, Type
from sklearn.base import  TransformerMixin
import matplotlib.pyplot as plt
import matplotlib as mpl

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


# Common interface
doc_param_center_medoid_data = \
"""\
X : array-like of shape (n_samples, n_features)
    Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.\
labels : array-like of shape (n_samples,)
    Cluster labels for each sample in ``X``. If `None`, no grouping is used.
component_x
    Index of the PCA component for the x-axis.
component_y
    Index of the PCA component for the y-axis.\
"""

doc_param_center_medoid_fig = \
"""\
ax
    Pre-defined Axes object to plot on. If `None`, a new Axes object is created.
figsize
    Figure size (width, height) in inches.
dot_alpha
    Alpha value of the plotted dots.
dot_size
    Size of the plotted dots.
legend
    Whether to show the legend.
palette
    Colormap for the labels. If `None`, a default colormap is used.\
"""

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
                                             param_to_check="n_components",
                                             method_to_check="transform")

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

    # TODO check functions, docstring, testing
    @ut.doc_params(doc_param_center_medoid_data=doc_param_center_medoid_data,
                   doc_param_center_medoid_fig=doc_param_center_medoid_fig)
    def center(self,
               X: ut.ArrayLike2D,
               labels: ut.ArrayLike1D = None,
               component_x: Optional[int] = 1,
               component_y: Optional[int] = 2,
               ax : Optional[plt.Axes] = None,
               figsize: Optional[Tuple[int, int]] = (7, 6),
               dot_alpha: Optional[float] = 0.75,
               dot_size: Optional[int] = 100,
               legend : Optional[bool] =True,
               palette : Optional[mpl.colors.ListedColormap] = None,
               ) -> Tuple[plt.Axes, pd.DataFrame]:
        """PCA plot of clustering with centers highlighted

        Parameters
        ----------
        {doc_param_center_medoid_data}
        {doc_param_center_medoid_fig}

        Returns
        -------
        ax
            Axes object with the PCA plot.
        df_components
            DataFrame with the PCA components.

        Notes
        -----
        - Ensure `X` and `labels` are in the same order to avoid mislabeling.

        See Also
        --------
        - See the :ref:`tutorial <palette_tutorial>` for more information.
        - See colormaps from matplotlib in :class:`matplotlib.colors.ListedColormap`.
        """
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
        ax, df_components = plot_center_or_medoid(X, labels=labels, plot_centers=True,
                                                  component_x=component_x, component_y=component_y,
                                                  model_class=self.model_class, model_kwargs=self.model_kwargs,
                                                  ax=ax, figsize=figsize,
                                                  dot_size=dot_size, dot_alpha=dot_alpha,
                                                  legend=legend, palette=palette)
        return ax, df_components

    # TODO check functions, docstring, testing
    @ut.doc_params(doc_param_center_medoid_data=doc_param_center_medoid_data,
                   doc_param_center_medoid_fig=doc_param_center_medoid_fig)
    def medoids(self,
                X: ut.ArrayLike2D,
                labels: ut.ArrayLike1D = None,
                component_x: Optional[int] = 1,
                component_y: Optional[int] = 2,
                metric: Optional[str] = "euclidean",
                ax: Optional[plt.Axes] = None,
                figsize: Optional[Tuple[int, int]] = (7, 6),
                dot_alpha: Optional[float] = 0.75,
                dot_size: Optional[int] = 100,
                legend: Optional[bool] = True,
                palette: Optional[mpl.colors.ListedColormap] = None,
                return_data : Optional[bool] = False
                ) -> Tuple[plt.Axes, pd.DataFrame]:
        """PCA plot of clustering with medoids highlighted

        Parameters
        ----------
        {doc_param_center_medoid_data}
        metric
            The distance metric for calculating medoid. Any metric from `scipy.spatial.distance` can be used.
        {doc_param_center_medoid_fig}
        return_data : bool, optional, default=False
            If `True`, returns PCA components DataFrame. If `False`, returns the Axes object.

        Returns
        -------
        ax
            Axes object with the PCA plot.
        df_components
            DataFrame with the PCA components.

        Notes
        -----
        - Ensure `X` and `labels` are in the same order to avoid mislabeling.
        """
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
        ax, df_components = plot_center_or_medoid(X, labels=labels, plot_centers=False, metric=metric,
                                                  component_x=component_x, component_y=component_y,
                                                  model_class=self.model_class, model_kwargs=self.model_kwargs,
                                                  ax=ax, figsize=figsize,
                                                  dot_size=dot_size, dot_alpha=dot_alpha,
                                                  legend=legend, palette=palette)
        if return_data:
           return df_components
        return ax


    # TODO check functions, docstring, testing
    @staticmethod
    def correlation(df_corr: Optional[pd.DataFrame] = None,
                    labels: Optional[List[str]] = None,
                    bar_position: str = "left",
                    bar_width: float = 0.1,
                    bar_spacing: float = 0.1,
                    bar_colors: Union[str, List[str]] = "gray",
                    bar_set_tick_labels: bool = True,
                    cluster_x : bool = True,
                    cluster_y : bool = False,
                    method : str = "average",
                    vmin: float = -1,
                    vmax: float = 1,
                    cmap: str = "viridis",
                    **kwargs_heatmap
                    ) -> plt.Axes:
        """
        Heatmap for correlation matrix with colored sidebar to label clusters.

        Parameters
        ----------
        df_corr : `array-like, shape (n_samples, n_clusters)`
            DataFrame with correlation matrix. `Rows` typically correspond to scales and `columns` to clusters.
        labels
            Labels determining the grouping and coloring of the side color bar.
            It should be of the same length as `df_corr` columns/rows.
            Defaults to None.
        bar_position
            Position of the colored sidebar (``left``, ``right``, ``top``, or ``down``). If ``None``, no sidebar is added.
        bar_width
            Width of the sidebar.
        bar_spacing
            Space between the heatmap and the side color bar.
        bar_colors
            Either a single color or a list of colors for each unique label in `labels`.
        bar_set_tick_labels
            Add text labels next to the bars without padding.
        cluster_x:
            If True, x-axis (samples) are clustered.
        cluster_y:
            If True, y-axis (clusters) are clustered.
        method:
            Linkage method from :func:`scipy.cluster.hierarchy.linkage` used for clustering.
        vmin
            Minimum value of the color scale in :func:`seaborn.heatmap`.
        vmax
            Maximum value of the color scale in :func:`seaborn.heatmap`.
        cmap
            Colormap to be used for the :func:`seaborn.heatmap`.
        **kwargs_heatmap
            Additional keyword arguments passed to :func:`seaborn.heatmap`.

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            Axes object with the correlation heatmap.

        Notes
        -----
        - Ensure `labels` and `df_corr` are in the same order to avoid mislabeling.
        - `bar_tick_labels=True` will remove tick labels and set them as text for optimal spacing
          so that they can not be adjusted or retrieved afterward (e.g., via `ax.get_xticklabels()`).

        See Also
        --------
        :func:`seaborn.heatmap`: Seaborn function for creating heatmaps.

        """
        ax = plot_correlation(df_corr=df_corr, labels_sorted=labels,
                              bar_position=bar_position,
                              bar_width=bar_width, bar_spacing=bar_spacing, bar_colors=bar_colors,
                              bar_set_tick_labels=bar_set_tick_labels,
                              cluster_x=cluster_x, cluster_y=cluster_y, method=method,
                              vmin=vmin, vmax=vmax, cmap=cmap, **kwargs_heatmap)
        plt.tight_layout()
        return ax