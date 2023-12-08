"""
This is a script for the frontend of the AAclustPlot class, used for plotting of the AAclust results.
"""
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Dict, Union, List, Tuple, Type
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.check_aaclust import check_metric
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


# Check correlation plot
def check_match_df_corr_labels(df_corr=None, labels=None):
    """"""
    labels = ut.check_labels(labels=labels)
    n_samples, n_clusters = df_corr.shape
    if n_samples != len(labels):
        raise ValueError(f"Number of 'labels' ({len(labels)}) must match with n_samples in 'df_corr' ({n_samples})")
    return labels


def check_method(method=None):
    """"""
    valid_methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    if method not in valid_methods:
        raise ValueError(f"'method' ({method}) should be one of following: {valid_methods}")


def check_match_df_corr_clust_x(df_corr=None, cluster_x=None):
    """"""
    all_vals = df_corr.to_numpy().flatten()[1:].tolist()
    if cluster_x:
        if len(set(all_vals)) == 1:
            raise ValueError(f"'df_corr' should not contain all same values if 'cluster_x' is True")
        if None in all_vals or np.NaN in all_vals:
            raise ValueError(f"'df_corr' should not contain missing values")


def check_bar_position(bar_position=None):
    """"""
    bar_position = ut.check_list_like(name="bar_position", val=bar_position, convert=True, accept_str=True)
    valid_positions = ['left', 'right', 'top', 'bottom']
    wrong_positions = [x for x in bar_position if x not in valid_positions]
    if len(wrong_positions) > 0:
        raise ValueError(f"Wrong 'bar_position' ({wrong_positions}). They should be as follows: {valid_positions}")
    return bar_position


def check_bar_colors(bar_colors=None):
    """"""
    if isinstance(bar_colors, str):
        ut.check_color(name='bar_colors', val=bar_colors, accept_none=False)
        bar_colors = [bar_colors]
    elif isinstance(bar_colors, list):
        for color in bar_colors:
            ut.check_color(name="element from 'bar_colors'", val=color, accept_none=False)
    else:
        raise ValueError("'bar_colors' should be string or list")
    return bar_colors


def check_match_bar_colors_labels(bar_colors=None, labels=None):
    """"""
    n_clusters = len(set(labels)) # Number of unique labels
    if len(bar_colors) == 1:
        bar_colors = bar_colors * n_clusters
    if len(bar_colors) < n_clusters:
        n_colors = len(bar_colors)
        bar_colors *= n_clusters
        warnings.warn(f"Length of 'bar_colors' (n={n_colors}) should be >= n_clusters (n={n_clusters})")
    return bar_colors[0:n_clusters]


# II Main Functions
class AAclustPlot:
    """Plot results of AAclust analysis.

    Dimensionality reduction is performed for visualization using decomposition models such as
    Principal Component Analysis (PCA).

    See Also
    --------
    * Scikit-learn `decomposition model classes <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition>`_.
    """
    def __init__(self,
                 model_class: Type[TransformerMixin] = PCA,
                 model_kwargs: Optional[Dict] = None):
        """
        Parameters
        ----------
        model_class
            A decomposition model class with ``n_components`` parameter.
        model_kwargs
            Keyword arguments to pass to the selected decomposition model.
        """
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
             figsize: Tuple[int, int] = (7, 6)
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Evaluates and plots ``n_clusters`` and clustering metrics ``BIC``, ``CH``, and ``SC`` for the provided data.

        The clustering evaluation metrics (BIC, CH, and SC) are ranked by the average of their independent rankings.

        Parameters
        ----------
        data_eval : array-like, shape (n_samples, n_features)
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
        axes
            Axes object(s) containing four subplots.

        Notes
        -----
        * The data is ranked in ascending order of the average ranking of the scale sets.

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
               component_x: int = 1,
               component_y: int = 2,
               ax: Optional[plt.Axes] = None,
               figsize: Tuple[int, int] = (7, 6),
               dot_alpha: float = 0.75,
               dot_size: int = 100,
               legend: bool = True,
               palette: Optional[mpl.colors.ListedColormap] = None,
               ) -> Tuple[plt.Axes, pd.DataFrame]:
        """PCA plot of clustering with centers highlighted

        Parameters
        ----------
        X : `array-like of shape (n_samples, n_features)`
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : `array-like of shape (n_samples,)`
            Cluster labels for each sample in ``X``. If ``None``, no grouping is used.
        component_x
            Index of the PCA component for the x-axis. Must be >= 1.
        component_y
            Index of the PCA component for the y-axis. Must be >= 1.
        ax
            Pre-defined Axes object to plot on. If ``None``, a new Axes object is created.
        figsize
            Figure size (width, height) in inches.
        dot_alpha
            Alpha value of the plotted dots.
        dot_size
            Size of the plotted dots.
        legend
            Whether to show the legend.
        palette
            Colormap for the labels or list of colors. If ``None``, a default colormap is used.

        Returns
        -------
        ax
            Axes object with the PCA plot.
        df_components
            DataFrame with the PCA components.

        Notes
        -----
        * Ensure `X` and `labels` are in the same order to avoid mislabeling.

        See Also
        --------
        * See the :ref:`tutorial <palette_tutorial>` for more information.
        * See colormaps from matplotlib in :class:`matplotlib.colors.ListedColormap`.
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="component_x", val=component_x, accept_none=False, min_val=1, just_int=True)
        ut.check_number_range(name="component_y", val=component_y, accept_none=False, min_val=1, just_int=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True, check_n_number=True)
        ut.check_number_range(name="dot_alpha", val=dot_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="dot_size", val=dot_size, accept_none=False, min_val=1, just_int=True)
        ut.check_bool(name="legend", val=legend)
        ut.check_palette(name="palette", val=palette, accept_none=True)
        # Plotting
        ax, df_components = plot_center_or_medoid(X, labels=labels, plot_centers=True,
                                                  component_x=component_x, component_y=component_y,
                                                  model_class=self.model_class, model_kwargs=self.model_kwargs,
                                                  ax=ax, figsize=figsize,
                                                  dot_size=dot_size, dot_alpha=dot_alpha,
                                                  legend=legend, palette=palette)
        return ax, df_components


    def medoids(self,
                X: ut.ArrayLike2D,
                labels: ut.ArrayLike1D = None,
                component_x: int = 1,
                component_y: int = 2,
                metric: str = "euclidean",
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[int, int] = (7, 6),
                dot_alpha: float = 0.75,
                dot_size: int = 100,
                legend: bool = True,
                palette: Optional[mpl.colors.ListedColormap] = None,
                ) -> Tuple[plt.Axes, pd.DataFrame]:
        """PCA plot of clustering with medoids highlighted

        Parameters
        ----------
        X : `array-like of shape (n_samples, n_features)`
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : `array-like of shape (n_samples,)`
            Cluster labels for each sample in ``X``. If ``None``, no grouping is used.
        component_x
            Index of the PCA component for the x-axis. Must be >= 1.
        component_y
            Index of the PCA component for the y-axis. Must be >= 1.
        metric
            The distance metric for calculating medoid. Any metric from `scipy.spatial.distance` can be used.
        ax
            Pre-defined Axes object to plot on. If ``None``, a new Axes object is created.
        figsize
            Figure size (width, height) in inches.
        dot_alpha
            Alpha value of the plotted dots.
        dot_size
            Size of the plotted dots.
        legend
            Whether to show the legend.
        palette
            Colormap for the labels or list of colors. If ``None``, a default colormap is used.

        Returns
        -------
        ax
            Axes object with the PCA plot.
        df_components
            DataFrame with the PCA components.

        Notes
        -----
        * Ensure `X` and `labels` are in the same order to avoid mislabeling.

        See Also
        --------
        * See the :ref:`tutorial <palette_tutorial>` for more information.
        * See colormaps from matplotlib in :class:`matplotlib.colors.ListedColormap`.
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="component_x", val=component_x, accept_none=False, min_val=1, just_int=True)
        ut.check_number_range(name="component_y", val=component_y, accept_none=False, min_val=1, just_int=True)
        check_metric(metric=metric)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_tuple(name="figsize", val=figsize, n=2, accept_none=True, check_n_number=True)
        ut.check_number_range(name="dot_alpha", val=dot_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="dot_size", val=dot_size, accept_none=False, min_val=1, just_int=True)
        ut.check_bool(name="legend", val=legend)
        ut.check_palette(name="palette", val=palette, accept_none=True)
        # Create plot
        ax, df_components = plot_center_or_medoid(X, labels=labels, plot_centers=False, metric=metric,
                                                  component_x=component_x, component_y=component_y,
                                                  model_class=self.model_class, model_kwargs=self.model_kwargs,
                                                  ax=ax, figsize=figsize,
                                                  dot_size=dot_size, dot_alpha=dot_alpha,
                                                  legend=legend, palette=palette)
        return ax, df_components

    @staticmethod
    def correlation(df_corr: pd.DataFrame = None,
                    labels: ut.ArrayLike1D = None,
                    cluster_x: bool = False,
                    method: str = "average",
                    xtick_label_rotation: int = 45,
                    ytick_label_rotation: int = 0,
                    bar_position: Union[str, List[str]] = "left",
                    bar_colors: Union[str, List[str]] = "tab:gray",
                    bar_width_x: float = 0.1,
                    bar_spacing_x: float = 0.1,
                    bar_width_y: float = 0.1,
                    bar_spacing_y: float = 0.1,
                    vmin: float = -1,
                    vmax: float = 1,
                    cmap: str = "viridis",
                    **kwargs_heatmap
                    ) -> plt.Axes:
        """
        Heatmap for correlation matrix with colored sidebar to label clusters.

        Parameters
        ----------
        df_corr
            DataFrame with correlation matrix. `Rows` typically correspond to scales and `columns` to clusters.
        labels : `array-like of shape (n_samples,)`
            Cluster labels determining the grouping and coloring of the side color bar.
            It should have the same length as number of rows in ``df_corr`` (n_samples).
        cluster_x
            If ``True``, x-axis (`clusters`) values are clustered.
        method
            Linkage method from :func:`scipy.cluster.hierarchy.linkage` used for clustering.
            Options are ``single``, ``complete``, ``average``, ``weighted``, ``centroid``, ``median``, and ``ward``.
        xtick_label_rotation
            Rotation of x-tick labels (names of `clusters`).
        ytick_label_rotation
            Rotation of y-tick labels (names of `samples`).
        bar_position
            Position of the colored sidebar (``left``, ``right``, ``top``, or ``down``). If ``None``, no sidebar is added.
        bar_colors
            Either a single color or a list of colors for each unique label in ``labels``.
        bar_width_x
            Width of the x-axis sidebar, must be >= 0.
        bar_spacing_x
            Space between the heatmap and the colored x-axis sidebar, must be >= 0.
        bar_width_y
            Width of the y-axis sidebar, must be >= 0.
        bar_spacing_y
            Space between the heatmap and the colored y-axis sidebar, must be >= 0.
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
        ax
            Axes object with the correlation heatmap.

        Notes
        -----
        * Ensure ``labels`` and ``df_corr`` are in the same order to avoid mislabeling.
        * ``bar_tick_labels=True`` will remove tick labels and set them as text for optimal spacing
          so that they can not be adjusted or retrieved afterward (e.g., via `ax.get_xticklabels()`).

        See Also
        --------
        * :func:`seaborn.heatmap`: Seaborn function for creating heatmaps.
        """
        # Check input
        ut.check_df(name="df_corr", df=df_corr, accept_none=False, accept_nan=False)
        labels = check_match_df_corr_labels(df_corr=df_corr, labels=labels)
        ut.check_bool(name="cluster_x", val=cluster_x)
        check_match_df_corr_clust_x(df_corr=df_corr, cluster_x=cluster_x)
        check_method(method=method)
        ut.check_number_val(name="xtick_label_rotation", val=xtick_label_rotation, just_int=True, accept_none=True)
        ut.check_number_val(name="ytick_label_rotation", val=ytick_label_rotation, just_int=True, accept_none=True)
        bar_position = check_bar_position(bar_position=bar_position)
        bar_colors = check_bar_colors(bar_colors=bar_colors)
        bar_colors = check_match_bar_colors_labels(bar_colors=bar_colors, labels=labels)
        dict_float_args = dict(bar_width_x=bar_width_x, bar_spacing_x=bar_spacing_x,
                               bar_width_y=bar_width_y, bar_spacing_y=bar_spacing_y)
        for name in dict_float_args:
            ut.check_number_range(name=name, val=dict_float_args[name], min_val=0, accept_none=False, just_int=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        ut.check_cmap(name="cmap", val=cmap, accept_none=False)
        # Plotting
        pairwise = [str(x) for x in list(df_corr)] == [str(x) for x in list(df_corr.T)]
        if pairwise:
            cluster_x = False
        ax = plot_correlation(df_corr=df_corr.copy(), labels=labels, pairwise=pairwise,
                              cluster_x=cluster_x, method=method,
                              xtick_label_rotation=xtick_label_rotation,
                              ytick_label_rotation=ytick_label_rotation,
                              bar_position=bar_position, bar_colors=bar_colors,
                              bar_width_x=bar_width_x, bar_spacing_x=bar_spacing_x,
                              bar_width_y=bar_width_y, bar_spacing_y=bar_spacing_y,
                              vmin=vmin, vmax=vmax, cmap=cmap,
                              **kwargs_heatmap)
        plt.tight_layout()
        return ax