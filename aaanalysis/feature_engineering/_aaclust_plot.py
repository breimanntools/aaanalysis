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

import aaanalysis.utils as ut

from ._backend.check_aaclust import check_metric
from ._backend.aaclust.aaclust_plot import plot_eval, plot_center_or_medoid, plot_correlation


# I Helper Functions
def check_match_df_eval_names(df_eval=None, names=None):
    """Validate the match between the number of samples in df_eval and the length of names"""
    n_samples = len(df_eval)
    if names is not None:
        if len(names) != n_samples:
            raise ValueError(f"n_samples does not match for 'data' ({n_samples}) and 'names' ({len(names)}).")
    else:
        names = [f"Set {i}" for i in range(1, n_samples + 1)]
    df_eval.index = names
    return df_eval


# Check correlation plot
def check_match_df_corr_labels(df_corr=None, labels=None):
    """Ensure the number of labels matches the number of samples in df_corr"""
    labels = ut.check_labels(labels=labels)
    n_samples, n_clusters = df_corr.shape
    if n_samples != len(labels):
        raise ValueError(f"Number of 'labels' ({len(labels)}) must match with n_samples in 'df_corr' ({n_samples})")
    return labels


def check_match_df_corr_labels_ref(df_corr=None, labels_ref=None, labels=None, pairwise=False, verbose=False):
    """Ensure the number of labels matches the number of samples in df_corr"""
    str_add = ""
    if labels_ref is None:
        if pairwise:
            return # Skip check
        columns = list(df_corr)
        if sum([isinstance(i, str) for i in columns]) != 0:
            raise ValueError("'labels_ref' must be provided if columns in 'df_corr' and strings.")
        labels_ref = list(df_corr)
        str_add = "(obtained from 'df_corr' columns if not given)"
    if set(labels) != set(labels_ref) and verbose:
        str_warn = (f"Warning: 'labels' and 'labels_ref' {str_add} does not match. "
                    f"Provide 'labels_ref' or adjust 'df_corr' columns."
                    f"\n 'labels': {set(labels)}, "
                    f"\n 'labels_ref': {set(labels_ref)}")
        warnings.warn(str_warn, UserWarning)

    labels_ref = ut.check_labels(labels=labels_ref, name="labels_ref")
    n_samples, n_clusters = df_corr.shape
    if n_clusters != len(labels_ref):
        raise ValueError(f"Number of 'labels_ref' ({len(labels_ref)}) must match with n_clusters in 'df_corr' ({n_samples})")
    return labels_ref


def check_method(method=None):
    """Validate the method parameter against a list of valid hierarchical clustering methods"""
    valid_methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    if method not in valid_methods:
        raise ValueError(f"'method' ({method}) should be one of: {valid_methods}")


def check_match_df_corr_clust_x(df_corr=None, cluster_x=None):
    """Ensure df_corr has variability and no missing values if cluster_x is enabled"""
    all_vals = df_corr.to_numpy().flatten()[1:].tolist()
    if cluster_x:
        if len(set(all_vals)) == 1:
            raise ValueError(f"'df_corr' should not contain all same values if 'cluster_x' is True")
        if None in all_vals or np.nan in all_vals:
            raise ValueError(f"'df_corr' should not contain missing values")


def check_bar_position(bar_position=None):
    """Validate and standardize bar_position to be one of the valid positions like 'left', 'right', 'top', or 'bottom'"""
    bar_position = ut.check_list_like(name="bar_position", val=bar_position, convert=True, accept_str=True)
    valid_positions = ['left', 'right', 'top', 'bottom']
    wrong_positions = [x for x in bar_position if x not in valid_positions]
    if len(wrong_positions) > 0:
        raise ValueError(f"Wrong 'bar_position' ({wrong_positions}). They should be as follows: {valid_positions}")
    return bar_position


def check_bar_colors(bar_colors=None):
    """Validate and standardize bar_colors to be either a single color or a list of valid colors."""
    if isinstance(bar_colors, str):
        ut.check_color(name='bar_colors', val=bar_colors, accept_none=False)
        bar_colors = [bar_colors]
    elif isinstance(bar_colors, list):
        for color in bar_colors:
            ut.check_color(name="element from 'bar_colors'", val=color, accept_none=False)
    else:
        raise ValueError("'bar_colors' should be string or list")
    return bar_colors


def check_warning_match_bar_colors_labels(bar_colors=None, labels=None):
    """Adjust bar_colors to match the number of unique labels and warn if there are insufficient colors provided"""
    n_clusters = len(set(labels))   # Number of unique labels
    if len(bar_colors) == 1:
        bar_colors = bar_colors * n_clusters
    if len(bar_colors) < n_clusters:
        n_colors = len(bar_colors)
        bar_colors *= n_clusters
        str_warn = f"Warning: Length of 'bar_colors' (n={n_colors}) should be >= n_clusters (n={n_clusters})"
        warnings.warn(str_warn, UserWarning)
    return bar_colors[0:n_clusters]


# II Main Functions
class AAclustPlot:
    """
    Plotting class for :class:`AAclust` (Amino Acid clustering) results [Breimann24a]_.

    This class performs dimensionality reduction for visualization using decomposition models such as
    Principal Component Analysis (PCA).
    """
    def __init__(self,
                 model_class: Type[TransformerMixin] = PCA,
                 model_kwargs: Optional[Dict] = None,
                 verbose: bool = True,
                 random_state: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        model_class : Type[TransformerMixin], default=PCA
            A decomposition model class with ``n_components`` parameter.
        model_kwargs : dict, optional
            Keyword arguments to pass to the selected decomposition model.
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        See Also
        --------
        * :class:`AAclust`: the respective computation class.
        * Scikit-learn `decomposition model classes <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition>`_.

        Examples
        --------
        .. include:: examples/aac_plot.rst
        """
        # Global parameters
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        # Model parameters
        ut.check_mode_class(model_class=model_class)
        model_kwargs = ut.check_model_kwargs(model_class=model_class,
                                             model_kwargs=model_kwargs,
                                             param_to_check="n_components",
                                             method_to_check="transform",
                                             random_state=random_state)
        # Internal attributes
        self._verbose = verbose
        self._random_state = random_state
        self._model_class = model_class
        self._model_kwargs = model_kwargs

    @staticmethod
    def eval(df_eval: pd.DataFrame = None,
             figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4),
             dict_xlims: Optional[dict] = None,
             ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots evaluation of ``n_clusters`` and clustering metrics ``BIC``, ``CH``, and ``SC`` from ``df_seq``.

        The clustering evaluation metrics (BIC, CH, and SC) are ranked by the average of their independent rankings.

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_datasets, n_metrics)
            DataFrame with evaluation measures for scale sets. Each `row` corresponds to a specific scale set
            and `columns` are as follows:

            - 'name': Name of clustering datasets.
            - 'n_clusters': Number of clusters.
            - 'BIC': Bayesian Information Criterion.
            - 'CH': Calinski-Harabasz Index.
            - 'SC': Silhouette Coefficient.

        figsize : tuple, default=(7, 6)
            Figure dimensions (width, height) in inches.
        dict_xlims : dict, optional
            A dictionary containing x-axis limits for subplots. Keys should be the subplot axis number ({0, 1, 2, 4})
            and values should be tuple specifying (``xmin``, ``xmax``). If ``None``, x-axis limits are auto-scaled.

        Returns
        -------
        fig : plt.Figure
            Figure object for evaluation plot
        axes : array of plt.Axes
            Array of Axes objects, each representing a subplot within the figure.

        Notes
        -----
        * The data is ranked in ascending order of the average ranking of the scale sets.

        See Also
        --------
        * :meth:`AAclust.eval` for details on evaluation measures.

        Examples
        --------
        .. include:: examples/aac_plot_eval.rst
        """
        # Check input
        ut.check_df(name="df_eval", df=df_eval, cols_requiered=ut.COLS_EVAL_AACLUST, accept_none=False, accept_nan=False)
        ut.check_dict_xlims(dict_xlims=dict_xlims, n_ax=4)
        ut.check_figsize(figsize=figsize, accept_none=True)
        # Plotting
        colors = ut.plot_get_clist_(n_colors=4)
        fig, axes = plot_eval(df_eval=df_eval,
                              dict_xlims=dict_xlims,
                              figsize=figsize,
                              colors=colors)
        return fig, axes

    def centers(self,
                X: ut.ArrayLike2D,
                labels: ut.ArrayLike1D = None,
                component_x: int = 1,
                component_y: int = 2,
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 6),
                legend: bool = True,
                dot_size: int = 100,
                dot_alpha: Union[int, float] = 0.75,
                palette: Optional[mpl.colors.ListedColormap] = None,
                ) -> Tuple[plt.Axes, pd.DataFrame]:
        """
        PCA plot of clustering with centers highlighted

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : array-like, shape (n_samples,)
            Cluster labels for each sample in ``X``. If ``None``, no grouping is used.
        component_x : int, default=1
            Index of the PCA component for the x-axis. Must be >= 1.
        component_y : int, default=2
            Index of the PCA component for the y-axis. Must be >= 1.
        ax : plt.Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new Axes object is created.
        figsize : tuple, default=(7, 6)
            Figure dimensions (width, height) in inches.
        legend : bool, default=True
            Whether to show the legend.
        dot_size : int, default=100
            Size of the plotted dots.
        dot_alpha : float or int, default=0.75
            The transparency alpha value [0-1] of the plotted dots.
        palette : list, optional
            Colormap for the labels or list of colors. If ``None``, a default colormap is used.

        Returns
        -------
        ax : plt.Axes
            PCA plot axes object.
        df_components : pd.DataFrame
            DataFrame with the PCA components.

        Notes
        -----
        * Ensure `X` and `labels` are in the same order to avoid mislabeling.

        See Also
        --------
        * See the :ref:`tutorial <palette_tutorial>` for more information.
        * See colormaps from matplotlib in :class:`matplotlib.colors.ListedColormap`.

        Examples
        --------
        .. include:: examples/aac_plot_centers.rst
        """
        # Check input
        X = ut.check_X(X=X)
        ut.check_X_unique_samples(X=X)
        labels = ut.check_labels(labels=labels)
        ut.check_match_X_labels(X=X, labels=labels)
        ut.check_number_range(name="component_x", val=component_x, accept_none=False, min_val=1, just_int=True)
        ut.check_number_range(name="component_y", val=component_y, accept_none=False, min_val=1, just_int=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="dot_alpha", val=dot_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="dot_size", val=dot_size, accept_none=False, min_val=1, just_int=True)
        ut.check_bool(name="legend", val=legend)
        ut.check_palette(name="palette", val=palette, accept_none=True)
        # Plotting
        ax, df_components = plot_center_or_medoid(X, labels=labels, plot_centers=True,
                                                  component_x=component_x, component_y=component_y,
                                                  model_class=self._model_class, model_kwargs=self._model_kwargs,
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
                figsize: Tuple[Union[int, float], Union[int, float]] = (7, 6),
                legend: bool = True,
                dot_size: int = 100,
                dot_alpha: Union[int, float] = 0.75,
                palette: Optional[mpl.colors.ListedColormap] = None,
                ) -> Tuple[plt.Axes, pd.DataFrame]:
        """
        PCA plot of clustering with medoids highlighted

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to scales and `columns` to amino acids.
        labels : array-like, shape (n_samples,)
            Cluster labels for each sample in ``X``. If ``None``, no grouping is used.
        component_x : int, default=1
            Index of the PCA component for the x-axis. Must be >= 1.
        component_y : int, default=1
            Index of the PCA component for the y-axis. Must be >= 1.
        metric : {'correlation', 'euclidean', 'manhattan', 'cosine'}, default='euclidean'
            The distance metric for calculating medoid.

            - ``correlation``: Pearson correlation (maximum)
            - ``euclidean``: Euclidean distance (minimum)
            - ``manhattan``: Manhattan distance (minimum)
            - ``cosine``: Cosine distance (minimum)

        ax : plt.Axes, optional
            Pre-defined Axes object to plot on. If ``None``, a new Axes object is created.
        figsize : tuple, default=(7, 6)
            Figure dimensions (width, height) in inches.
        legend : bool, default=True
            Whether to show the legend.
        dot_size : int, default=100
            Size of the plotted dots.
        dot_alpha : float or int, default=0.75
            The transparency alpha value [0-1] of the plotted dots.
        palette : list, optional
            Colormap for the labels or list of colors. If ``None``, a default colormap is used.

        Returns
        -------
        ax : plt.Axes
            PCA plot axes object.
        df_components : pd.DataFrame
            DataFrame with the PCA components.

        Notes
        -----
        * Ensure `X` and `labels` are in the same order to avoid mislabeling.

        See Also
        --------
        * See the :ref:`tutorial <palette_tutorial>` for more information.
        * See colormaps from matplotlib in :class:`matplotlib.colors.ListedColormap`.

        Examples
        --------
        .. include:: examples/aac_plot_medoids.rst
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
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="dot_alpha", val=dot_alpha, accept_none=False, min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="dot_size", val=dot_size, accept_none=False, min_val=1, just_int=True)
        ut.check_bool(name="legend", val=legend)
        ut.check_palette(name="palette", val=palette, accept_none=True)
        # Create plot
        ax, df_components = plot_center_or_medoid(X, labels=labels, plot_centers=False, metric=metric,
                                                  component_x=component_x, component_y=component_y,
                                                  model_class=self._model_class, model_kwargs=self._model_kwargs,
                                                  ax=ax, figsize=figsize,
                                                  dot_size=dot_size, dot_alpha=dot_alpha,
                                                  legend=legend, palette=palette)
        return ax, df_components

    def correlation(self,
                    df_corr: pd.DataFrame = None,
                    labels: ut.ArrayLike1D = None,
                    labels_ref: Optional[ut.ArrayLike1D] = None,
                    cluster_x: bool = False,
                    method: str = "average",
                    xtick_label_rotation: int = 90,
                    ytick_label_rotation: int = 0,
                    bar_position: Union[str, List[str]] = "left",
                    bar_colors: Union[str, List[str]] = "tab:gray",
                    bar_width_x: float = 0.1,
                    bar_spacing_x: float = 0.1,
                    bar_width_y: float = 0.1,
                    bar_spacing_y: float = 0.1,
                    vmin: float = -1.0,
                    vmax: float = 1.0,
                    cmap: str = "viridis",
                    kwargs_heatmap: Optional[dict] = None
                    ) -> plt.Axes:
        """
        Heatmap for correlation matrix with colored sidebar to label clusters.

        Parameters
        ----------
        df_corr : pd.DataFrame, shape (n_samples, n_clusters)
            DataFrame with correlation matrix. `Rows` typically correspond to scales and `columns` to clusters.
        labels : array-like, shape (n_samples,)
            Cluster labels determining the grouping and coloring of the side colorbar.
            It should have the same length as number of rows in ``df_corr`` (n_samples).
        labels_ref  : array-like, shape (n_clusters,), optional
            Cluster labels comprising unique values from 'labels'. Length must match with 'n_clusters' in ``df_corr``.
        cluster_x : bool, default=False
            If ``True``, x-axis (`clusters`) values are clustered. Disabled for pairwise correlation.
        method : {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'}, default='average'
            Linkage method from :func:`scipy.cluster.hierarchy.linkage` used for clustering.
        xtick_label_rotation : int, default=90
            Rotation of x-tick labels (names of `clusters`).
        ytick_label_rotation : int, default=0
            Rotation of y-tick labels (names of `samples`).
        bar_position : str or list of str, default='left'
            Position of the colored sidebar (``left``, ``right``, ``top``, or ``down``). If ``None``, no sidebar is added.
        bar_colors : str or list of str, default='tab:gray'
            Either a single color or a list of colors for each unique label in ``labels``.
        bar_width_x : float, default=0.1
            Width of the x-axis sidebar, must be >= 0.
        bar_spacing_x : float, default=0.1
            Space between the heatmap and the colored x-axis sidebar, must be >= 0.
        bar_width_y : float, default=0,1
            Width of the y-axis sidebar, must be >= 0.
        bar_spacing_y : float, default=0.1
            Space between the heatmap and the colored y-axis sidebar, must be >= 0.
        vmin : float, default=-1.0
            Minimum value of the color scale in :func:`seaborn.heatmap`.
        vmax : float, default=1.0
            Maximum value of the color scale in :func:`seaborn.heatmap`.
        cmap : str, default='viridis'
            Colormap to be used for the :func:`seaborn.heatmap`.
        kwargs_heatmap : dict, optional
            Dictionary with keyword arguments for adjusting heatmap (:func:`seaborn.heatmap`).

        Returns
        -------
        ax : plt.Axes
            Axes object with the correlation heatmap.

        Notes
        -----
        * Ensure ``labels`` and ``df_corr`` are in the same order to avoid mislabeling.
        * ``bar_tick_labels=True`` will remove tick labels and set them as text for optimal spacing
          so that they can not be adjusted or retrieved afterward (e.g., via `ax.get_xticklabels()`).

        See Also
        --------
        * :func:`seaborn.heatmap`: Seaborn function for creating heatmaps.

        Examples
        --------
        .. include:: examples/aac_plot_correlation.rst
        """
        # Check input
        ut.check_df(name="df_corr", df=df_corr, accept_none=False, accept_nan=False, accept_duplicates=True)
        labels = check_match_df_corr_labels(df_corr=df_corr, labels=labels)
        pairwise = [str(x) for x in list(df_corr)] == [str(x) for x in list(df_corr.T)]
        labels_ref = check_match_df_corr_labels_ref(df_corr=df_corr, labels_ref=labels_ref, labels=labels,
                                                    pairwise=pairwise, verbose=self._verbose)
        ut.check_bool(name="cluster_x", val=cluster_x)
        check_match_df_corr_clust_x(df_corr=df_corr, cluster_x=cluster_x)
        check_method(method=method)
        ut.check_number_val(name="xtick_label_rotation", val=xtick_label_rotation, just_int=True, accept_none=True)
        ut.check_number_val(name="ytick_label_rotation", val=ytick_label_rotation, just_int=True, accept_none=True)
        bar_position = check_bar_position(bar_position=bar_position)
        bar_colors = check_bar_colors(bar_colors=bar_colors)
        bar_colors = check_warning_match_bar_colors_labels(bar_colors=bar_colors, labels=labels)
        dict_float_args = dict(bar_width_x=bar_width_x, bar_spacing_x=bar_spacing_x,
                               bar_width_y=bar_width_y, bar_spacing_y=bar_spacing_y)
        for name in dict_float_args:
            ut.check_number_range(name=name, val=dict_float_args[name], min_val=0, accept_none=False, just_int=False)
        ut.check_vmin_vmax(vmin=vmin, vmax=vmax)
        ut.check_cmap(name="cmap", val=cmap, accept_none=False)
        ut.check_dict(name="kwargs_heatmap", val=kwargs_heatmap, accept_none=True)
        # Plotting
        if pairwise:
            cluster_x = False
        try:
            ax = plot_correlation(df_corr=df_corr.copy(), labels=labels, labels_ref=labels_ref,
                                  cluster_x=cluster_x, method=method,
                                  xtick_label_rotation=xtick_label_rotation,
                                  ytick_label_rotation=ytick_label_rotation,
                                  bar_position=bar_position, bar_colors=bar_colors,
                                  bar_width_x=bar_width_x, bar_spacing_x=bar_spacing_x,
                                  bar_width_y=bar_width_y, bar_spacing_y=bar_spacing_y,
                                  vmin=vmin, vmax=vmax, cmap=cmap,
                                  kwargs_heatmap=kwargs_heatmap)
        except Exception as e:
            str_error = f"Following error occurred due to sns.heatmap() function: {e}"
            raise ValueError(str_error)
        plt.tight_layout()
        return ax