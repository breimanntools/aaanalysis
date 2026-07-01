"""
This is a script for the frontend of the ShapModelPlot class, visualizing CPP-SHAP explanations.

It complements :class:`ShapModel` (which computes the SHAP values) with an explanation-similarity
clustermap and a stand-alone ``shap_to_feat_imp`` helper that turns a per-sample SHAP vector into a
normalized signed feature impact / absolute importance.
"""
from typing import Optional, List, Union, Tuple
from collections import Counter
import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.shap_model.sm_add_feat_impact import _comp_sample_shap_feat_impact
from ._backend.shap_model.sm_plot import comp_shap_correlation, plot_shap_clustermap


# I Helper Functions
def check_match_shap_values_labels(shap_values=None, labels=None, name="labels"):
    """Check shap_values is a 2D numeric array and labels match its number of samples."""
    shap_values = ut.check_array_like(name="shap_values", val=shap_values,
                                      dtype="numeric", expected_dim=2)
    n_samples = shap_values.shape[0]
    if n_samples < 2:
        raise ValueError(f"'shap_values' should contain >= 2 samples (rows), got {n_samples}.")
    labels = ut.check_labels(labels=labels, len_required=n_samples, accept_float=True)
    return shap_values, labels


def check_match_names_shap_values(names=None, n_samples=None):
    """Check names (unique, length n_samples) and set default 'Protein{i}' names."""
    if names is None:
        return [f"Protein{i}" for i in range(n_samples)]
    names = ut.check_list_like(name="names", val=names, accept_none=False)
    if len(names) != n_samples:
        raise ValueError(f"Length of 'names' (n={len(names)}) must match the number of "
                         f"samples in 'shap_values' (n={n_samples}).")
    if len(set(names)) != len(names):
        duplicated = sorted(x for x, c in Counter(names).items() if c > 1)
        raise ValueError(f"'names' should not contain duplicates: {duplicated}")
    return list(names)


def _get_dict_color(labels=None, labels_pred=None, dict_color=None):
    """Build (or validate) a class -> color mapping for the row/col sidebars."""
    classes = sorted(set(list(labels) + list(labels_pred)))
    if dict_color is None:
        colors = ut.plot_get_clist_(n_colors=max(len(classes), 3))
        dict_color = {c: colors[i] for i, c in enumerate(classes)}
    else:
        ut.check_dict(name="dict_color", val=dict_color, accept_none=False)
        missing = [c for c in classes if c not in dict_color]
        if len(missing) > 0:
            raise ValueError(f"'dict_color' is missing colors for label classes: {missing}")
    return dict_color


# II Main Functions
def shap_to_feat_imp(shap_values: ut.ArrayLike1D,
                     impact: bool = True,
                     ) -> np.ndarray:
    """
    Convert a per-sample SHAP-value vector into normalized feature impact or importance
    (**[pro]**, requires ``aaanalysis[pro]``).

    For one sample (or the mean SHAP vector of a group of same-class samples), the SHAP
    values are normalized so the sum of their absolute values equals 100%:

    - **feature impact** (``impact=True``): signed, ``shap / sum(|shap|) * 100`` — keeps the
      sign, so a feature that pushes the prediction up is positive and one that pushes it
      down is negative.
    - **feature importance** (``impact=False``): absolute, ``|shap| / sum(|shap|) * 100`` —
      magnitude only, the per-sample analogue of the SHAP value-based feature importance.

    This shares the normalization used internally by :meth:`ShapModel.add_feat_impact`
    (re-using its per-sample backend) so the two never diverge.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    shap_values : array-like, shape (n_features,)
        One-dimensional array of SHAP values for a single sample (or the mean SHAP vector of
        a group of same-class samples). Computation is only meaningful within one class.
    impact : bool, default=True
        If ``True``, return the signed feature impact; if ``False``, the absolute feature importance.

    Returns
    -------
    feat_imp : np.ndarray, shape (n_features,)
        Normalized feature impact (signed) or importance (absolute), summing in absolute
        value to 100.

    See Also
    --------
    * :meth:`ShapModel.add_feat_impact` for attaching impact/importance columns to a feature DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from aaanalysis.explainable_ai_pro import shap_to_feat_imp
    >>> shap_vec = np.array([0.2, -0.1, 0.3, -0.4])
    >>> impact = shap_to_feat_imp(shap_vec, impact=True)
    >>> float(np.round(np.abs(impact).sum(), 6))
    100.0
    """
    shap_values = ut.check_array_like(name="shap_values", val=shap_values,
                                      dtype="numeric", expected_dim=1)
    ut.check_bool(name="impact", val=impact)
    if np.nansum(np.abs(shap_values)) == 0:
        raise ValueError("'shap_values' are all zero; feature impact/importance is undefined.")
    if impact:
        # Re-use the per-sample backend used by ShapModel.add_feat_impact (no divergence)
        feat_imp = _comp_sample_shap_feat_impact(shap_values=shap_values.reshape(1, -1),
                                                 i=0, normalize=False)
        return np.asarray(feat_imp, dtype=float)
    abs_values = np.abs(shap_values)
    feat_imp = abs_values / np.nansum(abs_values) * 100
    return np.asarray(feat_imp, dtype=float)


class ShapModelPlot:
    """
    Plotting class for :class:`ShapModel` (**[pro]**, requires ``aaanalysis[pro]``) CPP-SHAP
    explanations [Breimann25]_.

    Visualizes per-sample SHapley Additive exPlanations (SHAP) [Lundberg20]_: :meth:`clustermap`
    clusters samples by *explanation similarity* (the pairwise Pearson correlation of their
    per-sample SHAP-value vectors), so proteins group by *why* the model scores them rather than
    by their raw features. Row/column color sidebars annotate each sample's class.

    .. versionadded:: 1.1.0

    Warnings
    --------
    * This class requires `SHAP`, which is automatically installed via `pip install aaanalysis[pro]`.

    """
    def __init__(self):
        """
        See Also
        --------
        * :class:`ShapModel`: the logic class whose SHAP values this visualizes.
        * :func:`shap_to_feat_imp`: convert a per-sample SHAP vector into feature impact/importance.
        """

    @staticmethod
    def clustermap(shap_values: ut.ArrayLike2D,
                   labels: ut.ArrayLike1D,
                   names: Optional[List[str]] = None,
                   labels_pred: Optional[ut.ArrayLike1D] = None,
                   dict_color: Optional[dict] = None,
                   method: str = "complete",
                   figsize: Tuple[Union[int, float], Union[int, float]] = (6, 6),
                   cmap: str = "GnBu",
                   vmin: Union[int, float] = -1,
                   vmax: Union[int, float] = 1,
                   tick_labels: int = 4,
                   tree_linewidth: Union[int, float] = 1.0,
                   ):
        """
        Plot an explanation-similarity clustermap of per-sample SHAP-value vectors.

        Samples are clustered by the pairwise Pearson correlation of their per-sample SHAP-value
        vectors, so proteins group by *why* the model scores them (similar feature attributions),
        not merely by their feature values. Hierarchical clustering uses the given linkage
        ``method`` on the correlation matrix. Row and column color sidebars annotate each sample's
        class (``labels_pred`` for rows, ``labels`` for columns), so concordant / discordant blocks
        between predicted and true class are visible.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        shap_values : array-like, shape (n_samples, n_features)
            Per-sample SHAP values (e.g., the ``shap_values`` attribute of a fitted :class:`ShapModel`).
            Rows are samples (proteins), columns are features.
        labels : array-like, shape (n_samples,)
            Class label per sample (typically, 1=positive, 0=negative), used for the column color sidebar.
        names : list of str, optional
            Unique sample names shown on the axes. If ``None``, defaults to ``Protein0``, ``Protein1``, ...
        labels_pred : array-like, shape (n_samples,), optional
            Predicted class label per sample, used for the row color sidebar. If ``None``, ``labels`` is reused.
        dict_color : dict, optional
            Mapping from class label to color for the sidebars. If ``None``, a default house palette is used.
        method : str, default='complete'
            Linkage method for hierarchical clustering (passed to :func:`scipy.cluster.hierarchy.linkage`).
        figsize : tuple, default=(6, 6)
            Figure dimensions (width, height) in inches.
        cmap : str, default='GnBu'
            Colormap for the correlation heatmap.
        vmin : int or float, default=-1
            Minimum correlation value for color scaling.
        vmax : int or float, default=1
            Maximum correlation value for color scaling.
        tick_labels : int, default=4
            Show every ``tick_labels``-th axis tick label (reduces clutter for many samples).
        tree_linewidth : int or float, default=1.0
            Line width of the dendrogram.

        Returns
        -------
        grid : seaborn.matrix.ClusterGrid
            The seaborn ``ClusterGrid``; access the figure via ``grid.fig`` and the heatmap axes via
            ``grid.ax_heatmap``. The clustering linkage is available as
            ``grid.dendrogram_col.linkage`` / ``grid.dendrogram_row.linkage``.

        See Also
        --------
        * :class:`ShapModel` for computing the ``shap_values``.
        * :meth:`ShapModelPlot.get_clusters` for the discrete cluster assignment per sample.

        Examples
        --------
        >>> import numpy as np
        >>> from aaanalysis.explainable_ai_pro import ShapModelPlot
        >>> rng = np.random.default_rng(0)
        >>> shap_values = rng.normal(size=(8, 20))
        >>> labels = [1, 1, 1, 1, 0, 0, 0, 0]
        >>> grid = ShapModelPlot.clustermap(shap_values, labels=labels)
        """
        # Check input
        shap_values, labels = check_match_shap_values_labels(shap_values=shap_values, labels=labels)
        n_samples = shap_values.shape[0]
        names = check_match_names_shap_values(names=names, n_samples=n_samples)
        if labels_pred is None:
            labels_pred = labels
        else:
            labels_pred = ut.check_labels(labels=labels_pred, len_required=n_samples, accept_float=True)
        ut.check_str(name="method", val=method)
        ut.check_figsize(figsize=figsize, accept_none=False)
        ut.check_str(name="cmap", val=cmap)
        ut.check_number_val(name="vmin", val=vmin)
        ut.check_number_val(name="vmax", val=vmax)
        ut.check_number_range(name="tick_labels", val=tick_labels, min_val=1, just_int=True)
        ut.check_number_range(name="tree_linewidth", val=tree_linewidth, min_val=0, just_int=False)
        dict_color = _get_dict_color(labels=labels, labels_pred=labels_pred, dict_color=dict_color)
        # Build correlation matrix and class-color sidebars
        df_cor = comp_shap_correlation(shap_values=shap_values, names=names)
        row_colors = pd.Series([dict_color[x] for x in labels_pred], index=names)
        col_colors = pd.Series([dict_color[x] for x in labels], index=names)
        # Plot
        grid = plot_shap_clustermap(df_cor=df_cor, dict_color=dict_color,
                                    row_colors=row_colors, col_colors=col_colors,
                                    method=method, figsize=figsize, cmap=cmap,
                                    vmin=vmin, vmax=vmax, tick_labels=tick_labels,
                                    tree_linewidth=tree_linewidth)
        return grid

    @staticmethod
    def get_clusters(shap_values: ut.ArrayLike2D,
                     names: Optional[List[str]] = None,
                     method: str = "complete",
                     n_clusters: Optional[int] = None,
                     color_threshold: Optional[Union[int, float]] = None,
                     ) -> pd.DataFrame:
        """
        Assign each sample to a cluster based on SHAP-explanation similarity.

        Clusters the samples with the same correlation-of-SHAP-vectors linkage used by
        :meth:`clustermap`, then cuts the dendrogram either into a fixed ``n_clusters`` or at a
        ``color_threshold`` distance. This is the library-grade, deterministic replacement for the
        original project's dendrogram-color parsing.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        shap_values : array-like, shape (n_samples, n_features)
            Per-sample SHAP values. Rows are samples, columns are features.
        names : list of str, optional
            Unique sample names. If ``None``, defaults to ``Protein0``, ``Protein1``, ...
        method : str, default='complete'
            Linkage method for hierarchical clustering.
        n_clusters : int, optional
            If given, cut the dendrogram into exactly ``n_clusters`` clusters. Mutually exclusive
            with ``color_threshold``.
        color_threshold : int or float, optional
            If given, cut the dendrogram at this distance. Mutually exclusive with ``n_clusters``.
            If both are ``None``, ``n_clusters=2`` is used.

        Returns
        -------
        df_clust : pd.DataFrame, shape (n_samples, 2)
            DataFrame with columns ``name`` and ``cluster`` (1-based cluster id per sample).

        See Also
        --------
        * :meth:`ShapModelPlot.clustermap` for the matching figure.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        shap_values = ut.check_array_like(name="shap_values", val=shap_values,
                                          dtype="numeric", expected_dim=2)
        n_samples = shap_values.shape[0]
        names = check_match_names_shap_values(names=names, n_samples=n_samples)
        ut.check_str(name="method", val=method)
        ut.check_number_range(name="n_clusters", val=n_clusters, min_val=1,
                              max_val=n_samples, just_int=True, accept_none=True)
        ut.check_number_range(name="color_threshold", val=color_threshold, min_val=0,
                              just_int=False, accept_none=True)
        if n_clusters is not None and color_threshold is not None:
            raise ValueError("Pass only one of 'n_clusters' or 'color_threshold', not both.")
        df_cor = comp_shap_correlation(shap_values=shap_values, names=names)
        link = linkage(df_cor.values, method=method)
        if color_threshold is not None:
            clusters = fcluster(link, t=color_threshold, criterion="distance")
        else:
            t = 2 if n_clusters is None else n_clusters
            clusters = fcluster(link, t=t, criterion="maxclust")
        df_clust = pd.DataFrame({"name": names, "cluster": [int(c) for c in clusters]})
        return df_clust
