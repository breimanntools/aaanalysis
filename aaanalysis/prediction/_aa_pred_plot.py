"""
This is a script for the frontend of the AAPredPlot class for visualizing AAPred results.
"""
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import aaanalysis.utils as ut
from ._backend.aa_pred.aa_pred_plot_comparison import plot_comparison_
from ._backend.aa_pred.aa_pred_plot_ranking import plot_ranking_
from ._backend.aa_pred.aa_pred_plot_clustermap import plot_clustermap_


# I Helper Functions
def _new_ax(ax=None, figsize=(6, 5)):
    """Return (fig, ax), creating a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def check_match_scores_labels(scores=None, labels=None):
    """Check that per-sample labels match the length of the scores array."""
    if labels is not None and len(labels) != len(scores):
        raise ValueError(f"'labels' (n={len(labels)}) should match length of 'scores' (n={len(scores)}).")


# II Main Functions
class AAPredPlot:
    """
    Plotting class for :class:`AAPred` evaluation and prediction results [Breimann25]_.

    The single home for prediction figures, grouped into three types:

    - **Positional** (one protein along its sequence): :meth:`window`, :meth:`domain`.
    - **Cohort** (many proteins / sequence-level scores): :meth:`hist`, :meth:`ranking`,
      :meth:`scatter`, :meth:`cutoff`, :meth:`clustermap`.
    - **Evaluation** (models / feature sets): :meth:`eval`, :meth:`comparison`.

    .. versionadded:: 1.1.0

    See Also
    --------
    * :class:`AAPred`: the logic class whose results this visualizes.
    * :func:`aaanalysis.plot_rank` for the per-protein rank scatter companion.
    """

    def __init__(self):
        """
        See Also
        --------
        * :class:`AAPred`: the logic class whose results this visualizes.

        Examples
        --------
        .. include:: examples/aapred_plot.rst
        """

    @staticmethod
    def eval(df_eval: pd.DataFrame,
             ax: Optional[Axes] = None,
             figsize: Tuple[Union[int, float], Union[int, float]] = (7, 5),
             dict_color: Optional[Dict[str, str]] = None,
             baseline: Optional[Union[int, float]] = None,
             ylabel: str = "Score",
             ) -> Tuple[Figure, Axes]:
        """
        Grouped bar plot comparing methods across metrics (hue = model).

        Each metric is a group on the x-axis and each model is a colored bar (the hue), so the
        different **methods** are compared side by side. Cross-validation bars carry ``score_std``
        error bars and held-out bars are hatched; pass ``baseline`` for a chance line. This is the
        plot for **method** comparison — to compare **CPP parameter combinations** (parameter
        ranges) instead, use the feature-optimization protocol :func:`aaanalysis.pipe.find_features`
        and its evaluation-grid heatmap :func:`aaanalysis.pipe.plot_eval`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_rows, 5)
            Evaluation table from :meth:`AAPred.eval` with columns ``model``, ``metric``,
            ``principle``, ``score``, and ``score_std``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, default=(7, 5)
            Figure size when ``ax`` is ``None``.
        dict_color : dict, optional
            Mapping ``model -> color`` (the bar hue). Defaults to the house categorical palette.
        baseline : int or float, optional
            If given, a horizontal reference line is drawn at this score (e.g. ``0.5`` for chance).
        ylabel : str, default="Score"
            Label for the y-axis.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the grouped bar plot.

        See Also
        --------
        * :func:`aaanalysis.pipe.find_features` and :func:`aaanalysis.pipe.plot_eval` for
          comparing CPP parameter combinations (a heatmap over the parameter grid).

        Examples
        --------
        .. include:: examples/aapred_plot_eval.rst
        """
        # Check input
        cols = ut.COLS_EVAL_PRED
        ut.check_df(name="df_eval", df=df_eval, cols_required=cols)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        if baseline is not None:
            ut.check_number_val(name="baseline", val=baseline, just_int=False)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # Grouped bar plot: metrics on the x-axis, one hued bar per model
        metrics = list(dict.fromkeys(df_eval[ut.COL_METRIC].tolist()))
        models = list(dict.fromkeys(df_eval[ut.COL_MODEL].tolist()))
        principles = list(dict.fromkeys(df_eval[ut.COL_PRINCIPLE].tolist()))
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        clist = ut.plot_get_clist_(n_colors=max(len(models), 2))
        dict_color = dict(dict_color) if dict_color is not None else {}
        dict_model_color = {m: dict_color.get(m, clist[i % len(clist)]) for i, m in enumerate(models)}
        n_groups = len(models) * len(principles)
        width = 0.8 / max(n_groups, 1)
        x = np.arange(len(metrics))
        idx = 0
        for model in models:
            for principle in principles:
                sub = df_eval[(df_eval[ut.COL_MODEL] == model) & (df_eval[ut.COL_PRINCIPLE] == principle)]
                heights = [float(sub[sub[ut.COL_METRIC] == m][ut.COL_SCORE].mean()) for m in metrics]
                errs = [float(sub[sub[ut.COL_METRIC] == m][ut.COL_SCORE_STD].mean()) for m in metrics]
                errs = [0 if np.isnan(e) else e for e in errs]
                hatch = "//" if principle == ut.STR_PRINCIPLE_HOLDOUT else None
                label = model if principle == principles[0] else f"{model} ({principle})"
                if len(principles) > 1:
                    label = f"{model} ({principle})"
                ax.bar(x + (idx - (n_groups - 1) / 2) * width, heights, width=width,
                       color=dict_model_color[model], edgecolor="black", linewidth=0.6,
                       hatch=hatch, yerr=errs, capsize=2.5, label=label)
                idx += 1
        if baseline is not None:
            ax.axhline(baseline, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def comparison(df_eval: pd.DataFrame,
                   group: str = "group",
                   condition: str = "condition",
                   value: str = "value",
                   baseline: Optional[Union[int, float]] = 50,
                   baseline_label: Optional[str] = None,
                   annotate: bool = True,
                   annotation_fmt: Optional[str] = None,
                   group_order: Optional[List[str]] = None,
                   condition_order: Optional[List[str]] = None,
                   colors: Optional[Union[List[str], Dict[str, str]]] = None,
                   bar_width: Union[int, float] = 0.8,
                   ax: Optional[Axes] = None,
                   figsize: Tuple[Union[int, float], Union[int, float]] = (7, 4.2),
                   xlabel: Optional[str] = None,
                   ylabel: str = "Score",
                   title: Optional[str] = None,
                   ylim: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                   fontsize_annotations: Union[int, float] = 10,
                   xtick_rotation: Union[int, float] = 0,
                   ) -> Tuple[Figure, Axes]:
        """
        Plot a grouped method x condition comparison barplot with value labels and a baseline.

        Draws the recurring "benchmark result" figure from a tidy eval frame in one call: each
        ``condition`` is an x-axis cluster, each ``group`` a colored bar within it (auto offsets /
        widths for *N* groups), with optional per-bar value labels and an optional dashed chance /
        baseline line.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_rows, n_cols)
            Tidy (long-form) frame with one row per (``group``, ``condition``); must contain the
            ``group``, ``condition``, and ``value`` columns. Repeated cells are averaged.
        group : str, default="group"
            Column whose distinct values become the colored bars within each cluster (the legend).
        condition : str, default="condition"
            Column whose distinct values become the x-axis clusters.
        value : str, default="value"
            Column with the numeric bar heights (e.g. balanced accuracy in percent).
        baseline : int or float, optional
            y-value of a dashed chance / baseline line. If ``None``, no line is drawn.
        baseline_label : str, optional
            Legend label for the baseline. ``None`` generates ``"chance (<baseline>)"``; ``""`` draws
            the line without a legend entry.
        annotate : bool, default=True
            If ``True``, write each bar's value above it.
        annotation_fmt : str, optional
            Format string for the value labels; if ``None``, chosen from the data scale.
        group_order : list of str, optional
            Order of the groups (bars within a cluster). Defaults to first-appearance order.
        condition_order : list of str, optional
            Order of the conditions (x-axis clusters). Defaults to first-appearance order.
        colors : list of str or dict, optional
            Bar colors aligned to ``group_order`` or a ``group -> color`` dict; defaults to the
            house categorical palette.
        bar_width : int or float, default=0.8
            Total width of each cluster (split across the groups). Must be in (0, 1].
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, default=(7, 4.2)
            Figure size when ``ax`` is ``None``.
        xlabel : str, optional
            x-axis label.
        ylabel : str, default="Score"
            y-axis label.
        title : str, optional
            Axes title.
        ylim : tuple, optional
            y-axis limits ``(bottom, top)``; if ``None``, the top leaves room for the value labels.
        fontsize_annotations : int or float, default=10
            Font size of the per-bar value labels.
        xtick_rotation : int or float, default=0
            Rotation (degrees) of the cluster tick labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the grouped comparison barplot.

        Examples
        --------
        .. include:: examples/aapred_plot_comparison.rst
        """
        # Check input
        ut.check_str(name="group", val=group)
        ut.check_str(name="condition", val=condition)
        ut.check_str(name="value", val=value)
        if len({group, condition, value}) < 3:
            raise ValueError(f"'group', 'condition', and 'value' should be three distinct columns, "
                             f"got group={group!r}, condition={condition!r}, value={value!r}.")
        ut.check_df(name="df_eval", df=df_eval, cols_required=[group, condition, value])
        if len(df_eval) == 0:
            raise ValueError("'df_eval' (0 rows) should contain at least one row.")
        if not pd.api.types.is_numeric_dtype(df_eval[value]):
            raise ValueError(f"'{value}' column of 'df_eval' should be numeric, "
                             f"got dtype '{df_eval[value].dtype}'.")
        ut.check_number_val(name="baseline", val=baseline, accept_none=True, just_int=False)
        ut.check_str(name="baseline_label", val=baseline_label, accept_none=True)
        ut.check_bool(name="annotate", val=annotate)
        ut.check_str(name="annotation_fmt", val=annotation_fmt, accept_none=True)
        ut.check_list_like(name="group_order", val=group_order, accept_none=True)
        ut.check_list_like(name="condition_order", val=condition_order, accept_none=True)
        ut.check_number_range(name="bar_width", val=bar_width, min_val=0, max_val=1, just_int=False)
        if bar_width == 0:
            raise ValueError("'bar_width' should be greater than 0.")
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        ut.check_number_range(name="fontsize_annotations", val=fontsize_annotations, min_val=0,
                              just_int=False)
        ut.check_number_val(name="xtick_rotation", val=xtick_rotation, just_int=False)
        if ylim is not None:
            ut.check_lim(name="ylim", val=ylim)
        # Plot
        fig, ax = plot_comparison_(df_eval=df_eval, group=group, condition=condition, value=value,
                                   baseline=baseline, baseline_label=baseline_label, annotate=annotate,
                                   annotation_fmt=annotation_fmt, group_order=group_order,
                                   condition_order=condition_order, colors=colors, bar_width=bar_width,
                                   ax=ax, figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title,
                                   ylim=ylim, fontsize_annotations=fontsize_annotations,
                                   xtick_rotation=xtick_rotation)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def ranking(df_pred: pd.DataFrame,
                col_name: str = "name",
                col_score: str = "score",
                col_group: Optional[str] = None,
                col_std: Optional[str] = None,
                colors: Optional[Dict[str, str]] = None,
                cutoffs: Optional[Tuple[Union[int, float], ...]] = (50, 80),
                top_n: Optional[int] = None,
                ascending: bool = False,
                ax: Optional[Axes] = None,
                figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                xlabel: str = "Prediction score",
                title: Optional[str] = None,
                ) -> Tuple[Figure, Axes]:
        """
        Plot ranked candidates as horizontal bars colored by class, with cut-off lines.

        Ranks proteins/samples by a prediction ``col_score`` (highest on top) and draws one
        horizontal bar each, colored by ``col_group`` (e.g. substrate vs non-substrate), with
        optional per-item error bars (``col_std``) and dashed confidence cut-off lines. The
        figure height grows with the number of items so each bar keeps a constant height.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_pred : pd.DataFrame, shape (n_samples, n_cols)
            Per-sample prediction frame; must contain ``col_name`` and ``col_score`` (and
            ``col_group`` / ``col_std`` when given).
        col_name : str, default="name"
            Column with the per-item labels shown as y-tick labels (e.g. gene names).
        col_score : str, default="score"
            Column with the numeric prediction score used to rank and size the bars.
        col_group : str, optional
            Column whose distinct values color the bars (adds a class legend). If ``None``, a
            single color is used.
        col_std : str, optional
            Column with per-item standard deviations, drawn as horizontal error bars.
        colors : dict, optional
            A ``group -> color`` mapping; defaults to the house categorical palette.
        cutoffs : tuple, optional
            x-positions of dashed confidence cut-off lines. ``None`` or empty draws none.
        top_n : int, optional
            If given, keep only the top ``top_n`` ranked items.
        ascending : bool, default=False
            Sort order of the score; ``False`` ranks the highest score first (on top).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, optional
            Figure size when ``ax`` is ``None``. If ``None``, the height scales with the number
            of items (``0.22 * n + 1``) and the width defaults to 5.
        xlabel : str, default="Prediction score"
            x-axis label.
        title : str, optional
            Axes title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the ranked-candidate bars.

        Examples
        --------
        .. include:: examples/aapred_plot_ranking.rst
        """
        # Check input
        ut.check_str(name="col_name", val=col_name)
        ut.check_str(name="col_score", val=col_score)
        ut.check_str(name="col_group", val=col_group, accept_none=True)
        ut.check_str(name="col_std", val=col_std, accept_none=True)
        cols_required = [c for c in [col_name, col_score, col_group, col_std] if c is not None]
        ut.check_df(name="df_pred", df=df_pred, cols_required=cols_required)
        if len(df_pred) == 0:
            raise ValueError("'df_pred' (0 rows) should contain at least one row.")
        if not pd.api.types.is_numeric_dtype(df_pred[col_score]):
            raise ValueError(f"'{col_score}' column of 'df_pred' should be numeric.")
        ut.check_bool(name="ascending", val=ascending)
        if top_n is not None:
            ut.check_number_range(name="top_n", val=top_n, min_val=1, just_int=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        # Plot
        fig, ax = plot_ranking_(df_pred=df_pred, col_name=col_name, col_score=col_score,
                                col_group=col_group, col_std=col_std, colors=colors,
                                cutoffs=cutoffs, top_n=top_n, ascending=ascending, ax=ax,
                                figsize=figsize, xlabel=xlabel, title=title)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def clustermap(data: ut.ArrayLike2D,
                   names: Optional[List[str]] = None,
                   labels: Optional[ut.ArrayLike1D] = None,
                   colors: Optional[Dict[str, str]] = None,
                   cmap: str = "GnBu",
                   figsize: Tuple[Union[int, float], Union[int, float]] = (9, 9),
                   cbar_label: str = "Pearson correlation (r)",
                   title: Optional[str] = None,
                   ) -> Tuple[Figure, Axes]:
        """
        Cluster samples by explanation similarity (correlation of per-sample importance vectors).

        Groups samples by *why* the model scores them: it correlates their per-sample
        importance / SHAP vectors (Pearson) and draws a hierarchically-clustered heatmap of the
        sample x sample correlation, with optional row/column class-color sidebars. Because it
        consumes provided importance vectors, it needs no optional dependency; compute the SHAP
        vectors with :class:`ShapModel` (``pro``) and pass them in.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Per-sample importance/explanation vectors (e.g. SHAP values), one row per sample.
        names : list of str, optional
            Per-sample labels shown as tick labels. Defaults to positional indices.
        labels : array-like, shape (n_samples,), optional
            Per-sample class labels used to color the row/column sidebars (adds a class legend).
        colors : dict, optional
            A ``label -> color`` mapping for the sidebars; defaults to the house palette.
        cmap : str, default="GnBu"
            Colormap for the correlation heatmap.
        figsize : tuple, default=(9, 9)
            Figure size.
        cbar_label : str, default="Pearson correlation (r)"
            Label of the colorbar.
        title : str, optional
            Figure title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The clustermap figure.
        ax : matplotlib.axes.Axes
            The heatmap axes of the clustermap.

        Examples
        --------
        .. include:: examples/aapred_plot_clustermap.rst
        """
        # Check input
        data = ut.check_X(X=data, min_n_samples=2, min_n_features=1)
        if names is not None:
            ut.check_list_like(name="names", val=names)
            if len(names) != data.shape[0]:
                raise ValueError(f"'names' (n={len(names)}) should match n_samples ({data.shape[0]}).")
        if labels is not None:
            # Labels here are purely cosmetic (sidebar coloring), so any hashable class
            # values are allowed (e.g. "substrate"/"non-substrate"), not only integers.
            labels = ut.check_list_like(name="labels", val=labels, accept_none=False)
            if len(labels) != data.shape[0]:
                raise ValueError(f"'labels' (n={len(labels)}) should match n_samples ({data.shape[0]}).")
        ut.check_str(name="cmap", val=cmap)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_str(name="cbar_label", val=cbar_label, accept_none=True)
        ut.check_str(name="title", val=title, accept_none=True)
        # Plot
        fig, ax = plot_clustermap_(data=data, names=names, labels=labels, colors=colors,
                                   cmap=cmap, figsize=figsize, cbar_label=cbar_label, title=title)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def hist(scores: ut.ArrayLike1D,
             labels: Optional[ut.ArrayLike1D] = None,
             ax: Optional[Axes] = None,
             figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4.5),
             bins: int = 20,
             thresholds: Optional[Union[int, float, List[Union[int, float]]]] = None,
             dict_color: Optional[Dict[Union[int, str], str]] = None,
             xlabel: str = "Prediction score",
             ylabel: str = "Number of samples",
             ) -> Tuple[Figure, Axes]:
        """
        Class-separated histogram of per-sample prediction scores.

        Shows how the positive and negative classes are distributed across the score range, with
        optional decision thresholds — the standard sanity check for a deployed classifier's margin.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        scores : array-like, shape (n_samples,)
            Per-sample prediction scores (e.g. from :meth:`AAPred.predict_proba`).
        labels : array-like, shape (n_samples,), optional
            Class labels used to color/separate the distribution. If ``None``, one histogram is drawn.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, default=(6, 4.5)
            Figure size when ``ax`` is ``None``.
        bins : int, default=20
            Number of histogram bins.
        thresholds : int, float, or list, optional
            One or more score values drawn as vertical dashed lines.
        dict_color : dict, optional
            Mapping ``label -> color``. Defaults to the locked positive/negative sample palette.
        xlabel, ylabel : str
            Axis labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the histogram.

        Examples
        --------
        .. include:: examples/aapred_plot_hist.rst
        """
        # Check input
        scores = ut.check_array_like(name="scores", val=scores, expected_dim=1)
        labels = ut.check_labels(labels=labels) if labels is not None else None
        check_match_scores_labels(scores=scores, labels=labels)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="bins", val=bins, min_val=1, just_int=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        list_thresholds = []
        if thresholds is not None:
            list_thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
            for i, t in enumerate(list_thresholds):
                ut.check_number_val(name=f"thresholds[{i}]", val=t, just_int=False)
        # Resolve colors
        dict_color = dict(dict_color) if dict_color is not None else {}
        default_cycle = [ut.COLOR_POS, ut.COLOR_NEG, ut.COLOR_REL_NEG, ut.COLOR_UNL]
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        bin_edges = np.linspace(float(np.min(scores)), float(np.max(scores)), bins + 1)
        if labels is None:
            ax.hist(scores, bins=bin_edges, color=ut.COLOR_POS, edgecolor="black", linewidth=0.6)
        else:
            for i, lab in enumerate(sorted(set(labels))):
                color = dict_color.get(lab, default_cycle[i % len(default_cycle)])
                ax.hist(np.asarray(scores)[np.asarray(labels) == lab], bins=bin_edges, alpha=0.7,
                        color=color, edgecolor="black", linewidth=0.6, label=str(lab))
            ax.legend(frameon=False)
        for t in list_thresholds:
            ax.axvline(t, color="0.3", linestyle="--", linewidth=1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def scatter(scores_x: ut.ArrayLike1D,
                scores_y: ut.ArrayLike1D,
                labels: Optional[ut.ArrayLike1D] = None,
                ax: Optional[Axes] = None,
                figsize: Tuple[Union[int, float], Union[int, float]] = (5.5, 5.5),
                dict_color: Optional[Dict[Union[int, str], str]] = None,
                marker_size: Union[int, float] = 30,
                diagonal: bool = True,
                xlabel: str = "Predictor 1 score",
                ylabel: str = "Predictor 2 score",
                ) -> Tuple[Figure, Axes]:
        """
        2D scatter comparing per-sample scores of two predictors.

        Each point is one sample placed at ``(scores_x, scores_y)`` and optionally colored by class;
        the ``y = x`` line marks agreement, so systematic disagreement between the predictors is visible.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        scores_x : array-like, shape (n_samples,)
            Per-sample scores of the first predictor (x-axis).
        scores_y : array-like, shape (n_samples,)
            Per-sample scores of the second predictor (y-axis).
        labels : array-like, shape (n_samples,), optional
            Class labels used to color the points.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, default=(5.5, 5.5)
            Figure size when ``ax`` is ``None``.
        dict_color : dict, optional
            Mapping ``label -> color``. Defaults to the locked positive/negative sample palette.
        marker_size : int or float, default=30
            Scatter marker size.
        diagonal : bool, default=True
            If ``True``, draw the ``y = x`` agreement line.
        xlabel, ylabel : str
            Axis labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the scatter plot.

        Examples
        --------
        .. include:: examples/aapred_plot_scatter.rst
        """
        # Check input
        scores_x = ut.check_array_like(name="scores_x", val=scores_x, expected_dim=1)
        scores_y = ut.check_array_like(name="scores_y", val=scores_y, expected_dim=1)
        if len(scores_x) != len(scores_y):
            raise ValueError(f"'scores_x' (n={len(scores_x)}) and 'scores_y' (n={len(scores_y)}) should match in length.")
        labels = ut.check_labels(labels=labels) if labels is not None else None
        check_match_scores_labels(scores=scores_x, labels=labels)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_dict_color(name="dict_color", val=dict_color, accept_none=True)
        ut.check_number_range(name="marker_size", val=marker_size, min_val=0, just_int=False)
        ut.check_bool(name="diagonal", val=diagonal)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # Resolve colors
        dict_color = dict(dict_color) if dict_color is not None else {}
        default_cycle = [ut.COLOR_POS, ut.COLOR_NEG, ut.COLOR_REL_NEG, ut.COLOR_UNL]
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        if diagonal:
            lo = float(min(np.min(scores_x), np.min(scores_y)))
            hi = float(max(np.max(scores_x), np.max(scores_y)))
            ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1, zorder=0)
        if labels is None:
            ax.scatter(scores_x, scores_y, s=marker_size, color=ut.COLOR_POS,
                       edgecolors="white", linewidths=0.3)
        else:
            sx, sy, la = np.asarray(scores_x), np.asarray(scores_y), np.asarray(labels)
            for i, lab in enumerate(sorted(set(labels))):
                color = dict_color.get(lab, default_cycle[i % len(default_cycle)])
                mask = la == lab
                ax.scatter(sx[mask], sy[mask], s=marker_size, color=color,
                           edgecolors="white", linewidths=0.3, label=str(lab))
            ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def cutoff(scores: ut.ArrayLike1D,
               ax: Optional[Axes] = None,
               figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4.5),
               n_steps: int = 101,
               color: Optional[str] = None,
               thresholds: Optional[Union[int, float, List[Union[int, float]]]] = None,
               xlabel: str = "Score cutoff",
               ylabel: str = "Samples above cutoff [%]",
               ) -> Tuple[Figure, Axes]:
        """
        Line plot of the percentage of samples scoring at or above each cutoff.

        The (non-increasing) curve is the survival function of the scores — it makes the trade-off
        between a strict and a permissive deployment threshold directly readable.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        scores : array-like, shape (n_samples,)
            Per-sample prediction scores (e.g. from :meth:`AAPred.predict_proba`).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, default=(6, 4.5)
            Figure size when ``ax`` is ``None``.
        n_steps : int, default=101
            Number of evenly spaced cutoffs between the min and max score.
        color : str, optional
            Line color. Defaults to the house feature-positive color.
        thresholds : int, float, or list, optional
            One or more cutoffs drawn as vertical dashed lines.
        xlabel, ylabel : str
            Axis labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the cutoff line.

        Examples
        --------
        .. include:: examples/aapred_plot_cutoff.rst
        """
        # Check input
        scores = ut.check_array_like(name="scores", val=scores, expected_dim=1)
        if len(scores) == 0:
            raise ValueError("'scores' (0 values) should contain at least one score.")
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_number_range(name="n_steps", val=n_steps, min_val=2, just_int=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        list_thresholds = []
        if thresholds is not None:
            list_thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
            for i, t in enumerate(list_thresholds):
                ut.check_number_val(name=f"thresholds[{i}]", val=t, just_int=False)
        # Compute
        scores = np.asarray(scores, dtype=float)
        cutoffs = np.linspace(float(np.min(scores)), float(np.max(scores)), n_steps)
        pct = np.array([100.0 * np.mean(scores >= c) for c in cutoffs])
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        ax.plot(cutoffs, pct, color=color or ut.COLOR_FEAT_POS, linewidth=2)
        for t in list_thresholds:
            ax.axvline(t, color="0.3", linestyle="--", linewidth=1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def window(df_window: pd.DataFrame,
               entry: Optional[str] = None,
               list_annotations: Optional[List[Dict]] = None,
               threshold: Optional[Union[int, float]] = None,
               ax: Optional[Axes] = None,
               figsize: Tuple[Union[int, float], Union[int, float]] = (10, 4),
               color: Optional[str] = None,
               xlabel: str = "Residue position",
               ylabel: str = "Prediction score",
               ) -> Tuple[Figure, Axes]:
        """
        Per-residue prediction profile from :meth:`AAPred.predict_window`.

        Draws the sliding-window score along the sequence, with an optional decision threshold and
        optional per-residue annotation tracks (topology, pLDDT, domains, ...) shown as color strips
        below the profile. Annotation values are user-provided arrays, so any track can be added.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_window : pd.DataFrame
            Output of :meth:`AAPred.predict_window` with columns ``entry``, ``position``, ``score``.
        entry : str, optional
            Protein to plot. Required only if ``df_window`` contains more than one ``entry``.
        list_annotations : list of dict, optional
            Per-residue annotation tracks. Each dict has ``values`` (array aligned to the plotted
            positions), ``label`` (str), and optional ``cmap`` (default ``viridis``).
        threshold : int or float, optional
            Score drawn as a horizontal dashed line.
        ax : matplotlib.axes.Axes, optional
            Axes to draw the profile on. If ``None``, a new figure is created (with extra track
            axes below when ``list_annotations`` is given).
        figsize : tuple, default=(10, 4)
            Figure size when ``ax`` is ``None``.
        color : str, optional
            Line color. Defaults to the house feature-negative (blue) color.
        xlabel, ylabel : str
            Axis labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The profile axes (the top axes when annotation tracks are present).

        Examples
        --------
        .. include:: examples/aapred_plot_window.rst
        """
        # Check input
        ut.check_df(name="df_window", df=df_window,
                    cols_required=[ut.COL_ENTRY, ut.COL_RESIDUE_POS, ut.COL_SCORE])
        ut.check_str(name="entry", val=entry, accept_none=True)
        ut.check_list_like(name="list_annotations", val=list_annotations, accept_none=True)
        if threshold is not None:
            ut.check_number_val(name="threshold", val=threshold, just_int=False)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # Resolve the entry
        entries = list(dict.fromkeys(df_window[ut.COL_ENTRY].tolist()))
        if entry is None:
            if len(entries) > 1:
                raise ValueError(f"'df_window' contains multiple entries {entries}; pass 'entry='.")
            entry = entries[0]
        elif entry not in entries:
            raise ValueError(f"'entry' ({entry}) not in 'df_window' entries {entries}.")
        sub = df_window[df_window[ut.COL_ENTRY] == entry].sort_values(ut.COL_RESIDUE_POS)
        pos = sub[ut.COL_RESIDUE_POS].to_numpy()
        score = sub[ut.COL_SCORE].to_numpy()
        # Layout: profile axes (+ track axes when annotations given)
        tracks = list(list_annotations) if list_annotations is not None else []
        if ax is None:
            if tracks:
                fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=figsize, sharex=True,
                                         gridspec_kw={"height_ratios": [6] + [0.6] * len(tracks)})
                ax, track_axes = axes[0], list(axes[1:])
            else:
                fig, ax = plt.subplots(figsize=figsize)
                track_axes = []
        else:
            fig = ax.figure
            track_axes = []
        # Draw the profile
        ax.plot(pos, score, color=color or ut.COLOR_FEAT_NEG, linewidth=1.2)
        if threshold is not None:
            ax.axhline(threshold, color="0.4", linestyle="--", linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_xlim(pos.min(), pos.max())
        # Draw annotation tracks
        for tax, track in zip(track_axes, tracks):
            values = np.asarray(track["values"], dtype=float).reshape(1, -1)
            tax.imshow(values, aspect="auto", cmap=track.get("cmap", "viridis"),
                       extent=[pos.min(), pos.max(), 0, 1])
            tax.set_yticks([])
            tax.set_ylabel(track.get("label", ""), rotation=0, ha="right", va="center", fontsize=8)
        (track_axes[-1] if track_axes else ax).set_xlabel(xlabel)
        if not track_axes:
            sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def domain(df_domain: pd.DataFrame,
               entry: Optional[str] = None,
               ax: Optional[Axes] = None,
               figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4.5),
               color: Optional[str] = None,
               xlabel: str = "Boundary offset [residues]",
               ylabel: str = "Prediction score",
               ) -> Tuple[Figure, Axes]:
        """
        Domain boundary-sensitivity plot from :meth:`AAPred.predict_domain`.

        Shows the prediction score as a function of the boundary shift, marking the highest-scoring
        definition — so how strongly the score depends on the exact domain boundary is visible.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_domain : pd.DataFrame
            Output of :meth:`AAPred.predict_domain` with columns ``entry``, ``offset``, ``score``,
            ``is_best``.
        entry : str, optional
            Protein to plot. Required only if ``df_domain`` contains more than one ``entry``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, default=(6, 4.5)
            Figure size when ``ax`` is ``None``.
        color : str, optional
            Line color. Defaults to the house feature-positive color.
        xlabel, ylabel : str
            Axis labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the sensitivity curve.

        Examples
        --------
        .. include:: examples/aapred_plot_domain.rst
        """
        # Check input
        ut.check_df(name="df_domain", df=df_domain,
                    cols_required=[ut.COL_ENTRY, ut.COL_OFFSET, ut.COL_SCORE])
        ut.check_str(name="entry", val=entry, accept_none=True)
        ut.check_ax(ax=ax, accept_none=True)
        ut.check_figsize(figsize=figsize, accept_none=True)
        ut.check_color(name="color", val=color, accept_none=True)
        ut.check_str(name="xlabel", val=xlabel, accept_none=True)
        ut.check_str(name="ylabel", val=ylabel, accept_none=True)
        # Resolve the entry
        entries = list(dict.fromkeys(df_domain[ut.COL_ENTRY].tolist()))
        if entry is None:
            if len(entries) > 1:
                raise ValueError(f"'df_domain' contains multiple entries {entries}; pass 'entry='.")
            entry = entries[0]
        elif entry not in entries:
            raise ValueError(f"'entry' ({entry}) not in 'df_domain' entries {entries}.")
        sub = df_domain[df_domain[ut.COL_ENTRY] == entry].sort_values(ut.COL_OFFSET)
        offsets = sub[ut.COL_OFFSET].to_numpy()
        score = sub[ut.COL_SCORE].to_numpy()
        # Draw
        fig, ax = _new_ax(ax=ax, figsize=figsize)
        ax.plot(offsets, score, color=color or ut.COLOR_FEAT_POS, marker="o", linewidth=1.6)
        i_best = int(np.argmax(score))
        ax.scatter([offsets[i_best]], [score[i_best]], s=110, marker="*",
                   color=ut.COLOR_FEAT_POS, edgecolors="black", zorder=5, label="best")
        ax.axvline(0, color="0.6", linestyle="--", linewidth=1)  # annotated boundary
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        return ut.FigAxResult(fig, ax)
