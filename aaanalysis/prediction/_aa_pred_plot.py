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

    Visualizes the two outputs of a prediction workflow: the model x metric evaluation table
    from :meth:`AAPred.eval` (:meth:`eval`), and the per-sample prediction scores from
    :meth:`AAPred.predict_proba` (:meth:`hist`, :meth:`scatter`, :meth:`cutoff`).

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
        Grouped bar plot of the model x metric evaluation table.

        Bars are grouped by metric and colored by model; cross-validation bars carry error bars
        from ``score_std`` and held-out bars (if present) are drawn hatched.

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
            Mapping ``model -> color``. Defaults to the house categorical palette.
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
        # Resolve layout
        metrics = list(dict.fromkeys(df_eval[ut.COL_METRIC].tolist()))
        models = list(dict.fromkeys(df_eval[ut.COL_MODEL].tolist()))
        principles = list(dict.fromkeys(df_eval[ut.COL_PRINCIPLE].tolist()))
        clist = ut.plot_get_clist_(n_colors=max(len(models), 2))
        dict_color = dict(dict_color) if dict_color is not None else {}
        dict_model_color = {m: dict_color.get(m, clist[i % len(clist)]) for i, m in enumerate(models)}
        # Draw grouped bars
        fig, ax = _new_ax(ax=ax, figsize=figsize)
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
