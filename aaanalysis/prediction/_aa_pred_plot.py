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
# Sample-prediction plot kinds dispatched by :meth:`AAPredPlot.predict`.
LIST_PREDICT_KINDS = ["window", "domain", "hist", "ranking", "scatter", "cutoff", "clustermap"]
# Evaluation plot kinds dispatched by :meth:`AAPredPlot.eval`.
LIST_EVAL_KINDS = ["eval", "comparison"]
# Per-kind figure-size defaults used when ``figsize=None``.
_DICT_PREDICT_FIGSIZE = {"window": (10, 4), "domain": (6, 4.5), "hist": (6, 4.5),
                         "ranking": None, "scatter": (5.5, 5.5), "cutoff": (6, 4.5),
                         "clustermap": (9, 9)}


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


def _default(val, fallback):
    """Return ``val`` unless it is ``None``, then ``fallback`` (per-kind default resolution)."""
    return fallback if val is None else val


# II Main Functions
class AAPredPlot:
    """
    Plotting class for :class:`AAPred` evaluation and prediction results [Breimann25]_.

    The single home for prediction figures, dispatched by ``kind`` from two methods:

    - :meth:`predict` visualizes **sample predictions**: positional profiles
      (``kind='window'``/``'domain'``), score cohorts (``kind='hist'``/``'ranking'``/
      ``'scatter'``/``'cutoff'``), and explanation similarity (``kind='clustermap'``).
    - :meth:`eval` visualizes **model/feature-set evaluation**: metric bars per model
      (``kind='eval'``) and grouped benchmark comparisons (``kind='comparison'``).

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
    def predict(data: Union[pd.DataFrame, ut.ArrayLike1D, ut.ArrayLike2D],
                kind: str = "window",
                ax: Optional[Axes] = None,
                figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                entry: Optional[str] = None,
                list_annotations: Optional[List[Dict]] = None,
                threshold: Optional[Union[int, float]] = None,
                color: Optional[str] = None,
                labels: Optional[ut.ArrayLike1D] = None,
                bins: int = 20,
                thresholds: Optional[Union[int, float, List[Union[int, float]]]] = None,
                dict_color: Optional[Dict[Union[int, str], str]] = None,
                col_name: str = "name",
                col_score: str = "score",
                col_group: Optional[str] = None,
                col_std: Optional[str] = None,
                colors: Optional[Dict[str, str]] = None,
                cutoffs: Optional[Tuple[Union[int, float], ...]] = (50, 80),
                top_n: Optional[int] = None,
                ascending: bool = False,
                title: Optional[str] = None,
                scores_y: Optional[ut.ArrayLike1D] = None,
                marker_size: Union[int, float] = 30,
                diagonal: bool = True,
                n_steps: int = 101,
                names: Optional[List[str]] = None,
                cmap: str = "GnBu",
                cbar_label: str = "Pearson correlation (r)",
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                ) -> Tuple[Figure, Axes]:
        """
        Visualize sample predictions, dispatched by ``kind``.

        One entry point for every sample-prediction figure; ``kind`` selects the renderer and
        ``data`` is its primary input:

        * ``'window'`` — per-residue profile from :meth:`AAPred.predict` (``level='window'``);
          ``data`` is the ``df_window`` frame (columns ``entry``, ``position``, ``score``).
          Uses ``entry``, ``list_annotations``, ``threshold``, ``color``, ``xlabel``, ``ylabel``.
        * ``'domain'`` — boundary-sensitivity curve from :meth:`AAPred.predict` (``level='domain'``);
          ``data`` is the ``df_domain`` frame (columns ``entry``, ``offset``, ``score``, ``is_best``).
          Uses ``entry``, ``color``, ``xlabel``, ``ylabel``.
        * ``'hist'`` — class-separated histogram of per-sample scores; ``data`` is the ``scores``
          array. Uses ``labels``, ``bins``, ``thresholds``, ``dict_color``, ``xlabel``, ``ylabel``.
        * ``'ranking'`` — ranked-candidate horizontal bars; ``data`` is a per-sample ``df_pred``.
          Uses ``col_name``, ``col_score``, ``col_group``, ``col_std``, ``colors``, ``cutoffs``,
          ``top_n``, ``ascending``, ``xlabel``, ``title``.
        * ``'scatter'`` — two-predictor agreement scatter; ``data`` is ``scores_x`` and the required
          ``scores_y`` the y-axis. Uses ``labels``, ``dict_color``, ``marker_size``, ``diagonal``,
          ``xlabel``, ``ylabel``.
        * ``'cutoff'`` — survival curve of the scores; ``data`` is the ``scores`` array. Uses
          ``n_steps``, ``color``, ``thresholds``, ``xlabel``, ``ylabel``.
        * ``'clustermap'`` — explanation-similarity clustermap; ``data`` is the per-sample
          importance matrix. Uses ``names``, ``labels``, ``colors``, ``cmap``, ``cbar_label``,
          ``title``, ``figsize`` (``ax`` is ignored: the clustermap owns its figure).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : pd.DataFrame or array-like
            Primary input for the selected ``kind`` (see above): a prediction frame
            (``'window'``/``'domain'``/``'ranking'``), a per-sample score array
            (``'hist'``/``'scatter'``/``'cutoff'``), or an importance matrix (``'clustermap'``).
        kind : str, default="window"
            Which sample-prediction figure to draw; one of ``window``, ``domain``, ``hist``,
            ``ranking``, ``scatter``, ``cutoff``, ``clustermap``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created (ignored for
            ``kind='clustermap'``, which always creates its own figure).
        figsize : tuple, optional
            Figure size when ``ax`` is ``None``. If ``None``, a per-kind default is used.
        entry : str, optional
            (``kind='window'``/``'domain'``) Protein to plot; required only when ``data`` holds
            more than one ``entry``.
        list_annotations : list of dict, optional
            (``kind='window'``) Per-residue annotation tracks; each dict has ``values`` (array
            aligned to the plotted positions), ``label`` (str) and optional ``cmap``.
        threshold : int or float, optional
            (``kind='window'``) Score drawn as a horizontal dashed decision line.
        color : str, optional
            (``kind='window'``/``'domain'``/``'cutoff'``) Line color; defaults to a house color.
        labels : array-like, optional
            (``kind='hist'``/``'scatter'``/``'clustermap'``) Per-sample class labels used to
            color/separate the data (adds a class legend / sidebars).
        bins : int, default=20
            (``kind='hist'``) Number of histogram bins.
        thresholds : int, float, or list, optional
            (``kind='hist'``/``'cutoff'``) One or more score values drawn as vertical dashed lines.
        dict_color : dict, optional
            (``kind='hist'``/``'scatter'``) Mapping ``label -> color``; defaults to the locked
            positive/negative sample palette.
        col_name : str, default="name"
            (``kind='ranking'``) Column with the per-item labels shown as y-tick labels.
        col_score : str, default="score"
            (``kind='ranking'``) Column with the numeric prediction score used to rank the bars.
        col_group : str, optional
            (``kind='ranking'``) Column whose distinct values color the bars (adds a class legend).
        col_std : str, optional
            (``kind='ranking'``) Column with per-item standard deviations, drawn as error bars.
        colors : dict, optional
            (``kind='ranking'``/``'clustermap'``) A ``group/label -> color`` mapping; defaults to
            the house categorical palette.
        cutoffs : tuple, optional
            (``kind='ranking'``) x-positions of dashed confidence cut-off lines.
        top_n : int, optional
            (``kind='ranking'``) If given, keep only the top ``top_n`` ranked items.
        ascending : bool, default=False
            (``kind='ranking'``) Sort order; ``False`` ranks the highest score first (on top).
        title : str, optional
            (``kind='ranking'``/``'clustermap'``) Axes/figure title.
        scores_y : array-like, optional
            (``kind='scatter'``, required there) Per-sample scores of the second predictor (y-axis).
        marker_size : int or float, default=30
            (``kind='scatter'``) Scatter marker size.
        diagonal : bool, default=True
            (``kind='scatter'``) If ``True``, draw the ``y = x`` agreement line.
        n_steps : int, default=101
            (``kind='cutoff'``) Number of evenly spaced cutoffs between the min and max score.
        names : list of str, optional
            (``kind='clustermap'``) Per-sample tick labels; defaults to positional indices.
        cmap : str, default="GnBu"
            (``kind='clustermap'``) Colormap for the correlation heatmap.
        cbar_label : str, default="Pearson correlation (r)"
            (``kind='clustermap'``) Label of the colorbar.
        xlabel : str, optional
            x-axis label; defaults to a per-kind label.
        ylabel : str, optional
            y-axis label; defaults to a per-kind label.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the requested plot.

        See Also
        --------
        * :meth:`AAPred.predict` for the predictions this visualizes.
        * :meth:`AAPredPlot.eval` for evaluation figures.

        Examples
        --------
        .. include:: examples/aapred_plot_predict.rst
        """
        if kind not in LIST_PREDICT_KINDS:
            raise ValueError(f"'kind' ('{kind}') must be one of {LIST_PREDICT_KINDS}.")
        figsize = figsize if figsize is not None else _DICT_PREDICT_FIGSIZE[kind]
        if kind == "window":
            return AAPredPlot._plot_window(
                df_window=data, entry=entry, list_annotations=list_annotations, threshold=threshold,
                ax=ax, figsize=figsize, color=color,
                xlabel=_default(xlabel, "Residue position"),
                ylabel=_default(ylabel, "Prediction score"))
        if kind == "domain":
            return AAPredPlot._plot_domain(
                df_domain=data, entry=entry, ax=ax, figsize=figsize, color=color,
                xlabel=_default(xlabel, "Boundary offset [residues]"),
                ylabel=_default(ylabel, "Prediction score"))
        if kind == "hist":
            return AAPredPlot._plot_hist(
                scores=data, labels=labels, ax=ax, figsize=figsize, bins=bins,
                thresholds=thresholds, dict_color=dict_color,
                xlabel=_default(xlabel, "Prediction score"),
                ylabel=_default(ylabel, "Number of samples"))
        if kind == "ranking":
            return AAPredPlot._plot_ranking(
                df_pred=data, col_name=col_name, col_score=col_score, col_group=col_group,
                col_std=col_std, colors=colors, cutoffs=cutoffs, top_n=top_n, ascending=ascending,
                ax=ax, figsize=figsize, xlabel=_default(xlabel, "Prediction score"), title=title)
        if kind == "scatter":
            if scores_y is None:
                raise ValueError("'kind'='scatter' requires 'scores_y' (the second predictor's scores).")
            return AAPredPlot._plot_scatter(
                scores_x=data, scores_y=scores_y, labels=labels, ax=ax, figsize=figsize,
                dict_color=dict_color, marker_size=marker_size, diagonal=diagonal,
                xlabel=_default(xlabel, "Predictor 1 score"),
                ylabel=_default(ylabel, "Predictor 2 score"))
        if kind == "cutoff":
            return AAPredPlot._plot_cutoff(
                scores=data, ax=ax, figsize=figsize, n_steps=n_steps, color=color,
                thresholds=thresholds, xlabel=_default(xlabel, "Score cutoff"),
                ylabel=_default(ylabel, "Samples above cutoff [%]"))
        # kind == "clustermap"
        return AAPredPlot._plot_clustermap(
            data=data, names=names, labels=labels, colors=colors, cmap=cmap, figsize=figsize,
            cbar_label=cbar_label, title=title)

    @staticmethod
    def eval(df_eval: pd.DataFrame,
             kind: str = "eval",
             ax: Optional[Axes] = None,
             figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
             dict_color: Optional[Dict[str, str]] = None,
             baseline: Optional[Union[int, float]] = None,
             group: str = "group",
             condition: str = "condition",
             value: str = "value",
             baseline_label: Optional[str] = None,
             annotate: bool = True,
             annotation_fmt: Optional[str] = None,
             group_order: Optional[List[str]] = None,
             condition_order: Optional[List[str]] = None,
             colors: Optional[Union[List[str], Dict[str, str]]] = None,
             bar_width: Union[int, float] = 0.8,
             xlabel: Optional[str] = None,
             ylabel: str = "Score",
             title: Optional[str] = None,
             ylim: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
             fontsize_annotations: Union[int, float] = 10,
             xtick_rotation: Union[int, float] = 0,
             ) -> Tuple[Figure, Axes]:
        """
        Visualize model / feature-set evaluation, dispatched by ``kind``.

        Two evaluation figures share one entry point:

        * ``'eval'`` — grouped bar plot comparing **models** across metrics (hue = model), from the
          long-format ``df_eval`` of :meth:`AAPred.eval` (columns ``model``, ``metric``,
          ``principle``, ``score``, ``score_std``). Cross-validation bars carry ``score_std`` error
          bars and held-out bars are hatched. Uses ``dict_color``, ``baseline``, ``ylabel``.
        * ``'comparison'`` — grouped ``condition`` x ``group`` benchmark barplot with per-bar value
          labels and an optional baseline, from a tidy ``df_eval`` with ``group`` / ``condition`` /
          ``value`` columns. Uses ``group``, ``condition``, ``value``, ``baseline``,
          ``baseline_label``, ``annotate``, ``annotation_fmt``, ``group_order``, ``condition_order``,
          ``colors``, ``bar_width``, ``xlabel``, ``ylabel``, ``title``, ``ylim``,
          ``fontsize_annotations``, ``xtick_rotation``.

        To compare **CPP parameter combinations** instead, use the feature-optimization protocol
        :func:`aaanalysis.pipe.find_features` and its evaluation-grid :func:`aaanalysis.pipe.plot_eval`.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame
            Evaluation table. For ``kind='eval'`` the :meth:`AAPred.eval` output (columns ``model``,
            ``metric``, ``principle``, ``score``, ``score_std``); for ``kind='comparison'`` a tidy
            frame with the ``group`` / ``condition`` / ``value`` columns.
        kind : str, default="eval"
            Which evaluation figure to draw; one of ``eval``, ``comparison``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created.
        figsize : tuple, optional
            Figure size when ``ax`` is ``None``. If ``None``, a per-kind default is used.
        dict_color : dict, optional
            (``kind='eval'``) Mapping ``model -> color`` (the bar hue).
        baseline : int or float, optional
            y-value of a dashed chance / baseline line (e.g. ``0.5`` for ``kind='eval'``, ``50`` for
            ``kind='comparison'``). If ``None``, no line is drawn.
        group : str, default="group"
            (``kind='comparison'``) Column whose distinct values become the colored bars (legend).
        condition : str, default="condition"
            (``kind='comparison'``) Column whose distinct values become the x-axis clusters.
        value : str, default="value"
            (``kind='comparison'``) Column with the numeric bar heights.
        baseline_label : str, optional
            (``kind='comparison'``) Legend label for the baseline; ``None`` generates
            ``"chance (<baseline>)"``; ``""`` draws the line without a legend entry.
        annotate : bool, default=True
            (``kind='comparison'``) If ``True``, write each bar's value above it.
        annotation_fmt : str, optional
            (``kind='comparison'``) Format string for the value labels; auto-chosen when ``None``.
        group_order : list of str, optional
            (``kind='comparison'``) Order of the groups (bars within a cluster).
        condition_order : list of str, optional
            (``kind='comparison'``) Order of the conditions (x-axis clusters).
        colors : list of str or dict, optional
            (``kind='comparison'``) Bar colors aligned to ``group_order`` or a ``group -> color`` dict.
        bar_width : int or float, default=0.8
            (``kind='comparison'``) Total width of each cluster (split across the groups); in (0, 1].
        xlabel : str, optional
            (``kind='comparison'``) x-axis label.
        ylabel : str, default="Score"
            y-axis label.
        title : str, optional
            (``kind='comparison'``) Axes title.
        ylim : tuple, optional
            (``kind='comparison'``) y-axis limits ``(bottom, top)``.
        fontsize_annotations : int or float, default=10
            (``kind='comparison'``) Font size of the per-bar value labels.
        xtick_rotation : int or float, default=0
            (``kind='comparison'``) Rotation (degrees) of the cluster tick labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes with the requested evaluation plot.

        See Also
        --------
        * :meth:`AAPred.eval` for the evaluation table this visualizes.
        * :func:`aaanalysis.pipe.find_features` and :func:`aaanalysis.pipe.plot_eval` for
          comparing CPP parameter combinations (a heatmap over the parameter grid).

        Examples
        --------
        .. include:: examples/aapred_plot_eval.rst
        """
        if kind not in LIST_EVAL_KINDS:
            raise ValueError(f"'kind' ('{kind}') must be one of {LIST_EVAL_KINDS}.")
        if kind == "eval":
            return AAPredPlot._eval_bars(
                df_eval=df_eval, ax=ax, figsize=_default(figsize, (7, 5)),
                dict_color=dict_color, baseline=baseline, ylabel=ylabel)
        # kind == "comparison"
        return AAPredPlot._plot_comparison(
            df_eval=df_eval, group=group, condition=condition, value=value, baseline=baseline,
            baseline_label=baseline_label, annotate=annotate, annotation_fmt=annotation_fmt,
            group_order=group_order, condition_order=condition_order, colors=colors,
            bar_width=bar_width, ax=ax, figsize=_default(figsize, (7, 4.2)), xlabel=xlabel,
            ylabel=ylabel, title=title, ylim=ylim, fontsize_annotations=fontsize_annotations,
            xtick_rotation=xtick_rotation)

    # III Private renderers (one per kind; kept as the original drawing logic)
    @staticmethod
    def _eval_bars(df_eval, ax=None, figsize=(7, 5), dict_color=None, baseline=None,
                   ylabel="Score"):
        """Grouped bar plot comparing methods across metrics (hue = model)."""
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
    def _plot_comparison(df_eval, group="group", condition="condition", value="value",
                         baseline=None, baseline_label=None, annotate=True, annotation_fmt=None,
                         group_order=None, condition_order=None, colors=None, bar_width=0.8,
                         ax=None, figsize=(7, 4.2), xlabel=None, ylabel="Score", title=None,
                         ylim=None, fontsize_annotations=10, xtick_rotation=0):
        """Grouped method x condition comparison barplot with value labels and a baseline."""
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
    def _plot_ranking(df_pred, col_name="name", col_score="score", col_group=None, col_std=None,
                      colors=None, cutoffs=(50, 80), top_n=None, ascending=False, ax=None,
                      figsize=None, xlabel="Prediction score", title=None):
        """Ranked candidates as horizontal bars colored by class, with cut-off lines."""
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
    def _plot_clustermap(data, names=None, labels=None, colors=None, cmap="GnBu", figsize=(9, 9),
                         cbar_label="Pearson correlation (r)", title=None):
        """Cluster samples by explanation similarity (correlation of importance vectors)."""
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
    def _plot_hist(scores, labels=None, ax=None, figsize=(6, 4.5), bins=20, thresholds=None,
                   dict_color=None, xlabel="Prediction score", ylabel="Number of samples"):
        """Class-separated histogram of per-sample prediction scores."""
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
    def _plot_scatter(scores_x, scores_y, labels=None, ax=None, figsize=(5.5, 5.5), dict_color=None,
                      marker_size=30, diagonal=True, xlabel="Predictor 1 score",
                      ylabel="Predictor 2 score"):
        """2D scatter comparing per-sample scores of two predictors."""
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
    def _plot_cutoff(scores, ax=None, figsize=(6, 4.5), n_steps=101, color=None, thresholds=None,
                     xlabel="Score cutoff", ylabel="Samples above cutoff [%]"):
        """Line plot of the percentage of samples scoring at or above each cutoff."""
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
    def _plot_window(df_window, entry=None, list_annotations=None, threshold=None, ax=None,
                     figsize=(10, 4), color=None, xlabel="Residue position",
                     ylabel="Prediction score"):
        """Per-residue prediction profile from AAPred.predict(level='window')."""
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
    def _plot_domain(df_domain, entry=None, ax=None, figsize=(6, 4.5), color=None,
                     xlabel="Boundary offset [residues]", ylabel="Prediction score"):
        """Domain boundary-sensitivity plot from AAPred.predict(level='domain')."""
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
