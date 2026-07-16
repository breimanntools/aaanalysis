"""
This is a script for the frontend of the ModelEvaluatorPlot class for visualizing cross-validated
model evaluation and paired comparison results.
"""
from typing import Optional, List, Tuple, Union
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import aaanalysis.utils as ut
from ._backend.model_evaluator.model_evaluator_plot import plot_scores, plot_compare


# II Main Functions
class ModelEvaluatorPlot:
    """
    Plotting class for :class:`ModelEvaluator` results.

    Visualizes the two ``ModelEvaluator`` tables: :meth:`scores` draws grouped confidence-interval
    bars of the cross-validated scores per (model, metric) from :meth:`ModelEvaluator.run`, and
    :meth:`compare` draws the paired model comparison from :meth:`ModelEvaluator.eval` as signed
    delta bars with bootstrap-CI whiskers and significance markers.

    Every plotting method returns a ``(fig, ax)`` pair (a thin tuple subclass): unpack as
    ``fig, ax = ...``. For backward compatibility, the returned object also forwards attribute
    access to ``ax``.

    .. versionadded:: 1.1.0

    See Also
    --------
    * :class:`ModelEvaluator` for the logic class whose results this visualizes.
    """

    def __init__(self):
        """
        See Also
        --------
        * :class:`ModelEvaluator`: the logic class whose evaluation and comparison this visualizes.
        """

    @staticmethod
    def scores(df_eval: pd.DataFrame,
               figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4),
               colors: Optional[List[str]] = None,
               metrics: Optional[List[str]] = None,
               ) -> Tuple[Figure, Axes]:
        """
        Plot cross-validated scores per (model, metric) as grouped bars with CI error bars.

        One bar group per metric holds one bar per model; the error bars show the bootstrap
        confidence interval of the mean (falling back to the fold std where the CI is ``NaN``),
        so models are compared at a glance with their uncertainty.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_models * n_metrics, 7)
            Evaluation table from :meth:`ModelEvaluator.run` with columns ``model``, ``metric``,
            ``score``, ``score_std``, ``ci_low``, ``ci_high``, and ``n_scores``. Error bars use the
            bootstrap CI (``ci_low`` / ``ci_high``), falling back to ``score_std`` where the CI is
            ``NaN``.
        figsize : tuple, default=(6, 4)
            Figure dimensions (width, height) in inches.
        colors : list of str, optional
            One color per model (in first-appearance order). Defaults to the package color list.
        metrics : list of str, optional
            Subset and order of metrics to show on the x-axis. Defaults to all metrics in
            ``df_eval`` (first-appearance order).

        Returns
        -------
        fig : Figure
            Figure object for the evaluation plot.
        ax : Axes
            Axes object of the grouped bar chart.

        See Also
        --------
        * :meth:`ModelEvaluator.run`: the respective computation method.

        Examples
        --------
        .. include:: examples/me_plot_scores.rst
        """
        # Check input
        ut.check_df(name="df_eval", df=df_eval, cols_required=ut.COLS_EVAL_MODELEVAL, accept_none=False)
        ut.check_figsize(figsize=figsize, accept_none=False)
        ut.check_list_colors(name="colors", val=colors, accept_none=True)
        metrics = ut.check_list_like(name="metrics", val=metrics, accept_none=True, accept_str=True)
        # Plotting
        n_models = df_eval[ut.COL_MODEL].nunique()
        if colors is None:
            colors = ut.plot_get_clist_(n_colors=n_models)
        elif len(colors) < n_models:
            raise ValueError(f"'colors' (n={len(colors)}) should provide at least one color per "
                             f"model (n_models={n_models}).")
        fig, ax = plot_scores(df_eval=df_eval, colors=colors, figsize=figsize, metrics=metrics)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def compare(df_eval: pd.DataFrame,
                figsize: Tuple[Union[int, float], Union[int, float]] = (6, 4),
                colors: Optional[List[str]] = None,
                alpha: float = 0.05,
                ) -> Tuple[Figure, Axes]:
        """
        Plot the paired model comparison as signed delta bars with CI whiskers.

        Each bar is one model pair's ``delta`` (``score_a - score_b``); bars are colored by sign,
        whiskers show the bootstrap CI, and a star marks a pair whose ``p_value`` is below
        ``alpha`` (a significant difference).

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_eval : pd.DataFrame, shape (n_pairs, 8)
            Comparison table from :meth:`ModelEvaluator.eval` with columns ``model_a``,
            ``model_b``, ``metric``, ``delta``, ``delta_std``, ``ci_low``, ``ci_high``, and
            ``p_value``.
        figsize : tuple, default=(6, 4)
            Figure dimensions (width, height) in inches.
        colors : list of str of length 2, optional
            Colors for positive and negative deltas (``[color_pos, color_neg]``). Defaults to the
            package positive/negative colors.
        alpha : float, default=0.05
            Significance level in ``(0, 1)``; pairs with ``p_value < alpha`` are starred.

        Returns
        -------
        fig : Figure
            Figure object for the comparison plot.
        ax : Axes
            Axes object of the delta bar chart.

        See Also
        --------
        * :meth:`ModelEvaluator.eval`: the respective computation method.

        Examples
        --------
        .. include:: examples/me_plot_compare.rst
        """
        # Check input
        ut.check_df(name="df_eval", df=df_eval, cols_required=ut.COLS_COMPARE_MODELEVAL, accept_none=False)
        ut.check_figsize(figsize=figsize, accept_none=False)
        ut.check_list_colors(name="colors", val=colors, accept_none=True, min_n=2, max_n=2)
        ut.check_number_range(name="alpha", val=alpha, min_val=0.0, max_val=1.0,
                              just_int=False, exclusive_limits=True)
        # Plotting
        if colors is None:
            colors = [ut.COLOR_POS, ut.COLOR_NEG]
        fig, ax = plot_compare(df_eval=df_eval, color_pos=colors[0], color_neg=colors[1],
                               figsize=figsize, alpha=alpha)
        return ut.FigAxResult(fig, ax)
