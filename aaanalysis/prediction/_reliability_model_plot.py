"""
This is a script for the frontend of the ReliabilityModelPlot class for reliability visualizations.
"""
from typing import Optional, Tuple, Union
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import aaanalysis.utils as ut

from ._backend.reliability.reliability_plot import (
    plot_reliability_diagram_, plot_ood_hist_, plot_trust_map_, plot_ranking_)


# I Helper Functions
def _check_df_cols(df, name, cols):
    """The plotted frame must be a DataFrame carrying the required columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"'{name}' should be a pandas DataFrame.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"'{name}' is missing required columns: {missing}.")

# II Main Functions
class ReliabilityModelPlot:
    """
    Visualize :class:`ReliabilityModel` outputs: calibration and the two trust axes.

    One figure asks "should I trust *this* prediction?": :meth:`ranking` shows each sample's score
    with its uncertainty interval, colored by trust status. Three look at a batch: the calibration
    curve (:meth:`reliability_diagram`), the spread of out-of-distribution scores
    (:meth:`ood_hist`), and a score against OOD-score map colored by the ``reliable`` flag
    (:meth:`trust_map`).

    .. warning::

        **Experimental.** This class is part of the new v1.1.0 prediction layer and is under active
        development; its API (signatures, defaults, return objects) may change between minor releases
        without the usual deprecation cycle. Pin a version if you depend on the current behaviour.

    .. versionadded:: 1.1.0

    See Also
    --------
    * :class:`ReliabilityModel` for the measures these methods visualize.
    """

    def __init__(self):
        pass

    @staticmethod
    def ranking(df_rel: pd.DataFrame,
                *, names: Optional[list] = None,
                figsize: Optional[Tuple[float, float]] = None,
                top_n: Optional[int] = None,
                title: Optional[str] = None,
                ax: Optional[Axes] = None,
                ) -> Tuple[Figure, Axes]:
        """
        Per-sample view: each prediction's score with its uncertainty, colored by trust status.

        Samples are ranked by score as horizontal bars, with the confidence interval as an error
        bar and the bar colored green (**reliable**), amber (in-domain but undecided), or red
        (**out-of-distribution**), so an over-confident score on an unfamiliar input stands out as
        a long red bar.

        Parameters
        ----------
        df_rel : pd.DataFrame
            Output of :meth:`ReliabilityModel.predict` (needs ``score``, ``ci_low`` / ``ci_high``,
            ``in_domain``, ``reliable``).
        names : list, optional
            Per-sample labels for the y-axis; row positions are used if ``None``.
        figsize : tuple, optional
            Figure size (used only when ``ax`` is ``None``); scales with the sample count by default.
        top_n : int, optional
            Show only the ``top_n`` highest-scoring samples; all are shown if ``None``.
        title : str, optional
            Axes title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created (or parent) figure.
        ax : matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        .. include:: examples/rm_plot_ranking.rst
        """
        _check_df_cols(df_rel, "df_rel", [ut.COL_SCORE, ut.COL_CI_LOW, ut.COL_CI_HIGH,
                                          ut.COL_IN_DOMAIN, ut.COL_RELIABLE])
        if names is not None and len(names) != len(df_rel):
            raise ValueError(f"'names' has {len(names)} entries but 'df_rel' has {len(df_rel)} rows.")
        if top_n is not None:
            ut.check_number_range(name="top_n", val=top_n, min_val=1, just_int=True)
        fig, ax = plot_ranking_(df_rel, names=names, figsize=figsize, top_n=top_n, title=title, ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def reliability_diagram(df_eval: pd.DataFrame,
                            *, figsize: Tuple[float, float] = (5, 5),
                            color: Union[str, Tuple[float, float, float]] = "tab:blue",
                            label: str = "model",
                            title: Optional[str] = None,
                            ax: Optional[Axes] = None,
                            ) -> Tuple[Figure, Axes]:
        """
        Calibration curve: mean predicted score against empirical positive rate, per bin.

        Points on the diagonal are perfectly calibrated; points below it mean the score
        overstates the true positive rate (over-confident), above it under-confident. When
        ``df_eval`` carries the Brier score and ECE rows
        (``ReliabilityModel.eval(add_metrics=True)``), both values are annotated in the curve's
        legend entry. To compare the raw and the calibrated curve, draw both frames onto the same
        ``ax`` with distinct ``label`` / ``color``; the perfect-calibration diagonal is drawn only
        once.

        Parameters
        ----------
        df_eval : pd.DataFrame
            Output of :meth:`ReliabilityModel.eval` (per-bin ``mean_score`` / ``empirical_pos``);
            either the raw-score or the calibrated-score table (``use_calibrated=True``).
        figsize : tuple of float, default=(5, 5)
            Figure width and height, each at least 1 (used only when ``ax`` is ``None``).
        color : str or tuple of float, default="tab:blue"
            Matplotlib line/marker color of the model curve. A named or hexadecimal color string,
            or an RGB tuple with three components from 0 to 255, is accepted.
        label : str, default="model"
            Legend label of the curve. The Brier score and expected calibration error are
            appended when present, so distinct labels identify raw and calibrated curves
            drawn on the same ``ax``.

            .. versionadded:: 1.2.0
        title : str, optional
            Axes title. If ``None``, do not set a title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are created; otherwise ``figsize``
            does not resize the supplied axes' figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created (or parent) figure.
        ax : matplotlib.axes.Axes
            The axes drawn on.

        Raises
        ------
        ValueError
            If ``df_eval`` is not a DataFrame or lacks the ``bin`` / ``mean_score`` /
            ``empirical_pos`` columns; ``figsize`` is not a tuple of two positive numbers;
            ``color`` is not a matplotlib color; ``label`` or non-``None`` ``title`` is not a
            string; or ``ax`` is not a matplotlib ``Axes``.

        Examples
        --------
        .. include:: examples/rm_plot_reliability_diagram.rst
        """
        _check_df_cols(df_eval, "df_eval", [ut.COL_BIN, ut.COL_MEAN_SCORE, ut.COL_EMPIRICAL_POS])
        ut.check_figsize(figsize=figsize, accept_none=False)
        ut.check_color(name="color", val=color, accept_none=False)
        ut.check_str(name="label", val=label)
        ut.check_str(name="title", val=title, accept_none=True)
        ut.check_ax(ax=ax, accept_none=True)
        fig, ax = plot_reliability_diagram_(df_eval, figsize=figsize, color=color, label=label,
                                            title=title, ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def ood_hist(df_rel: pd.DataFrame,
                 *, figsize: Tuple[float, float] = (6, 4.5),
                 bins: int = 20,
                 color: str = "tab:gray",
                 title: Optional[str] = None,
                 ax: Optional[Axes] = None,
                 ) -> Tuple[Figure, Axes]:
        """
        Histogram of the out-of-distribution score, with the in-domain boundary at ``1.0``.

        Mass to the right of the boundary is out-of-domain: predictions the model is extrapolating,
        not to be trusted at face value.

        Parameters
        ----------
        df_rel : pd.DataFrame
            Output of :meth:`ReliabilityModel.predict` (needs ``ood_score``).
        figsize : tuple, default=(6, 4.5)
            Figure size (used only when ``ax`` is ``None``).
        bins : int, default=20
            Number of histogram bins.
        color : str, default="tab:gray"
            Bar color.
        title : str, optional
            Axes title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created (or parent) figure.
        ax : matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        .. include:: examples/rm_plot_ood_hist.rst
        """
        _check_df_cols(df_rel, "df_rel", [ut.COL_OOD_SCORE])
        ut.check_number_range(name="bins", val=bins, min_val=1, just_int=True)
        fig, ax = plot_ood_hist_(df_rel, figsize=figsize, bins=bins, color=color, title=title, ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def trust_map(df_rel: pd.DataFrame,
                  *, figsize: Tuple[float, float] = (5.5, 5),
                  title: Optional[str] = None,
                  ax: Optional[Axes] = None,
                  ) -> Tuple[Figure, Axes]:
        """
        Score against OOD-score scatter colored by ``reliable``: the two trust axes at a glance.

        A point high on the x-axis (confident) but above the in-domain boundary (out-of-domain)
        is the untrustworthy, extrapolated prediction.

        Parameters
        ----------
        df_rel : pd.DataFrame
            Output of :meth:`ReliabilityModel.predict` (needs ``score``, ``ood_score``, ``reliable``).
        figsize : tuple, default=(5.5, 5)
            Figure size (used only when ``ax`` is ``None``).
        title : str, optional
            Axes title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created (or parent) figure.
        ax : matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        .. include:: examples/rm_plot_trust_map.rst
        """
        _check_df_cols(df_rel, "df_rel", [ut.COL_SCORE, ut.COL_OOD_SCORE, ut.COL_RELIABLE])
        fig, ax = plot_trust_map_(df_rel, figsize=figsize, title=title, ax=ax)
        return ut.FigAxResult(fig, ax)
