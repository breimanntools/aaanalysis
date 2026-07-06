"""
This is a script for the frontend of the ReliabilityModelPlot class for reliability visualizations.
"""
from typing import Optional, Tuple
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import aaanalysis.utils as ut

from ._backend.reliability.reliability_plot import (
    plot_reliability_diagram_, plot_ood_hist_, plot_trust_map_)


def _check_df_cols(df, name, cols):
    """The plotted frame must be a DataFrame carrying the required columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"'{name}' should be a pandas DataFrame.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"'{name}' is missing required columns: {missing}.")


class ReliabilityModelPlot:
    """
    Plot class for :class:`ReliabilityModel` outputs.

    Visualizes the two trust axes and the calibration quality: a calibration curve
    (:meth:`reliability_diagram`, from :meth:`ReliabilityModel.eval`), the out-of-distribution
    distribution (:meth:`ood_hist`), and a score-vs-OOD map colored by the ``reliable`` flag
    (:meth:`trust_map`, both from :meth:`ReliabilityModel.predict`).

    See Also
    --------
    * :class:`ReliabilityModel` for the measures these methods visualize.
    """

    def __init__(self):
        pass

    @staticmethod
    def reliability_diagram(df_eval: pd.DataFrame,
                            figsize: Tuple[float, float] = (5, 5),
                            color: str = "tab:blue",
                            title: Optional[str] = None,
                            ax: Optional[Axes] = None,
                            ) -> Tuple[Figure, Axes]:
        """
        Calibration curve — mean predicted score vs. empirical positive rate, per bin.

        Parameters
        ----------
        df_eval : pd.DataFrame
            Output of :meth:`ReliabilityModel.eval` (per-bin ``mean_score`` / ``empirical_pos``).
        figsize : tuple, default=(5, 5)
            Figure size (used only when ``ax`` is ``None``).
        color : str, default="tab:blue"
            Line/marker color of the model curve.
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
        .. include:: examples/rm_plot_reliability_diagram.rst
        """
        _check_df_cols(df_eval, "df_eval", ["bin", "mean_score", "empirical_pos"])
        fig, ax = plot_reliability_diagram_(df_eval, figsize=figsize, color=color, title=title, ax=ax)
        return ut.FigAxResult(fig, ax)

    @staticmethod
    def ood_hist(df_rel: pd.DataFrame,
                 figsize: Tuple[float, float] = (6, 4.5),
                 bins: int = 20,
                 color: str = "tab:gray",
                 title: Optional[str] = None,
                 ax: Optional[Axes] = None,
                 ) -> Tuple[Figure, Axes]:
        """
        Histogram of the out-of-distribution score, with the in-domain boundary at ``1.0``.

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
                  figsize: Tuple[float, float] = (5.5, 5),
                  title: Optional[str] = None,
                  ax: Optional[Axes] = None,
                  ) -> Tuple[Figure, Axes]:
        """
        Score vs. OOD-score scatter colored by ``reliable`` — the two trust axes at a glance.

        A point high on the x-axis (confident) but above the in-domain boundary (out-of-domain)
        is the untrustworthy, extrapolated prediction.

        Parameters
        ----------
        df_rel : pd.DataFrame
            Output of :meth:`ReliabilityModel.predict` (needs ``score``, ``ood_score``,
            ``reliable``; uses ``score_calibrated`` when present).
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
        _check_df_cols(df_rel, "df_rel",
                       [ut.COL_SCORE, ut.COL_OOD_SCORE, ut.COL_RELIABLE, ut.COL_SCORE_CAL])
        fig, ax = plot_trust_map_(df_rel, figsize=figsize, title=title, ax=ax)
        return ut.FigAxResult(fig, ax)
