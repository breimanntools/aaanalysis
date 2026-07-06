"""
This is a script for the backend of the ReliabilityModelPlot class (reliability visualizations).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import aaanalysis.utils as ut


def _fig_ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_reliability_diagram_(df_eval, figsize=(5, 5), color="tab:blue", title=None, ax=None):
    """Calibration curve: mean predicted score vs. empirical positive rate, per bin."""
    fig, ax = _fig_ax(ax, figsize)
    d = df_eval[df_eval["bin"] != "summary"].dropna(subset=["mean_score", "empirical_pos"])
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1.2, label="perfect calibration")
    ax.plot(d["mean_score"], d["empirical_pos"], "o-", color=color, label="model")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted score"), ax.set_ylabel("Empirical positive rate")
    ax.legend(frameon=False, loc="upper left")
    if title:
        ax.set_title(title)
    sns.despine(ax=ax)
    return fig, ax


def plot_ood_hist_(df_rel, figsize=(6, 4.5), bins=20, color="tab:gray", title=None, ax=None):
    """Histogram of the out-of-distribution score, with the in-domain boundary at 1.0."""
    fig, ax = _fig_ax(ax, figsize)
    ax.hist(df_rel[ut.COL_OOD_SCORE].to_numpy(), bins=bins, color=color, edgecolor="black",
            linewidth=0.6)
    ax.axvline(1.0, ls="--", color="tab:red", lw=1.4, label="in-domain boundary")
    ax.set_xlabel("OOD score (distance / training threshold)"), ax.set_ylabel("Number of samples")
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    sns.despine(ax=ax)
    return fig, ax


def plot_trust_map_(df_rel, figsize=(5.5, 5), title=None, ax=None):
    """Score vs. OOD-score scatter, colored by ``reliable`` — the two trust axes at a glance."""
    fig, ax = _fig_ax(ax, figsize)
    cal = df_rel[ut.COL_SCORE_CAL].to_numpy()
    x = np.where(np.isnan(cal), df_rel[ut.COL_SCORE].to_numpy(), cal)
    y = df_rel[ut.COL_OOD_SCORE].to_numpy()
    rel = df_rel[ut.COL_RELIABLE].to_numpy().astype(bool)
    kws = dict(s=32, edgecolors="white", linewidths=0.3)
    ax.scatter(x[rel], y[rel], color="tab:green", label="reliable", **kws)
    ax.scatter(x[~rel], y[~rel], color="tab:red", label="not reliable", **kws)
    ax.axhline(1.0, ls="--", color="0.4", lw=1.2, label="in-domain boundary")
    ax.set_xlim(0, 1), ax.set_xlabel("Prediction score"), ax.set_ylabel("OOD score")
    ax.legend(frameon=False, loc="best")
    if title:
        ax.set_title(title)
    sns.despine(ax=ax)
    return fig, ax
