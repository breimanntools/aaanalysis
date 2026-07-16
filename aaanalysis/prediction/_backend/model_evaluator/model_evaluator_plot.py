"""
This is a script for the backend of the ModelEvaluatorPlot class: cross-validation CI bars and
paired-comparison delta bars.
"""
import numpy as np
import matplotlib.pyplot as plt

import aaanalysis.utils as ut


# I Helper Functions
def _err_from_ci(scores, ci_low, ci_high, score_std):
    """Asymmetric error bars from the bootstrap CI, falling back to +/- std where CI is NaN."""
    low = np.where(np.isnan(ci_low), scores - score_std, ci_low)
    high = np.where(np.isnan(ci_high), scores + score_std, ci_high)
    lower = np.clip(scores - low, 0, None)
    upper = np.clip(high - scores, 0, None)
    return np.vstack([lower, upper])


# II Main Functions
def plot_scores(df_eval=None, colors=None, figsize=(6, 4), metrics=None):
    """Grouped bar chart of cross-validated scores per (model, metric) with CI error bars."""
    models = list(dict.fromkeys(df_eval[ut.COL_MODEL]))
    if metrics is None:
        metrics = list(dict.fromkeys(df_eval[ut.COL_METRIC]))
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(models):
        sub = df_eval[df_eval[ut.COL_MODEL] == model].set_index(ut.COL_METRIC).reindex(metrics)
        scores = sub[ut.COL_SCORE].to_numpy(dtype=float)
        yerr = _err_from_ci(scores, sub[ut.COL_CI_LOW].to_numpy(dtype=float),
                            sub[ut.COL_CI_HIGH].to_numpy(dtype=float),
                            sub[ut.COL_SCORE_STD].to_numpy(dtype=float))
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, scores, width=width, color=colors[i], label=str(model),
               edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, scores, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    if n_models > 1:
        ax.legend(frameon=False)
    return fig, ax


def plot_compare(df_eval=None, color_pos=None, color_neg=None, figsize=(6, 4), alpha=0.05):
    """Horizontal delta bars for the paired model comparison, colored by sign with CI whiskers.

    ``df_eval`` is the paired-comparison table from :meth:`ModelEvaluator.eval`; a filled marker
    on the bar flags a pair whose ``p_value`` is below ``alpha`` (significant difference).
    """
    df = df_eval.reset_index(drop=True)
    labels = [f"{a}\nvs {b}" for a, b in zip(df[ut.COL_MODEL_A], df[ut.COL_MODEL_B])]
    y = np.arange(len(df))
    delta = df[ut.COL_DELTA].to_numpy(dtype=float)
    lower = np.clip(delta - df[ut.COL_CI_LOW].to_numpy(dtype=float), 0, None)
    upper = np.clip(df[ut.COL_CI_HIGH].to_numpy(dtype=float) - delta, 0, None)
    xerr = np.vstack([lower, upper])
    bar_colors = [color_pos if d >= 0 else color_neg for d in delta]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, delta, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(delta, y, xerr=xerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    p_values = df[ut.COL_P_VALUE].to_numpy(dtype=float)
    for yi, (di, pi) in enumerate(zip(delta, p_values)):
        if np.isfinite(pi) and pi < alpha:
            ax.plot(di, yi, marker="*", color="black", markersize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    metric = df[ut.COL_METRIC].iloc[0] if len(df) else ""
    ax.set_xlabel(f"Delta {metric} (model_a - model_b)")
    return fig, ax
