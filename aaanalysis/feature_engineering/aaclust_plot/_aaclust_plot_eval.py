"""
This is a script for the AAclust plot_eval method.
"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# I Helper Functions
def _get_rank(data):
    """"""
    _df = data.copy()
    _df['BIC_rank'] = _df['BIC'].rank(ascending=False)
    _df['CH_rank'] = _df['CH'].rank(ascending=False)
    _df['SC_rank'] = _df['SC'].rank(ascending=False)
    return _df[['BIC_rank', 'CH_rank', 'SC_rank']].mean(axis=1).round(2)

# II Main Functions
def plot_eval(data=None, names=None, dict_xlims=None, figsize=None, columns=None, colors=None):
    """"""
    data = pd.DataFrame(data, columns=columns, index=names)
    data["rank"] = _get_rank(data)
    data = data.sort_values(by="rank", ascending=True)
    # Plotting
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=figsize)
    for i, col in enumerate(columns):
        ax = axes[i]
        sns.barplot(ax=ax, data=data, y=data.index, x=col, color=colors[i])
        # Customize subplots
        ax.set_ylabel("")
        ax.set_xlabel(col)
        ax.axvline(0, color='black')  # , linewidth=aa.plot_gcfs("axes.linewidth"))
        if dict_xlims and col in dict_xlims:
            ax.set_xlim(dict_xlims[col])
        if i == 0:
            ax.set_title("Number of clusters", weight="bold")
        elif i == 2:
            ax.set_title("Quality measures", weight="bold")
        sns.despine(ax=ax, left=True)
        ax.tick_params(axis='y', which='both', left=False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)
    return fig, axes
