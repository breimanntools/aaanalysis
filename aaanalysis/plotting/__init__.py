"""
Plotting utilities: shared styling, colors, and legends for every ``*Plot`` class.

Public objects: plot_get_clist, plot_get_cmap, plot_get_cdict, plot_settings,
plot_legend, plot_gcfs, plot_rank, plot_eval_heatmap.
A cross-cutting subpackage (not a pipeline stage): ``plot_settings`` sets the house
rcParams, the ``plot_get_*`` helpers supply the color list / map / dict, and
``plot_legend`` / ``plot_gcfs`` / ``plot_rank`` are reused by the paired plot classes
(``CPPPlot``, ``AAclustPlot``, …). Library plot code never calls ``plt.show()``.

See ``.claude/rules/plotting.md`` for the plotting conventions, ``CONTEXT.md`` for
domain terms.
"""
from ._plot_get_clist import plot_get_clist
from ._plot_get_cmap import plot_get_cmap
from ._plot_get_cdict import plot_get_cdict
from ._plot_settings import plot_settings
from ._plot_gcfs import plot_gcfs
from ._plot_legend import plot_legend
from ._plot_rank import plot_rank
from ._plot_eval_heatmap import plot_eval_heatmap


__all__ = [
    "plot_get_clist",
    "plot_get_cmap",
    "plot_get_cdict",
    "plot_settings",
    "plot_legend",
    "plot_gcfs",
    "plot_rank",
    "plot_eval_heatmap",
]
