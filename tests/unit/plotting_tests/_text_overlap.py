"""
Shared helpers for the dense-plot label-overlap gate (not collected as tests;
the leading underscore keeps pytest from picking this up as a test module).

The row/column labels of the dense CPP composites (``feature_map``, ``heatmap``,
``ranking``, ``profile``) are hand-placed ``ax.text`` artists rather than real
tick labels, so a reliable overlap measurement needs a *forced render* first
(``fig.savefig(BytesIO())``): a bare ``get_window_extent`` returns degenerate
~1px extents and reports false "0 overlaps".
"""
import io

import pandas as pd
from matplotlib.transforms import Bbox

import aaanalysis as aa


def get_label_overlaps(fig, names, min_frac=0.30):
    """Return label-vs-label bbox overlaps among the given target label strings.

    Only text artists whose (stripped) string is in ``names`` are compared, so
    legitimately layered composite axes (colorbar, importance bars, legend) do
    not produce false positives. An overlap counts when the intersection area is
    at least ``min_frac`` of the smaller of the two boxes.

    Returns a list of ``(name_i, name_j, fraction)`` tuples (empty = clean).
    """
    fig.savefig(io.BytesIO(), format="png", dpi=fig.dpi)
    r = fig.canvas.get_renderer()
    nameset = set(names)
    items = [(t.get_text().strip(), t.get_window_extent(renderer=r))
             for ax in fig.axes for t in ax.texts
             if t.get_visible() and t.get_text().strip() in nameset]
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            it = Bbox.intersection(items[i][1], items[j][1])
            if it and it.width > 0 and it.height > 0:
                f = it.width * it.height / min(items[i][1].width * items[i][1].height,
                                               items[j][1].width * items[j][1].height)
                if f >= min_frac:
                    bad.append((items[i][0], items[j][0], round(f, 2)))
    return bad


def make_dense_df_feat(n_subcat=74):
    """Build a valid high-row ``df_feat`` (one scale per subcategory, up to 74).

    Derives real category/subcategory/scale metadata from ``CPPPlot()._df_cat``
    so the frame passes ``feature_map`` validation, letting the row count (and
    thus the row-label density) be driven up to the full AAontology breadth.
    """
    dc = aa.CPPPlot()._df_cat
    subs = list(dict.fromkeys(dc["subcategory"]))[:n_subcat]
    rows = []
    for s in subs:
        r = dc[dc["subcategory"] == s].iloc[0]
        rows.append(dict(feature=f"TMD_C_JMD_C-Segment(3,4)-{r['scale_id']}",
                         category=r["category"], subcategory=s, scale_name=r["scale_name"],
                         scale_description=r["scale_description"], abs_auc=0.2, abs_mean_dif=0.3,
                         mean_dif=0.3, std_test=0.1, std_ref=0.1, p_val_mann_whitney=0.01,
                         p_val_fdr_bh=0.02, positions="31,32,33,34,35", feat_importance=1.0,
                         feat_importance_std=0.1))
    return pd.DataFrame(rows)
