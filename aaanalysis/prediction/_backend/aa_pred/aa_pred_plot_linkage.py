"""
This is a script for the backend of the shared sample-relation linkage used by
AAPredPlot.group_cluster: the sample x sample Pearson correlation of the per-sample
feature/importance vectors and its hierarchical linkage.

Computing the linkage here once (instead of letting ``seaborn.clustermap`` compute it
internally) lets every ``group_cluster`` kind draw exactly the same tree: the clustermap
receives it via ``row_linkage`` / ``col_linkage`` and the dendrogram kind draws it directly.
"""
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy


# I Helper Functions
# Linkage settings matching the ``seaborn.clustermap`` defaults the clustermap kind was built on,
# so passing the precomputed linkage leaves the clustermap figure unchanged.
_LINKAGE_METHOD = "average"
_LINKAGE_METRIC = "euclidean"


# II Main Functions
def sample_correlation_(data=None, names=None):
    """Sample x sample Pearson correlation of the per-sample vectors, as a labeled DataFrame.

    A sample whose vector has zero variance (e.g. an all-zero SHAP row) is treated as
    uncorrelated (0) with other samples and has self-correlation 1. The explicit
    centered dot-product avoids NumPy's divide-by-zero warning for valid input rows.
    """
    values = np.asarray(data, dtype=float)
    n = values.shape[0]
    if names is None:
        names = [str(i) for i in range(n)]
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = np.outer(norms, norms)
    corr = np.divide(centered @ centered.T, denominator, out=np.zeros((n, n)),
                     where=denominator != 0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=list(names), columns=list(names))


def sample_linkage_(corr_df=None, axis=0):
    """Hierarchical linkage of the correlation matrix rows (``axis=0``) or columns (``axis=1``).

    Mirrors how ``seaborn.clustermap`` clusters a matrix without ``fastcluster`` (average
    linkage on the Euclidean distance between correlation profiles; columns are clustered on
    the transpose), so the tree equals the one the clustermap computed on its own. With
    ``fastcluster`` installed, seaborn used its implementation instead, which can break exact
    ties between equidistant merges differently (same clusters, possibly another arrangement
    of tied branches).
    """
    values = corr_df.values if axis == 0 else corr_df.T.values
    return hierarchy.linkage(values, method=_LINKAGE_METHOD, metric=_LINKAGE_METRIC)
