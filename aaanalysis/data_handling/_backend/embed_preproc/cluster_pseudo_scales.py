"""
This is a script for the backend of EmbeddingPreprocessor.cluster_pseudo_scales.

Two independent AAclust runs at different correlation thresholds produce
coarser (``cat``) and finer (``subcat``) labels for each embedding dimension,
mirroring the AAontology two-level category hierarchy. The two runs are
independent: subcat labels do not necessarily nest within cat labels —
they're two views over the same pseudo-scales.

When ``df_stds_emb`` is supplied, the input to AAclust is the per-column
z-scored concatenation of per-AA (mean, std), expanding each dimension's
descriptor from 20 to 40 features. See the frontend Notes block for the
References that motivate this recipe.
"""
import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
def _zscore_columns(X):
    """Column-wise z-score across rows. Columns with zero std (constant across
    rows) become all-zero columns instead of NaN — they then contribute
    nothing to row-Pearson, which is the desired behavior."""
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma


def _build_std_aware_matrix(df_scales_emb, df_stds_emb):
    """Build the (D, 40) per-column z-scored concat of per-AA (mean, std).

    Both halves are z-scored across the D dimensions independently so neither
    dominates row-Pearson when AAclust later compares dimension descriptors.
    """
    M = df_scales_emb.T.values  # (D, 20)
    S = df_stds_emb.T.values    # (D, 20)
    return np.concatenate([_zscore_columns(M), _zscore_columns(S)], axis=1)


# II Main Functions
def cluster_pseudo_scales_(df_scales_emb=None, df_stds_emb=None, cat_min_th=None, subcat_min_th=None,
                           random_state=None, metric="correlation"):
    """Cluster pseudo-scales at two correlation thresholds via AAclust.

    Lazy-imports AAclust to avoid a circular import (data_handling is loaded
    before feature_engineering at top-level package init).

    When ``df_stds_emb`` is None, AAclust sees the standard (D, 20) per-AA
    mean matrix. When supplied, AAclust sees the (D, 40) z-scored concat of
    (mean, std) instead — see :func:`_build_std_aware_matrix`.

    ``metric`` is forwarded to AAclust and controls the *merge* step + medoid
    selection only — k-optimization itself remains Pearson-correlation-based.
    """
    # Lazy import to break the data_handling → feature_engineering cycle
    from aaanalysis.feature_engineering._aaclust import AAclust

    if df_stds_emb is None:
        X = df_scales_emb.T.values
    else:
        X = _build_std_aware_matrix(df_scales_emb, df_stds_emb)
    dim_names = list(df_scales_emb.columns)

    ac_cat = AAclust(verbose=False, random_state=random_state)
    ac_cat.fit(X=X, min_th=cat_min_th, names=dim_names, metric=metric)

    ac_subcat = AAclust(verbose=False, random_state=random_state)
    ac_subcat.fit(X=X, min_th=subcat_min_th, names=dim_names, metric=metric)

    # v1.1 category scheme: all PLM-derived dims share the top-level
    # ``'Embeddings'`` bucket (paired with ``ut.DICT_COLOR_CAT['Embeddings']``);
    # the two AAclust cluster labels move into subcategory as a structured
    # "cat:<i>|subcat:<j>" string so the redundancy filter still sees the
    # finer split via the subcategory column while CPPPlot resolves a single
    # color.
    df_cat_emb = pd.DataFrame({
        ut.COL_SCALE_ID: dim_names,
        ut.COL_CAT: ["Embeddings" for _ in dim_names],
        ut.COL_SUBCAT: [f"Embeddings_cat{c}_subcat{s}"
                        for c, s in zip(ac_cat.labels_, ac_subcat.labels_)],
        ut.COL_SCALE_NAME: dim_names,
        ut.COL_SCALE_DES: ["" for _ in dim_names],
    })
    return df_cat_emb
