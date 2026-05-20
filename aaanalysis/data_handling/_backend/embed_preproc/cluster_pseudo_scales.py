"""
This is a script for the backend of EmbeddingPreprocessor.cluster_pseudo_scales.

Two independent AAclust runs at different correlation thresholds produce
coarser (``cat``) and finer (``subcat``) labels for each embedding dimension,
mirroring the AAontology two-level category hierarchy. The two runs are
independent: subcat labels do not necessarily nest within cat labels —
they're two views over the same pseudo-scales.
"""
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
# (no helpers — single-function module)


# II Main Functions
# TODO consider the right similarity metric for clsutering pseudo-scales. Correlation is a common choice for expression profiles,
#  but maybe cosine similarity or Euclidean distance would be better for embedding dimensions? Could even make it an option.
# TODO similiartiy measures or clsutering algorihtms that take into account std might be usefull as well, e.g. to 
# avoid clustering together two dimensions that have similar means but very different variances across AAs.
def cluster_pseudo_scales_(df_scales_emb=None, cat_min_th=None, subcat_min_th=None, random_state=None):
    """Cluster pseudo-scales at two correlation thresholds via AAclust.

    Lazy-imports AAclust to avoid a circular import (data_handling is loaded
    before feature_engineering at top-level package init).
    """
    # Lazy import to break the data_handling → feature_engineering cycle
    from aaanalysis.feature_engineering._aaclust import AAclust

    # AAclust expects (n_samples, n_features) where n_samples = scales,
    # n_features = AAs. Our df_scales_emb is (20, D), so transpose.
    X = df_scales_emb.T.values
    dim_names = list(df_scales_emb.columns)

    ac_cat = AAclust(verbose=False, random_state=random_state)
    ac_cat.fit(X=X, min_th=cat_min_th, names=dim_names)

    ac_subcat = AAclust(verbose=False, random_state=random_state)
    ac_subcat.fit(X=X, min_th=subcat_min_th, names=dim_names)

    df_cat_emb = pd.DataFrame({
        ut.COL_SCALE_ID: dim_names,
        ut.COL_CAT: [f"PLM_cat_{c}" for c in ac_cat.labels_],
        ut.COL_SUBCAT: [f"PLM_subcat_{s}" for s in ac_subcat.labels_],
        ut.COL_SCALE_NAME: dim_names,
        ut.COL_SCALE_DES: ["" for _ in dim_names],
    })
    return df_cat_emb
