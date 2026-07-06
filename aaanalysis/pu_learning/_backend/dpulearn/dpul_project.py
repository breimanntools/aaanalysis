"""
This is a script for the backend of the dPULearn.project() method, mapping new samples into the
fitted PCA coordinate space (the ``PCi`` columns of ``dPULearn.df_pu_``).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import aaanalysis.utils as ut


# I Helper Functions
def get_pc_value_columns(df_pu=None):
    """Return the ``PCi`` value columns of ``df_pu`` (excludes the ``PCi_abs_dif`` helper columns)."""
    return [c for c in df_pu.columns if "PC" in c and "abs" not in c]


# II Main Functions
def project_into_pca_space(X_fit=None, df_pu=None, X_new=None, method="lstsq", alpha=1.0):
    """Project ``X_new`` into the PCA coordinate space fitted on ``X_fit`` and stored in ``df_pu``.

    dPULearn fits ``PCA().fit(X.T)`` (on the transpose), so ``df_pu`` holds ``pca.components_.T`` and
    there is no exact out-of-sample forward transform for a new sample's feature vector. Each ``method``
    reconstructs a linear map from the fit pairs ``(X_fit, df_pu[PC columns])`` and applies it to
    ``X_new``. Every map reproduces ``df_pu`` on the fit pool (when n_features >= n_samples), so it is
    exact for the fitted samples and an approximation (interpolation) for genuinely new points.
    """
    cols_pc = get_pc_value_columns(df_pu=df_pu)
    Z_fit = df_pu[cols_pc].to_numpy(dtype=float)
    X_fit = np.asarray(X_fit, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    if method == "lstsq":
        # Affine least-squares map [X | 1] -> Z; min-norm solution, exact on the fit pool.
        A_fit = np.hstack([X_fit, np.ones((len(X_fit), 1))])
        W, *_ = np.linalg.lstsq(A_fit, Z_fit, rcond=None)
        Z_new = np.hstack([X_new, np.ones((len(X_new), 1))]) @ W
    elif method == "components":
        # Exact PCA-geometry map: row-center by each sample's own feature mean, then min-norm linear
        # map. This min-norm solution equals the fitted PCA's U @ inv(Sigma) restricted to the stored
        # components, so it reproduces df_pu's PCi columns on the fit pool via the actual SVD geometry.
        X_fit_c = X_fit - X_fit.mean(axis=1, keepdims=True)
        M = np.linalg.pinv(X_fit_c) @ Z_fit
        Z_new = (X_new - X_new.mean(axis=1, keepdims=True)) @ M
    else:
        # Ridge: L2-regularized affine map; stabilizes extrapolation when n_features >> n_samples and
        # converges to the "lstsq" map as alpha -> 0. sklearn handles the (unpenalized) intercept.
        model = Ridge(alpha=alpha)
        model.fit(X_fit, Z_fit)
        Z_new = model.predict(X_new)
    df_proj = pd.DataFrame(np.asarray(Z_new, dtype=float), columns=cols_pc)
    return df_proj
