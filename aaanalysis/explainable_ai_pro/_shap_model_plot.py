"""
This is a script for the stand-alone ``shap_to_feat_imp`` helper.

It complements :class:`ShapModel` (which computes the SHAP values) with a helper that turns a
per-sample SHAP vector into a normalized signed feature impact / absolute importance.
"""
import numpy as np

import aaanalysis.utils as ut
from ._backend.shap_model.sm_add_feat_impact import _comp_sample_shap_feat_impact


# II Main Functions
def shap_to_feat_imp(shap_values: ut.ArrayLike1D,
                     impact: bool = True,
                     ) -> np.ndarray:
    """
    Convert a per-sample SHAP-value vector into normalized feature impact or importance
    (**[pro]**, requires ``aaanalysis[pro]``).

    For one sample (or the mean SHAP vector of a group of same-class samples), the SHAP
    values are normalized so the sum of their absolute values equals 100%:

    - **feature impact** (``impact=True``): signed, ``shap / sum(|shap|) * 100`` — keeps the
      sign, so a feature that pushes the prediction up is positive and one that pushes it
      down is negative.
    - **feature importance** (``impact=False``): absolute, ``|shap| / sum(|shap|) * 100`` —
      magnitude only, the per-sample analogue of the SHAP value-based feature importance.

    This shares the normalization used internally by :meth:`ShapModel.add_feat_impact`
    (re-using its per-sample backend) so the two never diverge.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    shap_values : array-like, shape (n_features,)
        One-dimensional array of SHAP values for a single sample (or the mean SHAP vector of
        a group of same-class samples). Computation is only meaningful within one class.
    impact : bool, default=True
        If ``True``, return the signed feature impact; if ``False``, the absolute feature importance.

    Returns
    -------
    feat_imp : np.ndarray, shape (n_features,)
        Normalized feature impact (signed) or importance (absolute), summing in absolute
        value to 100.

    See Also
    --------
    * :meth:`ShapModel.add_feat_impact` for attaching impact/importance columns to a feature DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from aaanalysis.explainable_ai_pro import shap_to_feat_imp
    >>> shap_vec = np.array([0.2, -0.1, 0.3, -0.4])
    >>> impact = shap_to_feat_imp(shap_vec, impact=True)
    >>> float(np.round(np.abs(impact).sum(), 6))
    100.0
    """
    shap_values = ut.check_array_like(name="shap_values", val=shap_values,
                                      dtype="numeric", expected_dim=1)
    ut.check_bool(name="impact", val=impact)
    if np.nansum(np.abs(shap_values)) == 0:
        raise ValueError("'shap_values' are all zero; feature impact/importance is undefined.")
    if impact:
        # Re-use the per-sample backend used by ShapModel.add_feat_impact (no divergence)
        feat_imp = _comp_sample_shap_feat_impact(shap_values=shap_values.reshape(1, -1),
                                                 i=0, normalize=False)
        return np.asarray(feat_imp, dtype=float)
    abs_values = np.abs(shap_values)
    feat_imp = abs_values / np.nansum(abs_values) * 100
    return np.asarray(feat_imp, dtype=float)
