"""
This is a script for the backend of ``NumericalFeature.feature_matrix``.

``get_feature_matrix_num_`` reconstructs the ``(n_samples, n_features)`` value
matrix ``X`` from pre-sliced per-part numerical tensors (``dict_num_parts``),
the numerical-mode analog of :func:`utils_feature.get_feature_matrix_`.

The value of feature ``PART-SPLIT-SCALE`` for sample ``i`` is:

    ``round(nanmean( arr[part][i, split_positions_i, scale_idx] ), 5)``

where ``split_positions_i`` are the **0-based residue indices** obtained by
applying the ``SPLIT`` to ``arange(L_part_i)`` (``L_part_i`` = the sample's real
residue count for that part), and ``scale_idx`` is the D-axis index of ``SCALE``
in ``df_scales.columns``. This is the SAME reconstruction ``CPP.run_num`` uses in
``recompute_feature_matrix`` (position buffer built from ``arange`` + per-split
``np.nanmean`` + ``np.round(_, 5)``), so the columns of ``X`` are byte-identical
to the values ``run_num`` computed for the same feature ids — verified against
``recompute_feature_matrix`` for uniform and variable-length parts.

Note that the ``positions`` column of ``run_num``'s ``df_feat`` is a *display*
numbering in TMD-JMD coordinate space (JMD-offset, e.g. ``21..30`` for a TMD),
NOT an index into the ``(L, D)`` tensor; value reconstruction therefore re-applies
the SPLIT to the residue axis rather than reading ``positions``.
"""
import numpy as np
from joblib import Parallel, delayed

import aaanalysis.utils as ut
from ..cpp._split import SplitVec
from ..cpp.utils_feature import _get_split_info


# I Helper Functions
def _feature_column_num(feature=None, dict_num_parts=None, part_lens=None,
                        scale_to_idx=None, sp_vec=None):
    """Compute the ``(n_samples,)`` value column for one ``PART-SPLIT-SCALE`` feature."""
    part_upper, split_str, scale = ut.split_feat_id(feat_id=feature)
    part = part_upper.lower()
    split_type, split_kwargs = _get_split_info(split=split_str)
    f_split = getattr(sp_vec, split_type.lower())
    arr_2d = dict_num_parts[part][:, :, scale_to_idx[scale]]  # (n, L_max) float64
    lens = part_lens[part]
    n_samples = arr_2d.shape[0]

    # Fast path: every sample shares the same real length for this part, so the
    # split positions are identical across samples -> one vectorized gather +
    # ``nanmean(axis=1)`` (element-wise identical to the per-sample nanmean).
    if n_samples > 0 and bool(np.all(lens == lens[0])):
        positions = np.asarray(
            f_split(seq=np.arange(int(lens[0]), dtype=np.int64), **split_kwargs),
            dtype=np.int64,
        )
        if positions.size == 0:
            return np.round(np.full(n_samples, np.nan, dtype=np.float64), 5)
        col = np.nanmean(arr_2d[:, positions], axis=1)
        return np.round(col, 5)

    # General path: positions depend on each sample's real length.
    col = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        positions = np.asarray(
            f_split(seq=np.arange(int(lens[i]), dtype=np.int64), **split_kwargs),
            dtype=np.int64,
        )
        col[i] = np.nanmean(arr_2d[i, positions]) if positions.size else np.nan
    return np.round(col, 5)


def _compute_features_num(features=None, dict_num_parts=None, part_lens=None,
                          scale_to_idx=None):
    """Fill an ``(n_samples, n_features)`` matrix for a chunk of features."""
    sp_vec = SplitVec(type_str=False)  # one shared instance, hoisted out of the loop
    n_samples = next(iter(dict_num_parts.values())).shape[0]
    X = np.empty((n_samples, len(features)), dtype=np.float64)
    for j, feature in enumerate(features):
        X[:, j] = _feature_column_num(
            feature=feature, dict_num_parts=dict_num_parts, part_lens=part_lens,
            scale_to_idx=scale_to_idx, sp_vec=sp_vec,
        )
    return X


# II Main Functions
def get_feature_matrix_num_(features=None, dict_num_parts=None, part_lens=None,
                            df_scales=None, n_jobs=None):
    """Build the ``(n_samples, n_features)`` numerical-mode feature matrix ``X``.

    Numerical-mode analog of ``utils_feature.get_feature_matrix_``: instead of an
    AA -> scale lookup, per-residue values come from the pre-sliced
    ``dict_num_parts`` tensors. Byte-identical to the values ``CPP.run_num``
    computes for the same feature ids (verified against ``recompute_feature_matrix``).

    Parameters
    ----------
    features : list of str
        Feature ids (``PART-SPLIT-SCALE``); output column order matches.
    dict_num_parts : dict[str, np.ndarray]
        Per-part ``(n_samples, L_part_max, D)`` NaN-padded numerical tensors, as
        produced by ``NumericalFeature.get_parts``.
    part_lens : dict[str, np.ndarray]
        Per-part ``(n_samples,)`` int64 real residue counts, from
        ``_derive_dict_part_lens(df_parts)`` — the SAME length source ``CPP.run_num``
        uses (non-gap character count of the string ``df_parts``). Applying each
        split to ``arange(part_lens[part][i])`` therefore lands on exactly the
        residues ``run_num`` selected.
    df_scales : pd.DataFrame
        Names the D dimensions; ``df_scales.columns`` order defines the SCALE ->
        D-index mapping.
    n_jobs : int, None, or -1
        CPU cores for the feature loop (resolved via ``ut.resolve_n_jobs``).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features) float64
        Per-sample per-feature values, rounded to 5 decimals.
    """
    features = [features] if isinstance(features, str) else list(features)
    scale_to_idx = {s: i for i, s in enumerate(list(df_scales.columns))}

    n_jobs = ut.resolve_n_jobs(n_jobs=n_jobs, n_work=len(features))
    if n_jobs == 1 or len(features) <= 1:
        return _compute_features_num(
            features=features, dict_num_parts=dict_num_parts,
            part_lens=part_lens, scale_to_idx=scale_to_idx,
        )

    feature_chunks = np.array_split(features, n_jobs)

    def _mp(chunk):
        return _compute_features_num(
            features=list(chunk), dict_num_parts=dict_num_parts,
            part_lens=part_lens, scale_to_idx=scale_to_idx,
        )

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_mp)(chunk) for chunk in feature_chunks)
    return np.concatenate(results, axis=1)
