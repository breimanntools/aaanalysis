"""
Cython-accelerated bit-exact feature-matrix builder.

Dispatches per (split_type) to the C kernels in ``_inner`` for Segment /
Pattern-N / Pattern-C; falls back to Phase-C Python for PeriodicPattern
(its position list is per-sample and not yet inlined in C).

The output is byte-identical to ``get_feature_matrix_fast_`` (which is
byte-identical to legacy ``get_feature_matrix_``). The Cython kernels
replicate numpy's pairwise-summation algorithm and apply ``np.round(_, 5)``
on the result, so per-sample means match ``np.mean(arr[i, positions])``
bit-for-bit.
"""
import os
import numpy as np
from joblib import Parallel, delayed

import aaanalysis.utils as ut
from .._split import SplitVec
from ..utils_feature import _get_split_info, post_check_vf_scale
from .._filters_num._get_feature_matrix_fast import (
    AALookupCache,
    _build_aa_idx_per_part,
    _build_scale_matrix_f64,
    _hoisted_gap_check,
)

from ._inner import (
    compute_segment_mean,
    compute_pattern_n_mean,
    compute_pattern_c_mean,
    compute_segment_nanmean,
    compute_pattern_n_nanmean,
    compute_pattern_c_nanmean,
)


# Minimum features per worker thread. Below this, threading overhead
# (~1–5 ms per dispatch on macOS) eats the per-thread savings — the Cython
# kernel finishes a feature in microseconds, so each thread needs a healthy
# chunk to amortize dispatch. Tuned empirically: at ~500 features/thread we
# spend <0.5% of total time on joblib bookkeeping.
_MIN_FEATURES_PER_THREAD = 500

# Floor: below this total feature count, the whole job finishes faster
# single-threaded than the overhead of spinning up any threads at all.
_MIN_FEATURES_FOR_THREADING = 2000


def _pick_n_jobs_cython(n_features=None, n_jobs=None):
    """Heuristic n_jobs picker tailored to the Cython kernels.

    Unlike the Python path (``get_feature_matrix_fast_``) where each
    per-feature ``np.mean`` is slow Python work that amortizes process-pool
    fork+pickle, the Cython kernels finish a single feature in
    microseconds. That changes the math:

    * For ``n_features < _MIN_FEATURES_FOR_THREADING``, total work is so
      small that even thread spin-up costs more than it saves. Use 1 thread.
    * Otherwise scale up to ``cpu_count``, but ensure each thread gets at
      least ``_MIN_FEATURES_PER_THREAD`` features so dispatch overhead stays
      below ~1 % of total kernel time.

    Realistic thread speedup is capped at ~1.5–3× because each kernel call
    still has a small GIL-held tail (``np.empty`` + ``np.round`` at the
    Python boundary). For higher speedup we'd need to hoist those out of
    the per-feature loop — out of scope here.
    """
    if n_jobs is not None:
        return n_jobs
    if n_features < _MIN_FEATURES_FOR_THREADING:
        return 1
    cpu = os.cpu_count() or 1
    return min(cpu, max(n_features // _MIN_FEATURES_PER_THREAD, 1))


def _compute_features_into_c(X_out=None, features=None,
                              aa_idx_per_part=None, seq_lens_per_part=None,
                              scale_matrix_f64=None, scale_to_idx=None,
                              sp_vec=None, accept_gaps=False):
    """Inline-by-split-type, but with the per-sample inner loop in Cython.

    Bit-identical to ``_compute_features_into`` (Phase C) — uses the SAME
    ``np.mean`` reduction order via the bit-exact Cython kernels in
    ``_inner.pyx``.

    The accept_gaps branch falls through to Phase-C Python because the
    Cython kernels assume no NaN values (canonical AAs only).
    """
    split_info_cache = {}

    for feat_idx, feat_name in enumerate(features):
        part_upper, split_str, scale = feat_name.split("-")
        part = part_upper.lower()

        cached = split_info_cache.get(split_str)
        if cached is None:
            split_type, split_kwargs = _get_split_info(split=split_str)
            f_split_num = getattr(sp_vec, split_type.lower())
            cached = (split_type, split_kwargs, f_split_num)
            split_info_cache[split_str] = cached
        split_type, split_kwargs, f_split_num = cached

        scale_idx = scale_to_idx[scale]
        aa_idx = aa_idx_per_part[part]
        seq_lens = seq_lens_per_part[part]
        arr_2d = np.ascontiguousarray(scale_matrix_f64[aa_idx, scale_idx])

        if split_type == "Segment":
            if accept_gaps:
                col = compute_segment_nanmean(
                    arr_2d, seq_lens, int(split_kwargs["i_th"]),
                    int(split_kwargs["n_split"]),
                )
            else:
                col = compute_segment_mean(
                    arr_2d, seq_lens, int(split_kwargs["i_th"]),
                    int(split_kwargs["n_split"]),
                )
        elif split_type == "Pattern" and split_kwargs["terminus"] == "N":
            positions = np.asarray(split_kwargs["list_pos"], dtype=np.int64) - 1
            if accept_gaps:
                col = compute_pattern_n_nanmean(arr_2d, positions)
            else:
                col = compute_pattern_n_mean(arr_2d, positions)
        elif split_type == "Pattern" and split_kwargs["terminus"] == "C":
            list_pos_arr = np.asarray(split_kwargs["list_pos"], dtype=np.int64)
            if accept_gaps:
                col = compute_pattern_c_nanmean(arr_2d, seq_lens, list_pos_arr)
            else:
                col = compute_pattern_c_mean(arr_2d, seq_lens, list_pos_arr)
        else:
            # PeriodicPattern (and any future split_type) — Phase-C fallback.
            col = _fallback_python_mean(
                arr_2d=arr_2d, seq_lens=seq_lens, split_type=split_type,
                split_kwargs=split_kwargs, f_split_num=f_split_num,
                accept_gaps=accept_gaps,
            )

        if accept_gaps:
            post_check_vf_scale(feature_values=col)
        X_out[:, feat_idx] = col


def _fallback_python_mean(arr_2d, seq_lens, split_type, split_kwargs,
                          f_split_num, accept_gaps):
    """Phase-C Python inner loop — used for accept_gaps + PeriodicPattern paths."""
    n = arr_2d.shape[0]
    seq_lens_int = seq_lens.tolist()
    col = np.empty(n, dtype=np.float64)
    for i in range(n):
        L_i = seq_lens_int[i]
        seq_vals = arr_2d[i, :L_i]
        segs = f_split_num(seq=seq_vals, **split_kwargs)
        if accept_gaps:
            vals = np.asarray(segs, dtype=np.float64)
            col[i] = np.nanmean(vals)
        else:
            col[i] = np.mean(segs)
    return np.round(col, 5)


def get_feature_matrix_c_(features=None, df_parts=None, df_scales=None,
                          accept_gaps=False, n_jobs=None,
                          aa_lookup_cache=None):
    """Cython-accelerated drop-in replacement for ``get_feature_matrix_fast_``.

    Byte-identical to legacy ``get_feature_matrix_`` (verified via
    ``test_get_feature_matrix_c_parity.py``).
    """
    features = [features] if isinstance(features, str) else list(features)

    if not accept_gaps:
        _hoisted_gap_check(df_parts=df_parts)

    if aa_lookup_cache is not None and aa_lookup_cache.matches(df_parts, df_scales):
        aa_idx_per_part = aa_lookup_cache.aa_idx_per_part
        seq_lens_per_part = aa_lookup_cache.seq_lens_per_part
        scale_matrix_f64 = aa_lookup_cache.scale_matrix_f64
        scale_to_idx = aa_lookup_cache.scale_to_idx
    else:
        dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col]))
                           for col in list(df_scales)}
        scale_to_idx = {s: i for i, s in enumerate(list(df_scales))}
        aa_idx_per_part, seq_lens_per_part, _, n_aa = _build_aa_idx_per_part(
            df_parts=df_parts, dict_all_scales=dict_all_scales,
        )
        scale_matrix_f64 = _build_scale_matrix_f64(
            dict_all_scales=dict_all_scales, n_aa=n_aa,
        )

    sp_vec = SplitVec(type_str=False)
    n_samples = len(df_parts)
    n_jobs = _pick_n_jobs_cython(n_features=len(features), n_jobs=n_jobs)

    if n_jobs == 1 or len(features) <= 1:
        X = np.empty((n_samples, len(features)), dtype=np.float64)
        _compute_features_into_c(
            X_out=X, features=features,
            aa_idx_per_part=aa_idx_per_part, seq_lens_per_part=seq_lens_per_part,
            scale_matrix_f64=scale_matrix_f64, scale_to_idx=scale_to_idx,
            sp_vec=sp_vec, accept_gaps=accept_gaps,
        )
        return X

    feature_chunks = np.array_split(features, n_jobs)

    def _mp(chunk):
        X_chunk = np.empty((n_samples, len(chunk)), dtype=np.float64)
        _compute_features_into_c(
            X_out=X_chunk, features=list(chunk),
            aa_idx_per_part=aa_idx_per_part, seq_lens_per_part=seq_lens_per_part,
            scale_matrix_f64=scale_matrix_f64, scale_to_idx=scale_to_idx,
            sp_vec=sp_vec, accept_gaps=accept_gaps,
        )
        return X_chunk

    # CRITICAL: ``prefer='threads'`` — the Cython kernels release the GIL
    # (``with nogil:`` in ``_inner.pyx``) so threads run in true parallel
    # without fork+pickle overhead. joblib's default ``loky`` backend uses
    # processes and pays ~50–200 ms per worker just to fork + serialize the
    # AA-lookup arrays; for a kernel that finishes the whole 29 K-feature
    # job in ~1 s that overhead is catastrophic. Threading also keeps the
    # shared ``aa_idx_per_part`` / ``scale_matrix_f64`` arrays in one
    # address space → no pickling, no copies.
    with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
        results = parallel(delayed(_mp)(chunk) for chunk in feature_chunks)
    return np.concatenate(results, axis=1)
