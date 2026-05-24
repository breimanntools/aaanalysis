"""
This is a script for the backend of CPP's numerical-mode pre-filter statistics
stage: ``pre_filtering_info_num`` consumes the (dict_part_vals, dict_part_lens)
contract from ``_filters_num._assign`` and emits the same (abs_mean_dif,
std_test, features) tuple as ``_filters._stat_filter`` — bit-identical numerics
in seq-mode by construction (same scale_matrix lookup, same nanmean buffer).

# DEV: PR2 fuses this stage with the pre-filter step (streaming pre-filter):
# the (n_samples, n_feats_part) per-sample matrix is kept in memory for
# survivors of the std_test mask so ``add_stat`` no longer recomputes via
# get_feature_matrix_. PR2 also vectorizes the split-position computation
# across the D axis (one numpy op per part instead of n_scales x n_parts).
"""
import os
import numpy as np
from joblib import Parallel, delayed

import aaanalysis.utils as ut
from .._split import SplitRange
from ._progress import (
    _resolve_shared,
    _reset_progress,
    _cleanup_mp_manager,
)
from ._recompute import build_position_buffer, iter_scale_chunks


# I Helper Functions
def _get_split_labels(split_type=None, split_type_args=None, spr=None):
    """Fetch split labels dynamically (mirrors _filters._stat_filter helper)."""
    if split_type_args is not None:
        labels_splits = getattr(spr, "labels_" + split_type.lower())(**split_type_args)
    else:
        labels_splits = getattr(spr, "labels_" + split_type.lower())()
    return labels_splits


def _get_f_split_num(split_type=None, split_type_args=None, len_seq_max=None, spr=None):
    """Iterative split function for ``vectorized=False`` — applies split per sample row."""
    f = getattr(spr, split_type.lower())

    def f_split(seq):
        # seq is a 1D float ndarray already trimmed to its true length by caller.
        splits = f(seq=seq, **(split_type_args or {}))
        X = np.full((len(splits), len_seq_max), np.nan, dtype=np.float64)
        for i, x in enumerate(splits):
            X[i, : len(x)] = x
        return np.nanmean(X, axis=-1)

    return f_split


def _get_vf_split_num(split_type=None, split_type_args=None, len_seq_max=None, spr=None,
                     list_splits=None, n_samples=None):
    """Vectorized split function for ``vectorized=True`` — single (n_samples, n_splits, L_max) buffer."""
    n_splits = len(list_splits)
    f = getattr(spr, split_type.lower())

    def vf_split(arr_seq, seq_lengths):
        # arr_seq: (n, L) float ndarray; seq_lengths: (n,) int ndarray.
        list_split_vals = [
            f(seq=arr_seq[i, : int(seq_lengths[i])], **(split_type_args or {}))
            for i in range(len(arr_seq))
        ]
        X = np.full((n_samples, n_splits, len_seq_max), np.nan, dtype=np.float64)
        for i, split in enumerate(list_split_vals):
            for j, x in enumerate(split):
                X[i, j, : len(x)] = x
        return np.nanmean(X, axis=-1)

    return vf_split


@ut.catch_runtime_warnings(suppress=True)
def _pre_filtering_info_num_split_type(
    dict_part_vals=None,
    dict_part_lens=None,
    list_scales=None,
    list_parts=None,
    split_type=None,
    split_kws=None,
    mask_ref=None,
    mask_test=None,
    len_seq_max=None,
    spr=None,
    vectorized=True,
    max_std_test=0.2,
    accept_gaps=False,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """One split-type's stats, iterating per (scale, part) on the (n, L, D) tensors."""
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)

    split_type_args = split_kws[split_type]

    n_samples = len(mask_test)
    mask_test_arr = np.asarray(mask_test, dtype=bool)
    mask_ref_arr = np.asarray(mask_ref, dtype=bool)

    # PR2.5+ vectorization: per ``(split_type, part)`` bucket, build the
    # per-sample split-position buffer ONCE (sentinel-sequence trick), then
    # chunk across scales for fancy-indexed gather + nanmean. Replaces the
    # per-(scale, part) ``vf_split`` Python loop. No cache — survivor values
    # are rebuilt in a focused second pass via ``_recompute``.
    kept_abs_mean_dif = []
    kept_std_test = []
    kept_feat_names = []

    for j, part in enumerate(list_parts):
        arr_3d = dict_part_vals[part]  # (n, L_part_max, D)
        seq_lens = dict_part_lens[part]
        L_part_max = arr_3d.shape[1]

        pos_buf, labels_splits, _ = build_position_buffer(
            split_type=split_type, split_type_args=split_type_args,
            seq_lens=seq_lens, L_max=L_part_max, spr=spr,
        )

        # Stream per-chunk to keep memory bounded — never materialize the full
        # (n_samples, n_splits, n_scales) means matrix.
        scale_indices = list(range(len(list_scales)))
        part_upper = part.upper()

        for chunk_start, chunk_means in iter_scale_chunks(
            arr_3d=arr_3d, scale_indices=scale_indices, pos_buf=pos_buf,
        ):
            # chunk_means shape: (n_samples, n_splits_part, Kc)
            test_vals = chunk_means[mask_test_arr]
            ref_vals = chunk_means[mask_ref_arr]
            test_means = np.mean(test_vals, axis=0)  # (n_splits, Kc)
            ref_means = np.mean(ref_vals, axis=0)
            std_test_chunk = np.std(test_vals, axis=0)
            abs_mean_dif_chunk = np.abs(test_means - ref_means)

            keep = std_test_chunk <= max_std_test
            if accept_gaps:
                keep &= ~np.isnan(abs_mean_dif_chunk)
            if not keep.any():
                continue

            kept_split_idx, kept_local_scale_idx = np.where(keep)
            kept_abs_mean_dif.append(
                abs_mean_dif_chunk[kept_split_idx, kept_local_scale_idx]
            )
            kept_std_test.append(
                std_test_chunk[kept_split_idx, kept_local_scale_idx]
            )
            # Translate local scale index → global scale name.
            for s_idx, local_sc_idx in zip(kept_split_idx, kept_local_scale_idx):
                global_scale = list_scales[chunk_start + local_sc_idx]
                kept_feat_names.append(
                    f"{part_upper}-{labels_splits[s_idx]}-{global_scale}"
                )

        if verbose:
            ut.print_progress(i=j, n_total=len(list_parts), **args_p)

    if kept_abs_mean_dif:
        out_abs_mean_dif = np.concatenate(kept_abs_mean_dif)
        out_std_test = np.concatenate(kept_std_test)
    else:
        out_abs_mean_dif = np.zeros((0,))
        out_std_test = np.zeros((0,))
    return out_abs_mean_dif, out_std_test, kept_feat_names


def _pre_filtering_info_num(
    dict_part_vals=None,
    dict_part_lens=None,
    list_scales=None,
    split_kws=None,
    list_parts=None,
    mask_ref=None,
    mask_test=None,
    len_seq_max=None,
    spr=None,
    vectorized=True,
    max_std_test=0.2,
    accept_gaps=False,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """Run all split types for one (possibly chunked) set of scales."""
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )

    args = dict(
        dict_part_vals=dict_part_vals,
        dict_part_lens=dict_part_lens,
        list_scales=list_scales,
        split_kws=split_kws,
        list_parts=list_parts,
        mask_test=mask_test,
        mask_ref=mask_ref,
        len_seq_max=len_seq_max,
        spr=spr,
        vectorized=vectorized,
        max_std_test=max_std_test,
        accept_gaps=accept_gaps,
        verbose=verbose,
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )

    feat_names = []
    abs_mean_dif = []
    std_test = []

    for split_type in split_kws:
        _abs_mean_dif, _std_test, _feat_names = _pre_filtering_info_num_split_type(
            split_type=split_type, **args
        )
        abs_mean_dif.append(_abs_mean_dif)
        std_test.append(_std_test)
        feat_names.extend(_feat_names)

    if abs_mean_dif:
        out_abs_mean_dif = np.concatenate(abs_mean_dif)
        out_std_test = np.concatenate(std_test)
    else:
        out_abs_mean_dif = np.zeros((0,))
        out_std_test = np.zeros((0,))
    return out_abs_mean_dif, out_std_test, np.array(feat_names, dtype=object)


# II Main Functions
def pre_filtering_info_num(df_parts=None, split_kws=None, dict_part_vals=None,
                           dict_part_lens=None, list_scales=None, labels=None,
                           label_test=1, label_ref=0, max_std_test=0.2,
                           accept_gaps=False, verbose=False, n_jobs=None,
                           vectorized=True):
    """Streaming pre-filter stats with in-stream mask (PR2.5).

    Drops features whose ``std_test > max_std_test`` (or whose ``abs_mean_dif``
    is NaN under ``accept_gaps``) immediately, so memory stays bounded at
    O(n_features_kept) rather than O(n_features_total). No per-sample cache —
    the (n_samples, n_pre_filter) survivor matrix is rebuilt in a focused
    second pass by ``_filters_num._recompute.recompute_feature_matrix_num``.

    Returns
    -------
    abs_mean_dif : np.ndarray, shape (n_features_kept,)
    std_test : np.ndarray, shape (n_features_kept,)
    feat_names : np.ndarray of str (object dtype), shape (n_features_kept,)
    """
    mask_ref = [x == label_ref for x in labels]
    mask_test = [x == label_test for x in labels]
    list_parts = list(df_parts)
    len_seq_max = df_parts.map(len).max().max()
    spr = SplitRange(split_type_str=False)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, len(list_scales))

    prefer_mp_progress = bool(n_jobs and n_jobs > 1)
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(prefer_multiprocessing=prefer_mp_progress)
    _reset_progress(shared_max_progress, shared_value_lock)

    args = dict(
        dict_part_lens=dict_part_lens,
        split_kws=split_kws,
        list_parts=list_parts,
        mask_test=mask_test,
        mask_ref=mask_ref,
        len_seq_max=len_seq_max,
        spr=spr,
        vectorized=vectorized,
        max_std_test=max_std_test,
        accept_gaps=accept_gaps,
        verbose=verbose,
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_mp_progress,
    )

    if n_jobs == 1:
        result = _pre_filtering_info_num(
            dict_part_vals=dict_part_vals, list_scales=list_scales, **args
        )
        if prefer_mp_progress:
            _cleanup_mp_manager()
        return result

    def _mp_pre_filtering_info(scale_idx_chunk, shared_max_progress, shared_value_lock, print_lock):
        chunk_idx = list(scale_idx_chunk)
        chunked_vals = {part: arr[:, :, chunk_idx] for part, arr in dict_part_vals.items()}
        chunk_scales = [list_scales[i] for i in chunk_idx]
        return _pre_filtering_info_num(
            dict_part_vals=chunked_vals,
            list_scales=chunk_scales,
            dict_part_lens=dict_part_lens,
            split_kws=split_kws,
            list_parts=list_parts,
            mask_ref=mask_ref,
            mask_test=mask_test,
            len_seq_max=len_seq_max,
            spr=spr,
            vectorized=vectorized,
            max_std_test=max_std_test,
            accept_gaps=accept_gaps,
            verbose=verbose,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
            print_lock=print_lock,
            prefer_multiprocessing=False,
        )

    scale_idx_chunks = np.array_split(np.arange(len(list_scales)), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_mp_pre_filtering_info)(chunk, shared_max_progress, shared_value_lock, print_lock)
            for chunk in scale_idx_chunks
        )

    if verbose:
        ut.print_end_progress(
            add_new_line=False,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
        )

    abs_mean_dif = np.concatenate([res[0] for res in results])
    std_test = np.concatenate([res[1] for res in results])
    feat_names = np.concatenate([res[2] for res in results])

    if prefer_mp_progress:
        _cleanup_mp_manager()
    return abs_mean_dif, std_test, feat_names


# ---------------------------------------------------------------------------
# Sample-batched pre-filter stats (Phase 1: unbounded n support)
# ---------------------------------------------------------------------------
def build_feature_index_map(list_parts=None, split_kws=None, list_scales=None):
    """Build canonical feature ordering + per-(split_type, part) 2D index tables.

    Returns
    -------
    features_canonical : list[str]
        Feature names in canonical order. Indices match the per-feature stats
        arrays produced by the sample-batched accumulator.
    indices_map : dict
        ``indices_map[split_type][part]`` is a ``(n_splits, n_scales)`` int64
        matrix giving the canonical feature index for each
        ``(split_label, scale_name)`` pair. Lets the per-chunk accumulator
        update step be fully vectorized via fancy-indexed numpy add.
    """
    spr = SplitRange(split_type_str=False)
    features_canonical = []
    indices_map = {}
    n_scales = len(list_scales)
    for split_type in split_kws:
        split_type_args = split_kws[split_type]
        labels_splits = getattr(spr, "labels_" + split_type.lower())(**split_type_args)
        n_splits = len(labels_splits)
        indices_map[split_type] = {}
        for part in list_parts:
            mat = np.zeros((n_splits, n_scales), dtype=np.int64)
            part_upper = part.upper()
            for s_idx, lbl in enumerate(labels_splits):
                for sc_idx, scale in enumerate(list_scales):
                    feat_name = f"{part_upper}-{lbl}-{scale}"
                    mat[s_idx, sc_idx] = len(features_canonical)
                    features_canonical.append(feat_name)
            indices_map[split_type][part] = mat
    return features_canonical, indices_map


@ut.catch_runtime_warnings(suppress=True)
def accumulate_partial_stats(
    dict_part_vals=None, dict_part_lens=None, list_parts=None, list_scales=None,
    split_kws=None, labels_batch=None, label_test=None, label_ref=None,
    sum_test=None, sum_sq_test=None, sum_ref=None, count_test=None, count_ref=None,
    indices_map=None,
):
    """Update per-feature stat accumulators with one batch's contribution.

    All ``sum_*``/``count_*`` arrays are mutated in-place. After processing
    all batches, ``finalize_stats`` combines them into the (abs_mean_dif,
    std_test, features) tuple compatible with ``pre_filtering``.
    """
    spr = SplitRange(split_type_str=False)
    mask_test_arr = np.asarray([x == label_test for x in labels_batch], dtype=bool)
    mask_ref_arr = np.asarray([x == label_ref for x in labels_batch], dtype=bool)
    n_test_batch = int(mask_test_arr.sum())
    n_ref_batch = int(mask_ref_arr.sum())

    for split_type in split_kws:
        split_type_args = split_kws[split_type]
        for part in list_parts:
            arr_3d = dict_part_vals[part]
            seq_lens = dict_part_lens[part]
            L_part_max = arr_3d.shape[1]
            pos_buf, labels_splits, _ = build_position_buffer(
                split_type=split_type, split_type_args=split_type_args,
                seq_lens=seq_lens, L_max=L_part_max, spr=spr,
            )
            indices_2d = indices_map[split_type][part]  # (n_splits, n_scales)
            for chunk_start, chunk_means in iter_scale_chunks(
                arr_3d=arr_3d, scale_indices=list(range(len(list_scales))),
                pos_buf=pos_buf,
            ):
                # chunk_means shape: (batch, n_splits_part, Kc)
                Kc = chunk_means.shape[2]
                idx_chunk = indices_2d[:, chunk_start : chunk_start + Kc]  # (n_splits, Kc)
                flat_idx = idx_chunk.ravel()
                test_vals = chunk_means[mask_test_arr]  # (n_test_batch, n_splits, Kc)
                ref_vals = chunk_means[mask_ref_arr]
                chunk_sum_test = test_vals.sum(axis=0).ravel()
                chunk_sum_sq_test = (test_vals * test_vals).sum(axis=0).ravel()
                chunk_sum_ref = ref_vals.sum(axis=0).ravel()
                sum_test[flat_idx] += chunk_sum_test
                sum_sq_test[flat_idx] += chunk_sum_sq_test
                sum_ref[flat_idx] += chunk_sum_ref
                count_test[flat_idx] += n_test_batch
                count_ref[flat_idx] += n_ref_batch


def finalize_stats(sum_test=None, sum_sq_test=None, sum_ref=None,
                  count_test=None, count_ref=None, features_canonical=None,
                  max_std_test=0.2, accept_gaps=False):
    """Combine batch accumulators into final stats; apply std_test + NaN masks.

    Returns ``(abs_mean_dif_kept, std_test_kept, feat_names_kept)`` matching
    the single-pass ``pre_filtering_info_num`` output contract.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_test = sum_test / count_test
        mean_ref = sum_ref / count_ref
        var_test = sum_sq_test / count_test - mean_test * mean_test
        var_test = np.maximum(var_test, 0.0)  # guard against tiny negatives from rounding
        std_test_arr = np.sqrt(var_test)
        abs_mean_dif_arr = np.abs(mean_test - mean_ref)
    keep = std_test_arr <= max_std_test
    if accept_gaps:
        keep &= ~np.isnan(abs_mean_dif_arr)
    kept_idx = np.flatnonzero(keep)
    feat_names_kept = [features_canonical[i] for i in kept_idx]
    return (
        abs_mean_dif_arr[kept_idx],
        std_test_arr[kept_idx],
        np.array(feat_names_kept, dtype=object),
    )
