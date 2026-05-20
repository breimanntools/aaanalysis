"""
This is a script for the backend of CPP's pre-filter statistics stage:
``pre_filtering_info`` computes per-feature absolute mean difference and
test-set standard deviation, which feed the pre-filtering threshold step.
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


# I Helper Functions
def _get_split_labels(split_type=None, split_type_args=None, spr=None):
    """Fetch split labels dynamically."""
    if split_type_args is not None:
        labels_splits = getattr(spr, "labels_" + split_type.lower())(**split_type_args)
    else:
        labels_splits = getattr(spr, "labels_" + split_type.lower())()
    return labels_splits


def _get_f_split(split_type=None, split_type_args=None, len_seq_max=None, spr=None):
    """
    Retrieve a function for sequence splitting, applied along an array axis.

    This function is memory-efficient and offers good performance. It is used when `vectorized=False`,
    meaning it applies the sequence splitting function iteratively along the given axis.
    """
    f = getattr(spr, split_type.lower())

    def f_split(seq):
        seq_len = seq[-1]
        splits = f(seq=seq[: int(seq_len)], **(split_type_args or {}))
        X = np.full((len(splits), len_seq_max), np.nan, dtype=np.float64)
        for i, x in enumerate(splits):
            X[i, : len(x)] = x
        return np.nanmean(X, axis=-1)

    return f_split


def _get_vf_split(split_type=None, split_type_args=None, len_seq_max=None, spr=None, list_splits=None, n_samples=None):
    """
    Retrieve a vectorized function for sequence splitting, applied to the entire dataset at once.

    This function is not memory-efficient, but provides better performance for large datasets.
    It is used when `vectorized=True`, allowing simultaneous processing of multiple sequences.
    """
    n_splits = len(list_splits)
    f = getattr(spr, split_type.lower())

    def vf_split(arr_seq):
        seq_lengths = arr_seq[:, -1].astype(int)
        list_split_vals = [f(seq=arr_seq[i, : seq_lengths[i]], **(split_type_args or {})) for i in range(len(arr_seq))]
        X = np.full((n_samples, n_splits, len_seq_max), np.nan, dtype=np.float64)
        for i, split in enumerate(list_split_vals):
            for j, x in enumerate(split):
                X[i, j, : len(x)] = x
        return np.nanmean(X, axis=-1)

    return vf_split


@ut.catch_runtime_warnings(suppress=True)
def _pre_filtering_info_split_type(
    dict_scale_part_vals=None,
    list_parts=None,
    split_type=None,
    split_kws=None,
    mask_ref=None,
    mask_test=None,
    len_seq_max=None,
    spr=None,
    vectorized=True,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """Optimized computation for absolute mean difference and standard deviation."""
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)

    split_type_args = split_kws[split_type]
    n_split_types = len(split_kws)
    i_split_types = list(split_kws).index(split_type)
    args_split = dict(split_type=split_type, split_type_args=split_type_args)
    list_splits = _get_split_labels(**args_split, spr=spr)

    feat_names = [
        f"{part.upper()}-{split}-{scale}"
        for scale in dict_scale_part_vals
        for part in list_parts
        for split in list_splits
    ]

    abs_mean_dif = np.zeros((len(feat_names)))
    std_test = np.zeros((len(feat_names)))

    args_f_split = dict(len_seq_max=len_seq_max, spr=spr)
    if vectorized:
        f_split = _get_vf_split(**args_split, **args_f_split, list_splits=list_splits, n_samples=len(mask_test))
    else:
        f_split = _get_f_split(**args_split, **args_f_split)

    start = 0
    for i, scale in enumerate(dict_scale_part_vals):
        for j, part in enumerate(list_parts):
            scale_part_vals = dict_scale_part_vals[scale][part]
            vals_split = f_split(scale_part_vals) if vectorized else np.apply_along_axis(f_split, axis=1, arr=scale_part_vals)

            vals_test = vals_split[mask_test]
            vals_ref = vals_split[mask_ref]

            _mean_test = np.mean(vals_test, axis=0).ravel()
            _mean_ref = np.mean(vals_ref, axis=0).ravel()
            _std_test = np.std(vals_test, axis=0).ravel()

            stop = start + len(list_splits)
            abs_mean_dif[start:stop] = np.abs(_mean_test - _mean_ref)
            std_test[start:stop] = _std_test
            start = stop

            if verbose:
                # Fixed: proper progress calculation across all scales and split types
                current_idx = i * len(list_parts) + j
                total_ops = len(dict_scale_part_vals) * len(list_parts)
                # Reduce progress tracking overhead: update every 5% or at completion
                progress_interval = max(1, total_ops // 20)
                if current_idx % progress_interval == 0 or current_idx == total_ops - 1:
                    ut.print_progress(i=current_idx, n_total=total_ops, **args_p)

    return abs_mean_dif, std_test, feat_names


def _pre_filtering_info(
    dict_scale_part_vals=None,
    split_kws=None,
    list_parts=None,
    mask_ref=None,
    mask_test=None,
    len_seq_max=None,
    spr=None,
    vectorized=True,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """Compute abs(mean_dif) and std(test) to rank features."""
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )

    args = dict(
        dict_scale_part_vals=dict_scale_part_vals,
        split_kws=split_kws,
        list_parts=list_parts,
        mask_test=mask_test,
        mask_ref=mask_ref,
        len_seq_max=len_seq_max,
        spr=spr,
        vectorized=vectorized,
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
        _abs_mean_dif, _std_test, _feat_names = _pre_filtering_info_split_type(split_type=split_type, **args)
        abs_mean_dif.append(_abs_mean_dif)
        std_test.append(_std_test)
        feat_names.extend(_feat_names)

    return np.concatenate(abs_mean_dif), np.concatenate(std_test), np.array(feat_names)


# II Main Functions
def pre_filtering_info(df_parts=None, split_kws=None, dict_scale_part_vals=None, labels=None,
                       label_test=1, label_ref=0, verbose=False, n_jobs=None, vectorized=True):
    """Compute abs(mean_dif) and std(test) for feature ranking."""
    mask_ref = [x == label_ref for x in labels]
    mask_test = [x == label_test for x in labels]
    list_parts = list(df_parts)
    len_seq_max = df_parts.map(len).max().max()
    spr = SplitRange(split_type_str=False)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, len(dict_scale_part_vals))

    prefer_mp_progress = bool(n_jobs and n_jobs > 1)
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(prefer_multiprocessing=prefer_mp_progress)
    _reset_progress(shared_max_progress, shared_value_lock)

    args = dict(
        split_kws=split_kws,
        list_parts=list_parts,
        mask_test=mask_test,
        mask_ref=mask_ref,
        len_seq_max=len_seq_max,
        spr=spr,
        vectorized=vectorized,
        verbose=verbose,
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_mp_progress,
    )

    if n_jobs == 1:
        result = _pre_filtering_info(dict_scale_part_vals=dict_scale_part_vals, **args)
        if prefer_mp_progress:
            _cleanup_mp_manager()
        return result

    def _mp_pre_filtering_info(scales_chunk, shared_max_progress, shared_value_lock, print_lock):
        chunked_scale_vals = {scale: dict_scale_part_vals[scale] for scale in scales_chunk}
        return _pre_filtering_info(
            dict_scale_part_vals=chunked_scale_vals,
            split_kws=split_kws,
            list_parts=list_parts,
            mask_ref=mask_ref,
            mask_test=mask_test,
            len_seq_max=len_seq_max,
            spr=spr,
            vectorized=vectorized,
            verbose=verbose,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
            print_lock=print_lock,
            prefer_multiprocessing=False,  # shared objects already provided
        )

    scale_chunks = np.array_split(list(dict_scale_part_vals.keys()), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_mp_pre_filtering_info)(chunk, shared_max_progress, shared_value_lock, print_lock)
            for chunk in scale_chunks
        )

    if verbose:
        ut.print_end_progress(add_new_line=False, shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)

    abs_mean_dif = np.concatenate([res[0] for res in results])
    std_test = np.concatenate([res[1] for res in results])
    feat_names = np.concatenate([res[2] for res in results])

    # Cleanup Manager if we created it
    if prefer_mp_progress:
        _cleanup_mp_manager()

    return abs_mean_dif, std_test, feat_names
