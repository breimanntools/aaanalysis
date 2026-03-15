""""
This is a script for the backend of the CPP.run() method.

This is the key algorithm of CPP and for AAanalysis.

Dev rules (important for macOS/Windows):
- Never create multiprocessing.Manager() (or start subprocesses) at import time.
  macOS/Windows default to "spawn" which re-imports modules in child processes.
  Import-time Manager() will crash with:
  "An attempt has been made to start a new process before the current process has finished..."
- Create multiprocessing shared objects lazily (only when needed) and only from the main process.
- Always allow passing shared_* objects explicitly to support true cross-process progress updates.
"""
import os
import threading
import multiprocessing as mp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import aaanalysis.utils as ut
from .utils_feature import get_feature_matrix_
from ._utils_feature_stat import add_stat_
from ._split import SplitRange


# ---------------------------------------------------------------------
# Progress sharing (thread fallback + optional multiprocessing shared)
# ---------------------------------------------------------------------
class _FloatBox:
    """Thread-safe fallback for multiprocessing.Value('d', x) using a `.value` attribute."""
    def __init__(self, v: float = 0.0):
        self.value = v


# Default safe globals (NO multiprocessing side effects at import time)
DEFAULT_SHARED_MAX_PROGRESS = _FloatBox(0.0)
DEFAULT_SHARED_VALUE_LOCK = threading.Lock()
DEFAULT_PRINT_LOCK = threading.Lock()

# Lazy-created manager/shared objects (only for true cross-process sync)
_MP_MANAGER = None
_MP_SHARED_MAX_PROGRESS = None
_MP_SHARED_VALUE_LOCK = None
_MP_PRINT_LOCK = None
_MP_MANAGER_REFCOUNT = 0  # Track usage to enable cleanup


def _is_main_process() -> bool:
    """True only in the original interpreter process (important for macOS spawn safety)."""
    return mp.current_process().name == "MainProcess"


def _get_mp_shared():
    """
    Lazily create a multiprocessing.Manager + shared objects.

    IMPORTANT:
    - Must only be called from the main process.
    - Never called at import time.
    """
    global _MP_MANAGER, _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK, _MP_MANAGER_REFCOUNT

    if not _is_main_process():
        # In workers spawned via "spawn", we must NOT try to start a Manager here.
        return None

    if _MP_MANAGER is None:
        _MP_MANAGER = mp.Manager()
        _MP_SHARED_MAX_PROGRESS = _MP_MANAGER.Value("d", 0.0)
        _MP_SHARED_VALUE_LOCK = _MP_MANAGER.Lock()
        _MP_PRINT_LOCK = _MP_MANAGER.Lock()
        _MP_MANAGER_REFCOUNT = 0

    _MP_MANAGER_REFCOUNT += 1
    return _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK


def _cleanup_mp_manager():
    """
    Cleanup multiprocessing Manager if no longer needed.
    Should be called after parallel operations complete.
    """
    global _MP_MANAGER, _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK, _MP_MANAGER_REFCOUNT

    if not _is_main_process():
        return

    if _MP_MANAGER is not None and _MP_MANAGER_REFCOUNT > 0:
        _MP_MANAGER_REFCOUNT -= 1
        # Only shutdown if refcount reaches 0 (all operations done)
        if _MP_MANAGER_REFCOUNT == 0:
            try:
                _MP_MANAGER.shutdown()
            except Exception:
                pass  # Ignore errors during cleanup
            _MP_MANAGER = None
            _MP_SHARED_MAX_PROGRESS = None
            _MP_SHARED_VALUE_LOCK = None
            _MP_PRINT_LOCK = None


def _resolve_shared(shared_max_progress=None, shared_value_lock=None, print_lock=None, prefer_multiprocessing=False):
    """
    Resolve shared objects for progress printing.

    Priority:
    1) If caller passes shared_* explicitly: use them.
    2) Else if prefer_multiprocessing=True and we are in main process: create/use Manager-based shared objects.
    3) Else: use thread-safe defaults.

    This design:
    - Avoids macOS spawn crashes (no Manager at import time, no Manager creation in workers).
    - Allows true cross-process shared progress (Manager) when requested.
    """
    if shared_max_progress is not None and shared_value_lock is not None and print_lock is not None:
        return shared_max_progress, shared_value_lock, print_lock

    if prefer_multiprocessing:
        mp_shared = _get_mp_shared()
        if mp_shared is not None:
            return mp_shared

    return DEFAULT_SHARED_MAX_PROGRESS, DEFAULT_SHARED_VALUE_LOCK, DEFAULT_PRINT_LOCK


def _reset_progress(shared_max_progress, shared_value_lock):
    """Reset shared progress to 0 in a safe way."""
    with shared_value_lock:
        shared_max_progress.value = 0.0


# ---------------------------------------------------------------------
# I Helper functions
# ---------------------------------------------------------------------
def _assign_scale_values_to_seq(
    df_parts=None,
    dict_all_scales=None,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """Assign scale values to each amino over each sequence"""
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)

    list_seq_max = df_parts.map(len).max().tolist()
    list_parts = list(df_parts)

    # Convert sequences into a padded NumPy matrix ("_" as padding)
    # Memory optimization: only create once per part, reuse across scales
    dict_seq_matrix = {}
    for part in list_parts:
        len_seq_max = max(list_seq_max)
        X_seq = np.full((len(df_parts), len_seq_max), "_", dtype="<U1")
        for i, seq in enumerate(df_parts[part]):
            X_seq[i, : len(seq)] = list(seq)
        dict_seq_matrix[part] = X_seq

    # Assign scale values
    seq_lengths = df_parts.map(len).values
    dict_scale_part_vals = {scale: {} for scale in dict_all_scales}
    
    # Progress tracking: update every 10% or at least every scale
    progress_interval = max(1, len(dict_all_scales) // 10) if verbose else len(dict_all_scales)
    
    for i, (scale, dict_scale) in enumerate(dict_all_scales.items()):
        for j, (part, len_seq_max) in enumerate(zip(list_parts, list_seq_max)):
            X_scale_parts = np.full((len(df_parts), len_seq_max), np.nan, dtype=np.float32)
            X_seq = dict_seq_matrix[part]
            # Efficiently replace amino acids with scale values in bulk
            # Optimization: use vectorized operations where possible
            for aa, value in dict_scale.items():
                mask = X_seq[:, :len_seq_max] == aa
                X_scale_parts[mask] = value
            # Append sequence lengths as the last column
            seq_length_column = seq_lengths[:, j][:, np.newaxis]
            X_scale_parts = np.concatenate([X_scale_parts, seq_length_column], axis=1)
            dict_scale_part_vals[scale][part] = X_scale_parts
        
        # Reduce progress tracking overhead: update less frequently
        if verbose and (i % progress_interval == 0 or i == len(dict_all_scales) - 1):
            ut.print_progress(i=i, n_total=len(dict_all_scales), **args_p)

    return dict_scale_part_vals


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


def filtering_info_(df=None, df_scales=None, check_cat=True):
    """Get datasets structures for filtering."""
    if check_cat:
        dict_c = dict(zip(df[ut.COL_FEATURE], df[ut.COL_CAT]))
    else:
        dict_c = dict()
    dict_p = dict(zip(df[ut.COL_FEATURE], [set(x) for x in df[ut.COL_POSITION]]))
    df_cor = df_scales.corr()
    return dict_c, dict_p, df_cor


# ---------------------------------------------------------------------
# II Main functions
# ---------------------------------------------------------------------
def assign_scale_values_to_seq(df_parts=None, df_scales=None, verbose=False, n_jobs=None):
    """Assign scale values to each sequence with optimized dictionary lookups."""
    list_scales = list(df_scales)
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col])) for col in list_scales}

    if n_jobs is None:
        n_samples, n_scales = len(df_parts), len(dict_all_scales)
        n_jobs = min(os.cpu_count() or 1, max(min(int(n_scales / 100), int(n_samples / 100)), 1))

    # Use multiprocessing-shared progress only if we truly use >1 process
    prefer_mp_progress = bool(n_jobs and n_jobs > 1)

    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        prefer_multiprocessing=prefer_mp_progress
    )
    args_print = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    args = dict(df_parts=df_parts, verbose=verbose, prefer_multiprocessing=prefer_mp_progress, **args_print)

    _reset_progress(shared_max_progress, shared_value_lock)

    if n_jobs == 1:
        out = _assign_scale_values_to_seq(dict_all_scales=dict_all_scales, **args)
        if verbose:
            ut.print_end_progress(add_new_line=False, shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)
        if prefer_mp_progress:
            _cleanup_mp_manager()
        return out

    def _mp_scale_assignment(scales_chunk, shared_max_progress, shared_value_lock, print_lock):
        # Workers use the passed-in shared objects; they must NOT create a Manager.
        chunked_dict_scales = {scale: dict_all_scales[scale] for scale in scales_chunk}
        return _assign_scale_values_to_seq(
            dict_all_scales=chunked_dict_scales,
            df_parts=df_parts,
            verbose=verbose,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
            print_lock=print_lock,
            prefer_multiprocessing=False,  # shared objects already provided
        )

    scale_chunks = np.array_split(list(dict_all_scales.keys()), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_mp_scale_assignment)(chunk, shared_max_progress, shared_value_lock, print_lock)
            for chunk in scale_chunks
        )

    dict_scale_part_vals = {}
    for r in results:
        dict_scale_part_vals.update(r)

    if verbose:
        ut.print_end_progress(add_new_line=False, shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)

    # Cleanup Manager if we created it
    if prefer_mp_progress:
        _cleanup_mp_manager()

    return dict_scale_part_vals


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


def pre_filtering(df=None, features=None, abs_mean_dif=None, std_test=None, max_std_test=0.2, n=10000, accept_gaps=False):
    """CPP pre-filtering based on thresholds."""
    if df is None:
        df = pd.DataFrame(zip(features, abs_mean_dif, std_test), columns=[ut.COL_FEATURE, ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST])
    df = df[df[ut.COL_STD_TEST] <= max_std_test]
    if accept_gaps:
        df = df[~df[ut.COL_ABS_MEAN_DIF].isna()]
    df = df.sort_values(by=[ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST, ut.COL_FEATURE], ascending=[False, True, True])
    df = df.reset_index(drop=True).head(n)
    return df


def filtering(df=None, df_scales=None, max_overlap=0.5, max_cor=0.5, n_filter=100, check_cat=True):
    """CPP filtering algorithm based on redundancy reduction in descending order of absolute AUC."""
    dict_c, dict_p, df_cor = filtering_info_(df=df, df_scales=df_scales, check_cat=check_cat)
    df = df.sort_values(by=[ut.COL_ABS_AUC, ut.COL_ABS_MEAN_DIF], ascending=False).copy().reset_index(drop=True)
    list_feat = list(df[ut.COL_FEATURE])
    list_top_feat = [list_feat.pop(0)]
    for feat in list_feat:
        add_flag = True
        if len(list_top_feat) == n_filter:
            break
        for top_feat in list_top_feat:
            if not check_cat or dict_c[feat] == dict_c[top_feat]:
                pos, top_pos = dict_p[feat], dict_p[top_feat]
                overlap = len(top_pos.intersection(pos)) / len(top_pos.union(pos))
                if overlap >= max_overlap or pos.issubset(top_pos):
                    scale, top_scale = feat.split("-")[2], top_feat.split("-")[2]
                    cor = df_cor[top_scale][scale]
                    if cor > max_cor:
                        add_flag = False
        if add_flag:
            list_top_feat.append(feat)
    return df[df[ut.COL_FEATURE].isin(list_top_feat)]


def add_stat(df_feat=None, df_parts=None, df_scales=None, labels=None, parametric=False, accept_gaps=False,
             label_test=1, label_ref=0, n_jobs=None, vectorized=True):
    """Add summary statistics for each feature to DataFrame."""
    features = list(df_feat[ut.COL_FEATURE])
    X = get_feature_matrix_(features=features, df_parts=df_parts, df_scales=df_scales, accept_gaps=accept_gaps, n_jobs=n_jobs)
    df_feat = add_stat_(df=df_feat, X=X, labels=labels, parametric=parametric,
                        label_test=label_test, label_ref=label_ref, n_jobs=n_jobs, vectorized=vectorized)
    return df_feat
