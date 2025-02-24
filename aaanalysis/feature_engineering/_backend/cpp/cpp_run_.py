"""
This is a script for the backend of the CPP.run() method.

This is the key algorithm of CPP and for AAanalysis.
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Manager

import aaanalysis.utils as ut
from .utils_feature import get_feature_matrix_
from ._utils_feature_stat import add_stat_
from ._split import SplitRange

# Multiprocessing shared progress bar
manager = Manager()
SHARED_MAX_PROGRESS = manager.Value('d', 0.0)
SHARED_VALUE_LOCK = manager.Lock()
PRINT_LOCK = manager.Lock()


# I Helper functions
# Assign scales values helper function
def _assign_scale_values_to_seq(df_parts=None, dict_all_scales=None, verbose=True,
                                shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """Assign scale values to each amino over each sequence"""
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    list_seq_max = df_parts.map(len).max().tolist()
    list_parts = list(df_parts)

    # Convert sequences into a padded NumPy matrix ("_" as padding)
    dict_seq_matrix = {}
    for part in list_parts:
        X_seq = np.full((len(df_parts), max(list_seq_max)), "_", dtype="<U1")
        for i, seq in enumerate(df_parts[part]):
            X_seq[i, :len(seq)] = list(seq)
        dict_seq_matrix[part] = X_seq

    # Assign scale values
    seq_lengths = df_parts.map(len).values
    dict_scale_part_vals = {scale: {} for scale in dict_all_scales}
    for i, (scale, dict_scale) in enumerate(dict_all_scales.items()):
        for j, (part, len_seq_max) in enumerate(zip(list_parts, list_seq_max)):
            X_scale_parts = np.full((len(df_parts), len_seq_max), np.nan, dtype=np.float32)
            X_seq = dict_seq_matrix[part]
            # Efficiently replace amino acids with scale values in bulk
            for aa, value in dict_scale.items():
                X_scale_parts[X_seq[:, :len_seq_max] == aa] = value
            # Append sequence lengths as the last column
            seq_length_column = seq_lengths[:, j][:, np.newaxis]
            X_scale_parts = np.concatenate([X_scale_parts, seq_length_column], axis=1)
            dict_scale_part_vals[scale][part] = X_scale_parts
        if verbose:
            ut.print_progress(i=i, n_total=len(dict_all_scales), **args_p)
    return dict_scale_part_vals


# Pre-filtering helper functions
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
        splits = f(seq=seq[:int(seq_len)], **(split_type_args or {}))
        # Pre-allocate array for storing splits
        X = np.full((len(splits), len_seq_max), np.nan, dtype=np.float64)
        for i, x in enumerate(splits):
            X[i, :len(x)] = x
        # Compute mean over the last axis
        return np.nanmean(X, axis=-1)
    return f_split


def _get_vf_split(split_type=None, split_type_args=None, len_seq_max=None, spr=None,
                  list_splits=None, n_samples=None):
    """
    Retrieve a vectorized function for sequence splitting, applied to the entire dataset at once.

    This function is not memory-efficient, but provides better performance for large datasets.
    It is used when `vectorized=True`, allowing simultaneous processing of multiple sequences.
    """
    n_splits = len(list_splits)
    f = getattr(spr, split_type.lower())

    def vf_split(arr_seq):
        seq_lengths = arr_seq[:, -1].astype(int)
        list_split_vals = [f(seq=arr_seq[i, :seq_lengths[i]], **(split_type_args or {})) for i in range(len(arr_seq))]
        # Pre-allocate array for storing splits
        X = np.full((n_samples, n_splits, len_seq_max), np.nan, dtype=np.float64)
        for i, split in enumerate(list_split_vals):
            for j, x in enumerate(split):
                X[i, j, :len(x)] = x
        # Compute mean along the last axis
        return np.nanmean(X, axis=-1)
    return vf_split


@ut.catch_runtime_warnings(suppress=True)
def _pre_filtering_info_split_type(dict_scale_part_vals=None, list_parts=None, split_type=None, split_kws=None,
                                   mask_ref=None, mask_test=None, len_seq_max=None, spr=None, vectorized=True,
                                   verbose=True, shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """Optimized computation for absolute mean difference and standard deviation."""
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    split_type_args = split_kws[split_type]
    n_split_types = len(split_kws)
    i_split_types = list(split_kws).index(split_type)
    args_split = dict(split_type=split_type, split_type_args=split_type_args)
    list_splits = _get_split_labels(**args_split, spr=spr)
    n_parts, n_splits, n_scales = len(list_parts), len(list_splits), len(dict_scale_part_vals)
    # Store feature names
    feat_names = [f"{part.upper()}-{split}-{scale}"
                  for scale in dict_scale_part_vals
                  for part in list_parts
                  for split in list_splits]

    # Process each scale (saved in pre-allocated NumPy arrays)
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
            if vectorized:
                vals_split = f_split(scale_part_vals)
            else:
                vals_split = np.apply_along_axis(f_split, axis=1, arr=scale_part_vals)
            # Get test and ref dataset
            vals_test = vals_split[mask_test]
            vals_ref = vals_split[mask_ref]
            # Compute mean and std
            _mean_test = np.mean(vals_test, axis=0).ravel()
            _mean_ref = np.mean(vals_ref, axis=0).ravel()
            _std_test = np.std(vals_test, axis=0).ravel()
            # Compute absolute mean difference and std test
            stop = start + n_splits
            abs_mean_dif[start:stop] = np.abs(_mean_test - _mean_ref)
            std_test[start:stop] = _std_test
            start = stop
            if verbose:
                ut.print_progress(i=i * (i_split_types + 1) + j, n_total=n_scales * n_split_types + j, **args_p)
    return abs_mean_dif, std_test, feat_names


def _pre_filtering_info(dict_scale_part_vals=None, split_kws=None, list_parts=None,
                        mask_ref=None, mask_test=None, len_seq_max=None, spr=None, vectorized=True,
                        verbose=True, shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """Compute abs(mean_dif) and std(test) to rank features, where mean_dif is the difference
    between the means of the test and the reference protein groups for a feature"""
    # Input (df_parts, split_kws, df_scales, checked in main method (CPP.run())
    args = dict(dict_scale_part_vals=dict_scale_part_vals, split_kws=split_kws, list_parts=list_parts,
                mask_test=mask_test, mask_ref=mask_ref, len_seq_max=len_seq_max,
                spr=spr, vectorized=vectorized,
                verbose=verbose, shared_max_progress=shared_max_progress,
                shared_value_lock=shared_value_lock, print_lock=print_lock)

    # Get split labels
    split_labels = []
    for split_type in split_kws:
        split_type_args = split_kws[split_type]
        split_labels.extend(_get_split_labels(split_type=split_type, split_type_args=split_type_args, spr=spr))

    n_parts, n_splits, n_scales = len(list_parts), len(split_labels), len(dict_scale_part_vals)
    feat_names = []
    abs_mean_dif = np.zeros((n_scales * n_parts * n_splits))
    std_test = np.zeros((n_scales * n_parts * n_splits))
    start, stop = 0, 0
    for i, split_type in enumerate(split_kws):
        _abs_mean_dif, _std_test, _feat_names = _pre_filtering_info_split_type(split_type=split_type, **args)
        stop += len(_feat_names)
        abs_mean_dif[start:stop] = _abs_mean_dif
        std_test[start:stop] = _std_test
        feat_names.extend(_feat_names)
        start = stop
    return abs_mean_dif, std_test, feat_names


# Filtering helper function
def filtering_info_(df=None, df_scales=None, check_cat=True):
    """Get datasets structures for filtering, two dictionaries with feature to scale category resp.
    feature positions and one datasets frame with paired pearson correlations of all scales"""
    if check_cat:
        dict_c = dict(zip(df[ut.COL_FEATURE], df[ut.COL_CAT]))
    else:
        dict_c = dict()
    dict_p = dict(zip(df[ut.COL_FEATURE], [set(x) for x in df[ut.COL_POSITION]]))
    df_cor = df_scales.corr()
    return dict_c, dict_p, df_cor


# II Main functions
# Assign scales to amino acids
def assign_scale_values_to_seq(df_parts=None, df_scales=None, verbose=False, n_jobs=None):
    """Assign scale values to each sequence with optimized dictionary lookups"""
    list_scales = list(df_scales)
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col])) for col in list_scales}
    args = dict(df_parts=df_parts, verbose=verbose, print_lock=PRINT_LOCK)
    args_print = dict(shared_max_progress=SHARED_MAX_PROGRESS, shared_value_lock=SHARED_VALUE_LOCK)
    if n_jobs is None:
        n_samples, n_scales = len(df_parts), len(dict_all_scales)
        n_jobs = min(os.cpu_count(), max(min(int(n_scales/100), int(n_samples/100)), 1))
    if n_jobs == 1:
        dict_scale_part_vals = _assign_scale_values_to_seq(dict_all_scales=dict_all_scales, **args, **args_print)
        if verbose:
            ut.print_end_progress(add_new_line=False, **args_print)
        return dict_scale_part_vals

    # Multi-processing function caller
    def _mp_scale_assignment(scales_chunk):
        chunked_dict_scales = {scale: dict_all_scales[scale] for scale in scales_chunk}
        return _assign_scale_values_to_seq(dict_all_scales=chunked_dict_scales, **args, **args_print)

    scale_chunks = np.array_split(list(dict_all_scales.keys()), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_mp_scale_assignment)(scales_chunk)
                           for scales_chunk in scale_chunks)

    dict_scale_part_vals = {}
    for _chunk_dict_scale_part_vals in results:
        dict_scale_part_vals.update(_chunk_dict_scale_part_vals)

    if verbose:
        ut.print_end_progress(add_new_line=False, **args_print)
    return dict_scale_part_vals


# Filtering methods
def pre_filtering_info(df_parts=None, split_kws=None, dict_scale_part_vals=None,
                       labels=None, label_test=1, label_ref=0,
                       verbose=False, n_jobs=None, vectorized=True):
    """Get n best features in descending order based on the abs(mean(group1) - mean(group0), with group 1 as target"""
    # Input (df_parts, split_kws, df_scales) checked in main method (CPP.run())
    mask_ref = [x == label_ref for x in labels]
    mask_test = [x == label_test for x in labels]
    list_parts = list(df_parts)
    len_seq_max = df_parts.map(len).max().max()
    spr = SplitRange(split_type_str=False)
    args = dict(split_kws=split_kws, list_parts=list_parts,
                mask_test=mask_test, mask_ref=mask_ref, len_seq_max=len_seq_max,
                spr=spr, vectorized=vectorized,
                verbose=verbose, shared_max_progress=SHARED_MAX_PROGRESS,
                shared_value_lock=SHARED_VALUE_LOCK, print_lock=PRINT_LOCK)
    if n_jobs is None:
        n_jobs = min(os.cpu_count(), len(dict_scale_part_vals))

    if n_jobs == 1:
        return _pre_filtering_info(dict_scale_part_vals=dict_scale_part_vals, **args)

    # Multi-processing function caller
    def _mp_pre_filtering_info(scales_chunk):
        chunked_scale_vals = {scale: dict_scale_part_vals[scale] for scale in scales_chunk}
        return _pre_filtering_info(dict_scale_part_vals=chunked_scale_vals, **args)

    scale_chunks = np.array_split(list(dict_scale_part_vals.keys()), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_mp_pre_filtering_info)(scales_chunk)
                           for scales_chunk in scale_chunks)
    if verbose:
        ut.print_end_progress(shared_max_progress=SHARED_MAX_PROGRESS,
                              shared_value_lock=SHARED_VALUE_LOCK,
                              add_new_line=False)
    # Concatenate results from all processes
    abs_mean_dif = np.concatenate([res[0] for res in results])
    std_test = np.concatenate([res[1] for res in results])
    feat_names = np.concatenate([res[2] for res in results])
    return abs_mean_dif, std_test, feat_names


def pre_filtering(df=None, features=None, abs_mean_dif=None, std_test=None,
                  max_std_test=0.2, n=10000, accept_gaps=False):
    """CPP pre-filtering based on thresholds."""
    if df is None:
        df = pd.DataFrame(zip(features, abs_mean_dif, std_test),
                          columns=[ut.COL_FEATURE, ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST])
    df = df[df[ut.COL_STD_TEST] <= max_std_test]
    if accept_gaps:
        # Remove features resulting in NaN features values due to sequence gaps
        df = df[~df[ut.COL_ABS_MEAN_DIF].isna()]
    df = df.sort_values(by=[ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST, ut.COL_FEATURE], ascending=[False, True, True])
    df = df.reset_index(drop=True).head(n)
    return df


def filtering(df=None, df_scales=None, max_overlap=0.5, max_cor=0.5, n_filter=100, check_cat=True):
    """CPP filtering algorithm based on redundancy reduction in descending order of absolute AUC."""
    dict_c, dict_p, df_cor = filtering_info_(df=df, df_scales=df_scales, check_cat=check_cat)
    df = df.sort_values(by=[ut.COL_ABS_AUC, ut.COL_ABS_MEAN_DIF], ascending=False).copy().reset_index(drop=True)
    list_feat = list(df[ut.COL_FEATURE])
    list_top_feat = [list_feat.pop(0)]  # List with best feature
    for feat in list_feat:
        add_flag = True
        # Stop condition for limit
        if len(list_top_feat) == n_filter:
            break
        # Compare features with all top features (added if low overlap & weak correlation or different category)
        for top_feat in list_top_feat:
            # If check_cat is False, the categories are not compared and only the position and correlation are checked
            if not check_cat or dict_c[feat] == dict_c[top_feat]:
                # Remove if feat positions high overlap or subset
                pos, top_pos = dict_p[feat], dict_p[top_feat]
                overlap = len(top_pos.intersection(pos))/len(top_pos.union(pos))
                if overlap >= max_overlap or pos.issubset(top_pos):
                    # Remove if high pearson correlation
                    scale, top_scale = feat.split("-")[2], top_feat.split("-")[2]
                    cor = df_cor[top_scale][scale]
                    if cor > max_cor:
                        add_flag = False
        if add_flag:
            list_top_feat.append(feat)
    df_top_feat = df[df[ut.COL_FEATURE].isin(list_top_feat)]
    return df_top_feat


# Adder method for CPP analysis
def add_stat(df_feat=None, df_parts=None, df_scales=None, labels=None, parametric=False, accept_gaps=False,
             label_test=1, label_ref=0, n_jobs=None, vectorized=True):
    """Add summary statistics for each feature to DataFrame."""
    # Add feature statistics
    features = list(df_feat[ut.COL_FEATURE])
    X = get_feature_matrix_(features=features,
                            df_parts=df_parts,
                            df_scales=df_scales,
                            accept_gaps=accept_gaps,
                            n_jobs=n_jobs)
    df_feat = add_stat_(df=df_feat, X=X, labels=labels, parametric=parametric,
                        label_test=label_test, label_ref=label_ref,
                        n_jobs=n_jobs, vectorized=vectorized)
    return df_feat
