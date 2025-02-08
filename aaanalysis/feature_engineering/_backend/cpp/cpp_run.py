"""
This is a script for the backend of the CPP.run() method.

This is the key algorithm of CPP and for AAanalysis.
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Manager, Lock

import warnings

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
def _get_split_labels(split_type=None, split_type_args=None):
    """Fetch split labels dynamically."""
    spr = SplitRange(split_type_str=False)
    if split_type_args is not None:
        labels_splits = getattr(spr, "labels_" + split_type.lower())(**split_type_args)
    else:
        labels_splits = getattr(spr, "labels_" + split_type.lower())()
    return labels_splits


def _get_f_split(split_type=None, split_type_args=None, len_seq_max=None):
    """Retrieve function for sequence splitting, optimizing lambda performance."""
    spr = SplitRange(split_type_str=False)
    f = getattr(spr, split_type.lower())

    def f_split(seq):
        """Apply splitting and return a NumPy array (without computing mean)."""
        splits = f(seq=seq, **(split_type_args or {}))
        arr = np.full((len(splits), len_seq_max), np.nan)  # Pre-allocate
        for i, x in enumerate(splits):
            arr[i, : len(x)] = x  # Insert split values
        return arr
    return f_split


# Pre-filtering helper functions
def _assign_scale_values_to_seq(df_parts=None, dict_all_scales=None, accept_gaps=None):
    """Assign scale values to each sequence with optimized dictionary lookups."""
    dict_scale_vals = {}
    for scale, dict_scale in dict_all_scales.items():
        if accept_gaps:
            f = lambda seq: np.array([dict_scale.get(a, np.nan) for a in seq])
        else:
            f = lambda seq: np.array([dict_scale[s] for s in seq])
        dict_scale_vals[scale] = np.array(df_parts.map(f))
    return dict_scale_vals

# TODO optimize (memory efficiency for CPP?)
"""
def _assign_scale_values_to_seq(df_parts=None, dict_all_scales=None, accept_gaps=None):
    dict_scale_vals = {}
    for scale, dict_scale in dict_all_scales.items():
        # Convert dict_scale to a NumPy array for fast lookups if all keys are ASCII (optional)
        keys, values = zip(*dict_scale.items())
        key_to_idx = {k: i for i, k in enumerate(keys)}
        values_arr = np.array(values)
        def map_seq(seq):
            seq_arr = np.array(list(seq))  # Convert sequence to array
            if accept_gaps:
                return np.array([dict_scale.get(a, np.nan) for a in seq_arr])  # With NaN handling
            try:
                return values_arr[[key_to_idx[s] for s in seq_arr]]  # Vectorized lookup
            except KeyError:
                return np.array([dict_scale[s] for s in seq_arr])  # Fallback for missing keys
        dict_scale_vals[scale] = df_parts.apply(map_seq)
    return dict_scale_vals
"""


def _pre_filtering_info_split_type(dict_scale_vals=None, list_parts=None, split_type=None, split_kws=None,
                                   mask_ref=None, mask_test=None, len_seq_max=None, verbose=True,
                                   shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """Optimized computation for absolute mean difference and standard deviation."""
    split_type_args = split_kws[split_type]
    n_split_types = len(split_kws)
    i_split_types = list(split_kws).index(split_type)
    args_split = dict(split_type=split_type, split_type_args=split_type_args)
    list_splits = _get_split_labels(**args_split)

    n_parts, n_splits, n_scales = len(list_parts), len(list_splits), len(dict_scale_vals)

    # Store feature names
    feat_names = [f"{part.upper()}-{split}-{scale}"
                  for scale in dict_scale_vals
                  for part in list_parts
                  for split in list_splits]

    # Process each scale (saved in pre-allocated NumPy arrays)
    abs_mean_dif = np.zeros((len(feat_names)))
    std_test = np.zeros((len(feat_names)))
    f_split = _get_f_split(**args_split, len_seq_max=len_seq_max)
    ufunc = np.frompyfunc(f_split, 1, 1)
    for i, (_, scale_vals) in enumerate(dict_scale_vals.items()):
        # Print progress bar with shared object for consistency during multiprocessing
        if verbose:
            ut.print_progress(i=i * (i_split_types + 1),
                              n_total=n_scales * n_split_types,
                              shared_max_progress=shared_max_progress,
                              shared_value_lock=shared_value_lock,
                              print_lock=print_lock)
        # Apply splitting and compute mean via vectorized
        vals_split = np.nanmean(np.array(ufunc(scale_vals).tolist()), axis=-1)
        # Get test and ref dataset
        vals_test = vals_split[mask_test]
        vals_ref = vals_split[mask_ref]
        # Compute mean and std
        _mean_test = np.mean(vals_test, axis=0).ravel()
        _mean_ref = np.mean(vals_ref, axis=0).ravel()
        _std_test = np.std(vals_test, axis=0).ravel()
        # Compute absolute mean difference and std test
        start = i * n_parts * n_splits
        stop = (i+1) * n_parts * n_splits
        abs_mean_dif[start:stop] = np.abs(_mean_test - _mean_ref)
        std_test[start:stop] = _std_test
    return abs_mean_dif, std_test, feat_names


def _pre_filtering_info(dict_scale_vals=None, split_kws=None, list_parts=None,
                        mask_ref=None, mask_test=None, len_seq_max=None, verbose=True,
                        shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """Compute abs(mean_dif) and std(test) to rank features, where mean_dif is the difference
    between the means of the test and the reference protein groups for a feature"""
    # Input (df_parts, split_kws, df_scales, checked in main method (CPP.run())
    args = dict(list_parts=list_parts, dict_scale_vals=dict_scale_vals,
                mask_test=mask_test, mask_ref=mask_ref, len_seq_max=len_seq_max,
                verbose=verbose, shared_max_progress=shared_max_progress,
                shared_value_lock=shared_value_lock, print_lock=print_lock)

    # Get split labels
    split_labels = []
    for split_type in split_kws:
        split_type_args = split_kws[split_type]
        split_labels.extend(_get_split_labels(split_type=split_type, split_type_args=split_type_args))

    n_parts, n_splits, n_scales = len(split_labels), len(list_parts), len(dict_scale_vals)
    feat_names = []
    abs_mean_dif = np.zeros((n_scales * n_parts * n_splits))
    std_test = np.zeros((n_scales * n_parts * n_splits))
    start = 0
    stop = 0
    for i, split_type in enumerate(split_kws):
        _abs_mean_dif, _std_test, _feat_names = _pre_filtering_info_split_type(split_type=split_type,
                                                                               split_kws=split_kws,
                                                                               **args)
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
# Filtering methods
def pre_filtering_info(df_parts=None, split_kws=None, df_scales=None,
                       labels=None, label_test=1, label_ref=0,
                       accept_gaps=False, verbose=False, n_jobs=None):
    """Get n best features in descending order based on the abs(mean(group1) - mean(group0),
    where group 1 is the target group"""
    # Input (df_parts, split_kws, df_scales) checked in main method (CPP.run())
    mask_ref = [x == label_ref for x in labels]
    mask_test = [x == label_test for x in labels]

    # Assign scale values
    list_scales = list(df_scales)
    list_parts = list(df_parts)
    len_seq_max = df_parts.map(len).max().max()
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col])) for col in list_scales}
    dict_scale_vals = _assign_scale_values_to_seq(df_parts=df_parts,
                                                  dict_all_scales=dict_all_scales,
                                                  accept_gaps=accept_gaps)

    args = dict(list_parts=list_parts, split_kws=split_kws, len_seq_max=len_seq_max,
                mask_test=mask_test, mask_ref=mask_ref,
                verbose=verbose, shared_max_progress=SHARED_MAX_PROGRESS,
                shared_value_lock=SHARED_VALUE_LOCK, print_lock=PRINT_LOCK)

    # Determine number of jobs
    if n_jobs is None:
        n_jobs = min(os.cpu_count(), len(dict_scale_vals))
    # Run one job
    if n_jobs == 1:
        return _pre_filtering_info(**args, dict_scale_vals=dict_scale_vals)

    # Run in parallel across scales
    scale_chunks = np.array_split(list(dict_scale_vals.keys()), n_jobs)

    # Define a worker that does pre-filtering info and returns its result
    def _mp_pre_filtering_info(scales_chunk):
        chunked_scale_vals = {scale: dict_scale_vals[scale] for scale in scales_chunk}
        return _pre_filtering_info(dict_scale_vals=chunked_scale_vals, **args)

    results = Parallel(n_jobs=n_jobs)(delayed(_mp_pre_filtering_info)(scales_chunk)
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


def pre_filtering(features=None, abs_mean_dif=None, std_test=None, max_std_test=0.2, n=10000, accept_gaps=False):
    """CPP pre-filtering based on thresholds."""
    df = pd.DataFrame(zip(features, abs_mean_dif, std_test),
                      columns=[ut.COL_FEATURE, ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST])
    df = df[df[ut.COL_STD_TEST] <= max_std_test]
    if accept_gaps:
        # Remove features resulting in NaN features values due to sequence gaps
        df = df[~df[ut.COL_ABS_MEAN_DIF].isna()]
    df = df.sort_values(by=ut.COL_ABS_MEAN_DIF, ascending=False).head(n)
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
             label_test=1, label_ref=0, n_jobs=None):
        """
        Add summary statistics for each feature to DataFrame.

        Notes
        -----
        P-values are calculated Mann-Whitney U test (non-parametric) or T-test (parametric) as implemented in SciPy.
        For multiple hypothesis correction, the Benjamini-Hochberg FDR correction is applied on all given features
        as implemented in SciPy.
        """
        # Add feature statistics
        features = list(df_feat[ut.COL_FEATURE])
        X = get_feature_matrix_(features=features,
                                df_parts=df_parts,
                                df_scales=df_scales,
                                accept_gaps=accept_gaps,
                                n_jobs=n_jobs)
        df_feat = add_stat_(df=df_feat, X=X, labels=labels, parametric=parametric,
                            label_test=label_test, label_ref=label_ref, n_jobs=n_jobs)
        return df_feat
