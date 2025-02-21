"""
This script implements the Bridge Layer, an interface between the frontend and backend
for the `CPP.run()` method. It enables both single-processing and batch-processing modes
to accommodate different computational and memory constraints.

`CPP.run()` is the key algorithm of AAanalysis.

DEV: Bridge layers should be used only in exceptional cases to preserve the
primary backend-frontend architecture.
"""
import numpy as np
import pandas as pd
import aaanalysis.utils as ut

from .cpp.sequence_feature import get_features_
from .cpp.utils_feature import get_positions_, add_scale_info_
from .cpp.cpp_run_ import assign_scale_values_to_seq, pre_filtering_info, pre_filtering, filtering, add_stat


# I Helper Functions
def _get_n_pre_filter(n_pre_filter=None, n_filter=None, n_feat=None, pct_pre_filter=None):
    """Get number of feature to pre-filter"""
    if n_pre_filter is None:
        n_pre_filter = int(n_feat * (pct_pre_filter / 100))
        n_pre_filter = n_filter if n_pre_filter < n_filter else n_pre_filter
    pct_pre_filter = np.round((n_pre_filter / n_feat * 100), 2)
    return n_pre_filter, pct_pre_filter


# II Main Functions
# Single processing implementation
def cpp_run_single(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None, accept_gaps=True,
                   labels=None, label_test=1, label_ref=0, n_filter=100, n_pre_filter=None, pct_pre_filter=5,
                   max_std_test=0.2, max_overlap=0.5, max_cor=0.5, check_cat=True, parametric=False,
                   start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True):
    """Perform CPP algorithm on entire dataset (fast but potentially memory intensive for large datasets)"""
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    # Get feature settings
    n_feat = len(get_features_(list_parts=list(df_parts),
                               split_kws=split_kws,
                               list_scales=list(df_scales)))
    n_filter = n_feat if n_feat < n_filter else n_filter

    # Assign scales values to parts
    if verbose:
        start_message = (f"1. CPP creates {n_feat} features for {len(df_parts)} samples"
                         f"\n1.1 Assigning scales values to parts")
        ut.print_start_progress(start_message=start_message)
    dict_scale_part_vals = assign_scale_values_to_seq(df_parts=df_parts, df_scales=df_scales,
                                                      verbose=verbose, n_jobs=n_jobs)

    if verbose:
        start_message = f"\n1.2 Applying splitting to parts"
        ut.print_start_progress(start_message=start_message)

    # Compute pre-filtering information (Combining splits, parts, and scales)
    abs_mean_dif, std_test, features = pre_filtering_info(df_parts=df_parts, split_kws=split_kws,
                                                          dict_scale_part_vals=dict_scale_part_vals, labels=labels,
                                                          label_test=label_test, label_ref=label_ref,
                                                          verbose=verbose, n_jobs=n_jobs, vectorized=vectorized)

    # Pre-filtering: Select best n % of feature (filter_pct) based std(test set) and mean_dif
    n_pre_filter, pct_pre_filter = _get_n_pre_filter(n_pre_filter=n_pre_filter, n_filter=n_filter,
                                                     n_feat=int(len(features)), pct_pre_filter=pct_pre_filter)
    if verbose:
        end_message = (f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest "
                       f"'{ut.COL_ABS_MEAN_DIF}' and 'max_std_test' <= {max_std_test}")
        ut.print_end_progress(end_message=end_message)
    df = pre_filtering(features=features, abs_mean_dif=abs_mean_dif, std_test=std_test, n=n_pre_filter,
                       max_std_test=max_std_test, accept_gaps=accept_gaps)
    features = df[ut.COL_FEATURE].to_list()

    # Add feature information
    df = add_stat(df_feat=df, df_scales=df_scales, df_parts=df_parts, labels=labels, parametric=parametric,
                  accept_gaps=accept_gaps, label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                  vectorized=vectorized)
    feat_positions = get_positions_(features=features, start=start, **args_len)
    df[ut.COL_POSITION] = feat_positions
    df = add_scale_info_(df_feat=df, df_cat=df_cat)

    # Filtering using CPP algorithm
    if verbose:
        ut.print_out(f"3. CPP filtering algorithm")
    df_feat = filtering(df=df, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                        max_overlap=max_overlap, max_cor=max_cor)
    # Adjust df_feat
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat


# Batch processing implementation
def cpp_run_batch(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None, accept_gaps=True,
                  labels=None, label_test=1, label_ref=0, n_filter=100, n_pre_filter=None, pct_pre_filter=5,
                  max_std_test=0.2, max_overlap=0.5, max_cor=0.5, check_cat=True, parametric=False,
                  start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True, n_batches=10):
    """Perform CPP algorithm on batches of datasets (slower but memory consumption adjustable by number of batches)"""
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    # Get feature settings
    n_feat = len(get_features_(list_parts=list(df_parts),
                               split_kws=split_kws,
                               list_scales=list(df_scales)))
    n_filter = n_feat if n_feat < n_filter else n_filter
    if verbose:
        ut.print_out(f"1. CPP creates {n_feat} features for {len(df_parts)} samples in {n_batches} batches")

    # Batch processing of feature computation
    scale_batches = np.array_split(np.array(list(df_scales)), n_batches)
    list_batch_pre_filter_info = []
    for i, scales_batch in enumerate(scale_batches):
        df_scales_batch = df_scales[list(scales_batch)]
        # Assign scales values to parts
        if verbose:
            str_start = "" if i == 0 else "\n"
            start_message = f"{str_start}1.1 Assigning scales values to parts ({i + 1}/{n_batches} batch)"
            ut.print_start_progress(start_message=start_message)
        dict_scale_part_vals = assign_scale_values_to_seq(df_parts=df_parts, df_scales=df_scales_batch,
                                                          verbose=verbose, n_jobs=n_jobs)
        if verbose:
            str_message = (f"\n1.2 Applying splitting to parts "
                           f"({i + 1}/{n_batches} batch)")
            ut.print_start_progress(start_message=str_message)
        # Compute pre-filtering information (Combining splits, parts, and scales)
        abs_mean_dif, std_test, features = pre_filtering_info(df_parts=df_parts, split_kws=split_kws,
                                                              dict_scale_part_vals=dict_scale_part_vals, labels=labels,
                                                              label_test=label_test, label_ref=label_ref,
                                                              verbose=verbose, n_jobs=n_jobs, vectorized=vectorized)
        list_batch_pre_filter_info.append((abs_mean_dif, std_test, features))
    abs_mean_dif_merged, std_test_merged, features_merged = zip(*list_batch_pre_filter_info)
    abs_mean_dif = np.concatenate(abs_mean_dif_merged)
    std_test = np.concatenate(std_test_merged)
    features = np.concatenate(features_merged)

    # Pre-filtering: Select best n % of feature (filter_pct) based std(test set) and mean_dif
    n_pre_filter, pct_pre_filter = _get_n_pre_filter(n_pre_filter=n_pre_filter, n_filter=n_filter,
                                                     n_feat=int(len(features)), pct_pre_filter=pct_pre_filter)
    if verbose:
        end_message = (f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest "
                       f"'{ut.COL_ABS_MEAN_DIF}' and 'max_std_test' <= {max_std_test}")
        ut.print_end_progress(end_message=end_message)
    df = pre_filtering(features=features, abs_mean_dif=abs_mean_dif,
                       std_test=std_test, n=n_pre_filter,
                       max_std_test=max_std_test, accept_gaps=accept_gaps)
    features = df[ut.COL_FEATURE].to_list()

    # Batch processing of additional CPP information
    feature_batches = np.array_split(np.array(features), n_batches)
    list_batch_dfs = []
    for i, feature_batch in enumerate(feature_batches):
        _df = df[df[ut.COL_FEATURE].isin(feature_batch)]
        # Add feature information
        _df = add_stat(df_feat=_df, df_scales=df_scales, df_parts=df_parts, labels=labels, parametric=parametric,
                       accept_gaps=accept_gaps, label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                       vectorized=vectorized)

        feat_positions = get_positions_(features=feature_batch, start=start, **args_len)
        _df[ut.COL_POSITION] = feat_positions
        _df = add_scale_info_(df_feat=_df, df_cat=df_cat)
        list_batch_dfs.append(_df)
    df_merged = pd.concat(list_batch_dfs, ignore_index=True)

    # Filtering using CPP algorithm
    if verbose:
        ut.print_out(f"3. CPP filtering algorithm")
    df_feat = filtering(df=df_merged, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                        max_overlap=max_overlap, max_cor=max_cor)
    # Adjust df_feat
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat
