"""
This is a script for utility functions for CPP and SequenceFeature objects and backend.
"""
import os
import numpy as np
from itertools import repeat
import multiprocessing as mp

import pandas as pd

from aaanalysis.feature_engineering._backend.cpp._part import Parts
from aaanalysis.feature_engineering._backend.cpp._split import Split
import aaanalysis.utils as ut


# I Helper Functions
def check_dict_part_pos(dict_part_pos=None):
    """Check if dict_part_pos is valid"""
    list_parts = list(dict_part_pos.keys())
    wrong_parts = [x for x in list_parts if x not in ut.LIST_ALL_PARTS]
    if len(wrong_parts) > 0:
        error = f"Following parts from 'dict_part_pos' are not valid: {wrong_parts}." \
                f"\n Parts should be as follows: {ut.LIST_ALL_PARTS}"
        raise ValueError(error)


# Get positions
def _get_dict_part_pos(tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Get dictionary for part to positions."""
    ut.check_number_range(name="start", val=start, min_val=0, just_int=True)
    ut.check_args_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    pa = Parts()
    jmd_n = list(range(0, jmd_n_len))
    tmd = list(range(jmd_n_len, tmd_len+jmd_n_len))
    jmd_c = list(range(jmd_n_len + tmd_len, jmd_n_len + tmd_len + jmd_c_len))
    # Change int to string and adjust length
    jmd_n = [i + start for i in jmd_n]
    tmd = [i + start for i in tmd]
    jmd_c = [i + start for i in jmd_c]
    dict_part_pos = pa.get_dict_part_seq(tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_c)
    return dict_part_pos


def _get_positions(dict_part_pos=None, features=None, as_str=True):
    """Get list of positions for given feature names."""
    check_dict_part_pos(dict_part_pos=dict_part_pos)
    features = ut.check_features(features=features, parts=list(dict_part_pos.keys()))
    sp = Split(type_str=False)
    list_pos = []
    for feat_id in features:
        part, split, scale = feat_id.split("-")
        split_type, split_kwargs = ut.check_split(split=split)
        f_split = getattr(sp, split_type.lower())
        pos = sorted(f_split(seq=dict_part_pos[part.lower()], **split_kwargs))
        if as_str:
            pos = str(pos).replace("[", "").replace("]", "").replace(" ", "")
        list_pos.append(pos)
    return list_pos


# Get df positions
def _get_df_pos_long(df=None, y="category", col_value=None):
    """Get """
    if col_value is None:
        df_feat = df[[ut.COL_FEATURE, y]].set_index(ut.COL_FEATURE)
    else:
        df_feat = df[[ut.COL_FEATURE, y, col_value]].set_index(ut.COL_FEATURE)
    # Columns = scale categories, rows = features
    df_pos_long = pd.DataFrame(df[ut.COL_POSITION].str.split(",").tolist())
    df_pos_long.index = df[ut.COL_FEATURE]
    df_pos_long = df_pos_long.stack().reset_index(level=1).drop("level_1", axis=1).rename({0: ut.COL_POSITION}, axis=1)
    df_pos_long = df_pos_long.join(df_feat)
    df_pos_long[ut.COL_POSITION] = df_pos_long[ut.COL_POSITION].astype(int)
    return df_pos_long


# Get feature matrix
def _feature_value(df_parts=None, split=None, dict_scale=None, accept_gaps=False):
    """Helper function to create feature values for feature matrix"""
    sp = Split()
    # Get vectorized split function
    split_type, split_kwargs = ut.check_split(split=split)
    f_split = getattr(sp, split_type.lower())
    # Vectorize split function using anonymous function
    vf_split = np.vectorize(lambda x: f_split(seq=x, **split_kwargs))
    # Get vectorized scale function
    vf_scale = ut.get_vf_scale(dict_scale=dict_scale, accept_gaps=accept_gaps)
    # Combine part split and scale to get feature values
    part_split = vf_split(df_parts)
    feature_value = np.round(vf_scale(part_split), 5)  # feature values
    return feature_value


def _feature_matrix(feat_names, dict_all_scales, df_parts, accept_gaps):
    """Helper function to create feature matrix via multiple processing"""
    X = np.empty([len(df_parts), len(feat_names)])
    for i, feat_name in enumerate(feat_names):
        part, split, scale = feat_name.split("-")
        dict_scale = dict_all_scales[scale]
        X[:, i] = _feature_value(split=split,
                                 dict_scale=dict_scale,
                                 df_parts=df_parts[part.lower()],
                                 accept_gaps=accept_gaps)
    return X


# II Main Functions
def get_list_parts(features=None):
    """Get list of parts to cover all features"""
    features = ut.check_list_like(name="features", val=features, convert=True, accept_str=True)
    # Features are PART-SPLIT-SCALE combinations
    list_parts = list(set([x.split("-")[0].lower() for x in features]))
    return list_parts


def get_df_parts_(df_seq=None, list_parts=None, jmd_n_len=None, jmd_c_len=None):
    """Create DataFrame with sequence parts"""
    seq_info_in_df = set(ut.COLS_SEQ_TMD_POS_KEY).issubset(set(df_seq))
    pa = Parts()
    dict_parts = {}
    for i, row in df_seq.iterrows():
        entry = row[ut.COL_ENTRY]
        if jmd_c_len is not None and jmd_n_len is not None and seq_info_in_df:
            seq, start, stop = row[ut.COLS_SEQ_TMD_POS_KEY].values
            parts = pa.create_parts(seq=seq, tmd_start=start, tmd_stop=stop, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
            jmd_n, tmd, jmd_c = parts.jmd_n, parts.tmd, parts.jmd_c
        else:
            jmd_n, tmd, jmd_c = row[ut.COLS_PARTS].values
        dict_part_seq = pa.get_dict_part_seq(tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_c)
        dict_part_seq = {part: dict_part_seq[part] for part in list_parts}
        dict_parts[entry] = dict_part_seq
    df_parts = pd.DataFrame.from_dict(dict_parts).T
    # DEV: the following line sorts index if list_parts contains just one element
    # df_parts = pd.DataFrame.from_dict(dict_parts, orient="index")
    return df_parts


def get_positions_(features=None, start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10):
    """Create list with positions for given feature names"""
    features = ut.check_list_like(name="features", val=features, convert=True, accept_str=True)
    dict_part_pos = _get_dict_part_pos(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    feat_positions = _get_positions(dict_part_pos=dict_part_pos, features=features)
    return feat_positions


def get_amino_acids_(features=None, tmd_seq="", jmd_n_seq="", jmd_c_seq=""):
    """"""
    features = ut.check_list_like(name="features", val=features, convert=True, accept_str=True)
    pos = get_positions_(features=features, tmd_len=len(tmd_seq), jmd_n_len=len(jmd_n_seq),
                         jmd_c_len=len(jmd_c_seq), start=0)
    seq = jmd_n_seq + tmd_seq + jmd_c_seq
    f_seg = lambda x: "".join([seq[int(p)] for p in x.split(",")])
    f_pat = lambda x: "-".join([seq[int(p)] for p in x.split(",")])
    feat_aa = [f_seg(pos) if "Segment" in feat else f_pat(pos) for feat, pos in zip(features, pos)]
    return feat_aa


def get_feature_matrix_(features=None, df_parts=None, df_scales=None, accept_gaps=False, n_jobs=None):
    """Create feature matrix for given feature ids and sequence parts."""
    # Create feature matrix using parallel processing
    features = ut.check_list_like(name="features", val=features, convert=True, accept_str=True)
    dict_all_scales = ut.get_dict_all_scales(df_scales=df_scales)
    n_processes = min([os.cpu_count(), len(features)]) if n_jobs is None else n_jobs
    features = features.to_list() if isinstance(features, pd.Series) else features
    feat_chunks = np.array_split(features, n_processes)
    args = zip(feat_chunks, repeat(dict_all_scales), repeat(df_parts), repeat(accept_gaps))
    with mp.get_context("spawn").Pool(processes=n_processes) as pool:
        result = pool.starmap(_feature_matrix, args)
    feat_matrix = np.concatenate(result, axis=1)
    return feat_matrix


def get_df_pos_(df_feat=None, y="category", value_type="count", col_value=None, start=None, stop=None):
    """Get df with counts for each combination of column values and positions"""
    list_y_cat = sorted(set(df_feat[y]))
    normalize_for_pos = value_type != "mean"
    if normalize_for_pos:
        df_feat[col_value] = df_feat[col_value] / [len(x.split(",")) for x in df_feat[ut.COL_POSITION]]
    # Get df with features for each position
    df_pos_long = _get_df_pos_long(df=df_feat, y=y, col_value=col_value)
    # Get dict with values of categories for each position
    dict_pos_val = {p: [] for p in range(start, stop+1)}
    dict_cat_val = {c: 0 for c in list_y_cat}
    for p in dict_pos_val:
        if value_type == "count":
            dict_val = dict(df_pos_long[df_pos_long[ut.COL_POSITION] == p][y].value_counts())
        elif value_type == "mean":
            # TOOD check for axis
            dict_val = dict(df_pos_long[df_pos_long[ut.COL_POSITION] == p].groupby(y).mean()[col_value])
        elif value_type == "sum":
            dict_val = dict(df_pos_long[df_pos_long[ut.COL_POSITION] == p].groupby(y).sum()[col_value])
        else:
            dict_val = dict(df_pos_long[df_pos_long[ut.COL_POSITION] == p].groupby(y).std()[col_value])
        dict_pos_val[p] = {**dict_cat_val, **dict_val}
    # Get df with values (e.g., counts) of each category and each position
    df_pos = pd.DataFrame(dict_pos_val)
    df_pos = df_pos.T[list_y_cat].T     # Filter and order categories
    return df_pos


