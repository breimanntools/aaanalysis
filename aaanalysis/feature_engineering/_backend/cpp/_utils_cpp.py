"""
This is a script for utility functions for CPP and SequenceFeature objects and backend.
"""
import os
import numpy as np
from itertools import repeat
import multiprocessing as mp

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


# Get feature
def _get_feature_components(feat_name=None, dict_all_scales=None):
    """Convert feature name into three feature components of part, split, and scale given as dictionary"""
    if feat_name is None or dict_all_scales is None:
        raise ValueError("'feature_name' and 'dict_all_scales' must be given")
    part, split, scale = feat_name.split("-")
    if scale not in dict_all_scales:
        raise ValueError("'scale' from 'feature_name' is not in 'dict_all_scales")
    dict_scale = dict_all_scales[scale]
    return part, split, dict_scale


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



# II Main Functions
def get_dict_part_pos(tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=0, start=1):
    """Get dictionary for part to positions.

    Parameters
    ----------
    tmd_len: length of TMD
    jmd_n_len: length of JMD-N
    jmd_c_len: length of JMD-C
    ext_len: length of extending part (starting from C and N terminal part of TMD)
    start: position label of first position

    Returns
    -------
    dict_part_pos: dictionary with parts to positions of parts
    """
    ut.check_number_range(name="start", val=start, min_val=1, just_int=True)
    ut.check_args_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
    pa = Parts()
    jmd_n = list(range(0, jmd_n_len))
    tmd = list(range(jmd_n_len, tmd_len+jmd_n_len))
    jmd_c = list(range(jmd_n_len + tmd_len, jmd_n_len + tmd_len + jmd_c_len))
    # Change int to string and adjust length
    jmd_n = [i + start for i in jmd_n]
    tmd = [i + start for i in tmd]
    jmd_c = [i + start for i in jmd_c]
    dict_part_pos = pa.get_dict_part_seq(tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_c, ext_len=ext_len)
    return dict_part_pos

def _get_positions(dict_part_pos=None, features=None, as_str=True):
    """Get list of positions for given feature names.

    Parameters
    ----------
    dict_part_pos: dictionary with parts to positions of parts
    features: list with feature ids
    as_str: bool whether to return positions as string or list

    Returns
    -------
    list_pos: list with positions for each feature in feat_names
    """
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


# Get feature matrix
def _feature_matrix(feat_names, dict_all_scales, df_parts, accept_gaps):
   """Helper function to create feature matrix via multiple processing"""
   X = np.empty([len(df_parts), len(feat_names)])
   for i, feat_name in enumerate(feat_names):
       part, split, dict_scale = _get_feature_components(feat_name=feat_name,
                                                          dict_all_scales=dict_all_scales)
       X[:, i] = _feature_value(split=split,
                                          dict_scale=dict_scale,
                                          df_parts=df_parts[part.lower()],
                                          accept_gaps=accept_gaps)
   return X


def get_feat_matrix(features=None, df_parts=None, df_scales=None, accept_gaps=False, n_jobs=None):
    """Create feature matrix for given feature ids and sequence parts."""
    # Create feature matrix using parallel processing
    dict_all_scales = ut.get_dict_all_scales(df_scales=df_scales)
    n_processes = min([os.cpu_count(), len(features)]) if n_jobs is None else n_jobs
    feat_chunks = np.array_split(features, n_processes)
    args = zip(feat_chunks, repeat(dict_all_scales), repeat(df_parts), repeat(accept_gaps))
    with mp.get_context("spawn").Pool(processes=n_processes) as pool:
        result = pool.starmap(_feature_matrix, args)
    feat_matrix = np.concatenate(result, axis=1)
    return feat_matrix

