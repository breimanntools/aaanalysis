"""
Script for SequenceFeature() object that combines scales, splits, and parts to create
    feature names, feature values, or a feature matrix for ML or CPP pipelines.
"""
import pandas as pd

from ._utils_cpp import _get_positions, get_dict_part_pos, get_feat_matrix
from ._part import Parts
from ._split import SplitRange

import aaanalysis.utils as ut


# I Helper Functions


# II Main functions
# Parts and splits
def get_df_parts(df_seq=None, list_parts=None, jmd_n_len=None, jmd_c_len=None, ext_len=None):
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
        dict_part_seq = pa.get_dict_part_seq(tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_c, ext_len=ext_len)
        dict_part_seq = {part: dict_part_seq[part] for part in list_parts}
        dict_parts[entry] = dict_part_seq
    df_parts = pd.DataFrame.from_dict(dict_parts, orient="index")
    return df_parts


def get_split_kws(n_split_min=1, n_split_max=15, steps_pattern=None, n_min=2, n_max=4, len_max=15,
                  steps_periodicpattern=None, split_types=None):
    """Get split kws for CPP class"""
    if steps_pattern is None:
        # Differences between interacting amino acids in helix (without gaps) include 6, 7 ,8 to include gaps
        steps_pattern = [3, 4]
    if steps_periodicpattern is None:
        steps_periodicpattern = [3, 4]  # Differences between interacting amino acids in helix (without gaps)
    split_kws = {ut.STR_SEGMENT: dict(n_split_min=n_split_min, n_split_max=n_split_max),
                 ut.STR_PATTERN: dict(steps=steps_pattern, n_min=n_min, n_max=n_max, len_max=len_max),
                 ut.STR_PERIODIC_PATTERN: dict(steps=steps_periodicpattern)}
    split_kws = {x: split_kws[x] for x in split_types}
    ut.check_split_kws(split_kws=split_kws)
    return split_kws


# Features
def feat_matrix(features=None, df_parts=None, df_scales=None, accept_gaps=False, n_jobs=None):
    """Create feature matrix for given feature ids and sequence parts."""
    # Create feature matrix using parallel processing
    _feat_matrix = get_feat_matrix(features=features, df_parts=df_parts,
                                  df_scales=df_scales, accept_gaps=accept_gaps, n_jobs=n_jobs)
    labels = df_parts.index.tolist()
    return _feat_matrix, labels  # X, y


def get_features(list_parts=None, split_kws=None, df_scales=None):
    """Create list of all feature ids for given Parts, Splits, and Scales"""
    scales = list(df_scales)
    spr = SplitRange()
    features = []
    for split_type in split_kws:
        args = split_kws[split_type]
        labels_s = getattr(spr, "labels_" + split_type.lower())(**args)
        features.extend(["{}-{}-{}".format(p.upper(), s, sc) for p in list_parts for s in labels_s for sc in scales])
    return features


def get_feat_names(features=None, df_cat=None, tmd_len=20, jmd_c_len=10, jmd_n_len=10, ext_len=0, start=1):
    """Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions])."""
    # Get feature names
    dict_part_pos = get_dict_part_pos(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                          ext_len=ext_len, start=start)
    list_positions = _get_positions(dict_part_pos=dict_part_pos, features=features)
    dict_scales = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SCALE_NAME]))
    feat_names = []
    for feat_id, pos in zip(features, list_positions):
        part, split, scale = feat_id.split("-")
        split_type = split.split("(")[0]
        if split_type == ut.STR_SEGMENT and len(pos.split(",")) > 2:
            pos = pos.split(",")[0] + "..." + pos.split(",")[-1]
        if split_type == ut.STR_PERIODIC_PATTERN:
            step = split.split("+")[1].split(",")[0]
            pos = pos.split(",")[0] + ".." + step + ".." + pos.split(",")[-1]
        feat_names.append(f"{dict_scales[scale]} [{pos}]")
    return feat_names
