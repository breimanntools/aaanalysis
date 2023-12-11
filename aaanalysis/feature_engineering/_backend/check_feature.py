"""
This is a script for the common SequenceFeature and CPP checking functions
"""
import pandas as pd
import numpy as np
import warnings

import aaanalysis.utils as ut
from .cpp.utils_feature import get_parts


# Helper functions
def _get_max_pos_split(split=None):
    """Get maximum position requiered for split"""
    if ut.STR_SEGMENT in split:
        n_max = int(split.split(",")[1].replace(")", ""))
    elif ut.STR_PERIODIC_PATTERN in split:
        n_max = int(split.split(",")[-1].replace(")", ""))
    elif ut.STR_PATTERN:
        n_max = int(split.split(",")[-1].replace(")", ""))
    else:
        raise ValueError(f"Wrong 'split' ({split})")
    return n_max


def _get_max_pos_split_kws(split_kws=None):
    """Get maximum position requiered for splits basd on split_kws"""
    list_n_max = []
    if ut.STR_SEGMENT in split_kws:
        n_max = split_kws[ut.STR_SEGMENT]["n_split_max"]
        list_n_max.append(n_max)
    if ut.STR_PATTERN in split_kws:
        n_max = split_kws[ut.STR_PATTERN]["len_max"]
        list_n_max.append(n_max)
    if ut.STR_PERIODIC_PATTERN in split_kws:
        n_max = split_kws[ut.STR_PERIODIC_PATTERN]["steps"][0]
        list_n_max.append(n_max)
    if len(list_n_max) == 0:
        raise ValueError(f"Wrong 'split_kws' ({split_kws})")
    n_max = max(list_n_max)
    return n_max


# II Main Functions
# Check splits
# TODO check if can be simplified
def check_split_kws(split_kws=None, accept_none=True):
    """Check if argument dictionary for splits is a valid input"""
    # Split dictionary with data types
    split_kws_types = {ut.STR_SEGMENT: dict(n_split_min=int, n_split_max=int),
                       ut.STR_PATTERN: dict(steps=list, n_min=int, n_max=int, len_max=int),
                       ut.STR_PERIODIC_PATTERN: dict(steps=list)}
    if accept_none and split_kws is None:
        return None     # Skip check
    if not isinstance(split_kws, dict):
        raise ValueError(f"'split_kws' should be type dict (not {split_kws})")
    # Check if split_kws contains wrong split_types
    wrong_split_types = [x for x in split_kws if x not in ut.LIST_SPLIT_TYPES]
    if len(wrong_split_types) > 0:
        error = f"Following keys are invalid: {wrong_split_types}." \
                "\n  'split_kws' should have following structure: {split_kws_types}."
        raise ValueError(error)
    if len(split_kws) == 0:
        raise ValueError("'split_kws' should be not empty")
    # Check if arguments are valid and have valid type
    for split_type in split_kws:
        for arg in split_kws[split_type]:
            if arg not in split_kws_types[split_type]:
                error = f"'{arg}' arg in '{split_type}' of 'split_kws' is invalid." \
                        "\n  'split_kws' should have following structure: {split_kws_types}."
                raise ValueError(error)
            arg_val = split_kws[split_type][arg]
            arg_type = type(arg_val)
            target_arg_type = split_kws_types[split_type][arg]
            if target_arg_type != arg_type:
                error = f"Type of '{arg}':'{arg_val}' ({arg_type}) should be {target_arg_type}"
                raise ValueError(error)
            if arg_type is list:
                wrong_type = [x for x in arg_val if type(x) is not int]
                if len(wrong_type) > 0:
                    error = f"All list elements ({arg_val}) of '{arg}' should have type int."
                    raise ValueError(error)
    # Check Segment
    if ut.STR_SEGMENT in split_kws:
        segment_args = split_kws[ut.STR_SEGMENT]
        if segment_args["n_split_min"] > segment_args["n_split_max"]:
            raise ValueError(f"For '{ut.STR_SEGMENT}', 'n_split_min' should be smaller or equal to 'n_split_max'")
    # Check Pattern
    if ut.STR_PATTERN in split_kws:
        pattern_args = split_kws[ut.STR_PATTERN]
        if pattern_args["n_min"] > pattern_args["n_max"]:
            raise ValueError(f"For '{ut.STR_PATTERN}', 'n_min' should be smaller or equal to 'n_max'")
        if pattern_args["steps"] != sorted(pattern_args["steps"]):
            raise ValueError(f"For '{ut.STR_PATTERN}', 'steps' should be ordered in ascending order.")
        if pattern_args["steps"][0] >= pattern_args["len_max"]:
            raise ValueError(f"For '{ut.STR_PATTERN}', 'len_max' should be greater than the smallest step in 'steps'.")
    # Check PeriodicPattern
    if ut.STR_PERIODIC_PATTERN in split_kws:
        periodicpattern_args = split_kws[ut.STR_PERIODIC_PATTERN]
        if periodicpattern_args["steps"] != sorted(periodicpattern_args["steps"]):
            raise ValueError(f"For '{ut.STR_PERIODIC_PATTERN}', 'steps' should be ordered in ascending order.")


# Check parts
def check_parts_len(tmd_len=None, jmd_n_len=None, jmd_c_len=None, accept_none_len=False,
                    tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None):
    """Check length parameters and if they are matching with sequences if provided"""
    tmd_seq = ut.check_str(name="tmd_seq", val=tmd_seq, accept_none=True, return_empty_string=True)
    jmd_n_seq = ut.check_str(name="jmd_n_seq", val=jmd_n_seq, accept_none=True, return_empty_string=True)
    jmd_c_seq = ut.check_str(name="jmd_c_seq", val=jmd_c_seq, accept_none=True, return_empty_string=True)
    # If sequences is not None, set length to sequence length
    if len(jmd_n_seq + tmd_seq + jmd_c_seq) > 0:
        tmd_len, jmd_n_len, jmd_c_len = len(tmd_seq), len(jmd_n_seq), len(jmd_c_seq)
    else:
        tmd_seq = jmd_n_seq = jmd_c_seq = None
    # Check lengths
    ext_len = ut.options["ext_len"]
    ut.check_number_range(name="tmd_len", val=tmd_len, accept_none=accept_none_len, min_val=1, just_int=True)
    ut.check_number_range(name="jmd_n_len", val=jmd_n_len, accept_none=accept_none_len, min_val=0, just_int=True)
    ut.check_number_range(name="jmd_c_len", val=jmd_c_len, accept_none=accept_none_len, min_val=0, just_int=True)
    ut.check_number_range(name="ext_len", val=ext_len, min_val=0, accept_none=True, just_int=True)
    # Check len_ext
    if ext_len is not None and ext_len != 0:
        # Check if ext_len exceeds either jmd_n_len or jmd_c_len
        if ext_len > jmd_n_len:
            raise ValueError(f"'ext_len' ({ext_len}) must be <= length of jmd_n ({jmd_n_len})")
        if ext_len > jmd_c_len:
            raise ValueError(f"'ext_len' ({ext_len}) must be <= length of jmd_c ({jmd_c_len})")
    # Create out.dictionaries
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_seq = dict(tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
    return args_len, args_seq


def check_match_features_seq_parts(features=None, tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                                   tmd_len=20, jmd_n_len=10, jmd_c_len=10):
    """Check if sequence lengths do match with length requirements of features"""
    # Check match of part length and features
    if None in [tmd_seq, jmd_n_seq, jmd_c_seq] or len(jmd_n_seq + tmd_seq + jmd_c_seq) == 0:
        jmd_n, tmd, jmd_c = get_parts(start=1, jmd_n_len=jmd_n_len, tmd_len=tmd_len, jmd_c_len=jmd_c_len)
        dict_part_seq = ut.get_dict_part_seq(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c)
        for feature in features:
            part, split, scale = feature.split("-")
            n_max = _get_max_pos_split(split=split)
            seq = dict_part_seq[part.lower()]
            if len(seq) < n_max:
                raise ValueError(f"Sequence length (n={len(seq)}) too short for '{feature}' feature (n_max={n_max})")
    # Check match of sequence part length and features
    else:
        dict_part_seq = ut.get_dict_part_seq(tmd=tmd_seq, jmd_c=jmd_c_seq, jmd_n=jmd_n_seq)
        for feature in features:
            part, split, scale = feature.split("-")
            n_max = _get_max_pos_split(split=split)
            seq = dict_part_seq[part.lower()]
            if len(seq) < n_max:
                raise ValueError(
                    f"Sequence 'part' ({seq}, n={len(seq)}) too short for '{feature}' feature (n_max={n_max})")

# Check df_seq
# TODO check if can be simplified
def check_df_seq(df_seq=None, jmd_n_len=None, jmd_c_len=None):
    """Check columns from df_seq"""
    ut.check_df(name="df_seq", df=df_seq, cols_requiered=[ut.COL_ENTRY])
    seq_in_df = ut.COL_SEQ in set(df_seq)
    seq_pos_in_df = set(ut.COLS_SEQ_POS).issubset(set(df_seq))
    seq_parts_in_df = set(ut.COLS_SEQ_PARTS).issubset(set(df_seq))
    if "start" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'start' in columns. Change column to '{ut.COL_TMD_START}'.")
    if "stop" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'stop' in columns. Change column to '{ut.COL_TMD_STOP}'.")
    if not (seq_in_df or seq_pos_in_df or seq_parts_in_df):
        raise ValueError(f"'df_seq' should contain ['{ut.COL_SEQ}'] and {ut.COLS_SEQ_POS}, or {ut.COLS_SEQ_PARTS}")
    # Check data type in part or sequence columns
    else:
        if seq_pos_in_df or seq_in_df:
            error = f"Sequence column ('{ut.COL_SEQ}') should only contain strings"
            dict_wrong_seq = {ut.COL_SEQ: [x for x in df_seq[ut.COL_SEQ].values if type(x) != str]}
        else:
            error = f"Part columns ('{ut.COLS_SEQ_PARTS}') should only contain strings"
            dict_wrong_seq = {part: [x for x in df_seq[part].values if type(x) != str] for part in ut.COLS_SEQ_PARTS}
        # Filter empty lists
        dict_wrong_seq = {part: dict_wrong_seq[part] for part in dict_wrong_seq if len(dict_wrong_seq[part]) > 0}
        n_wrong_entries = sum([len(dict_wrong_seq[part]) for part in dict_wrong_seq])
        if n_wrong_entries > 0:
            error += f"\n   but following non-strings exist in given columns: {dict_wrong_seq}"
            raise ValueError(error)
    # Check if only sequence given -> Convert sequence to tmd
    if seq_in_df and not seq_parts_in_df:
        if seq_pos_in_df:
            for entry, start, stop in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_TMD_START], df_seq[ut.COL_TMD_STOP]):
                ut.check_number_range(name=f"tmd_start [{entry}]", val=start, just_int=True)
                ut.check_number_range(name=f"tmd_start [{entry}]", val=stop, just_int=True)
            tmd_start = [int(x) for x in df_seq[ut.COL_TMD_START]]
            tmd_stop = [int(x) for x in df_seq[ut.COL_TMD_STOP]]
        else:
            tmd_start = 1 if jmd_n_len is None else 1 + jmd_n_len
            tmd_stop = [len(x)-1 for x in df_seq[ut.COL_SEQ]]
            if jmd_c_len is not None:
                tmd_stop = [x - jmd_c_len for x in tmd_stop]
        df_seq[ut.COL_TMD_START] = tmd_start
        df_seq[ut.COL_TMD_STOP] = tmd_stop
        seq_pos_in_df = set(ut.COLS_SEQ_POS).issubset(set(df_seq))
    # Check parameter combinations
    if [jmd_n_len, jmd_c_len].count(None) == 1:
        raise ValueError("'jmd_n_len' and 'jmd_c_len' should both be given (not None) or None")
    if not seq_parts_in_df and seq_pos_in_df and jmd_n_len is None and jmd_c_len is None:
        error = f"'jmd_n_len' and 'jmd_c_len' should not be None if " \
                f"sequence information ({ut.COLS_SEQ_POS}) are given."
        raise ValueError(error)
    if not seq_pos_in_df and jmd_n_len is not None and jmd_c_len is not None:
        error = f"If not all sequence information ({ut.COLS_SEQ_POS}) are given," \
                f"'jmd_n_len' and 'jmd_c_len' should be None."
        raise ValueError(error)
    if not seq_parts_in_df and seq_pos_in_df and (jmd_c_len is None or jmd_n_len is None):
        error = f"If part columns ({ut.COLS_SEQ_PARTS}) are not in 'df_seq' but sequence information ({ut.COLS_SEQ_POS}), " \
                "\n'jmd_n_len' and 'jmd_c_len' should be given (not None)."
        raise ValueError(error)
    return df_seq


# Check df_parts
def check_df_parts(df_parts=None, accept_none=False):
    """Check if df_parts is a valid input"""
    ut.check_df(name="df_parts", df=df_parts, accept_none=accept_none)
    if df_parts is None and accept_none:
        return # Skip check
    if len(list(df_parts)) == 0 or len(df_parts) == 0:
        raise ValueError("'df_parts' should not be empty pd.DataFrame")
    ut.check_list_parts(list_parts=list(df_parts))
    # Check if columns are unique
    if len(list(df_parts)) != len(set(df_parts)):
        raise ValueError("Column names in 'df_parts' must be unique. Drop duplicates!")
    # Check if index is unique
    if len(list(df_parts.index)) != len(set(df_parts.index)):
        raise ValueError("Index in 'df_parts' must be unique. Drop duplicates!")
    # Check if columns contain strings
    dict_dtype = dict(df_parts.dtypes)
    cols_wrong_type = [col for col in dict_dtype if dict_dtype[col] not in [object, str]]
    if len(cols_wrong_type) > 0:
        error = "'df_parts' should contain sequences with type string." \
                f"\n  Following columns contain no values with type string: {cols_wrong_type}"
        raise ValueError(error)


def check_match_df_parts_features(df_parts=None, features=None):
    """Check if df_parts does match with length requirements of features"""
    if df_parts is None:
        return # Skip check
    for feature in features:
        part, split, scale = feature.split("-")
        n_min = _get_max_pos_split(split=split)
        if any(df_parts[part.lower()].map(len) < n_min):
            mask = df_parts[part.lower()].map(len) < n_min
            list_seq = df_parts[mask][part.lower()].to_list()
            if len(list_seq) == 1:
                seq = list_seq[0]
                raise ValueError(f"Sequence 'part' ('{seq}', n={len(seq)}) is too short"
                                 f"\n  for '{feature}' feature (n_min={n_min})")
            else:
                raise ValueError(
                    f"For '{feature}' feature (n_min={n_min}),"
                    f"\n  following sequence 'parts' are too short: {list_seq}")


def check_match_df_parts_list_parts(df_parts=None, list_parts=None):
    """"""
    if list_parts is not None and df_parts is not None:
        list_parts = ut.check_list_like(name="list_parts", val=list_parts, accept_str=True, convert=True)
        missing_parts = [part.lower() for part in list_parts if part.lower() is not list(df_parts)]
        raise ValueError(f"'part' ({missing_parts}) must be in columns of 'df_parts': {list(df_parts)}")


def check_match_df_parts_split_kws(df_parts=None, split_kws=None):
    """"""
    n_max = _get_max_pos_split_kws(split_kws=split_kws)
    for part in list(df_parts):
        if any(df_parts[part.lower()].map(len) < n_max):
            mask = df_parts[part.lower()].map(len) < n_max
            list_seq = df_parts[mask][part.lower()].to_list()
            if len(list_seq) == 1:
                seq = list_seq[0]
                raise ValueError(
                    f"'{part}' part contains too short sequence ('{seq}', n={len(seq)})"
                    f"\n  for '{split_kws}' split_kws (n_max={n_max})")
            else:
                seq = list_seq[0]
                raise ValueError(
                    f"For split_kws (n_max={n_max}): '{split_kws}',"
                    f"\n  following '{part}' part contains too short sequences (e.g., '{seq}', n={len(seq)}).")


# Check df_scales & df_cat
# TODO check if can be simplified
def check_df_scales(df_scales=None, accept_none=False):
    """Check if df_scales is a valid input and matching to df_parts"""
    ut.check_df(name="df_scales", df=df_scales, accept_none=accept_none)
    if df_scales is None and accept_none:
        return  # Skip check
    # Check if columns are unique
    if len(list(df_scales)) != len(set(df_scales)):
        raise ValueError("Column names in 'df_scales' must be unique. Drop duplicates!")
    # Check if index is unique
    if len(list(df_scales.index)) != len(set(df_scales.index)):
        raise ValueError("Index in 'df_scales' must be unique. Drop duplicates!")
    # Check if columns contain number
    dict_dtype = dict(df_scales.dtypes)
    cols_wrong_type = [col for col in dict_dtype if dict_dtype[col] not in [np.number, int, float]]
    if len(cols_wrong_type) > 0:
        error = "'df_scales' should only contain numbers." \
                f"\n  Following columns contain no numerical values: {cols_wrong_type}"
        raise ValueError(error)
    # Check if NaN in df
    cols_nans = [x for x in list(df_scales) if df_scales[x].isnull().any()]
    if len(cols_nans) > 0:
        error = "'df_scales' should not contain NaN." \
                f"\n  Following columns contain NaN: {cols_nans}"
        raise ValueError(error)


def check_match_df_scales_features(df_scales=None, features=None):
    """Check if scale ids from df_scales does match with features"""
    if df_scales is None:
        return # Skip check
    scales_feat = [x.split("-")[2] for x in features]
    scales = list(df_scales)
    missing_scales_in_df_scales = [x for x in scales_feat if x not in scales]
    if len(missing_scales_in_df_scales) > 0:
        raise ValueError(f"Following scale ids are missing in 'df_scales': {missing_scales_in_df_scales}")


# Check df_cat
def check_df_cat(df_cat=None, accept_none=True):
    """"""
    if df_cat is None and accept_none:
        return # Skip check
    cols_cat = [ut.COL_SCALE_ID, ut.COL_CAT, ut.COL_SUBCAT]
    ut.check_df(name="df_cat", df=df_cat, cols_requiered=cols_cat, accept_none=accept_none)


def check_match_df_cat_features(df_cat=None, features=None):
    """Check if scale ids from df_cat does match with features"""
    if df_cat is None:
        return # Skip check
    scales_feat = [x.split("-")[2] for x in features]
    scales_cat = list(df_cat[ut.COL_SCALE_ID])
    missing_scales_in_df_cat = [x for x in scales_feat if x not in scales_cat]
    if len(missing_scales_in_df_cat) > 0:
        raise ValueError(f"Following scale ids are missing in 'df_cat': {missing_scales_in_df_cat}")


# Check matching of df_scales with df_parts and df_scales
def check_match_df_parts_df_scales(df_parts=None, df_scales=None, accept_gaps=False):
    """Check if characters from df_parts match with scales from df_scales"""
    if df_parts is not None and df_scales is not None:
        f = lambda x: set(x)
        vf = np.vectorize(f)
        char_parts = set().union(*vf(df_parts.values).flatten())
        char_scales = list(set(df_scales.index))
        if accept_gaps:
            char_scales.append(ut.STR_AA_GAP)
        missing_char = [x for x in char_parts if x not in char_scales]
        # Replace gaps by default amino acid gap
        if accept_gaps:
            for col in list(df_parts):
                for mc in missing_char:
                    df_parts[col] = df_parts[col].str.replace(mc, ut.STR_AA_GAP)
        elif len(missing_char) > 0:
            error = f"Not all characters in sequences from 'df_parts' are covered!"\
                    f"\n  Following characters are missing in 'df_scales': {missing_char}." \
                    f"\n  Consider enabling 'accept_gaps'"
            raise ValueError(error)
    return df_parts


def check_match_df_scales_df_cat(df_cat=None, df_scales=None, verbose=True):
    """Check if scale ids matches from df_cat and df_scales"""
    if df_scales is not None and df_cat is not None:
        scales_cat = list(df_cat[ut.COL_SCALE_ID])
        scales = list(df_scales)
        overlap_scales = [x for x in scales if x in scales_cat]
        difference_scales = list(set(scales).difference(set(scales_cat)))
        # Adjust df_cat and df_scales
        df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(overlap_scales)]
        df_scales = df_scales[overlap_scales]
        if verbose and len(difference_scales) > 0:
            str_warning = f"Scales from 'df_scales' and 'df_cat' do not overlap completely."
            missing_scales_in_df_scales = [x for x in scales_cat if x not in scales]
            missing_scales_in_df_cat = [x for x in scales if x not in scales_cat]
            if len(missing_scales_in_df_scales) > 0:
                str_warning += f"\n Following scale ids are missing in 'df_scales': {missing_scales_in_df_scales}"
            else:
                str_warning += f"\n Following scale ids are missing in 'df_cat': {missing_scales_in_df_cat}"
            warnings.warn(str_warning)
    return df_scales, df_cat
