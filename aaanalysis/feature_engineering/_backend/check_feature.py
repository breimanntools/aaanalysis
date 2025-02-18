"""
This is a script for the common SequenceFeature and CPP checking functions
"""
import pandas as pd
import numpy as np
import warnings

import aaanalysis.utils as ut
from .cpp.utils_feature import get_part_positions

# TODO!! check tmd_start and tmd_stop within sequence length !!


# Helper functions
def _return_empty_string(val=None):
    val = "" if val is None else val
    return val


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
def check_split_kws(split_kws=None, accept_none=True):
    """Check if argument dictionary for splits is a valid input"""
    # Define the expected structure and types for split_kws
    split_kws_types = {ut.STR_SEGMENT: dict(n_split_min=int, n_split_max=int),
                       ut.STR_PATTERN: dict(steps=list, n_min=int, n_max=int, len_max=int),
                       ut.STR_PERIODIC_PATTERN: dict(steps=list)}
    if accept_none and split_kws is None:
        return None     # Skip check
    if not isinstance(split_kws, dict):
        raise ValueError(f"'split_kws' should be type dict (not {split_kws})")
    if len(split_kws) == 0:
        raise ValueError("'split_kws' should be not empty")
    # Check if split_kws contains wrong split_types
    invalid_split_types = [x for x in split_kws if x not in ut.LIST_SPLIT_TYPES]
    if len(invalid_split_types) > 0:
        error = f"Following 'split_types' are invalid: {invalid_split_types}." \
                f"\n  'split_kws' should have following structure: {split_kws_types}."
        raise ValueError(error)
    # Validate split types and argument structure
    for split_type in split_kws:
        expected_args = split_kws_types[split_type]
        missing_args = [arg for arg in expected_args if arg not in split_kws[split_type]]
        if missing_args:
            raise ValueError(f"Missing required arguments for '{split_type}': {missing_args}")
        for arg in split_kws[split_type]:
            if arg not in split_kws_types[split_type]:
                error = f"'{arg}' arg in '{split_type}' of 'split_kws' is invalid." \
                        f"\n  'split_kws' should have following structure: {split_kws_types}."
                raise ValueError(error)
            arg_val = split_kws[split_type][arg]
            expected_arg_type = split_kws_types[split_type][arg]
            if not isinstance(arg_val, expected_arg_type):
                error = f"Type of '{arg}':'{arg_val}' ('{type(arg_val)}') should be '{expected_arg_type}'"
                raise ValueError(error)
            if isinstance(arg_val, list):
                wrong_type = [x for x in arg_val if type(x) is not int]
                if len(wrong_type) > 0:
                    error = f"All list elements ({arg_val}) of '{arg}' should have type int."
                    raise ValueError(error)
    # Check Segment
    if ut.STR_SEGMENT in split_kws:
        segment_args = split_kws[ut.STR_SEGMENT]
        n_split_min, n_split_max = segment_args["n_split_min"], segment_args["n_split_max"]
        ut.check_number_range(name=f"{ut.STR_SEGMENT}[n_split_min]", val=n_split_min, just_int=True, min_val=1)
        ut.check_number_range(name=f"{ut.STR_SEGMENT}[n_split_max]", val=n_split_max, just_int=True, min_val=1)
        if n_split_min > n_split_max:
            raise ValueError(f"For '{ut.STR_SEGMENT}', 'n_split_min' ({n_split_min}) should be smaller "
                             f"or equal to 'n_split_max' ({n_split_max})")
    # Check Pattern
    if ut.STR_PATTERN in split_kws:
        pattern_args = split_kws[ut.STR_PATTERN]
        n_min, n_max = pattern_args["n_min"], pattern_args["n_max"]
        steps_pattern, len_max = pattern_args["steps"], pattern_args["len_max"]
        ut.check_number_range(name=f"{ut.STR_PATTERN}[n_min]", val=n_min, just_int=True, min_val=1)
        ut.check_number_range(name=f"{ut.STR_PATTERN}[n_max]", val=n_max, just_int=True, min_val=1)
        ut.check_number_range(name=f"{ut.STR_PATTERN}[len_max]", val=n_max, just_int=True, min_val=1)
        if n_min > n_max:
            raise ValueError(f"For '{ut.STR_PATTERN}', 'n_min' ({n_min}) should be smaller or equal to 'n_max' ({n_max})")
        if not isinstance(steps_pattern, list) or len(steps_pattern) < 1:
            raise ValueError(f"'steps_pattern' ({steps_pattern}) should be non-empty list of with at"
                             f" least 1 non-negative integers")
        if steps_pattern != sorted(steps_pattern):
            raise ValueError(f"For '{ut.STR_PATTERN}', 'steps_pattern' ({steps_pattern}) should be ordered in ascending order.")
        if steps_pattern[0] >= len_max:
            raise ValueError(f"For '{ut.STR_PATTERN}', 'len_max' ({len_max}) should be greater than the smallest step "
                             f"in 'steps_pattern' ({steps_pattern}).")
        for i, step in enumerate(steps_pattern):
            ut.check_number_range(name=f"{ut.STR_PATTERN}[steps_pattern](step{i+1})", val=step, just_int=True,
                                  min_val=1)
    # Check PeriodicPattern
    if ut.STR_PERIODIC_PATTERN in split_kws:
        periodicpattern_args = split_kws[ut.STR_PERIODIC_PATTERN]
        steps_periodicpattern = periodicpattern_args["steps"]
        if not isinstance(steps_periodicpattern, list) or len(steps_periodicpattern) != 2:
            raise ValueError(f"'steps_periodicpattern' ({steps_periodicpattern}) should be list of"
                             f" with exactly 2 non-negative integers")
        if steps_periodicpattern != sorted(steps_periodicpattern):
            raise ValueError(f"For '{ut.STR_PERIODIC_PATTERN}', 'steps_periodicpattern' ({steps_periodicpattern}) "
                             f"should be ordered in ascending order.")
        step1, step2 = steps_periodicpattern
        ut.check_number_range(name=f"{ut.STR_PERIODIC_PATTERN}[steps_periodicpattern](step1)",
                              val=step1, just_int=True, min_val=1)
        ut.check_number_range(name=f"{ut.STR_PERIODIC_PATTERN}[steps_periodicpattern](step2)",
                              val=step2, just_int=True, min_val=1)


# Check parts
def check_parts_len(tmd_len=None, jmd_n_len=None, jmd_c_len=None,
                    accept_none_tmd_len=False, accept_none_jmd_len=False,
                    tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                    check_jmd_seq_len_consistent=False):
    """Check length parameters and if they are matching with sequences if provided"""
    ut.check_str(name="tmd_seq", val=tmd_seq, accept_none=True)
    ut.check_str(name="jmd_n_seq", val=jmd_n_seq, accept_none=True)
    ut.check_str(name="jmd_c_seq", val=jmd_c_seq, accept_none=True)
    tmd_seq = _return_empty_string(val=tmd_seq)
    jmd_n_seq = _return_empty_string(val=jmd_n_seq)
    jmd_c_seq = _return_empty_string(val=jmd_c_seq)
    # If sequences is not None, set length to sequence length
    if len(jmd_n_seq + tmd_seq + jmd_c_seq) > 0:
        if check_jmd_seq_len_consistent:
            if jmd_n_len is None or jmd_n_len != len(jmd_n_seq):
                raise ValueError(f"Not matching of 'jmd_n_len' ({jmd_n_len}) and 'jmd_n_seq' ({jmd_n_seq})")
            if jmd_c_len is None or jmd_c_len != len(jmd_c_seq):
                raise ValueError(f"Not matching of 'jmd_c_len' ({jmd_c_len}) and 'jmd_n_seq' ({jmd_c_seq})")
        tmd_len, jmd_n_len, jmd_c_len = len(tmd_seq), len(jmd_n_seq), len(jmd_c_seq)
    else:
        tmd_seq = jmd_n_seq = jmd_c_seq = None
    # Check lengths
    ext_len = ut.options["ext_len"]
    ut.check_number_range(name="tmd_len", val=tmd_len, accept_none=accept_none_tmd_len, min_val=1, just_int=True)
    ut.check_number_range(name="jmd_n_len", val=jmd_n_len, accept_none=accept_none_jmd_len, min_val=0, just_int=True)
    ut.check_number_range(name="jmd_c_len", val=jmd_c_len, accept_none=accept_none_jmd_len, min_val=0, just_int=True)
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
        jmd_n, tmd, jmd_c = get_part_positions(start=1, jmd_n_len=jmd_n_len, tmd_len=tmd_len, jmd_c_len=jmd_c_len)
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
def _get_tmd_positions(row):
    """Get position of tmd from sequence"""
    tmd, seq = row[ut.COL_TMD], row[ut.COL_SEQ]
    tmd_start = seq.find(tmd)
    tmd_stop = tmd_start + len(tmd) if tmd_start != -1 else -1
    if tmd_start == -1 or tmd_start == tmd_stop:
        raise ValueError(f"'{ut.COL_TMD}' is not contained in '{ut.COL_SEQ}' for '{row[ut.COL_ENTRY]}' entry")
    tmd_start += 1
    return pd.Series([tmd_start, tmd_stop])


def check_match_df_seq_jmd_len(df_seq=None, jmd_n_len=None, jmd_c_len=None):
    """Check matching of df_seq and jmd lengths."""
    df_seq = df_seq.copy()
    pos_based = set(ut.COLS_SEQ_POS).issubset(set(df_seq))
    part_based = set(ut.COLS_SEQ_PARTS).issubset(set(df_seq))
    seq_based = ut.COL_SEQ in list(df_seq)
    seq_tmd_based = set(ut.COLS_SEQ_TMD).issubset(set(df_seq))
    # 'jmd_n_len' and 'jmd_c_len' should be already checked if None or int by interface
    if [jmd_n_len, jmd_c_len].count(None) >= 1:
        if not part_based:
            raise ValueError(f"'jmd_n_len' and 'jmd_c_len' should be both given if '{ut.COLS_SEQ_PARTS}' are not provided")
        else:
            if [jmd_n_len, jmd_c_len].count(None) == 1:
                raise ValueError(f"'jmd_n_len' and 'jmd_c_len' should be both given or both None "
                                 f"if '{ut.COLS_SEQ_PARTS}' are provided")
    # Get 'tmd_start' and 'tmd_stop'
    elif [jmd_n_len, jmd_c_len].count(None) == 0:
        if part_based and not pos_based:
            df_seq[ut.COL_SEQ] = df_seq[ut.COL_JMD_N] + df_seq[ut.COL_TMD] + df_seq[ut.COL_JMD_C]
            df_seq[[ut.COL_TMD_START, ut.COL_TMD_STOP]] = df_seq.apply(_get_tmd_positions, axis=1)
    if not pos_based and not part_based:
        if seq_tmd_based:
            df_seq[[ut.COL_TMD_START, ut.COL_TMD_STOP]] = df_seq.apply(_get_tmd_positions, axis=1)
        elif seq_based:
            tmd_start = 1 + jmd_n_len
            list_seq = []
            list_tmd_stop = []
            for seq in df_seq[ut.COL_SEQ]:
                # If 'jmd_n_len' and 'jmd_c_len' exceed the sequence length, sequence is adjusted using gaps.
                dif_jmd_n_len_seq = jmd_n_len - len(seq)
                if dif_jmd_n_len_seq >= 0:
                    # Add one gap for TMD
                    seq += ut.STR_AA_GAP * (dif_jmd_n_len_seq + 1)
                dif_jmd_len_seq = jmd_c_len + jmd_n_len - len(seq)
                if dif_jmd_len_seq >= 0:
                    # If no jmd_n, add gaps to N-terminus of sequence
                    if jmd_n_len == 0:
                        seq = ut.STR_AA_GAP * (dif_jmd_len_seq + 1) + seq
                    # If jmd_n, add gaps to C-terminus of sequence
                    else:
                        seq += ut.STR_AA_GAP * (dif_jmd_len_seq + 1)
                    tmd_stop = tmd_start
                else:
                    tmd_stop = len(seq) - jmd_c_len
                list_seq.append(seq)
                list_tmd_stop.append(tmd_stop)
            df_seq[ut.COL_TMD_START] = tmd_start
            df_seq[ut.COL_TMD_STOP] = list_tmd_stop
            df_seq[ut.COL_SEQ] = list_seq
    return df_seq


# Check df_parts matching
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
    """Check match between df_parts and list of parts"""
    if list_parts is not None and df_parts is not None:
        list_parts = ut.check_list_like(name="list_parts", val=list_parts, accept_str=True, convert=True)
        missing_parts = [part.lower() for part in list_parts if part.lower() is not list(df_parts)]
        raise ValueError(f"'part' ({missing_parts}) must be in columns of 'df_parts': {list(df_parts)}")


def check_match_df_parts_split_kws(df_parts=None, split_kws=None):
    """Check if df_parts and split_kws match regarding the sequence size"""
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
def check_df_scales(df_scales=None, accept_none=False):
    """Check if df_scales is a valid input and matching to df_parts"""
    ut.check_df(name=ut.FILE_DF_SCALES, df=df_scales, accept_none=accept_none)
    if df_scales is None and accept_none:
        return  # Skip check
    # Check if columns are unique
    if len(list(df_scales)) != len(set(df_scales)):
        raise ValueError("Column names in 'df_scales' must be unique. Drop duplicates!")
    # Check if index is unique
    if len(list(df_scales.index)) != len(set(df_scales.index)):
        raise ValueError("Index in 'df_scales' must be unique. Drop duplicates!")
    # Check if columns contain numbers
    dict_dtype = dict(df_scales.dtypes)
    cols_wrong_type = [col for col in dict_dtype if not np.issubdtype(dict_dtype[col], np.number)]
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
    """Check if df_cat is valid"""
    if df_cat is None and accept_none:
        return  # Skip check
    cols_cat = [ut.COL_SCALE_ID, ut.COL_CAT, ut.COL_SUBCAT, ut.COL_SCALE_NAME]
    ut.check_df(name=ut.FILE_DF_CAT, df=df_cat, cols_requiered=cols_cat, accept_none=accept_none)


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
        if len(difference_scales) > 0:
            missing_scales_in_df_scales = [x for x in scales_cat if x not in scales]
            missing_scales_in_df_cat = [x for x in scales if x not in scales_cat]
            # Error since all scale ids must be covered by df_cat
            if missing_scales_in_df_cat:
                raise ValueError(f"Following scale ids from 'df_scales' are missing in 'df_cat': {missing_scales_in_df_cat}")
            # Only warning (excepted)
            if verbose and len(missing_scales_in_df_scales) > 0:
                warnings.warn(f"Following scale ids from 'df_cat' are missing in 'df_scales': {missing_scales_in_df_scales}")
    return df_scales, df_cat
