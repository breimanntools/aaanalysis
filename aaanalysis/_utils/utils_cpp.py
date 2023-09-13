"""
This is a script with utility functions and settings for CPP project.
"""
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import aaanalysis._utils._utils_constants as ut_c
import aaanalysis._utils._utils_check as ut_check
import aaanalysis._utils._utils_output as ut_o

# Settings

# Default Split names
STR_SEGMENT = "Segment"
STR_PATTERN = "Pattern"
STR_PERIODIC_PATTERN = "PeriodicPattern"

# DEFAULT Signs
STR_AA_GAP = "-"

# Default column names for cpp analysis
LIST_ALL_PARTS = ["tmd", "tmd_e", "tmd_n", "tmd_c", "jmd_n", "jmd_c", "ext_c", "ext_n",
                  "tmd_jmd", "jmd_n_tmd_n", "tmd_c_jmd_c", "ext_n_tmd_n", "tmd_c_ext_c"]
LIST_PARTS = ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"]

SPLIT_DESCRIPTION = "\n a) {}(i-th,n_split)" \
                    "\n b) {}(N/C,p1,p2,...,pn)" \
                    "\n c) {}(N/C,i+step1/step2,start)" \
                    "\nwith i-th<=n_split, and p1<p2<...<pn," \
                    "\nwhere all numbers should be non-negative integers, and N/C means N or C."\
    .format(STR_SEGMENT, STR_PATTERN, STR_PERIODIC_PATTERN)


# II Main Functions
# General check functions
def check_color(name=None, val=None, accept_none=False):
    """Check if color valid for matplotlib"""
    base_colors = list(mcolors.BASE_COLORS.keys())
    tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
    css4_colors = list(mcolors.CSS4_COLORS.keys())
    all_colors = base_colors + tableau_colors + css4_colors
    if accept_none:
        all_colors.append("none")
    if val not in all_colors:
        error = f"'{name}' ('{val}') is not a valid color. Chose from following: {all_colors}"
        raise ValueError(error)


def check_y_categorical(df=None, y=None):
    """Check if y in df"""
    list_cat_columns = [col for col, data_type in zip(list(df), df.dtypes)
                        if data_type != float and "position" not in col and col != "feature"]
    if y not in list_cat_columns:
        raise ValueError("'y' should be one of following columns with categorical values "
                         "of 'df': {}".format(list_cat_columns))


def check_labels(labels=None, df=None, name_df=None):
    """Check if y not None and just containing 0 and 1"""
    if labels is None:
        raise ValueError("'labels' should not be None")
    if set(labels) == {0} or set(labels) == {1}:
        raise ValueError(f"'labels' contain just one class {set(labels)}.")
    if set(labels) != {0, 1}:
        wrong_labels = [x for x in set(labels) if x not in [0, 1]]
        raise ValueError(f"'labels' should only contain 0 and 1 as class labels and not following: {wrong_labels}")
    if df is not None:
        if len(labels) != len(df):
            raise ValueError(f"'labels' does not match with '{name_df}'")


def check_ylim(df=None, ylim=None, val_col=None, retrieve_plot=False, scaling_factor=1.1):
    """"""
    if ylim is not None:
        ut_check.check_tuple(name="ylim", val=ylim, n=2)
        ut_check.check_float(name="ylim:min", val=ylim[0], just_float=False)
        ut_check.check_float(name="ylim:max", val=ylim[1], just_float=False)
        max_val = round(max(df[val_col]), 3)
        max_y = ylim[1]
        if max_val >= max_y:
            error = "Maximum of 'ylim' ({}) must be higher than maximum" \
                    " value of given datasets ({}).".format(max_y, max_val)
            raise ValueError(error)
    else:
        if retrieve_plot:
            ylim = plt.ylim()
            ylim = (ylim[0] * scaling_factor, ylim[1] * scaling_factor)
    return ylim


# Sequence check function
def _check_seq(seq, len_, name_seq, name_len, verbose):
    """"""
    if seq is None:
        return len_
    else:
        if type(seq) != str:
            raise ValueError(f"'{name_seq}' should be string (type={type(seq)})")
        if len_ is not None:
            # Waring sequence length doesn't match the corresponding length parameter
            if len(seq) != len_ and verbose:
                warning_msg = f"The length of {seq} ({len(seq)}) does not match {name_len} ({len_})."
                ut_o.print_red(f"Warning: {warning_msg}")
        return len(seq)


def check_args_len(tmd_len=None, jmd_n_len=None, jmd_c_len=None, ext_len=None,
                   tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None, verbose=False,
                   accept_tmd_none=False):
    """Check length parameters and if they are matching with sequences if provided"""
    # Check lengths
    tmd_seq_given = tmd_seq is not None or accept_tmd_none # If tmd_seq is given, tmd_len can be None
    ut_check.check_non_negative_number(name="tmd_len", val=tmd_len, accept_none=tmd_seq_given, min_val=1)
    ut_check.check_non_negative_number(name="jmd_n_len", val=jmd_n_len, accept_none=True, min_val=1)
    ut_check.check_non_negative_number(name="jmd_c_len", val=jmd_c_len, accept_none=True, min_val=1)
    ut_check.check_non_negative_number(name="ext_len", val=ext_len, accept_none=True)
    # Check if lengths are matching
    if ext_len is not None:
        if jmd_n_len is None:
            raise ValueError(f"'jmd_n_len' should not be None if 'ext_len' ({ext_len}) is given")
        if jmd_c_len is None:
            raise ValueError(f"'jmd_c_len' should not be None if 'ext_len' ({ext_len}) is given")
        if jmd_n_len is not None and ext_len > jmd_n_len:
            raise ValueError(f"'ext_len' ({ext_len}) must be <= jmd_n_len ({jmd_n_len})")
        if jmd_c_len is not None and ext_len > jmd_c_len:
            raise ValueError(f"'ext_len' ({ext_len}) must be <= jmd_c_len ({jmd_c_len})")
    # Check if lengths and sequences match
    tmd_len = _check_seq(tmd_seq, tmd_len, "tmd_seq", "tmd_len", verbose)
    jmd_n_len = _check_seq(jmd_n_seq, jmd_n_len, "jmd_n_seq", "jmd_n_len", verbose)
    jmd_c_len = _check_seq(jmd_c_seq, jmd_c_len, "jmd_c_seq", "jmd_c_len", verbose)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    return args_len

# TODO
def check_df_seq(df_seq=None, jmd_n_len=None, jmd_c_len=None):
    """Get features from df"""
    # TODO check
    if df_seq is None or not isinstance(df_seq, pd.DataFrame):
        raise ValueError("Type of 'df_seq' ({}) must be pd.DataFrame".format(type(df_seq)))
    if ut_c.COL_ENTRY not in list(df_seq):
        raise ValueError("'{}' must be in 'df_seq'".format(ut_c.COL_ENTRY))
    seq_info_in_df = set(ut_c.COLS_SEQ_INFO).issubset(set(df_seq))
    parts_in_df = set(ut_c.COLS_PARTS).issubset(set(df_seq))
    seq_in_df = ut_c.COL_SEQ in set(df_seq)
    if "start" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'start' in columns. Change column to '{ut_c.COL_TMD_START}'.")
    if "stop" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'stop' in columns. Change column to '{ut_c.COL_TMD_STOP}'.")
    if not (seq_info_in_df or parts_in_df or seq_in_df):
        raise ValueError(f"'df_seq' should contain ['{ut_c.COL_SEQ}'], {ut_c.COLS_SEQ_INFO}, or {ut_c.COLS_PARTS}")
    # Check data type in part or sequence columns
    else:
        if seq_info_in_df or seq_in_df:
            error = f"Sequence column ('{ut_c.COL_SEQ}') should only contain strings"
            dict_wrong_seq = {ut_c.COL_SEQ: [x for x in df_seq[ut_c.COL_SEQ].values if type(x) != str]}
        else:
            cols = ut_c.COLS_PARTS
            error = f"Part columns ('{cols}') should only contain strings"
            dict_wrong_seq = {part: [x for x in df_seq[part].values if type(x) != str] for part in ut_c.COLS_PARTS}
        # Filter empty lists
        dict_wrong_seq = {part: dict_wrong_seq[part] for part in dict_wrong_seq if len(dict_wrong_seq[part]) > 0}
        n_wrong_entries = sum([len(dict_wrong_seq[part]) for part in dict_wrong_seq])
        if n_wrong_entries > 0:
            error += f"\n   but following non-strings exist in given columns: {dict_wrong_seq}"
            raise ValueError(error)
    # Check if only sequence given -> Convert sequence to tmd
    if seq_in_df and not parts_in_df:
        if seq_info_in_df:
            for entry, start, stop in zip(df_seq[ut_c.COL_ENTRY], df_seq[ut_c.COL_TMD_START], df_seq[ut_c.COL_TMD_STOP]):
                ut_check.check_non_negative_number(name=f"tmd_start [{entry}]", val=start)
                ut_check.check_non_negative_number(name=f"tmd_start [{entry}]", val=stop,)
            tmd_start = [int(x) for x in df_seq[ut_c.COL_TMD_START]]
            tmd_stop = [int(x) for x in df_seq[ut_c.COL_TMD_STOP]]
        else:
            tmd_start = 1 if jmd_n_len is None else 1 + jmd_n_len
            tmd_stop = [len(x)-1 for x in df_seq[ut_c.COL_SEQ]]
            if jmd_c_len is not None:
                tmd_stop = [x - jmd_c_len for x in tmd_stop]
        df_seq[ut_c.COL_TMD_START] = tmd_start
        df_seq[ut_c.COL_TMD_STOP] = tmd_stop
        seq_info_in_df = set(ut_c.COLS_SEQ_INFO).issubset(set(df_seq))
    # Check parameter combinations
    if [jmd_n_len, jmd_c_len].count(None) == 1:
        raise ValueError("'jmd_n_len' and 'jmd_c_len' should both be given (not None) or None")
    if not parts_in_df and seq_info_in_df and jmd_n_len is None and jmd_c_len is None:
        error = f"'jmd_n_len' and 'jmd_c_len' should not be None if " \
                f"sequence information ({ut_c.COLS_SEQ_INFO}) are given."
        raise ValueError(error)
    if not seq_info_in_df and jmd_n_len is not None and jmd_c_len is not None:
        error = f"If not all sequence information ({ut_c.COLS_SEQ_INFO}) are given," \
                f"'jmd_n_len' and 'jmd_c_len' should be None."
        raise ValueError(error)
    if not parts_in_df and seq_info_in_df and (jmd_c_len is None or jmd_n_len is None):
        error = "If part columns ({}) are not in 'df_seq' but sequence information ({}), " \
                "\n'jmd_n_len' and 'jmd_c_len' should be given (not None).".format(ut_c.COLS_PARTS, ut_c.COLS_SEQ_INFO)
        raise ValueError(error)
    return df_seq


# Scale check functions
def check_df_scales(df_scales=None, df_parts=None, accept_none=False, accept_gaps=False):
    """Check if df_scales is a valid input and matching to df_parts"""
    ut_check.check_bool(name="accept_gaps", val=accept_gaps)
    if accept_none and df_scales is None:
        return  # Skip check
    if not isinstance(df_scales, pd.DataFrame):
        raise ValueError("'df_scales' should be type pd.DataFrame (not {})".format(type(df_scales)))
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
        error = "'df_scales' should contain numbers." \
                "\n  Following columns contain no numerical values: {}".format(cols_wrong_type)
        raise ValueError(error)
    # Check if NaN in df
    cols_nans = [x for x in list(df_scales) if df_scales[x].isnull().any()]
    if len(cols_nans) > 0:
        error = "'df_scales' should not contain NaN." \
                "\n  Following columns contain NaN: {}".format(cols_nans)
        raise ValueError(error)
    if df_parts is not None:
        f = lambda x: set(x)
        vf = np.vectorize(f)
        char_parts = set().union(*vf(df_parts.values).flatten())
        char_scales = list(set(df_scales.index))
        if accept_gaps:
            char_scales.append(STR_AA_GAP)
        missing_char = [x for x in char_parts if x not in char_scales]
        if accept_gaps:
            for col in list(df_parts):
                for mc in missing_char:
                    df_parts[col] = df_parts[col].str.replace(mc, STR_AA_GAP)
        elif len(missing_char) > 0:
            error = f"Not all characters in sequences from 'df_parts' are covered!"\
                    f"\n  Following characters are missing in 'df_scales': {missing_char}." \
                    f"\n    Consider enabling 'accept_gaps'"
            raise ValueError(error)
    return df_parts


def check_df_cat(df_cat=None, df_scales=None, accept_none=True, verbose=True):
    """Check if df_cat is a valid input"""
    if accept_none and df_cat is None:
        return None     # Skip check
    if not isinstance(df_cat, pd.DataFrame):
        raise ValueError("'df_cat' should be type pd.DataFrame (not {})".format(type(df_cat)))
    # Check columns
    for col in [ut_c.COL_SCALE_ID, ut_c.COL_CAT, ut_c.COL_SUBCAT]:
        if col not in df_cat:
            raise ValueError(f"'{col}' not in 'df_cat'")
    # Check scales from df_cat and df_scales do match
    if df_scales is not None:
        scales_cat = list(df_cat[ut_c.COL_SCALE_ID])
        scales = list(df_scales)
        overlap_scales = [x for x in scales if x in scales_cat]
        difference_scales = list(set(scales).difference(set(scales_cat)))
        # Adjust df_cat and df_scales
        df_cat = df_cat[df_cat[ut_c.COL_SCALE_ID].isin(overlap_scales)]
        df_scales = df_scales[overlap_scales]
        if verbose and len(difference_scales) > 0:
            str_warning = f"Scales from 'df_scales' and 'df_cat' do not overlap completely."
            missing_scales_in_df_scales = [x for x in scales_cat if x not in scales]
            missing_scales_in_df_cat = [x for x in scales if x not in scales_cat]
            if len(missing_scales_in_df_scales) > 0:
                str_warning += f"\n Following scale ids are missing in 'df_scales': {missing_scales_in_df_scales}"
            else:
                str_warning += f"\n Following scale ids are missing in 'df_cat': {missing_scales_in_df_cat}"
            print(f"Warning: {str_warning}")
    return df_cat, df_scales


# Part check functions
def check_list_parts(list_parts=None, all_parts=False):
    """Check if parts from list_parts are columns of df_seq"""
    if list_parts is None:
        list_parts = LIST_ALL_PARTS if all_parts else LIST_PARTS
    if type(list_parts) is str:
        list_parts = [list_parts]
    if type(list_parts) != list:
        raise ValueError(f"'list_parts' must be list with selection of following parts: {LIST_ALL_PARTS}")
    # Check for invalid parts
    wrong_parts = [x for x in list_parts if x not in LIST_ALL_PARTS]
    if len(wrong_parts) > 0:
        str_part = "part" if len(wrong_parts) == 1 else "parts"
        error = f"{wrong_parts} not valid {str_part}.\n  Select from following parts: {LIST_ALL_PARTS}"
        raise ValueError(error)
    return list_parts


def check_df_parts(df_parts=None, verbose=True):
    """Check if df_parts is a valid input"""
    if df_parts is None:
        warning = "Warning 'df_part' should just be None if you want to use CPP for plotting of already existing features"
        if verbose:
            print(warning)
        #raise ValueError("'df_part' should not be None")
    else:
        if not (isinstance(df_parts, pd.DataFrame)):
            raise ValueError(f"'df_parts' ({type(df_parts)}) must be type pd.DataFrame")
        if len(list(df_parts)) == 0 or len(df_parts) == 0:
            raise ValueError("'df_parts' should not be empty pd.DataFrame")
        check_list_parts(list_parts=list(df_parts))
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
                    "\n  Following columns contain no values with type string: {}".format(cols_wrong_type)
            raise ValueError(error)


# Split check functions
def check_split_kws(split_kws=None, accept_none=True):
    """Check if argument dictionary for splits is a valid input"""
    # Split dictionary with data types
    split_kws_types = {STR_SEGMENT: dict(n_split_min=int, n_split_max=int),
                       STR_PATTERN: dict(steps=list, n_min=int, n_max=int, len_max=int),
                       STR_PERIODIC_PATTERN: dict(steps=list)}
    if accept_none and split_kws is None:
        return None     # Skip check
    if not isinstance(split_kws, dict):
        raise ValueError("'split_kws' should be type dict (not {})".format(type(split_kws)))
    # Check if split_kws contains wrong split_types
    split_types = [STR_SEGMENT, STR_PATTERN, STR_PERIODIC_PATTERN]
    wrong_split_types = [x for x in split_kws if x not in split_types]
    if len(wrong_split_types) > 0:
        error = "Following keys are invalid: {}." \
                "\n  'split_kws' should have following structure: {}.".format(wrong_split_types, split_kws_types)
        raise ValueError(error)
    if len(split_kws) == 0:
        raise ValueError("'split_kws' should be not empty")
    # Check if arguments are valid and have valid type
    for split_type in split_kws:
        for arg in split_kws[split_type]:
            if arg not in split_kws_types[split_type]:
                error = "'{}' arg in '{}' of 'split_kws' is invalid." \
                        "\n  'split_kws' should have following structure: {}.".format(arg, split_type, split_kws_types)
                raise ValueError(error)
            arg_val = split_kws[split_type][arg]
            arg_type = type(arg_val)
            target_arg_type = split_kws_types[split_type][arg]
            if target_arg_type != arg_type:
                error = "Type of '{}':'{}' ({}) should be {}".format(arg, arg_val, arg_type, target_arg_type)
                raise ValueError(error)
            if arg_type is list:
                wrong_type = [x for x in arg_val if type(x) is not int]
                if len(wrong_type) > 0:
                    error = "All list elements ({}) of '{}' should have type int.".format(arg_val, arg)
                    raise ValueError(error)
    # Check Segment
    if STR_SEGMENT in split_kws:
        segment_args = split_kws[STR_SEGMENT]
        if segment_args["n_split_min"] > segment_args["n_split_max"]:
            raise ValueError("For '{}', 'n_split_min' should be smaller or equal to 'n_split_max'".format(STR_SEGMENT))
    # Check Pattern
    if STR_PATTERN in split_kws:
        pattern_args = split_kws[STR_PATTERN]
        if pattern_args["n_min"] > pattern_args["n_max"]:
            raise ValueError("For '{}', 'n_min' should be smaller or equal to 'n_max'".format(STR_PATTERN))
        if pattern_args["steps"] != sorted(pattern_args["steps"]):
            raise ValueError("For '{}', 'steps' should be ordered in ascending order.".format(STR_PATTERN))
        if pattern_args["steps"][0] >= pattern_args["len_max"]:
            raise ValueError("For '{}', 'len_max' should be greater than the smallest step in 'steps'.".format(STR_PATTERN))
    # Check PeriodicPattern
    if STR_PERIODIC_PATTERN in split_kws:
        ppattern_args = split_kws[STR_PERIODIC_PATTERN]
        if ppattern_args["steps"] != sorted(ppattern_args["steps"]):
            raise ValueError("For '{}', 'steps' should be ordered in ascending order.".format(STR_PERIODIC_PATTERN))


def check_split(split=None):
    """Check split and convert split name to split type and split arguments"""
    if type(split) is not str:
        raise ValueError("'split' must have type 'str'")
    split = split.replace(" ", "")  # remove whitespace
    try:
        # Check Segment
        if STR_SEGMENT in split:
            split_type = STR_SEGMENT
            i_th, n_split = [int(x) for x in split.split("(")[1].replace(")", "").split(",")]
            # Check if values non-negative integers
            for name, val in zip(["i_th", "n_split"], [i_th, n_split]):
                ut_check.check_non_negative_number(name=name, val=val)
            # Check if i-th and n_split are valid
            if i_th > n_split:
                raise ValueError
            split_kwargs = dict(i_th=i_th, n_split=n_split)
        # Check PeriodicPattern
        elif STR_PERIODIC_PATTERN in split:
            split_type = STR_PERIODIC_PATTERN
            start = split.split("i+")[1].replace(")", "").split(",")
            step1, step2 = [int(x) for x in start.pop(0).split("/")]
            start = int(start[0])
            # Check if values non-negative integers
            for name, val in zip(["start", "step1", "step2"], [start, step1, step2]):
                ut_check.check_non_negative_number(name=name, val=val)
            # Check if terminus valid
            terminus = split.split("i+")[0].split("(")[1].replace(",", "")
            if terminus not in ["N", "C"]:
                raise ValueError
            split_kwargs = dict(terminus=terminus, step1=step1, step2=step2, start=start)
        # Check pattern
        elif STR_PATTERN in split:
            split_type = STR_PATTERN
            list_pos = split.split("(")[1].replace(")", "").split(",")
            terminus = list_pos.pop(0)
            # Check if values non-negative integers
            list_pos = [int(x) for x in list_pos]
            for val in list_pos:
                name = "pos" + str(val)
                ut_check.check_non_negative_number(name=name, val=val)
            # Check if terminus valid
            if terminus not in ["N", "C"]:
                raise ValueError
            # Check if arguments are in order
            if not sorted(list_pos) == list_pos:
                raise ValueError
            split_kwargs = dict(terminus=terminus, list_pos=list_pos)
        else:
            raise ValueError
        tuple_split = (split_type, split_kwargs)
        return tuple_split
    except:
        error = "Wrong split annotation for '{}'. Splits should be denoted as follows:".format(split, SPLIT_DESCRIPTION)
        raise ValueError(error)


# Feature name check function
def check_features(features=None, parts=None, df_scales=None):
    """Check if feature names are valid for df_parts and df_scales

    Parameters
    ----------
    features: str, list of strings, pd.Series
    parts: list or DataFrame with parts, optional
    df_scales: DataFrame with scales, optional
    """
    if isinstance(features, str):
        features = [features]
    if isinstance(features, pd.Series):
        features = list(features)
    # Check type of features list
    if features is None or type(features) is not list:
        error = f"'features' ({type(features)}) should be given as list" \
                f" of feature names with following form:\n  PART-SPLIT-SCALE"
        raise ValueError(error)
    # Check elements of features list
    feat_with_wrong_n_components = [x for x in features if type(x) is not str or len(x.split("-")) != 3]
    if len(feat_with_wrong_n_components) > 0:
        error = "Following elements from 'features' are not valid: {}" \
                "\n  Form of feature names should be PART-SPLIT-SCALE ".format(feat_with_wrong_n_components)
        raise ValueError(error)
    # Check splits
    list_splits = list(set([x.split("-")[1] for x in features]))
    for split in list_splits:
        check_split(split=split)
    # Check parts
    list_parts = list(set([x.split("-")[0] for x in features]))
    if parts is None:
        wrong_parts = [x.lower() for x in list_parts if x.lower() not in LIST_ALL_PARTS]
        if len(wrong_parts) > 0:
            error = f"Following parts from 'features' are not valid {wrong_parts}. " \
                    f"Chose from following: {LIST_ALL_PARTS}"
            raise ValueError(error)
    if parts is not None:
        if isinstance(parts, pd.DataFrame):
            parts = list(parts)
        if not isinstance(parts, list):
            parts = list(parts)
        missing_parts = [x.lower() for x in list_parts if x.lower() not in parts]
        if len(missing_parts) > 0:
            raise ValueError("Following parts from 'features' are not in 'df_parts: {}".format(missing_parts))
    # Check scales
    if df_scales is not None:
        list_scales = list(set([x.split("-")[2] for x in features]))
        missing_scales = [x for x in list_scales if x not in list(df_scales)]
        if len(missing_scales) > 0:
            raise ValueError("Following scales from 'features' are not in 'df_scales: {}".format(missing_scales))
    return features


def check_df_feat(df_feat=None, df_cat=None):
    """Check if df not empty pd.DataFrame"""
    # Check df
    if not isinstance(df_feat, pd.DataFrame):
        raise ValueError(f"'df_feat' should be type pd.DataFrame (not {type(df_feat)})")
    if len(df_feat) == 0 or len(list(df_feat)) == 0:
        raise ValueError("'df_feat' should be not empty")
    # Check if feature column in df_feat
    if ut_c.COL_FEATURE not in df_feat:
        raise ValueError(f"'{ut_c.COL_FEATURE}' must be column in 'df_feat'")
    list_feat = list(df_feat[ut_c.COL_FEATURE])
    for feat in list_feat:
        if feat.count("-") != 2:
            raise ValueError(f"'{feat}' is no valid feature")
    # Check if df_feat matches df_cat
    if df_cat is not None:
        scales = set([x.split("-")[2] for x in list_feat])
        list_scales = list(df_cat[ut_c.COL_SCALE_ID])
        missing_scales = [x for x in scales if x not in list_scales]
        if len(missing_scales) > 0:
            raise ValueError(f"Following scales occur in 'df_feat' but not in 'df_cat': {missing_scales}")
    return df_feat.copy()


# Scale functions
def get_dict_all_scales(df_scales=None):
    """Get nested dictionary where each scales is a key for a amino acid scale value dictionary"""
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col])) for col in list(df_scales)}
    return dict_all_scales


def get_vf_scale(dict_scale=None, accept_gaps=False):
    """Vectorized function to calculate the mean for a feature"""
    if not accept_gaps:
        # Vectorized scale function
        vf_scale = np.vectorize(lambda x: np.mean([dict_scale[a] for a in x]))
    else:
        # Except NaN derived from 'X' in sequence if not just 'X' in sequence (3x slower)
        def get_mean_excepting_nan(x):
            vals = np.array([dict_scale.get(a, np.NaN) for a in x])
            # TODO!! check if working with nan possible
            #if np.isnan(vals).all():
            #    raise ValueError(f"Not all values in sequence split ('{x}') should result in NaN")
            return vals
        vf_scale = np.vectorize(lambda x: np.nanmean(get_mean_excepting_nan(x)))
    return vf_scale


