"""
Config with folder structure. Most imported modules contain checking functions for code validation
"""
import os
import platform

# Import utility functions for specific purposes
from aaanalysis._utils._utils_constants import *
from aaanalysis._utils._utils_check import *
from aaanalysis._utils._utils_output import *

# Import utility function for specific modules
from aaanalysis._utils.utils_aaclust import *
from aaanalysis._utils.utils_cpp import *


# I Folder structure
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = _folder_path(FOLDER_PROJECT, '_data')
URL_DATA = "https://github.com/breimanntools/aaanalysis/tree/master/aaanalysis/data/"


# II MAIN FUNCTIONS
# Check key dataframes using constants and general checking functions (df_seq, df_parts, df_cat, df_scales, df_feat)
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


# Scale check functions
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


# Check further important data types
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