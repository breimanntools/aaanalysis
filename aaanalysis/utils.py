"""
This is the main script for utility functions, folder structure, and constants.
Most imported modules contain checking functions for code validation.
"""
import os
import platform
from functools import lru_cache
import pandas as pd
import numpy as np

from .config import options

# Import utility functions explicitly
from ._utils.check_data import (check_X, check_X_unique_samples, check_labels, check_match_X_labels,
                                check_array_like, check_superset_subset,
                                check_col_in_df)
from ._utils.check_models import check_mode_class, check_model_kwargs
from ._utils.check_type import (check_number_range, check_number_val, check_str, check_bool,
                                check_dict, check_tuple, check_list_like, check_str_in_list,
                                check_ax)

from ._utils.utils_cpp import (check_color, check_y_categorical, check_labels_, check_ylim,
                               check_args_len, check_args_len, check_list_parts,
                               check_split_kws, check_split,
                               get_dict_all_scales, get_vf_scale,
                               STR_SEGMENT, STR_PATTERN, STR_PERIODIC_PATTERN, STR_AA_GAP,
                               LIST_PARTS, LIST_ALL_PARTS, SPLIT_DESCRIPTION)

from ._utils.new_types import ArrayLike1D, ArrayLike2D

from ._utils.decorators import (catch_runtime_warnings, CatchRuntimeWarnings,
                                catch_convergence_warning, ClusteringConvergenceException,
                                catch_invalid_divide_warning,
                                doc_params)

from ._utils.utils_output import (print_out, print_start_progress, print_progress, print_finished_progress)
from ._utils.utils_ploting import plot_gco, plot_add_bars


# Folder structure
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS-depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = _folder_path(FOLDER_PROJECT, '_data')
URL_DATA = "https://github.com/breimanntools/aaanalysis/tree/master/aaanalysis/data/"


# I Constants
# Default scale datasets for protein analysis
STR_SCALES = "scales"   # Min-max normalized scales (from AAontology)
STR_SCALES_RAW = "scales_raw"   # Raw scales (from AAontology)
STR_SCALES_PC = "scales_pc"     # AAclust pc-based scales (pc: principal component)
STR_SCALE_CAT = "scales_cat"  # AAontology
STR_TOP60 = "top60"    # AAclustTop60
STR_TOP60_EVAL = "top60_eval"  # AAclustTop60 evaluation
NAMES_SCALE_SETS = [STR_SCALES, STR_SCALES_RAW, STR_SCALE_CAT,
                    STR_SCALES_PC, STR_TOP60, STR_TOP60_EVAL]

# AAclust
METRIC_CORRELATION = "correlation"
LIST_METRICS = [METRIC_CORRELATION, "manhattan",  "euclidean", "cosine"]
STR_UNCLASSIFIED = "Unclassified"
COL_N_CLUST = "n_clusters"
COL_BIC = "BIC"
COL_CH = "CH"
COL_SC = "SC"
COL_RANK = "rank"
COLS_EVAL_AACLUST = [COL_N_CLUST, COL_BIC, COL_CH, COL_SC]

# Column names for primary df
# df_seq
COL_ENTRY = "entry"     # ACC, protein entry, uniprot id
COL_NAME = "name"       # Entry name, Protein name, Uniprot Name
COL_LABEL = "label"
COL_SEQ = "sequence"
COLS_PARTS = ["jmd_n", "tmd", "jmd_c"]
COL_TMD_START = "tmd_start"
COL_TMD_STOP = "tmd_stop"
COLS_SEQ_KEY = [COL_ENTRY, COL_SEQ, COL_LABEL]
COLS_SEQ_TMD_POS_KEY = [COL_SEQ, COL_TMD_START, COL_TMD_STOP]  # TODO adjust to COL_ENTRY
COLS_SEQ_TMD_PART_KEY = [COL_ENTRY, COL_SEQ] + COLS_PARTS

# df_part

# df_scales
# Column for df_cat (as defined in AAontology, retrieved by aa.load_scales(name="scale_cat"))
COL_SCALE_ID = "scale_id"
COL_CAT = "category"
COL_SUBCAT = "subcategory"
COL_SCALE_NAME = "scale_name"
COL_SCALE_DES = "scale_description"


# Columns for df_feat
COL_FEATURE = "feature"
# COL_CAT, COL_SUBCAT, COL_SCALE_NAME, COL_SCALE_DES
COL_ABS_AUC = "abs_auc"
COL_ABS_MEAN_DIF = "abs_mean_dif"
COL_MEAN_DIF = "mean_dif"
COL_STD_TEST = "std_test"
COL_STD_REF = "std_ref"
COL_PVAL_MW = "p_val_mann_whitney"
COL_PVAL_FDR = "p_val_fdr_bh"
COL_POSITION = "positions"

# Columns for df_feat after processing with explainable AI methods
COL_FEAT_IMPORTANCE = "feat_importance"
COO_FEAT_IMP_STD = "feat_importance_std"
COL_FEAT_IMPACT = "feat_impact"


# Column name datasets (DOM_GSEC)


# Column names cpp features


# Standard colors
COLOR_SHAP_POS = '#FF0D57'  # (255, 13, 87)
COLOR_SHAP_NEG = '#1E88E5'  # (30, 136, 229)
COLOR_FEAT_POS = '#9D2B39'  # (157, 43, 57) Mean difference
COLOR_FEAT_NEG = '#326599'  # (50, 101, 133) Mean difference
COLOR_FEAT_IMP = '#7F7F7F'  # (127, 127, 127) feature importance
COLOR_TMD = '#00FA9A'       # (0, 250, 154)
COLOR_JMD = '#0000FF'       # (0, 0, 255)

DICT_COLOR = {"SHAP_POS": COLOR_SHAP_POS,
              "SHAP_NEG": COLOR_SHAP_NEG,
              "FEAT_POS": COLOR_FEAT_POS,
              "FEAT_NEG": COLOR_FEAT_NEG,
              "FEAT_IMP": COLOR_FEAT_IMP,
              "TMD": COLOR_TMD,
              "JMD": COLOR_JMD}

DICT_COLOR_CAT = {"ASA/Volume": "tab:blue",
                  "Composition": "tab:orange",
                  "Conformation": "tab:green",
                  "Energy": "tab:red",
                  "Others": "tab:gray",
                  "Polarity": "gold",
                  "Shape": "tab:cyan",
                  "Structure-Activity": "tab:brown"}

# Parameter options for cmaps and color dicts
STR_CMAP_CPP = "CPP"
STR_CMAP_SHAP = "SHAP"
STR_DICT_COLOR = "DICT_COLOR"
STR_DICT_CAT = "DICT_CAT"

# II Helper functions

# III MAIN FUNCTIONS
# Caching for data loading for better performance (data loaded ones)
@lru_cache(maxsize=None)
def read_excel_cached(name, index_col=None):
    """Load cached dataframe to save loading time"""
    df = pd.read_excel(name, index_col=index_col)
    return df.copy()

@lru_cache(maxsize=None)
def read_csv_cached(name, sep=None):
    """Load cached dataframe to save loading time"""
    df = pd.read_csv(name, sep=sep)
    return df.copy()

# Main check functions
def check_verbose(verbose):
    if verbose is None:
        # System level verbosity
        verbose = options['verbose']
    else:
        check_bool(name="verbose", val=verbose)
    return verbose


def check_metric(metric=None):
    """"""
    if metric not in LIST_METRICS:
        error = f"'metric' should be None or one of following: {LIST_METRICS}"
        raise ValueError(error)


# TODO check each of this checking function (make simpler)
# Check key dataframes using constants and general checking functions
# df_seq, df_parts, df_cat, df_scales, df_feat, and features
def check_df_seq(df_seq=None, jmd_n_len=None, jmd_c_len=None):
    """Get features from df"""
    # TODO check
    if df_seq is None or not isinstance(df_seq, pd.DataFrame):
        raise ValueError("Type of 'df_seq' ({}) must be pd.DataFrame".format(type(df_seq)))
    if COL_ENTRY not in list(df_seq):
        raise ValueError("'{}' must be in 'df_seq'".format(COL_ENTRY))
    seq_info_in_df = set(COLS_SEQ_TMD_POS_KEY).issubset(set(df_seq))
    parts_in_df = set(COLS_PARTS).issubset(set(df_seq))
    seq_in_df = COL_SEQ in set(df_seq)
    if "start" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'start' in columns. Change column to '{COL_TMD_START}'.")
    if "stop" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'stop' in columns. Change column to '{COL_TMD_STOP}'.")
    if not (seq_info_in_df or parts_in_df or seq_in_df):
        raise ValueError(f"'df_seq' should contain ['{COL_SEQ}'], {COLS_SEQ_TMD_POS_KEY}, or {COLS_PARTS}")
    # Check data type in part or sequence columns
    else:
        if seq_info_in_df or seq_in_df:
            error = f"Sequence column ('{COL_SEQ}') should only contain strings"
            dict_wrong_seq = {COL_SEQ: [x for x in df_seq[COL_SEQ].values if type(x) != str]}
        else:
            cols = COLS_PARTS
            error = f"Part columns ('{cols}') should only contain strings"
            dict_wrong_seq = {part: [x for x in df_seq[part].values if type(x) != str] for part in COLS_PARTS}
        # Filter empty lists
        dict_wrong_seq = {part: dict_wrong_seq[part] for part in dict_wrong_seq if len(dict_wrong_seq[part]) > 0}
        n_wrong_entries = sum([len(dict_wrong_seq[part]) for part in dict_wrong_seq])
        if n_wrong_entries > 0:
            error += f"\n   but following non-strings exist in given columns: {dict_wrong_seq}"
            raise ValueError(error)
    # Check if only sequence given -> Convert sequence to tmd
    if seq_in_df and not parts_in_df:
        if seq_info_in_df:
            for entry, start, stop in zip(df_seq[COL_ENTRY], df_seq[COL_TMD_START], df_seq[COL_TMD_STOP]):
                check_number_range(name=f"tmd_start [{entry}]", val=start, just_int=True)
                check_number_range(name=f"tmd_start [{entry}]", val=stop, just_int=True)
            tmd_start = [int(x) for x in df_seq[COL_TMD_START]]
            tmd_stop = [int(x) for x in df_seq[COL_TMD_STOP]]
        else:
            tmd_start = 1 if jmd_n_len is None else 1 + jmd_n_len
            tmd_stop = [len(x)-1 for x in df_seq[COL_SEQ]]
            if jmd_c_len is not None:
                tmd_stop = [x - jmd_c_len for x in tmd_stop]
        df_seq[COL_TMD_START] = tmd_start
        df_seq[COL_TMD_STOP] = tmd_stop
        seq_info_in_df = set(COLS_SEQ_TMD_POS_KEY).issubset(set(df_seq))
    # Check parameter combinations
    if [jmd_n_len, jmd_c_len].count(None) == 1:
        raise ValueError("'jmd_n_len' and 'jmd_c_len' should both be given (not None) or None")
    if not parts_in_df and seq_info_in_df and jmd_n_len is None and jmd_c_len is None:
        error = f"'jmd_n_len' and 'jmd_c_len' should not be None if " \
                f"sequence information ({COLS_SEQ_TMD_POS_KEY}) are given."
        raise ValueError(error)
    if not seq_info_in_df and jmd_n_len is not None and jmd_c_len is not None:
        error = f"If not all sequence information ({COLS_SEQ_TMD_POS_KEY}) are given," \
                f"'jmd_n_len' and 'jmd_c_len' should be None."
        raise ValueError(error)
    if not parts_in_df and seq_info_in_df and (jmd_c_len is None or jmd_n_len is None):
        error = "If part columns ({}) are not in 'df_seq' but sequence information ({}), " \
                "\n'jmd_n_len' and 'jmd_c_len' should be given (not None).".format(COLS_PARTS, COLS_SEQ_TMD_POS_KEY)
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
        if not isinstance(df_parts, pd.DataFrame):
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
    for col in [COL_SCALE_ID, COL_CAT, COL_SUBCAT]:
        if col not in df_cat:
            raise ValueError(f"'{col}' not in 'df_cat'")
    # Check scales from df_cat and df_scales do match
    if df_scales is not None:
        scales_cat = list(df_cat[COL_SCALE_ID])
        scales = list(df_scales)
        overlap_scales = [x for x in scales if x in scales_cat]
        difference_scales = list(set(scales).difference(set(scales_cat)))
        # Adjust df_cat and df_scales
        df_cat = df_cat[df_cat[COL_SCALE_ID].isin(overlap_scales)]
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
    check_bool(name="accept_gaps", val=accept_gaps)
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
    if COL_FEATURE not in df_feat:
        raise ValueError(f"'{COL_FEATURE}' must be column in 'df_feat'")
    list_feat = list(df_feat[COL_FEATURE])
    for feat in list_feat:
        if feat.count("-") != 2:
            raise ValueError(f"'{feat}' is no valid feature")
    # Check if df_feat matches df_cat
    if df_cat is not None:
        scales = set([x.split("-")[2] for x in list_feat])
        list_scales = list(df_cat[COL_SCALE_ID])
        missing_scales = [x for x in scales if x not in list_scales]
        if len(missing_scales) > 0:
            raise ValueError(f"Following scales occur in 'df_feat' but not in 'df_cat': {missing_scales}")
    return df_feat.copy()


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
