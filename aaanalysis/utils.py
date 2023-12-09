"""
This is the main script for utility functions, folder structure, and constants.
Most imported modules contain checking functions for code validation.
"""
import os
import platform
from functools import lru_cache
import pandas as pd
import numpy as np
import warnings

from .config import options

# Import utility functions explicitly
from ._utils.check_type import (check_number_range, check_number_val, check_str, check_bool,
                                check_dict, check_tuple, check_list_like, check_str_in_list,
                                check_ax)
from ._utils.check_data import (check_X, check_X_unique_samples,
                                check_labels, check_match_X_labels,
                                check_array_like, check_superset_subset,
                                check_df)
from ._utils.check_models import check_mode_class, check_model_kwargs
from ._utils.check_plots import (check_vmin_vmax, check_color, check_cmap, check_palette,
                                 check_ylim, check_y_categorical)

from ._utils.new_types import ArrayLike1D, ArrayLike2D

from ._utils.decorators import (catch_runtime_warnings, CatchRuntimeWarnings,
                                catch_convergence_warning, ClusteringConvergenceException,
                                catch_invalid_divide_warning)

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


# Constants
FONT_AA = "DejaVu Sans Mono"
STR_AA_GAP = "-"

# Part names
LIST_ALL_PARTS = ["tmd", "tmd_e", "tmd_n", "tmd_c", "jmd_n", "jmd_c", "ext_c", "ext_n",
                  "tmd_jmd", "jmd_n_tmd_n", "tmd_c_jmd_c", "ext_n_tmd_n", "tmd_c_ext_c"]
LIST_PARTS = ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"]

# Split names
STR_SEGMENT = "Segment"
STR_PATTERN = "Pattern"
STR_PERIODIC_PATTERN = "PeriodicPattern"
LIST_SPLIT_TYPES = [STR_SEGMENT, STR_PATTERN, STR_PERIODIC_PATTERN]
SPLIT_DESCRIPTION = f"\n a) {STR_SEGMENT}(i-th,n_split)" \
                    f"\n b) {STR_PATTERN}(N/C,p1,p2,...,pn)" \
                    f"\n c) {STR_PERIODIC_PATTERN}(N/C,i+step1/step2,start)" \
                    f"\nwith i-th<=n_split, and p1<p2<...<pn," \
                    f" where all numbers should be non-negative integers, and N/C means 'N' or 'C'."

# Scale dataset names
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
COL_JMD_N = "jmd_n"
COL_TMD = "tmd"
COL_JMD_C = "jmd_c"
COLS_PARTS = [COL_JMD_N, COL_TMD, COL_JMD_C]
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
COL_AA_TEST = "amino_acids_test"
COL_AA_REF = "amino_acids_ref"

# Columns for df_feat after processing with explainable AI methods
COL_FEAT_IMPORT = "feat_importance"
COL_FEAT_IMP_STD = "feat_importance_std"
COL_FEAT_IMPACT = "feat_impact"

COLS_FEAT_SCALES = [COL_CAT, COL_SUBCAT, COL_SCALE_NAME]
COLS_FEAT_STAT = [COL_ABS_AUC, COL_ABS_MEAN_DIF, COL_MEAN_DIF, COL_STD_TEST, COL_STD_REF]
COLS_FEAT_WEIGHT = [COL_FEAT_IMPORT, COL_FEAT_IMP_STD, COL_FEAT_IMPACT]
DICT_VALUE_TYPE = {COL_ABS_AUC: "mean",
                   COL_ABS_MEAN_DIF: "mean",
                   COL_MEAN_DIF: "mean",
                   COL_STD_TEST: "mean",
                   COL_STD_REF: "mean",
                   COL_FEAT_IMPORT: "sum",
                   COL_FEAT_IMP_STD: "mean",
                   COL_FEAT_IMPACT: "sum"}

# Labels
LABEL_FEAT_VAL = "Feature value"
LABEL_HIST_COUNT = "Number of proteins"
LABEL_HIST_DEN = "Relative density"

LABEL_FEAT_IMPORT_CUM = "Cumulative feature importance\n(normalized) [%]"
LABEL_FEAT_IMPACT_CUM = "Cumulative feature impact\n(normalized) [%]"
LABEL_CBAR_FEAT_IMPACT_CUM = "Cumulative feature impact"

LABEL_FEAT_IMPORT = "Importance [%]"
LABEL_FEAT_IMPACT = "Impact [%]"
LABEL_FEAT_RANKING = "Feature ranking"
LABEL_SCALE_CAT = "Scale category"
LABEL_MEAN_DIF = "Mean difference"

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


# I Helper functions
def _retrieve_string_starting_at_end(seq, start=None, end=None):
    """Reverse_string_start_end"""
    def reverse_string(s):
        return s[::-1]
    reversed_seq = reverse_string(seq)
    reversed_seq_part = reversed_seq[start:end]
    seq = reverse_string(reversed_seq_part)
    return seq


def get_dict_part_seq(tmd=None, jmd_n=None, jmd_c=None):
    """Get dictionary for part to sequence"""
    # Length of extending part (starting from C and N terminal part of TMD)
    ext_len = options["ext_len"]
    tmd_n = tmd[0:round(len(tmd) / 2)]
    tmd_c = tmd[round(len(tmd) / 2):]
    # Historical feature parts (can be set via aa.options["ext_len"] = 4
    ext_n = _retrieve_string_starting_at_end(jmd_n, start=0, end=ext_len)  # helix_stop motif for TMDs
    ext_c = jmd_c[0:ext_len]  # anchor for TMDs
    tmd_e = ext_n + tmd + ext_c
    part_seq_dict = {'tmd': tmd, 'tmd_e': tmd_e,
                     'tmd_n': tmd_n, 'tmd_c': tmd_c,
                     'jmd_n': jmd_n, 'jmd_c': jmd_c,
                     'ext_n': ext_n, 'ext_c': ext_c,
                     'tmd_jmd': jmd_n + tmd + jmd_c,
                     'jmd_n_tmd_n': jmd_n + tmd_n, 'tmd_c_jmd_c': tmd_c + jmd_c,
                     'ext_n_tmd_n': ext_n + tmd_n, 'tmd_c_ext_c': tmd_c + ext_c}
    return part_seq_dict


# II Main functions
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


# Check parts
def check_parts_len(tmd_len=None, jmd_n_len=None, jmd_c_len=None, accept_none_len=False,
                    tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None):
    """Check length parameters and if they are matching with sequences if provided"""
    tmd_seq = check_str(name="tmd_seq", val=tmd_seq, accept_none=True, return_empty_string=True)
    jmd_n_seq = check_str(name="jmd_n_seq", val=jmd_n_seq, accept_none=True, return_empty_string=True)
    jmd_c_seq = check_str(name="jmd_c_seq", val=jmd_c_seq, accept_none=True, return_empty_string=True)
    # If sequences is not None, set length to sequence length
    if len(jmd_n_seq + tmd_seq + jmd_c_seq) > 0:
        tmd_len, jmd_n_len, jmd_c_len = len(tmd_seq), len(jmd_n_seq), len(jmd_c_seq)
    else:
        tmd_seq = jmd_n_seq = jmd_c_seq = None
    # Check lengths
    ext_len = options["ext_len"]
    check_number_range(name="tmd_len", val=tmd_len, accept_none=accept_none_len, min_val=1, just_int=True)
    check_number_range(name="jmd_n_len", val=jmd_n_len, accept_none=accept_none_len, min_val=0, just_int=True)
    check_number_range(name="jmd_c_len", val=jmd_c_len, accept_none=accept_none_len, min_val=0, just_int=True)
    check_number_range(name="ext_len", val=ext_len, min_val=0, accept_none=True, just_int=True)
    # Check len_ext
    if ext_len is not None and ext_len != 0:
        # Check if ext_len exceeds either jmd_n_len or jmd_c_len
        if ext_len > jmd_n_len:
            raise ValueError(f"'ext_len' ({ext_len}) must be <= length of jmd_n ({jmd_n_len})")
        if ext_len > jmd_c_len:
            raise ValueError(f"'ext_len' ({ext_len}) must be <= length of jmd_c ({jmd_c_len})")
    # Create output dictionaries
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    args_seq = dict(tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
    return args_len, args_seq


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


# Check features
def _check_part(part=None, feature=None, list_parts=None):
    """Check if feature PART is valid"""
    list_parts = check_list_like(name="list_parts", val=list_parts, accept_str=True)
    list_parts = [x.lower() for x in list_parts]
    part = part.replace(" ", "")  # remove whitespace
    error = f"Wrong 'PART' for '{feature}'. Features should be 'PART-SPLIT-SCALE', with parts from: {list_parts}."
    if part.lower() in LIST_ALL_PARTS:
        error += f"\n Or include '{part.lower()}' in parts."
    if part.lower() not in list_parts:
        raise ValueError(error)


def _check_split(split=None, feature=None):
    """Check feature SPLIT has valid form"""
    split = split.replace(" ", "")  # remove whitespace
    error = f"Wrong 'SPLIT' for '{feature}'. Features should be 'PART-SPLIT-SCALE', with splits denoted as follows: {SPLIT_DESCRIPTION}"
    # Check Segment
    if STR_SEGMENT in split:
        try:
            i_th, n_split = [int(x) for x in split.split("(")[1].replace(")", "").split(",")]
        except:
            error += "\n Error: Wrong Segment."
            raise ValueError(error)
        # Check if values non-negative integers
        for name, val in zip(["i_th", "n_split"], [i_th, n_split]):
            check_number_range(name=name, val=val, just_int=True)
        # Check if i-th and n_split are valid
        if i_th > n_split:
            error += "\n Error: i-th segment should be smaller than 'n_splits'."
            raise ValueError(error)
    # Check PeriodicPattern
    elif STR_PERIODIC_PATTERN in split:
        try:
            start = split.split("i+")[1].replace(")", "").split(",")
            step1, step2 = [int(x) for x in start.pop(0).split("/")]
            start = int(start[0])
        except:
            error += "\n Error: Wrong PeriodicPattern."
            raise ValueError(error)
        # Check if values non-negative integers
        for name, val in zip(["start", "step1", "step2"], [start, step1, step2]):
            check_number_range(name=name, val=val, min_val=1, just_int=True)
        # Check if terminus valid
        terminus = split.split("i+")[0].split("(")[1].replace(",", "")
        if terminus not in ["N", "C"]:
            error += "\n Error: Terminus should be 'N' or 'C'."
            raise ValueError(error)
    # Check pattern (must be after PeriodicPattern due to string matching)
    elif STR_PATTERN in split:
        try:
            list_pos = split.split("(")[1].replace(")", "").split(",")
            terminus = list_pos.pop(0)
            list_pos = [int(x) for x in list_pos]
        except:
            error += "\n Error: Wrong Pattern."
            raise ValueError(error)
        # Check if contain at least one position
        if len(list_pos) < 1:
            error += "\n Error: Steps should contain at least 1 element."
            raise ValueError(error)
        # Check if values non-negative integers
        for val in list_pos:
            name = "pos" + str(val)
            check_number_range(name=name, val=val, min_val=1, just_int=True)
        # Check if terminus valid
        if terminus not in ["N", "C"]:
            error += "\n Error: Terminus should be 'N' or 'C'."
            raise ValueError(error)
        # Check if arguments are in order
        if not sorted(list_pos) == list_pos:
            error += "\n Error: Positions are in wrong order."
            raise ValueError(error)
    else:
        raise ValueError(error)

def _check_scale(scale=None, feature=None, list_scales=None):
    """"""
    list_scales = check_list_like(name="list_scales", val=list_scales, accept_str=True)
    if list_scales is not None:
        error = f"Wrong 'SCALE' for '{feature}'. Features should be 'PART-SPLIT-SCALE', with scales from: {LIST_ALL_PARTS}"
        if scale not in list_scales:
            raise ValueError(error)


def check_features(features=None, list_parts=None, list_scales=None):
    """Check if feature names are valid for list of parts  and df_scales
    """
    features = check_list_like(name="features", val=features, accept_none=False, accept_str=True, convert=True)
    list_parts = check_list_parts(list_parts=list_parts, all_parts=True)
    # Check elements of features list
    features_wrong_n_components = [x for x in features if type(x) is not str or len(x.split("-")) != 3]
    if len(features_wrong_n_components) > 0:
        error = (f"Following elements from 'features' are not valid: {features_wrong_n_components}"
                 f"\n  Form of features should be 'PART-SPLIT-SCALE'")
        raise ValueError(error)
    # Check part, split, and scale
    for feature in features:
        part, split, scale = feature.split("-")
        _check_part(part=part, feature=feature, list_parts=list_parts)
        _check_split(split=split, feature=feature)
        _check_scale(scale=scale, feature=feature, list_scales=list_scales)
    return features


# Check matching features with parts
def _get_min_pos_split(split=None):
    """"""
    if STR_SEGMENT in split:
        n_min = int(split.split(",")[1].replace(")", ""))
    elif STR_PERIODIC_PATTERN in split:
        n_min = int(split.split(",")[-1].replace(")", ""))
    elif STR_PERIODIC_PATTERN:
        n_min = int(split.split(",")[1].replace(")", ""))
    else:
        raise ValueError(f"Wrong 'split' ({split})")
    return n_min


def check_match_features_seq_parts(features=None, tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None,
                                   tmd_len=20, jmd_n_len=10, jmd_c_len=10):
    """Check if sequence lengths do match with length requirements of features"""
    # Check match of part length and features
    if None in [tmd_seq, jmd_n_seq, jmd_c_seq] or len(jmd_n_seq + tmd_seq + jmd_c_seq) == 0:
        dict_part_seq = get_dict_part_seq(tmd=list(range(0, tmd_len)),
                                          jmd_c=list(range(0, jmd_c_len)),
                                          jmd_n=list(range(0, jmd_n_len)))
        for feature in features:
            part, split, scale = feature.split("-")
            n_min = _get_min_pos_split(split=split)
            seq = dict_part_seq[part.lower()]
            if len(seq) < n_min:
                raise ValueError(f"Sequence length (n={len(seq)}) too short for '{feature}' feature (n_min={n_min})")
    # Check match of sequence part length and features
    else:
        dict_part_seq = get_dict_part_seq(tmd=tmd_seq, jmd_c=jmd_c_seq, jmd_n=jmd_n_seq)
        for feature in features:
            part, split, scale = feature.split("-")
            n_min = _get_min_pos_split(split=split)
            seq = dict_part_seq[part.lower()]
            if len(seq) < n_min:
                raise ValueError(
                    f"Sequence 'part' ({seq}, n={len(seq)}) too short for '{feature}' feature (n_min={n_min})")


def check_match_features_df_parts(features=None, df_parts=None):
    """Check if df_parts does match with length requirements of features"""
    for feature in features:
        part, split, scale = feature.split("-")
        n_min = _get_min_pos_split(split=split)
        if any(df_parts[part.lower()].map(len) < n_min):
            mask =  df_parts[part.lower()].map(len) < n_min
            list_seq = df_parts[mask][part.lower()].to_list()
            if len(list_seq) == 1:
                seq = list_seq[0]
                raise ValueError(f"Sequence 'part' ('{seq}', n={len(seq)}) is too short for '{feature}' feature (n_min={n_min})")
            else:
                raise ValueError(
                    f"For '{feature}' feature (n_min={n_min}), following sequence 'parts' are too short: {list_seq}")


# TODO check each of this checking function (make simpler)
# Check key dataframes using constants and general checking functions
# df_seq, df_parts, df_scales, df_cat, df_feat, and features
def check_df_seq(df_seq=None, jmd_n_len=None, jmd_c_len=None):
    """Get features from df"""
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
        error = f"If part columns ({COLS_PARTS}) are not in 'df_seq' but sequence information ({COLS_SEQ_TMD_POS_KEY}), " \
                "\n'jmd_n_len' and 'jmd_c_len' should be given (not None)."
        raise ValueError(error)
    return df_seq


def check_df_parts(df_parts=None, verbose=True):
    """Check if df_parts is a valid input"""
    if df_parts is None:
        warning = "'df_part' should just be None if you want to use CPP for plotting of already existing features"
        if verbose:
            warnings.warn(warning)
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
                    f"\n  Following columns contain no values with type string: {cols_wrong_type}"
            raise ValueError(error)


# Scale check functions
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
        error = "'df_scales' should only contain numbers." \
                f"\n  Following columns contain no numerical values: {cols_wrong_type}"
        raise ValueError(error)
    # Check if NaN in df
    cols_nans = [x for x in list(df_scales) if df_scales[x].isnull().any()]
    if len(cols_nans) > 0:
        error = "'df_scales' should not contain NaN." \
                f"\n  Following columns contain NaN: {cols_nans}"
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

def check_df_feat(df_feat=None, df_cat=None, list_parts=None):
    """Check if df not empty pd.DataFrame"""
    # Check df
    cols_feat = COLS_FEAT_SCALES + COLS_FEAT_STAT + [COL_FEATURE]
    check_df(df=df_feat, name="df_feat", cols_requiered=cols_feat)
    if len(df_feat) == 0 or len(list(df_feat)) == 0:
        raise ValueError("'df_feat' should be not empty")
    # Check features
    features = list(df_feat[COL_FEATURE])
    check_features(features=features, list_parts=list_parts)
    # Check if df_feat matches df_cat
    if df_cat is not None:
        scales = set([x.split("-")[2] for x in features])
        list_scales = list(df_cat[COL_SCALE_ID])
        missing_scales = [x for x in scales if x not in list_scales]
        if len(missing_scales) > 0:
            raise ValueError(f"Following scales occur in 'df_feat' but not in 'df_cat': {missing_scales}")
    return df_feat.copy()

