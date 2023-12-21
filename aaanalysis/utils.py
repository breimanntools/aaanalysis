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

# Import utility functions explicitly (can be imported from this utils file from other modules)
from ._utils.check_type import (check_number_range, check_number_val, check_str, check_bool,
                                check_dict, check_tuple, check_list_like, check_str_in_list,
                                check_ax)
from ._utils.check_data import (check_X, check_X_unique_samples,
                                check_labels, check_match_X_labels, check_match_X_list_labels,
                                check_match_list_labels_names_datasets,
                                check_array_like, check_superset_subset,
                                check_df)
from ._utils.check_models import check_mode_class, check_model_kwargs
from ._utils.check_plots import (check_vmin_vmax, check_color, check_cmap, check_palette,
                                 check_ylim, check_y_categorical)

from ._utils.new_types import ArrayLike1D, ArrayLike2D
from ._utils.decorators import (catch_runtime_warnings, CatchRuntimeWarnings,
                                catch_convergence_warning, ClusteringConvergenceException,
                                catch_invalid_divide_warning)
from ._utils.utils_metrics import (auc_adjusted_, kullback_leibler_divergence_, bic_score_)
from ._utils.utils_output import (print_out, print_start_progress, print_progress, print_finished_progress)
from ._utils.utils_ploting import plot_gco, plot_get_clist, plot_add_bars


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


# Column names for primary df
# df_seq
COL_ENTRY = "entry"     # ACC, protein entry, uniprot id
COL_NAME = "name"       # Entry name, Protein name, Uniprot Name
COL_LABEL = "label"
COL_SEQ = "sequence"
COL_JMD_N = "jmd_n"
COL_TMD = "tmd"
COL_JMD_C = "jmd_c"
COL_TMD_START = "tmd_start"
COL_TMD_STOP = "tmd_stop"
COLS_SEQ_INFO = [COL_ENTRY, COL_SEQ, COL_LABEL]
COLS_SEQ_POS = [COL_SEQ, COL_TMD_START, COL_TMD_STOP]
COLS_SEQ_PARTS = [COL_JMD_N, COL_TMD, COL_JMD_C]
COLS_SEQ_TMD = [COL_SEQ, COL_TMD]

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


# Columns of df_eval
# AAclust (evaluation)
METRIC_CORRELATION = "correlation"
LIST_METRICS = [METRIC_CORRELATION, "manhattan",  "euclidean", "cosine"]
STR_UNCLASSIFIED = "Unclassified"
COL_N_CLUST = "n_clusters"
COL_BIC = "BIC"
COL_CH = "CH"
COL_SC = "SC"
COL_RANK = "rank"
COLS_EVAL_AACLUST = [COL_N_CLUST, COL_BIC, COL_CH, COL_SC]

# dPULearn (evaluation)
COL_N_REL_NEG = "n_rel_neg"
COL_AVG_STD = "avg_STD"
COL_AVG_IQR = "avg_IQR"
COL_AVG_ABS_AUC_POS = "avg_abs_AUC_pos"
COL_AVG_KLD_POS = "avg_KLD_pos"
COL_AVG_ABS_AUC_UNL = "avg_abs_AUC_unl"
COL_AVG_KLD_UNL = "avg_KLD_unl"
COL_AVG_ABS_AUC_NEG = "avg_abs_AUC_neg"
COL_AVG_KLD_NEG = "avg_KLD_neg"
COLS_EVAL_DPULEARN = [COL_N_REL_NEG, COL_AVG_STD, COL_AVG_IQR, COL_AVG_ABS_AUC_POS,
                      COL_AVG_KLD_POS,
                      COL_AVG_ABS_AUC_UNL, COL_AVG_KLD_UNL,
                      COL_AVG_ABS_AUC_NEG, COL_AVG_KLD_NEG]

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


# Adjust df_eval
def add_names_to_df_eval(df_eval=None, names_datasets=None):
    """Add names column to df_eval"""
    if names_datasets is None:
        n_datasets = len(df_eval)
        names_datasets = [f"Set {i}" for i in range(1, n_datasets + 1)]
    df_eval.insert(0, COL_NAME, names_datasets)
    return df_eval


# Main check functions
# Check system level (option) parameters
def check_verbose(verbose):
    """Check if general verbosity is on or off. Adjusted based on options setting and value provided to object"""
    if verbose is None:
        # System level verbosity
        verbose = options['verbose']
    else:
        check_bool(name="verbose", val=verbose)
    return verbose


# Check df_seq
def check_df_seq(df_seq=None, accept_none=False):
    """Check columns from df_seq is valid regarding four distinct formats, differentiated by their respective columns:
        a) Position-based format: ['sequence', 'tmd_start', 'tmd_stop']
        b) Part-based format: ['jmd_n', 'tmd', 'jmd_c']
        c) Sequence-based format: ['sequence']
        d) Sequence-TMD-based format: ['sequence', 'tmd']
    """
    if df_seq is None:
        if accept_none:
            return None
        else:
            raise ValueError("'df_seq' should not be None")
    check_df(name="df_seq", df=df_seq, cols_requiered=[COL_ENTRY])
    pos_based = set(COLS_SEQ_POS).issubset(set(df_seq))
    part_based = set(COLS_SEQ_PARTS).issubset(set(df_seq))
    seq_based = COL_SEQ in list(df_seq)
    seq_tmd_based = set(COLS_SEQ_TMD).issubset(set(df_seq))
    if not (seq_based or pos_based or part_based or seq_tmd_based):
        raise ValueError(f"'df_seq' should contain one of the following sets of columns:"
                         f" a) {COLS_SEQ_POS} (Position-based format)"
                         f" b) {COLS_SEQ_PARTS} (Part-based format)"
                         f" c) {[COL_SEQ]} (Sequence-based format)"
                         f" d) {COLS_SEQ_TMD} (Sequence-TMD-based format)")
    # Check tmd_start & tmd_stop columns
    if "start" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'tmd_start' in columns. Change column to '{COL_TMD_START}'.")
    if "stop" in list(df_seq):
        raise ValueError(f"'df_seq' should not contain 'tmd_stop' in columns. Change column to '{COL_TMD_STOP}'.")
    # Check if different formats are valid
    if pos_based:
        for entry, tmd_start, tmd_stop in zip(df_seq[COL_ENTRY], df_seq[COL_TMD_START], df_seq[COL_TMD_STOP]):
            check_number_range(name=f"'tmd_start'={tmd_start} (entry: '{entry}')", val=tmd_start, just_int=True)
            check_number_range(name=f"'tmd_stop'={tmd_stop} (entry: '{entry}')", val=tmd_stop, just_int=True)
    if part_based:
        for col in COLS_SEQ_PARTS:
            if not all(isinstance(x, str) for x in df_seq[col]):
                raise ValueError(f"'{col}' should only contain strings.")
    if seq_based:
        if not all(isinstance(x, str) for x in df_seq[COL_SEQ]):
            raise ValueError(f"'{COL_SEQ}' should only contain strings.")
    if seq_tmd_based:
        if any([tmd not in seq for tmd, seq in zip(df_seq[COL_TMD], df_seq[COL_SEQ])]):
            raise ValueError(f"Parts in '{COL_TMD}' should be contained in '{COL_SEQ}'")
    # Check matching of parts with 'sequence', 'tmd_start', and 'tmd_stop'
    if part_based and pos_based:
        for i, row in df_seq.iterrows():
            entry = row[COL_ENTRY]
            jmd_n, tmd, jmd_c = row[COL_JMD_N], row[COL_TMD], row[COL_JMD_C]
            tmd_jmd = jmd_n + tmd + jmd_c
            seq, start, stop = row[COL_SEQ], row[COL_TMD_START], row[COL_TMD_STOP]
            if tmd_jmd not in seq:
                raise ValueError(f"For '{entry}' entry, '{COL_JMD_N}', '{COL_TMD}', and '{COL_JMD_C}' "
                                 f"do not match with '{COL_SEQ}'")
            if seq[start-1:stop] != tmd:
                raise ValueError(f"For '{entry}' entry, '{COL_TMD_START}' and '{COL_TMD_STOP}' "
                                 f"do not match with '{COL_TMD}'")


# Check parts
def check_list_parts(list_parts=None, return_default=True, all_parts=False, accept_none=False):
    """Check if parts from list_parts are columns of df_seq"""
    try:
        list_parts = check_list_like(name="list_parts", val=list_parts, accept_none=True, accept_str=True)
    except ValueError:
        raise ValueError(f"'list_parts' must be list with selection of following parts: {LIST_ALL_PARTS}")
    if list_parts is None:
        if return_default:
            list_parts = LIST_ALL_PARTS if all_parts else LIST_PARTS
            return list_parts
        elif accept_none:
            return  # skip further checks
        else:
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
    list_scales = check_list_like(name="list_scales", val=list_scales, accept_none=True, accept_str=True)
    if list_scales is not None:
        error = f"Wrong 'SCALE' for '{feature}'. Features should be 'PART-SPLIT-SCALE', with scales from: {LIST_ALL_PARTS}"
        if scale not in list_scales:
            raise ValueError(error)


def check_features(features=None, list_parts=None, list_scales=None):
    """Check if feature names are valid for list of parts  and df_scales
    """
    features = check_list_like(name="features", val=features, accept_none=False, accept_str=True, convert=True)
    list_parts = check_list_parts(list_parts=list_parts, all_parts=True, return_default=True)
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
        # Scales is only checked if list_scales is provided
        _check_scale(scale=scale, feature=feature, list_scales=list_scales)
    return features


def check_df_feat(df_feat=None, df_cat=None, list_parts=None):
    """Check if df not empty pd.DataFrame"""
    # Check df
    cols_feat = [COL_FEATURE] + COLS_FEAT_SCALES + COLS_FEAT_STAT
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


def check_col_cat(col_cat=None):
    """Check if col_cat valid column from df_feat"""
    if col_cat not in COLS_FEAT_SCALES:
        raise ValueError(f"'col_cat' {col_cat} should be one of the following: {COLS_FEAT_SCALES}")


def check_col_value(col_value=None):
    """Check if col_value valid column from df_feat"""
    cols_feat = COLS_FEAT_STAT + COLS_FEAT_WEIGHT
    if col_value not in cols_feat:
        raise ValueError(f"'col_value' {col_value} should be one of the following: {cols_feat}")
