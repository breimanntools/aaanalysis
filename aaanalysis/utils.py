"""
This is the core script for utility functions, folder structure, and constants.

Utility functions are explicitly imported here such that other modules can import them (via ut.).
These comprise options, datatypes, decorators, check functions, internal utility functions,
and backend of external utility functions.
"""
import os
import platform
from functools import lru_cache
import pandas as pd
import seaborn as sns
import numpy as np

# Options
from .config import (options,
                     check_verbose,
                     check_n_jobs,
                     check_random_state,
                     check_jmd_n_len,
                     check_jmd_c_len)

# Data types
from ._utils.utils_types import (ArrayLike1D,
                                 ArrayLike2D,
                                 VALID_INT_TYPES,
                                 VALID_FLOAT_TYPES,
                                 VALID_INT_FLOAT_TYPES)

# Decorators
from ._utils.decorators import (catch_backend_processing_error,
                                BackendProcessingError,
                                catch_runtime_warnings,
                                CatchRuntimeWarnings,
                                catch_convergence_warning,
                                ClusteringConvergenceException,
                                catch_invalid_divide_warning,
                                catch_undefined_metric_warning,
                                CatchUndefinedMetricWarning)

# Check functions
from ._utils.check_type import (check_number_range,
                                check_number_val,
                                check_str,
                                check_str_options,
                                check_bool,
                                check_dict,
                                check_tuple,
                                check_list_like)
from ._utils.check_data import (check_X,
                                check_X_unique_samples,
                                check_labels,
                                check_match_X_labels,
                                check_match_X_list_labels,
                                check_match_list_labels_names_datasets,
                                check_array_like,
                                check_superset_subset,
                                check_df,
                                check_warning_consecutive_index,
                                check_file_path_exists,
                                check_is_fasta)
from ._utils.check_models import (check_mode_class,
                                  check_model_kwargs,
                                  check_match_list_model_classes_kwargs)
from ._utils.check_plots import (check_fig,
                                 check_ax,
                                 check_figsize,
                                 check_fontsize_args,
                                 check_vmin_vmax,
                                 check_lim,
                                 check_dict_xlims,
                                 check_color,
                                 check_list_colors,
                                 check_dict_color,
                                 check_cmap,
                                 check_palette)

# Internal utility functions
from ._utils.utils_output import (print_out,
                                  print_start_progress,
                                  print_progress,
                                  print_end_progress)
from ._utils.utils_plot_elements import (plot_add_bars,
                                         adjust_spine_to_middle,
                                         ticks_0,
                                         adjust_tuple_elements)
from ._utils.utils_plot_parts import (add_tmd_jmd_bar,
                                      add_tmd_jmd_text,
                                      add_tmd_jmd_xticks,
                                      highlight_tmd_area)

# External (system-level) utility functions (only backend)
from ._utils.plotting import (plot_gco,
                              plot_get_clist_,
                              plot_legend_)
from ._utils.metrics import (auc_adjusted_,
                             kullback_leibler_divergence_,
                             bic_score_)


# Folder structure
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS-depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = _folder_path(FOLDER_PROJECT, '_data')
URL_DATA = "https://github.com/breimanntools/aaanalysis/tree/master/aaanalysis/data/"

# File names
FILE_DF_SCALES = "df_scales"
FILE_DF_CAT = "df_cat"

# Constants
FONT_AA = "DejaVu Sans Mono"
STR_AA_GAP = "-"
LIST_CANONICAL_AA = ['N', 'A', 'I', 'V', 'K', 'Q', 'R', 'M', 'H', 'F', 'E', 'D', 'C', 'G', 'L', 'T', 'S', 'Y', 'W', 'P']
DTYPE = np.float64

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

STR_FILE_TYPE = "tsv"

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

# df_cluster
# COL_ENTRY = "entry"
COL_CLUST = "cluster"
COL_REP_IDEN = "identity_with_rep"
COL_IS_REP = "is_representative"

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
COL_FEAT_IMPORT_STD = "feat_importance_std"
COL_FEAT_IMPACT = "feat_impact"
COL_FEAT_IMPACT_STD = "feat_impact_std"

COLS_FEAT_SCALES = [COL_CAT, COL_SUBCAT, COL_SCALE_NAME]
COLS_FEAT_STAT = [COL_ABS_AUC, COL_ABS_MEAN_DIF, COL_MEAN_DIF, COL_STD_TEST, COL_STD_REF]
COLS_FEAT_WEIGHT = [COL_FEAT_IMPORT, COL_FEAT_IMPORT_STD, COL_FEAT_IMPACT]
DICT_VALUE_TYPE = {COL_ABS_AUC: "mean",
                   COL_ABS_MEAN_DIF: "mean",
                   COL_MEAN_DIF: "mean",
                   COL_STD_TEST: "mean",
                   COL_STD_REF: "mean",
                   COL_FEAT_IMPORT: "sum",
                   COL_FEAT_IMPORT_STD: "mean",
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

# CPP evaluation
COL_N_FEAT = "n_features"
COL_AVG_ABS_AUC = "avg_ABS_AUC"
COL_RANGE_ABS_AUC = "range_ABS_AUC"   # Quintiles (min, 25%, median, 75%, max)
COL_AVG_MEAN_DIF = "avg_MEAN_DIF"
COL_AVG_STD_TEST = "avg_STD_TEST"
# COL_N_CLUST = "n_clusters"
COL_AVG_N_FEAT_PER_CLUST = "avg_n_feat_per_clust"
COL_STD_N_FEAT_PER_CLUST = "std_n_feat_per_clust"
COL_EVAL_CPP = [COL_N_FEAT,
                COL_AVG_ABS_AUC, COL_AVG_MEAN_DIF, COL_AVG_STD_TEST,
                COL_N_CLUST, COL_AVG_N_FEAT_PER_CLUST, COL_STD_N_FEAT_PER_CLUST]

# dPULearn (evaluation)
COL_SELECTION_VIA = "selection_via"
COL_N_REL_NEG = "n_rel_neg"
COL_AVG_STD = "avg_STD"
COL_AVG_IQR = "avg_IQR"
COL_AVG_ABS_AUC_POS = "avg_abs_AUC_pos"
COL_AVG_KLD_POS = "avg_KLD_pos"
COL_AVG_ABS_AUC_UNL = "avg_abs_AUC_unl"
COL_AVG_KLD_UNL = "avg_KLD_unl"
COL_AVG_ABS_AUC_NEG = "avg_abs_AUC_neg"
COL_AVG_KLD_NEG = "avg_KLD_neg"
COLS_EVAL_DPULEARN_SIMILARITY = [COL_AVG_STD, COL_AVG_IQR]
COLS_EVAL_DPULEARN_DISSIMILARITY = [COL_AVG_ABS_AUC_POS, COL_AVG_KLD_POS,
                                    COL_AVG_ABS_AUC_UNL, COL_AVG_KLD_UNL,
                                    COL_AVG_ABS_AUC_NEG, COL_AVG_KLD_NEG]
COLS_EVAL_DPULEARN = [COL_N_REL_NEG] + COLS_EVAL_DPULEARN_SIMILARITY + COLS_EVAL_DPULEARN_DISSIMILARITY

# Labels
LABEL_FEAT_VAL = "Feature value"
LABEL_HIST_COUNT = "Number of proteins"
LABEL_HIST_DEN = "Relative density"

LABEL_FEAT_NUMBER = "Number of features\n(per residue position)"
LABEL_FEAT_IMPORT_CUM = "Cumulative feature importance\n(normalized) [%]"
LABEL_FEAT_IMPACT_CUM = "Cumulative feature impact\n(normalized) [%]"
LABEL_CBAR_FEAT_IMPACT_CUM = "Cumulative feature impact"
LABEL_FEAT_POS = ("                  Feature                  \n"
                  "Scale (subcategory)  +  Positions                 ")

_LABEL_FEAT_POS = ("      Feature     \n"
                  "       Scale  +  Positions")

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
COLOR_POS = "#389d2b"    # (56, 157, 43)
COLOR_UNL = "tab:gray"
COLOR_NEG = "#ad4570"   # (173,69,112)
COLOR_REL_NEG = "#ad9745" # (173, 151, 69)

DICT_COLOR = {"SHAP_POS": COLOR_SHAP_POS,
              "SHAP_NEG": COLOR_SHAP_NEG,
              "FEAT_POS": COLOR_FEAT_POS,
              "FEAT_NEG": COLOR_FEAT_NEG,
              "FEAT_IMP": COLOR_FEAT_IMP,
              "TMD": COLOR_TMD,
              "JMD": COLOR_JMD,
              "SAMPLES_POS": COLOR_POS,
              "SAMPLES_UNL": COLOR_UNL,
              "SAMPLES_NEG": COLOR_NEG,
              "SAMPLES_REL_NEG": COLOR_REL_NEG
              }

DICT_COLOR_CAT = {"ASA/Volume": "tab:blue",
                  "Composition": "tab:orange",
                  "Conformation": "tab:green",
                  "Energy": "tab:red",
                  "Others": "tab:gray",
                  "Polarity": "gold",
                  "Shape": "tab:cyan",
                  "Structure-Activity": "tab:brown"}

LIST_CAT = ['ASA/Volume', 'Composition', 'Conformation', 'Energy',
            'Others', 'Polarity', 'Shape', 'Structure-Activity']


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
def read_csv_cached(name, sep="\t", index_col=None):
    """Load cached dataframe to save loading time"""
    df = pd.read_csv(name, sep=sep, index_col=index_col)
    return df.copy()


def load_default_scales(scale_cat=False):
    """Load default scales sets or categories. Copy is always returned to maintain data integrity."""
    if scale_cat:
        if options[FILE_DF_CAT] is None:
            df_cat = read_csv_cached(FOLDER_DATA + f"{STR_SCALE_CAT}.{STR_FILE_TYPE}")
            options[FILE_DF_CAT] = df_cat
            return df_cat.copy()
        else:
            return options[FILE_DF_CAT].copy()
    else:
        if options[FILE_DF_SCALES] is None:
            df_scales = read_csv_cached(FOLDER_DATA + f"{STR_SCALES}.{STR_FILE_TYPE}", index_col=0)
            df_scales = df_scales.astype(float)
            options[FILE_DF_SCALES] = df_scales
            return df_scales.copy()
        else:
            return options[FILE_DF_SCALES].copy()


# Adjust df_eval
def add_names_to_df_eval(df_eval=None, names=None):
    """Add names column to df_eval"""
    if names is None:
        n_datasets = len(df_eval)
        names = [f"Set {i}" for i in range(1, n_datasets + 1)]
    df_eval.insert(0, COL_NAME, names)
    return df_eval


# Plotting utilities
def get_color_dif(mean_dif=0):
    return COLOR_FEAT_NEG if mean_dif < 0 else COLOR_FEAT_POS


# DEV: Exceptionally placed here due to dependency on constants
def plot_get_cdict_(name=STR_DICT_COLOR):
    """Return DICT_COLOR or DICT_COLOR_CAT"""
    if name == STR_DICT_COLOR:
        return DICT_COLOR
    elif name == STR_DICT_CAT:
        return DICT_COLOR_CAT
    else:
        raise ValueError(f"'name' must be '{STR_DICT_COLOR}' or '{STR_DICT_CAT}'")


def _get_diverging_cmap(cmap="ReBu_r", n_colors=101, facecolor_dark=False, only_pos=False, only_neg=False):
    """Generate a diverging colormap based on the provided cmap."""
    n = min(int(np.floor(1 + n_colors/20)), 5)
    color_0 = [(0, 0, 0)] if facecolor_dark else [(1, 1, 1)]
    if only_neg:
        cmap = sns.color_palette(palette=cmap, n_colors=(n_colors * 2) + n)
        cmap = cmap[0:n_colors-1] + color_0
    elif only_pos:
        cmap = sns.color_palette(palette=cmap, n_colors=(n_colors * 2) + n)
        cmap = color_0 + cmap[-n_colors+1:]
    else:
        n_cmap = n_colors + n * 2
        n_colors_half = int(np.floor(n_colors / 2))
        n_sub = (n_colors + 1) % 2
        n_cmap_half = int(np.floor(n_cmap / 2))
        cmap = sns.color_palette(cmap, n_colors=n_cmap)
        cmap_low, cmap_high = cmap[:n_cmap_half - n], cmap[n + n_cmap_half:]
        cmap = cmap_low[0:n_colors_half-n_sub] + color_0 + cmap_high[-n_colors_half:]
    return cmap


def _get_shap_cmap(n_colors=101, facecolor_dark=True, only_pos=False, only_neg=False):
    """Generate a diverging colormap for feature values."""
    n = min((int(np.floor(1 + n_colors/5)), 20))
    color_0 = [(0, 0, 0)] if facecolor_dark else [(1, 1, 1)]

    if only_neg:
        cmap = sns.light_palette(COLOR_SHAP_NEG, input="hex", reverse=True, n_colors=n_colors + n)
        cmap = cmap[0:n_colors-1] + color_0
    elif only_pos:
        cmap = sns.light_palette(COLOR_SHAP_POS, input="hex", n_colors=n_colors + n)
        cmap = color_0 + cmap[-n_colors+1:]
    else:
        n_colors_half = int(np.floor(n_colors / 2))
        n_sub = (n_colors + 1) % 2
        n_cmap_half = n_colors_half + n
        cmap_low = sns.light_palette(COLOR_SHAP_NEG, input="hex", reverse=True, n_colors=n_cmap_half)
        cmap_high = sns.light_palette(COLOR_SHAP_POS, input="hex", n_colors=n_cmap_half)
        cmap = cmap_low[0:n_colors_half-n_sub] + color_0 + cmap_high[-n_colors_half:]
    return cmap


def plot_get_cmap_(cmap="CPP", n_colors=101, facecolor_dark=None, only_pos=False, only_neg=False):
    """Get colormap for CPP or CPP-SHAP plots"""
    args = dict(n_colors=n_colors, facecolor_dark=facecolor_dark,
                only_neg=only_neg, only_pos=only_pos)
    if cmap == STR_CMAP_CPP:
        cmap = _get_diverging_cmap("RdBu_r", **args)
    elif cmap == STR_CMAP_SHAP:
        cmap = _get_shap_cmap(**args)
    else:
        cmap = _get_diverging_cmap(cmap=cmap, **args)
    return cmap


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
    check_warning_consecutive_index(name="df_seq", df=df_seq)
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
        for entry, tmd_start, tmd_stop, seq in zip(df_seq[COL_ENTRY], df_seq[COL_TMD_START],
                                                   df_seq[COL_TMD_STOP], df_seq[COL_SEQ]):
            # Check if valid integers
            args = dict(min_val=1, just_int=True)
            check_number_range(name=f"'tmd_start'={tmd_start} (entry: '{entry}')", val=tmd_start, **args)
            check_number_range(name=f"'tmd_stop'={tmd_stop} (entry: '{entry}')", val=tmd_stop, **args)
            # Check if tmd_start smaller than tmd_stop
            if tmd_start > tmd_stop:
                raise ValueError(f"'tmd_start'={tmd_start} should be <= 'tmd_stop'={tmd_stop} (entry: '{entry}')")
            # Check if tmd_start and tmd_stop smaller than sequence length
            len_seq = len(seq)  # TODO check first if seq is string and not float (if nan is given)
            if tmd_start > len_seq:
                raise ValueError(f"'tmd_start'={tmd_start} should be <= sequence length (n={len_seq}) (entry: '{entry}')")
            if tmd_stop > len_seq:
                raise ValueError(f"'tmd_stop'={tmd_stop} should be <= sequence length (n={len_seq}) (entry: '{entry}')")
            # Check if tmd length matches with sequence length
            len_tmd = tmd_stop - tmd_start
            if len_tmd > len_seq:
                raise ValueError(f"TMD (n={len_tmd}) is longer than sequence (n={len_seq}) (entry: '{entry}')")
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
        list_parts = check_list_like(name="list_parts", val=list_parts, accept_none=True,
                                     accept_str=True, convert=True)
    except ValueError:
        raise ValueError(f"'list_parts' must be list with selection of following parts: {LIST_ALL_PARTS}")
    if list_parts is None:
        if return_default:
            list_parts = LIST_ALL_PARTS if all_parts else LIST_PARTS
            # Remove ext parts if ext is 0
            if options["ext_len"] == 0:
                list_parts = [l for l in list_parts if "ext" not in l and l != "tmd_e"]
            return list_parts
        elif accept_none:
            return  # skip further checks
        else:
            raise ValueError(f"'list_parts' must be list with selection of following parts: {LIST_ALL_PARTS}")
    # Check if empty list
    if len(list_parts) == 0:
        raise ValueError(f"'list_parts' should not be empty list.")
    # Check for invalid parts
    wrong_parts = [x for x in list_parts if x not in LIST_ALL_PARTS]
    if len(wrong_parts) > 0:
        str_part = "part" if len(wrong_parts) == 1 else "parts"
        error = f"{wrong_parts} not valid {str_part}.\n  Select from following parts: {LIST_ALL_PARTS}"
        raise ValueError(error)
    return list_parts


def check_df_parts(df_parts=None, accept_none=False):
    """Check if df_parts is a valid input"""
    check_df(name="df_parts", df=df_parts, accept_none=accept_none)
    if df_parts is None and accept_none:
        return  # Skip check
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


# Check features
def _check_part(part=None, feature=None, list_parts=None):
    """Check if feature PART is valid"""
    list_parts = check_list_like(name="list_parts", val=list_parts, accept_str=True)
    list_parts = [x.lower() for x in list_parts]
    part = part.replace(" ", "")  # remove whitespace
    error = f"Wrong 'PART' for '{feature}'. Features should be 'PART-SPLIT-SCALE', with parts from: {list_parts}."
    if part.lower() in LIST_ALL_PARTS:
        error += f"\n Or include '{part.lower()}' in parts or set 'ext_len' option > 0."
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
    """Check if scale in feature is valid"""
    list_scales = check_list_like(name="list_scales", val=list_scales, accept_none=True, accept_str=True)
    if list_scales is not None:
        error = f"Wrong 'SCALE' for '{feature}'. Features should be 'PART-SPLIT-SCALE', with scales from: {LIST_ALL_PARTS}"
        if scale not in list_scales:
            raise ValueError(error)


def check_features(features=None, list_parts=None, list_scales=None):
    """Check if feature names are valid for list of parts  and df_scales"""
    features = check_list_like(name="features", val=features, accept_none=False, accept_str=True, convert=True)
    list_parts = check_list_parts(list_parts=list_parts, all_parts=True, return_default=True)
    # Check elements of features list
    features_wrong_n_components = [x for x in features if type(x) is not str or len(x.split("-")) != 3]
    if len(features_wrong_n_components) > 0:
        error = (f"Following elements from 'features' are not valid: {features_wrong_n_components}"
                 f"\n  Form of features should be 'PART-SPLIT-SCALE'")
        raise ValueError(error)
    # Check part, split, and scale
    if len(features) == 0:
        raise ValueError("'features' should not be empty.")
    for feature in features:
        part, split, scale = feature.split("-")
        _check_part(part=part, feature=feature, list_parts=list_parts)
        _check_split(split=split, feature=feature)
        # Scales is only checked if list_scales is provided
        _check_scale(scale=scale, feature=feature, list_scales=list_scales)
    return features


def check_df_feat(df_feat=None, df_cat=None, list_parts=None, shap_plot=None,
                  cols_requiered=None, cols_nan_check=None):
    """Check if df not empty pd.DataFrame"""
    # Check df
    cols_feat = [COL_FEATURE] + COLS_FEAT_SCALES + COLS_FEAT_STAT
    if cols_requiered is not None:
        cols_feat += [cols_requiered] if type(cols_requiered) is str else cols_requiered
    check_df(df=df_feat, name="df_feat", cols_requiered=cols_feat, cols_nan_check=cols_nan_check)
    if len(df_feat) == 0 or len(list(df_feat)) == 0:
        raise ValueError("'df_feat' should be not empty")
    duplicated_columns = df_feat.columns[df_feat.columns.duplicated()]
    if len(duplicated_columns) > 0:
        raise ValueError(f"'df_feat' should not contain duplicated columns: {duplicated_columns}")
    # Check features
    features = list(df_feat[COL_FEATURE]) #.values.tolist()
    check_features(features=features, list_parts=list_parts)
    # Check if df_feat matches df_cat
    if df_cat is not None:
        scales = set([x.split("-")[2] for x in features])
        list_scales = list(df_cat[COL_SCALE_ID])
        missing_scales = [x for x in scales if x not in list_scales]
        if len(missing_scales) > 0:
            raise ValueError(f"Following scales occur in 'df_feat' but not in 'df_cat': {missing_scales}")
    # Check if feat_importance or feat_impact column is in df_feat
    if shap_plot is not None:
        if shap_plot:
            col_feat_impact = [x for x in list(df_feat) if COL_FEAT_IMPACT in x]
            if len(col_feat_impact) == 0:
                raise ValueError(f"If 'shap_plot' is True, At least on '{COL_FEAT_IMPACT}' column must be "
                                 f"in 'df_feat' columns: {list(df_feat)}")
        else:
            if COL_FEAT_IMPORT not in list(df_feat):
                raise ValueError(f"If 'shap_plot' is False, '{COL_FEAT_IMPORT}' must be in 'df_feat' columns: {list(df_feat)}")
    df_feat = df_feat.reset_index(drop=True).copy()
    return df_feat
