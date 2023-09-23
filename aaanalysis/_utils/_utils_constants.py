"""
This is a script containing constant names, column names, or colors.

AAanalysis comprises these primary pd.DataFrames: df_seq, df_part, df_cat, df_scales, df_feat

"""

# Default scale datasets for protein analysis
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