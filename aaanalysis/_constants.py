"""This is a script for the shared constant definitions of AAanalysis: column names,
part / split / scale vocabularies, colours, and option strings.

Extracted from ``utils.py`` so the import barrel stays thin. This module imports only the
standard library and numpy -- never ``aaanalysis.utils`` -- so it can be imported first
with no circular dependency; ``utils.py`` re-exports everything here via
``from ._constants import *``, leaving every ``ut.X`` call site unchanged.
"""
import os
import platform

import numpy as np


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
LIST_CANONICAL_AA = list("ACDEFGHIKLMNPQRSTVWY")
DTYPE = np.float64

# Part names
LIST_ALL_PARTS = ["tmd", "tmd_e", "tmd_n", "tmd_c", "jmd_n", "jmd_c", "ext_c", "ext_n",
                  "tmd_jmd", "jmd_n_tmd_n", "tmd_c_jmd_c", "ext_n_tmd_n", "tmd_c_ext_c"]
LIST_PARTS = ["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"]

# Canonical, human-readable label per sequence part (the PART field of a
# PART-SPLIT-SCALE feature id). Single source of the part-label vocabulary used by
# SequenceFeature.get_feature_descriptions; keys cover every part in LIST_ALL_PARTS.
# ('region' is deliberately avoided here — reserved for the #27 region abstraction.)
DICT_PART_LABEL = {"tmd": "TMD",
                   "tmd_e": "extended TMD",
                   "tmd_n": "TMD-N",
                   "tmd_c": "TMD-C",
                   "jmd_n": "JMD-N",
                   "jmd_c": "JMD-C",
                   "ext_c": "C-terminal extension",
                   "ext_n": "N-terminal extension",
                   "tmd_jmd": "JMD-N+TMD+JMD-C",
                   "jmd_n_tmd_n": "JMD-N+TMD-N",
                   "tmd_c_jmd_c": "TMD-C+JMD-C",
                   "ext_n_tmd_n": "N-terminal extension+TMD-N",
                   "tmd_c_ext_c": "TMD-C+C-terminal extension"}

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
STR_TOP_EXPLAIN = "top_explain"  # interpretability-tiered selection table (internal, not a load name)
STR_SUBCAT = "subcat"  # subcategory overview (interpretability, tier, counts, descriptions)
NAMES_SCALE_SETS = [STR_SCALES, STR_SCALES_RAW, STR_SCALE_CAT,
                    STR_SCALES_PC, STR_TOP60, STR_TOP60_EVAL, STR_SUBCAT]
# Valid grids for the interpretability-tier selector of load_scales
LIST_TOP_EXPLAIN_N = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
LIST_TOP_EXPLAIN_MIN_TH = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
COL_POS = "pos"   # AAWindowSampler — per-row 1-based positive position(s); list[int] or scalar
COL_SS = "ss"     # get_dssp — per-residue secondary-structure codes; list[str] (length matches sequence) or None
COL_DSSP_OK = "dssp_ok"   # get_dssp — DSSP success flag (bool)
COL_EMBEDDINGS_OK = "embeddings_ok"   # fetch_embeddings — per-entry success flag (bool)

# EmbeddingPreprocessor.fetch_embeddings — option bundles
LIST_EMBED_MODES = ["protein", "residue"]
LIST_EMBED_SOURCES = ["auto", "compute"]      # 'uniprot' reserved
LIST_POOLING = ["mean", "max", "cls"]
LIST_EMBED_DEVICES = ["auto", "cpu", "cuda", "mps"]

# AAWindowSampler — output column names
COL_ENTRY_WIN = "entry_win"
COL_WINDOW = "window"
COL_SOURCE_POS = "source_position"
COL_LABELS = "labels"
COL_ROLE = "role"
COL_STRATEGY = "strategy"
COL_ARM = "arm"
COLS_SEGMENTS = [COL_ENTRY_WIN, COL_ENTRY, COL_SEQ, COL_WINDOW, COL_SOURCE_POS,
                 COL_LABEL, COL_ROLE, COL_STRATEGY]
COLS_SEQUENCES = [COL_ENTRY, COL_SEQ, COL_LABELS]

# AAWindowSampler — output modes
OUT_SEGMENTS = "segments"
OUT_SEQUENCES = "sequences"
LIST_OUTPUT_MODES = [OUT_SEGMENTS, OUT_SEQUENCES]

# AAWindowSampler — role tags (canonical; users may pass any custom string)
ROLE_TEST = "Test"
ROLE_NEG = "Negative"
ROLE_UNL = "Unlabeled"
ROLE_CTRL = "Control"
LIST_ROLES = [ROLE_TEST, ROLE_NEG, ROLE_UNL, ROLE_CTRL]

# AAWindowSampler — strategy tags (stored in output for provenance)
STRATEGY_SAME = "same_protein"
STRATEGY_DIFF = "different_protein"
STRATEGY_SYNTH_PREFIX = "synthetic"
STRATEGY_MOTIF_MATCHED = "motif_matched"
STRATEGY_TEST = "test"   # provenance of known positive (test) windows; not a sampled strategy
# Strategy tags double as the ``method`` keys accepted by
# AAWindowSampler.sample_benchmark_set arms.
LIST_STRATEGIES = [STRATEGY_SAME, STRATEGY_DIFF, STRATEGY_SYNTH_PREFIX,
                   STRATEGY_MOTIF_MATCHED]

# AAWindowSampler — synthetic sampling modes (built-in; AAontology presets live in
# the backend ``sample_synthetic.py`` next to the ``PRESETS`` registry; the
# polymorphic-mode segments below are tag fragments for list/dict shapes)
MODE_UNIFORM = "uniform"
MODE_GLOBAL_FREQ = "global_freq"
MODE_POSITION_SPECIFIC = "position_specific"
MODE_SCRAMBLED = "scrambled"
LIST_SYNTH_MODES_BUILTIN = [MODE_UNIFORM, MODE_GLOBAL_FREQ,
                            MODE_POSITION_SPECIFIC, MODE_SCRAMBLED]
STR_SYNTH_MIX = "mix"
STR_SYNTH_CUSTOM = "custom"

# AAWindowSampler — motif-match filter modes (used with ``motif_pwm`` /
# ``motif_score_threshold`` on sample_same_protein / sample_different_protein)
STR_MOTIF_IN = "in"
STR_MOTIF_OUT = "out"
LIST_MOTIF_MATCHES = [STR_MOTIF_IN, STR_MOTIF_OUT]

# get_dssp — secondary-structure encoding modes
SS_MODE_3 = "ss3"
SS_MODE_8 = "ss8"
LIST_SS_MODES = [SS_MODE_3, SS_MODE_8]

# get_dssp — gap-handling policy when aligning DSSP output to df_seq[sequence]
GAP_PAD = "pad"
GAP_OMIT = "omit"
LIST_GAP_HANDLING = [GAP_PAD, GAP_OMIT]

# get_dssp — DSSP 8-state -> 3-state reduction (H/G/I -> H ; E/B -> E ; rest -> C).
# DSSP's blank (no SS) maps to "C" in 3-state; "-" stays "-" (alignment gap).
STR_SS_GAP = "-"
DICT_DSSP_3STATE = {"H": "H", "G": "H", "I": "H",
                    "E": "E", "B": "E",
                    "T": "C", "S": "C", " ": "C",
                    STR_SS_GAP: STR_SS_GAP}

# df_part

# df_scales
# Column for df_cat (as defined in AAontology, retrieved by aa.load_scales(name="scale_cat"))
COL_SCALE_ID = "scale_id"
COL_CAT = "category"
COL_SUBCAT = "subcategory"
COL_SCALE_NAME = "scale_name"
COL_SCALE_DES = "scale_description"
COL_INTERPRET_GRADE = "interpret_grade"  # 1-10 interpretability grade (1 = best); lives on df_subcat
COL_TOP_EXPLAIN = "top_explain"            # interpretability tier threshold (NaN for unclassified subcats)
# Columns for df_subcat (subcategory overview, retrieved by aa.load_scales(name="subcat"))
COL_CLUSTER = "cluster"
COL_N_SCALES = "n_scales"
COL_N_SCALES_AAINDEX = "n_scales_aaindex"
COL_SUBCAT_DES = "subcategory_description"
COL_KEY_REFERENCES = "key_references"
COLS_SUBCAT = [COL_CAT, COL_SUBCAT, COL_CLUSTER, COL_INTERPRET_GRADE, COL_TOP_EXPLAIN,
               COL_N_SCALES, COL_N_SCALES_AAINDEX, COL_SUBCAT_DES, COL_KEY_REFERENCES]

# df_annot (canonical per-residue annotation schema; AnnotationPreprocessor)
COL_PROTEIN_ID = "protein_id"   # UniProt accession (mirrors COL_ENTRY)
COL_START = "start"             # 1-based, UniProt-canonical frame
COL_STOP = "end"               # 1-based inclusive; single residue -> start==end
COL_AA = "aa"                   # expected residue identity (encode-time guard)
COL_FEATURE_TYPE = "feature_type"   # registry key (e.g. 'phospho', 'binding')
COL_SOURCE = "source"           # 'UniProt' | user source name
COL_EVIDENCE = "evidence"       # ECO code (e.g. 'ECO:0000269')
COL_SCORE = "score"             # nullable float in [0, 1]
COL_BOND_ID = "bond_id"         # pairing id for DISULFID / CROSSLNK endpoints
COLS_ANNOT = [COL_PROTEIN_ID, COL_START, COL_STOP, COL_AA, COL_FEATURE_TYPE,
              COL_CAT, COL_SOURCE, COL_EVIDENCE, COL_SCORE, COL_BOND_ID]

# Evidence allow-sets (ECO codes); AnnotationPreprocessor evidence= toggle
STR_ECO_EXPERIMENTAL = "ECO:0000269"   # experimental, manual assertion
STR_ECO_COMBINATORIAL = "ECO:0007744"  # combinatorial, manual assertion
LIST_ECO_EXPERIMENTAL = [STR_ECO_EXPERIMENTAL]
LIST_ECO_MANUAL = [STR_ECO_EXPERIMENTAL, STR_ECO_COMBINATORIAL]

# df_cluster
# COL_ENTRY = "entry"
COL_CLUST = "cluster"
COL_REP_IDEN = "identity_with_rep"
COL_IS_REP = "is_representative"
COL_DIST_TO_REP = "dist_to_rep"

# Columns for df_feat
COL_FEATURE = "feature"
# COL_CAT, COL_SUBCAT, COL_SCALE_NAME, COL_SCALE_DES
COL_ABS_AUC = "abs_auc"
COL_ABS_MEAN_DIF = "abs_mean_dif"
COL_MEAN_DIF = "mean_dif"
COL_STD_TEST = "std_test"
COL_STD_REF = "std_ref"
COL_PVAL_MW = "p_val_mann_whitney"
COL_PVAL_TTEST = "p_val_ttest_indep"
COL_PVAL_FDR = "p_val_fdr_bh"
COL_POSITION = "positions"
COL_AA_TEST = "amino_acids_test"
COL_AA_REF = "amino_acids_ref"
COL_FEAT_DES = "feature_description"  # optional, additive: one readable sentence per feature id
COL_SELECTION_FREQUENCY = "selection_frequency"  # optional: bootstrap stability score [0-1], present only when bootstrap=True

# Columns for df_feat after processing with explainable AI methods
COL_FEAT_IMPORT = "feat_importance"
COL_FEAT_IMPORT_STD = "feat_importance_std"
COL_FEAT_IMPACT = "feat_impact"
COL_FEAT_IMPACT_STD = "feat_impact_std"

COLS_FEAT_SCALES = [COL_CAT, COL_SUBCAT, COL_SCALE_NAME]
COLS_FEAT_SCALES_FULL = COLS_FEAT_SCALES + [COL_SCALE_DES]  # incl. scale_description
COLS_FEAT_STAT = [COL_ABS_AUC, COL_ABS_MEAN_DIF, COL_MEAN_DIF, COL_STD_TEST, COL_STD_REF]
COLS_FEAT_WEIGHT = [COL_FEAT_IMPORT, COL_FEAT_IMPORT_STD, COL_FEAT_IMPACT]

# Protein engineering (AAMut / SeqMut) — output column names
COL_FROM_AA = "from_aa"             # substituted-from amino acid (single letter)
COL_TO_AA = "to_aa"                 # substituted-to amino acid (single letter)
COL_MUTATION = "mutation"           # HGVS-like label "<from_aa><pos><to_aa>", e.g. "M123V"
COL_DELTA = "delta"                 # AAMut — signed per-scale substitution delta (to_aa - from_aa)
COL_ABS_DELTA = "abs_delta"         # AAMut — magnitude of the per-scale delta
COL_DELTA_CPP = "delta_cpp"         # SeqMut — Sum|dX| feature-space magnitude of a mutation
COL_SHIFT_SCORE = "shift_score"     # SeqMut — signed shift toward the test-class profile
COL_SEQ_MUT = "sequence_mut"        # SeqMut.mutate — full mutated sequence
COL_REGION = "region"               # SeqMut — part a scanned position falls in (jmd_n/tmd/jmd_c)
COL_IS_DISRUPTIVE = "is_disruptive"  # SeqMut.eval — disruptive flag (|delta_cpp| >= threshold)
COL_N_MUT = "n_mut"                 # SeqMut.eval — number of scanned mutations
COL_N_DISRUPTIVE = "n_disruptive"   # SeqMut.eval — number flagged disruptive
COL_FRAC_DISRUPTIVE = "frac_disruptive"  # SeqMut.eval — n_disruptive / n_mut
COL_MEAN_DELTA_CPP = "mean_delta_cpp"    # SeqMut.eval — mean |delta_cpp| over scanned mutations
# SeqMut (model-based / ML-guided) — predicted-impact columns, present only when a fitted
# model is bound to SeqMut. delta_pred is "the change of prediction score per mutation".
COL_DELTA_PRED = "delta_pred"       # SeqMut (model) — ΔP% = (P(mut) − P(wt))*100, target-class prediction shift
COL_WT_PRED = "wt_pred"             # SeqMut (model) — wild-type prediction score (%) for the target class (per-entry)
COL_WT_PRED_STD = "wt_pred_std"     # SeqMut (model) — std of the wild-type prediction score (%), NaN if unavailable
COL_VARIANT = "variant"            # SeqMut.combine — combined-variant label, e.g. "R20K+K27P"
COLS_AAMUT = [COL_FROM_AA, COL_TO_AA, COL_SCALE_ID, COL_CAT, COL_SUBCAT, COL_DELTA, COL_ABS_DELTA]
COLS_SEQMUT_SCAN = [COL_ENTRY, COL_POS, COL_FROM_AA, COL_TO_AA, COL_MUTATION,
                    COL_REGION, COL_DELTA_CPP, COL_SHIFT_SCORE]
COLS_SEQMUT_EVAL = [COL_ENTRY, COL_REGION, COL_N_MUT, COL_N_DISRUPTIVE,
                    COL_FRAC_DISRUPTIVE, COL_MEAN_DELTA_CPP]
# SeqMut.combine — base columns (COL_DELTA_PRED appended when a model is bound)
COLS_SEQMUT_VARIANT = [COL_ENTRY, COL_VARIANT, COL_N_MUT, COL_SEQ_MUT,
                       COL_DELTA_CPP, COL_SHIFT_SCORE]
# SeqMut.suggest — optional weighting of the shift score by a df_feat column
LIST_SHIFT_WEIGHTS = [COL_FEAT_IMPORT, COL_ABS_AUC]

# Protein engineering (SeqOpt) — multi-objective directed-evolution optimizer.
# NSGA-II output columns (COL_RANK is shared, defined in the eval block below; COL_VARIANT,
# COL_N_MUT, COL_SEQ_MUT, COL_ENTRY are reused from the SeqMut block above).
COL_GENERATION = "generation"       # SeqOpt — 0-based evolve-score-select round index
COL_CROWDING = "crowding"           # SeqOpt — NSGA-II crowding distance within a front
COL_HYPERVOLUME = "hypervolume"     # SeqOpt.eval — objective-space volume dominated by the front
COL_N_FRONT = "n_front"             # SeqOpt.eval — number of variants on the first (rank=0) front
COL_SPREAD = "spread"               # SeqOpt.eval — objective-space diversity of the front
COL_CONVERGENCE = "convergence"     # SeqOpt.eval — generational distance to a reference front
# Fixed lower-bound columns of df_pareto (one column per objective is inserted between
# COL_SEQ_MUT and COL_RANK at run time; COL_RANK defined in the eval block below).
COLS_PARETO_BASE = [COL_ENTRY, COL_VARIANT, COL_N_MUT, COL_SEQ_MUT]
COLS_SEQOPT_EVAL = [COL_HYPERVOLUME, COL_N_FRONT, COL_SPREAD]
# SeqOpt option vocabularies (Validate-block check_str_options targets).
LIST_SEQOPT_MODES = ["impact", "importance"]            # SHAP-guided (adaptive) | feat_importance (greedy)
LIST_SEQOPT_ALGORITHMS = ["nsga2", "greedy"]            # population NSGA-II | importance-ordered greedy walk
LIST_SEQOPT_CROSSOVER = ["uniform", "one_point", "two_point"]
LIST_SEQOPT_MUTATION = ["substitution", "shift"]
LIST_SEQOPT_VARIATION = ["and", "or"]                   # varAnd (crossover AND mutation) | varOr (one of)
LIST_SEQOPT_SURVIVAL = ["mu_plus_lambda", "mu_comma_lambda", "ea_simple"]
LIST_SEQOPT_PENALTY = ["delta", "closest_valid"]        # DeltaPenalty | ClosestValidPenalty semantics
LIST_SEQOPT_INIT = ["random", "suggest"]                # random seeding | warm-start from SeqMut.suggest
LIST_OBJECTIVE_GOALS = ["max", "min"]
# Built-in objective sources (a callable(df_variant)->array is also accepted at run time).
LIST_OBJECTIVE_SOURCES = [COL_DELTA_PRED, COL_DELTA_CPP, COL_SHIFT_SCORE, COL_N_MUT]

# Canonical, deterministic df_feat column order (issue #18). This is a LOWER BOUND
# on the known/fixed columns, not an exhaustive schema: the dynamic p-value column
# (COL_PVAL_MW vs COL_PVAL_TTEST per 'parametric'), the post-hoc explainable-AI
# columns (feat_importance*, feat_impact*) and the per-substrate SHAP columns
# (feat_impact_'name', mean_dif_'name', ...) are NOT listed here — sort_cols_feat
# appends any unlisted column after 'positions' in stable order (never dropped).
LIST_COLS_FEAT = ([COL_FEATURE] + COLS_FEAT_SCALES_FULL + COLS_FEAT_STAT
                  + [COL_PVAL_MW, COL_PVAL_FDR] + [COL_POSITION])

# TreeModel — post-fit feature-selection strategies. RFE is the
# fit-time engine (fit(use_rfe=True)), not a selection strategy.
STRATEGY_TOP_K = "top_k"
STRATEGY_THRESHOLD = "threshold"
STRATEGY_FREQUENCY = "frequency"
LIST_SELECTION_STRATEGIES = [STRATEGY_TOP_K, STRATEGY_THRESHOLD, STRATEGY_FREQUENCY]

# CPP bootstrap / stability selection (CPP.__init__ bootstrap=True) — which class group is
# resampled per round. 'reference' fixes the (usually smaller/cleaner) test group and resamples
# only the reference group, isolating the dominant instability source; 'both' resamples both
# groups; 'test' fixes the reference group and resamples only the test group.
RESAMPLE_BOTH = "both"
RESAMPLE_REFERENCE = "reference"
RESAMPLE_TEST = "test"
LIST_RESAMPLE = [RESAMPLE_BOTH, RESAMPLE_REFERENCE, RESAMPLE_TEST]
# Tuned defaults for CPP(bootstrap=True)'s ``bootstrap_kws`` config dict.
DICT_BOOTSTRAP_DEFAULTS = {"rounds": 20, "resample": RESAMPLE_REFERENCE, "frac": 0.8}

# CPP.simplify — interpretability-guided scale swapping
STRATEGY_GREEDY = "greedy"            # per-feature swap, RF+CV non-regression gate
STRATEGY_CONSOLIDATE = "consolidate"  # set-level consolidation (staged)
STRATEGY_SWAP_ALL = "swap_all"        # swap-all then one CV comparison (staged)
LIST_SIMPLIFY_STRATEGIES = [STRATEGY_GREEDY, STRATEGY_CONSOLIDATE, STRATEGY_SWAP_ALL]
ON_UNIMPROVABLE_KEEP = "keep"
ON_UNIMPROVABLE_DROP = "drop"
ON_UNIMPROVABLE_DROP_IF_PERF = "drop_if_perf_allows"
LIST_ON_UNIMPROVABLE = [ON_UNIMPROVABLE_KEEP, ON_UNIMPROVABLE_DROP, ON_UNIMPROVABLE_DROP_IF_PERF]
TIE_BREAK_INTERPRETABILITY = "interpretability"
TIE_BREAK_PERFORMANCE = "performance"
LIST_REDUNDANCY_TIE_BREAK = [TIE_BREAK_INTERPRETABILITY, TIE_BREAK_PERFORMANCE]
# CPP.simplify — per-feature candidate-search breadth. 'exact' tests every
# eligible candidate (default; byte-identical to the historical path); 'fast' is
# an opt-in heuristic that caps the candidates evaluated per feature for speed.
CANDIDATE_SEARCH_EXACT = "exact"
CANDIDATE_SEARCH_FAST = "fast"
LIST_CANDIDATE_SEARCH = [CANDIDATE_SEARCH_EXACT, CANDIDATE_SEARCH_FAST]
# CPP.simplify — cross-validation-gate model presets (SVM default; also accepts a
# custom sklearn estimator instance).
MODEL_SVM = "svm"
MODEL_RF = "rf"
MODEL_LOG_REG = "log_reg"
MODEL_EXTRA_TREES = "extra_trees"
LIST_CV_MODELS = [MODEL_SVM, MODEL_RF, MODEL_LOG_REG]
# Prediction-model registry names (AAPred). Resolved to sklearn estimators by
# ut.get_cv_model_. Kept deliberately small — the four standard families the package
# already uses (matching pipe.predict_samples' default set). Any other estimator
# (MLP, gradient boosting, xgboost, a voting/stacking ensemble, ...) is used by
# passing a configured sklearn estimator instance instead of a name.
LIST_PRED_MODELS = [MODEL_SVM, MODEL_RF, MODEL_EXTRA_TREES, MODEL_LOG_REG]
DICT_VALUE_TYPE = {COL_ABS_AUC: "mean",
                   COL_ABS_MEAN_DIF: "mean",
                   COL_MEAN_DIF: "mean",
                   COL_STD_TEST: "mean",
                   COL_STD_REF: "mean",
                   COL_FEAT_IMPORT: "sum",
                   COL_FEAT_IMPORT_STD: "mean",
                   COL_FEAT_IMPACT: "sum"}

# df_feat interface contract — the machine-readable "data dictionary" pinning the CPP
# output schema for downstream consumers (e.g. ProtXplain). Each entry maps a column
# name to a (dtype, required, nullable, semantics) tuple:
#   dtype     — coarse kind: 'str' | 'float' | 'int'
#   required  — True only for the canonical lower-bound columns (LIST_COLS_FEAT) that
#               every CPP.run() output carries; False = optional/dynamic (test-dependent
#               p-value variant, diagnostic, or appended post-fit by TreeModel/ShapModel)
#   nullable  — whether NaN is a valid value for the column
#   semantics — one-line value meaning
# Internal contract only (no public accessor; strict-semver caution): rendered to the
# df_feat contract doc and guarded by tests/unit/api_tests/test_df_feat_contract.py.
DICT_DF_FEAT = {
    COL_FEATURE:         ("str",   True,  False, "Opaque PART-SPLIT-SCALE feature id (e.g. 'TMD_C_JMD_C-Segment(3,4)-KLEP840101'); split with split_feat_id, never parse by hand."),
    COL_CAT:             ("str",   True,  False, "AAontology scale category of the feature's scale."),
    COL_SUBCAT:          ("str",   True,  False, "AAontology scale subcategory."),
    COL_SCALE_NAME:      ("str",   True,  False, "Human-readable scale name."),
    COL_SCALE_DES:       ("str",   True,  False, "One-sentence scale description."),
    COL_ABS_AUC:         ("float", True,  False, "Absolute adjusted AUC, range [-0.5, 0.5]; primary feature ranking statistic."),
    COL_ABS_MEAN_DIF:    ("float", True,  False, "Absolute mean difference between test and reference group, range [0, 1]."),
    COL_MEAN_DIF:        ("float", True,  False, "Signed mean difference (test - reference), range [-1, 1]; the sign gives the direction."),
    COL_STD_TEST:        ("float", True,  False, "Standard deviation of the feature in the test group."),
    COL_STD_REF:         ("float", True,  False, "Standard deviation of the feature in the reference group."),
    COL_PVAL_MW:         ("float", True,  False, "Mann-Whitney U p-value (default, non-parametric). Named 'p_val_ttest_indep' instead when parametric=True."),
    COL_PVAL_FDR:        ("float", True,  False, "Benjamini-Hochberg FDR-corrected p-value."),
    COL_POSITION:        ("str",   True,  False, "Comma-separated 1-based residue positions the feature spans."),
    # Optional / dynamic — present depending on settings or appended downstream.
    COL_PVAL_TTEST:      ("float", False, False, "Independent t-test p-value; replaces 'p_val_mann_whitney' when parametric=True."),
    COL_AA_TEST:         ("str",   False, False, "Amino acids at the feature positions in the test group (diagnostic)."),
    COL_AA_REF:          ("str",   False, False, "Amino acids at the feature positions in the reference group (diagnostic)."),
    COL_FEAT_DES:        ("str",   False, True,  "Optional readable one-sentence feature description."),
    COL_SELECTION_FREQUENCY: ("float", False, False, "Bootstrap selection frequency in [0, 1] (fraction of resampling rounds a feature was selected); present only when CPP(bootstrap=True)."),
    COL_FEAT_IMPORT:     ("float", False, False, "Feature importance from TreeModel.fit (post-fit)."),
    COL_FEAT_IMPORT_STD: ("float", False, False, "Standard deviation of the feature importance across CV rounds (post-fit)."),
    COL_FEAT_IMPACT:     ("float", False, False, "SHAP-based signed feature impact from ShapModel (post-fit, pro)."),
    COL_FEAT_IMPACT_STD: ("float", False, False, "Standard deviation of the feature impact (post-fit, pro)."),
}


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

# AAPred (model evaluation / deployment)
COL_MODEL = "model"             # model class short name (e.g. 'RandomForestClassifier')
COL_METRIC = "metric"           # performance metric name (e.g. 'balanced_accuracy')
COL_PRINCIPLE = "principle"     # evaluation principle: 'cv' (cross-validation) | 'holdout'
COL_SCORE_STD = "score_std"     # std of the score (across CV folds; NaN for a single holdout estimate)
COL_GROUP = "group"             # per-sample/per-protein group label used for coloring
COL_OFFSET = "offset"           # AAPred.predict(level='domain') — boundary shift applied to tmd_start/tmd_stop
COL_RESIDUE_POS = "position"    # AAPred.predict(level='window') — 1-based anchor position scored
COL_PRED_LABEL = "predicted_label"  # AAPred.predict — class label when a threshold is given
STR_PRINCIPLE_CV = "cv"
STR_PRINCIPLE_HOLDOUT = "holdout"
LIST_PRINCIPLES = [STR_PRINCIPLE_CV, STR_PRINCIPLE_HOLDOUT]
LIST_METRICS_PRED = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
COLS_EVAL_PRED = [COL_MODEL, COL_METRIC, COL_PRINCIPLE, COL_SCORE, COL_SCORE_STD]

# Prediction reliability (ReliabilityModel) — per-sample trust columns
COL_CI_LOW = "ci_low"                # lower confidence-interval bound of the score
COL_CI_HIGH = "ci_high"              # upper confidence-interval bound of the score
COL_OOD_SCORE = "ood_score"          # applicability-domain distance, relative to the training threshold (1.0 = threshold)
COL_IN_DOMAIN = "in_domain"          # bool: inside the training applicability domain (ood_score <= 1)
COL_AD_KNN = "ad_knn_dist"           # mean distance to k nearest training samples (standardized space)
COL_AD_MAHALANOBIS = "ad_mahalanobis"    # Mahalanobis distance to the training center
COL_AD_LEVERAGE = "ad_leverage"      # leverage (hat value) relative to the training feature space
COL_SCORE_CAL = "score_calibrated"   # calibrated positive-class probability (NaN if not calibrated)
COL_MARGIN = "margin"                # |p - 0.5| * 2 on the calibrated score — aleatoric sharpness (1 = decisive, 0 = coin-flip)
COL_ENTROPY = "entropy"              # binary entropy of the calibrated score (0 = decisive, 1 = coin-flip)
COL_CONFORMAL_SET = "conformal_set"  # split-conformal prediction set: 'neg' | 'pos' | 'both' (ambiguous) | 'none' (abstain)
COL_RELIABLE = "reliable"            # bool headline: in_domain AND a confident conformal singleton
STR_CONF_NEG, STR_CONF_POS = "neg", "pos"
STR_CONF_BOTH, STR_CONF_NONE = "both", "none"
COLS_RELIABILITY = [COL_SCORE, COL_SCORE_STD, COL_CI_LOW, COL_CI_HIGH, COL_OOD_SCORE, COL_IN_DOMAIN,
                    COL_AD_KNN, COL_AD_MAHALANOBIS, COL_AD_LEVERAGE, COL_SCORE_CAL, COL_MARGIN,
                    COL_ENTROPY, COL_CONFORMAL_SET, COL_RELIABLE]

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
# Linked-selection highlight (CPPStructurePlot.interactive): the same colour marks the picked
# residue in the 3D structure and its feature-map column, so the link reads at a glance.
COLOR_LINK_HIGHLIGHT = '#00E5FF'  # (0, 229, 255) bright cyan, distinct from the SHAP ramp
COLOR_FEAT_POS = '#9D2B39'  # (157, 43, 57) Mean difference
COLOR_FEAT_NEG = '#326599'  # (50, 101, 133) Mean difference
COLOR_FEAT_IMP = '#7F7F7F'  # (127, 127, 127) feature importance
COLOR_TMD = '#00FA9A'       # (0, 250, 154)
COLOR_JMD = '#0000FF'       # (0, 0, 255)
COLOR_POS = "#389d2b"    # (56, 157, 43)
COLOR_UNL = "tab:gray"
COLOR_NEG = "#ad4570"   # (173,69,112)
COLOR_REL_NEG = "#ad9745" # (173, 151, 69)

# Public, named aliases for the canonical sample-group colors (positive / negative /
# unlabeled / reliable-negative). They mirror the ``DICT_COLOR["SAMPLES_*"]`` entries
# exactly, so users can reference a named constant (``aa.COLOR_SAMPLES_POS``) instead
# of indexing ``plot_get_cdict("DICT_COLOR")`` by string key.
COLOR_SAMPLES_POS = COLOR_POS
COLOR_SAMPLES_NEG = COLOR_NEG
COLOR_SAMPLES_UNL = COLOR_UNL
COLOR_SAMPLES_REL_NEG = COLOR_REL_NEG

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

# pLDDT confidence palette (AlphaFold-DB), ordered low -> high confidence. Read
# high -> low it is the familiar blue -> cyan -> yellow -> orange ramp used to
# paint per-residue AlphaFold model confidence onto a structure.
COLOR_PLDDT_VERY_LOW = '#FF7D45'    # orange, pLDDT < 50
COLOR_PLDDT_LOW = '#FFDB13'         # yellow, 50 <= pLDDT < 70
COLOR_PLDDT_CONFIDENT = '#65CBF3'   # cyan,   70 <= pLDDT < 90
COLOR_PLDDT_VERY_HIGH = '#0053D6'   # blue,   pLDDT >= 90
COLOR_STRUCT_MISSING = '#BFBFBF'    # gray for residues without a mapped value
COLOR_STRUCT_HIGHLIGHT = '#9D2A71'  # magenta for the hovered/selected residue (linked view)

# Continuous low -> high ramp consumed by the pLDDT structure colouring.
LIST_COLOR_PLDDT = [COLOR_PLDDT_VERY_LOW, COLOR_PLDDT_LOW,
                    COLOR_PLDDT_CONFIDENT, COLOR_PLDDT_VERY_HIGH]

DICT_COLOR_PLDDT = {"very_low": COLOR_PLDDT_VERY_LOW,
                    "low": COLOR_PLDDT_LOW,
                    "confident": COLOR_PLDDT_CONFIDENT,
                    "very_high": COLOR_PLDDT_VERY_HIGH}

DICT_COLOR_CAT = {"ASA/Volume": "tab:blue",
                  "Composition": "tab:orange",
                  "Conformation": "tab:green",
                  "Energy": "tab:red",
                  "Others": "tab:gray",
                  "Polarity": "gold",
                  "Shape": "tab:cyan",
                  "Structure-Activity": "tab:brown",
                  # Category buckets for source-of-feature classes (one per
                  # *Preprocessor) — see `StructurePreprocessor`,
                  # `EmbeddingPreprocessor`, `AnnotationPreprocessor`.
                  "Structure":        "#2E6E5E",   # deep teal-green
                  "Embeddings":       "#6B4FB5",   # indigo-violet
                  "PTMs":             "#B36BCB",   # lilac-magenta
                  "Functional sites": "#2C6E9E"}   # deep ocean-blue

LIST_CAT = ['ASA/Volume', 'Composition', 'Conformation', 'Energy',
            'Others', 'Polarity', 'Shape', 'Structure-Activity',
            'Structure', 'Embeddings', 'PTMs', 'Functional sites']


# Parameter options for cmaps and color dicts
STR_CMAP_CPP = "CPP"
STR_CMAP_SHAP = "SHAP"
STR_DICT_COLOR = "DICT_COLOR"
STR_DICT_CAT = "DICT_CAT"
