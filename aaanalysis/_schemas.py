"""This is a script for the data dictionary / interface contract of the key AAanalysis
DataFrames.

``DICT_DF_SCHEMAS`` is the machine-readable contract that documents, for every important
DataFrame, what columns exist, their dtype / nullability / uniqueness, the allowed values
or numeric range, a one-line meaning, and an example value. It is the single source of
truth rendered to the documentation (via ``render_schemas_rst``) and guarded by the
contract tests in ``tests/unit/api_tests/test_df_schemas.py``.

The module imports only ``._constants`` (column-name + vocabulary constants) -- never
``aaanalysis.utils`` -- so it stays circular-free; ``utils.py`` re-exports
``DICT_DF_SCHEMAS`` so it is reachable as ``ut.DICT_DF_SCHEMAS``.

Each frame entry has:
    description : str
    columns     : dict[col_name -> field record]   (column-table frames)
and, where relevant, one of:
    formats         : dict[name -> required column set]  (df_seq's conditional formats)
    dynamic_columns : dict                                (df_parts: columns are part names)
    matrix          : dict(index, columns, values, dtype) (df_scales / df_logo / X)

Each field record (built by ``_field``) has:
    dtype, required, nullable, unique, description
and optionally: range, allowed_values, example, validation.
"""
from ._constants import (
    COL_ENTRY, COL_SEQ, COL_LABEL, COL_NAME, COL_TMD_START, COL_TMD_STOP,
    COL_JMD_N, COL_TMD, COL_JMD_C,
    COL_SCALE_ID, COL_CAT, COL_SUBCAT, COL_SCALE_NAME, COL_SCALE_DES,
    COL_CLUSTER, COL_INTERPRET_GRADE, COL_TOP_EXPLAIN, COL_N_SCALES,
    COL_N_SCALES_AAINDEX, COL_SUBCAT_DES, COL_KEY_REFERENCES,
    COL_FEATURE, COL_ABS_AUC, COL_ABS_MEAN_DIF, COL_MEAN_DIF, COL_STD_TEST,
    COL_STD_REF, COL_PVAL_MW, COL_PVAL_TTEST, COL_PVAL_FDR, COL_POSITION,
    COL_AA_TEST, COL_AA_REF, COL_FEAT_DES,
    COL_FEAT_IMPORT, COL_FEAT_IMPORT_STD, COL_FEAT_IMPACT, COL_FEAT_IMPACT_STD,
    COL_FROM_AA, COL_TO_AA, COL_MUTATION, COL_DELTA, COL_ABS_DELTA,
    COL_POS, COL_REGION, COL_DELTA_CPP, COL_SHIFT_SCORE,
    COL_DELTA_PRED, COL_WT_PRED, COL_WT_PRED_STD, COL_VARIANT, COL_SEQ_MUT,
    COL_N_MUT, COL_N_DISRUPTIVE, COL_FRAC_DISRUPTIVE, COL_MEAN_DELTA_CPP,
    COL_RANK, COL_GENERATION, COL_CROWDING,
    COL_HYPERVOLUME, COL_N_FRONT, COL_SPREAD,
    COL_PROTEIN_ID, COL_START, COL_STOP, COL_AA, COL_FEATURE_TYPE, COL_SOURCE,
    COL_EVIDENCE, COL_SCORE, COL_BOND_ID,
    LIST_CAT, LIST_ALL_PARTS, LIST_CANONICAL_AA, COLS_SEQ_POS, COLS_SEQ_PARTS,
    COLS_SEQ_TMD,
)

# Field-record keys (kept positionless / dict-based on purpose; documented above).
FIELD_KEYS = ("dtype", "required", "nullable", "unique", "range",
              "allowed_values", "example", "validation", "description")


def _field(dtype, description, *, required=True, nullable=False, unique=False,
           range=None, allowed_values=None, example=None, validation=None):
    """Build one column's field record; optional keys are omitted when None."""
    rec = {"dtype": dtype, "required": required, "nullable": nullable,
           "unique": unique, "description": description}
    if range is not None:
        rec["range"] = range
    if allowed_values is not None:
        rec["allowed_values"] = allowed_values
    if example is not None:
        rec["example"] = example
    if validation is not None:
        rec["validation"] = validation
    return rec


DICT_DF_SCHEMAS = {
    # ---------------------------------------------------------------- inputs
    "df_seq": {
        "description": (
            "Sequence-level input table; one row per protein/sequence. Accepts four "
            "column formats (besides the always-required 'entry'): position-based "
            f"{COLS_SEQ_POS}, part-based {COLS_SEQ_PARTS}, sequence-based ['sequence'], "
            f"or sequence-TMD-based {COLS_SEQ_TMD}."),
        "formats": {
            "position_based": COLS_SEQ_POS,
            "part_based": COLS_SEQ_PARTS,
            "sequence_based": [COL_SEQ],
            "sequence_tmd_based": COLS_SEQ_TMD,
        },
        "columns": {
            COL_ENTRY: _field("str", "Unique protein/sequence identifier.",
                              required=True, unique=True, example="P05067"),
            COL_SEQ: _field("str", "Amino acid sequence (one-letter code).",
                            required=False, validation="amino_acid_sequence",
                            example="MLPGLALLLL..."),
            COL_TMD_START: _field("int", "1-based start position of the TMD in 'sequence' "
                                  "(position-based format).", required=False,
                                  range=[1, None], example=37),
            COL_TMD_STOP: _field("int", "1-based stop position of the TMD in 'sequence', "
                                 "inclusive (position-based format).", required=False,
                                 range=[1, None], example=59),
            COL_JMD_N: _field("str", "N-terminal juxtamembrane domain subsequence "
                              "(part-based format).", required=False, example="NSPFYYDWHS"),
            COL_TMD: _field("str", "Transmembrane domain subsequence (part-/sequence-TMD "
                            "formats).", required=False, example="LQVGGLICAGVL"),
            COL_JMD_C: _field("str", "C-terminal juxtamembrane domain subsequence "
                              "(part-based format).", required=False, example="KCKCKFGQKS"),
            COL_LABEL: _field("int", "Class/group label. Binary contrasts by convention "
                              "(reference=0, test=1); multi-class via the "
                              "SequenceFeature.get_labels_* helpers.", required=False,
                              example=1),
            COL_NAME: _field("str", "Human-readable protein name.", required=False,
                             nullable=True, example="APP"),
        },
    },
    "df_parts": {
        "description": (
            "Sequence parts table consumed by CPP / SequenceFeature; one row per "
            "sequence. Columns are DYNAMIC: one column per selected sequence part, named "
            "by the part vocabulary; each value is the part's amino acid subsequence."),
        "dynamic_columns": {
            "name_from": "LIST_ALL_PARTS",
            "allowed_names": list(LIST_ALL_PARTS),
            "dtype": "str",
            "nullable": False,
            "description": "Amino acid subsequence of the named sequence part.",
            "example": "NSPFYYDWHSLQVGGL",
        },
    },
    # ------------------------------------------------------------- reference
    "df_scales": {
        "description": (
            "Amino acid scale matrix (AAontology / AAindex-derived). Loaded via "
            "load_scales; consumed by CPP / SequenceFeature."),
        "matrix": {
            "index": "amino acid (one-letter; 20 canonical)",
            "columns": "scale id (e.g. 'KLEP840101')",
            "values": "min-max normalised scale value in [0, 1]",
            "dtype": "float",
        },
    },
    "df_cat": {
        "description": (
            "AAontology scale metadata (one row per scale id). Loaded via "
            "load_scales(name='scales_cat')."),
        "columns": {
            COL_SCALE_ID: _field("str", "Amino acid scale identifier; the join key to "
                                 "df_scales columns.", required=True, unique=True,
                                 example="KLEP840101"),
            COL_CAT: _field("str", "AAontology main category.", required=True,
                            allowed_values=list(LIST_CAT), example="Energy"),
            COL_SUBCAT: _field("str", "AAontology subcategory.", required=True,
                               example="Charge"),
            COL_SCALE_NAME: _field("str", "Human-readable scale name.", required=True,
                                   example="Charge"),
            COL_SCALE_DES: _field("str", "One-sentence scale description.", required=True,
                                  example="Net charge (Klein et al., 1984)."),
        },
    },
    "df_subcat": {
        "description": (
            "AAontology subcategory overview (one row per subcategory). Loaded via "
            "load_scales(name='subcat')."),
        "columns": {
            COL_CAT: _field("str", "AAontology main category.", required=True,
                            allowed_values=list(LIST_CAT), example="Energy"),
            COL_SUBCAT: _field("str", "AAontology subcategory.", required=True,
                               unique=True, example="Charge"),
            COL_CLUSTER: _field("str", "Cluster the subcategory belongs to.",
                                required=True, example="Electrostatics"),
            COL_INTERPRET_GRADE: _field("int", "Interpretability grade, 1-10 (1 = best).",
                                        required=True, range=[1, 10], example=2),
            COL_TOP_EXPLAIN: _field("float", "Interpretability-tier threshold; NaN for "
                                    "unclassified subcategories.", required=True,
                                    nullable=True, example=0.5),
            COL_N_SCALES: _field("int", "Number of scales in the subcategory.",
                                 required=True, range=[1, None], example=12),
            COL_N_SCALES_AAINDEX: _field("int", "Number of those scales originating from "
                                         "AAindex.", required=True, range=[0, None],
                                         example=10),
            COL_SUBCAT_DES: _field("str", "Subcategory description.", required=True,
                                   example="Net charge of the residue."),
            COL_KEY_REFERENCES: _field("str", "Key literature references.", required=True,
                                       nullable=True, example="Klein84"),
        },
    },
    # --------------------------------------------------------------- outputs
    "df_feat": {
        "description": (
            "CPP feature table (the primary downstream contract; see also the simple "
            "DICT_DF_FEAT). One row per feature. Columns 1-13 are the canonical lower "
            "bound (LIST_COLS_FEAT); optional/dynamic columns follow in stable order."),
        "columns": {
            COL_FEATURE: _field("str", "Opaque PART-SPLIT-SCALE feature id; split with "
                                "split_feat_id, never parse by hand.", required=True,
                                unique=True, validation="feature_id",
                                example="TMD_C_JMD_C-Segment(3,4)-KLEP840101"),
            COL_CAT: _field("str", "AAontology category of the feature's scale.",
                            required=True, allowed_values=list(LIST_CAT), example="Energy"),
            COL_SUBCAT: _field("str", "AAontology subcategory.", required=True,
                               example="Charge"),
            COL_SCALE_NAME: _field("str", "Human-readable scale name.", required=True,
                                   example="Charge"),
            COL_SCALE_DES: _field("str", "One-sentence scale description.", required=True,
                                  example="Net charge (Klein et al., 1984)."),
            COL_ABS_AUC: _field("float", "Absolute adjusted AUC; primary ranking "
                                "statistic.", required=True, range=[-0.5, 0.5],
                                example=0.244),
            COL_ABS_MEAN_DIF: _field("float", "Absolute mean difference between test and "
                                     "reference group.", required=True, range=[0, 1],
                                     example=0.104),
            COL_MEAN_DIF: _field("float", "Signed mean difference (test - reference); the "
                                 "sign gives the direction.", required=True,
                                 range=[-1, 1], example=0.104),
            COL_STD_TEST: _field("float", "Standard deviation in the test group.",
                                 required=True, range=[0, None], example=0.107),
            COL_STD_REF: _field("float", "Standard deviation in the reference group.",
                                required=True, range=[0, None], example=0.110),
            COL_PVAL_MW: _field("float", "Mann-Whitney U p-value (default). Named "
                                "'p_val_ttest_indep' when parametric=True.", required=True,
                                range=[0, 1], example=0.0),
            COL_PVAL_FDR: _field("float", "Benjamini-Hochberg FDR-corrected p-value.",
                                 required=True, range=[0, 1], example=0.0),
            COL_POSITION: _field("str", "Comma-separated 1-based residue positions the "
                                 "feature spans.", required=True, example="31,32,33,34,35"),
            COL_PVAL_TTEST: _field("float", "Independent t-test p-value; replaces "
                                   "'p_val_mann_whitney' when parametric=True.",
                                   required=False, range=[0, 1], example=0.01),
            COL_AA_TEST: _field("str", "Amino acids at the feature positions in the test "
                                "group (diagnostic).", required=False, example="K,R,K"),
            COL_AA_REF: _field("str", "Amino acids at the feature positions in the "
                               "reference group (diagnostic).", required=False,
                               example="A,L,G"),
            COL_FEAT_DES: _field("str", "Optional readable one-sentence feature "
                                 "description.", required=False, nullable=True,
                                 example="Charge of the C-terminal TMD segment."),
            COL_FEAT_IMPORT: _field("float", "Feature importance from TreeModel.fit "
                                    "(post-fit).", required=False, range=[0, None],
                                    example=0.97),
            COL_FEAT_IMPORT_STD: _field("float", "Std of the feature importance across CV "
                                        "rounds (post-fit).", required=False,
                                        range=[0, None], example=1.44),
            COL_FEAT_IMPACT: _field("float", "SHAP-based signed feature impact from "
                                    "ShapModel (post-fit, pro).", required=False,
                                    example=0.12),
            COL_FEAT_IMPACT_STD: _field("float", "Std of the feature impact (post-fit, "
                                        "pro).", required=False, range=[0, None],
                                        example=0.03),
        },
    },
    "df_eval": {
        "description": (
            "Evaluation table; one row per evaluated set. Columns depend on the "
            "evaluating class (AAclust vs dPULearn); 'name' labels the row when present."),
        "columns": {
            COL_NAME: _field("str", "Label of the evaluated set.", required=False,
                             example="Set 1"),
            "n_clusters": _field("int", "AAclust: number of clusters.", required=False,
                                 range=[1, None], example=8),
            "BIC": _field("float", "AAclust: Bayesian Information Criterion.",
                          required=False, example=-1234.5),
            "CH": _field("float", "AAclust: Calinski-Harabasz score.", required=False,
                         range=[0, None], example=42.1),
            "SC": _field("float", "AAclust: Silhouette Coefficient.", required=False,
                         range=[-1, 1], example=0.34),
            "n_rel_neg": _field("int", "dPULearn: number of reliable negatives "
                                "identified.", required=False, range=[0, None],
                                example=50),
            "avg_STD": _field("float", "dPULearn: average within-group standard "
                              "deviation.", required=False, range=[0, None], example=0.21),
            "avg_KLD_pos": _field("float", "dPULearn: average KL divergence vs the "
                                  "positive set.", required=False, example=0.12),
        },
    },
    "df_mut": {
        "description": (
            "AAMut per-scale mutation effects (one row per scale for a substitution)."),
        "columns": {
            COL_FROM_AA: _field("str", "Substituted-from amino acid (one letter).",
                                required=True, allowed_values=list(LIST_CANONICAL_AA),
                                example="M"),
            COL_TO_AA: _field("str", "Substituted-to amino acid (one letter).",
                              required=True, allowed_values=list(LIST_CANONICAL_AA),
                              example="V"),
            COL_SCALE_ID: _field("str", "Amino acid scale identifier.", required=True,
                                 example="KLEP840101"),
            COL_CAT: _field("str", "AAontology category.", required=True,
                            allowed_values=list(LIST_CAT), example="Energy"),
            COL_SUBCAT: _field("str", "AAontology subcategory.", required=True,
                               example="Charge"),
            COL_DELTA: _field("float", "Signed per-scale substitution delta "
                              "(to_aa - from_aa).", required=True, example=-0.31),
            COL_ABS_DELTA: _field("float", "Magnitude of the per-scale delta.",
                                  required=True, range=[0, None], example=0.31),
        },
    },
    "df_seqmut_scan": {
        "description": (
            "SeqMut.scan per-mutation scan (one row per scanned single mutation)."),
        "columns": {
            COL_ENTRY: _field("str", "Protein/sequence identifier.", required=True,
                              example="P05067"),
            COL_POS: _field("int", "1-based mutated position.", required=True,
                            range=[1, None], example=123),
            COL_FROM_AA: _field("str", "Original amino acid.", required=True,
                                allowed_values=list(LIST_CANONICAL_AA), example="M"),
            COL_TO_AA: _field("str", "Substituted amino acid.", required=True,
                              allowed_values=list(LIST_CANONICAL_AA), example="V"),
            COL_MUTATION: _field("str", "HGVS-like label '<from><pos><to>'.",
                                 required=True, example="M123V"),
            COL_REGION: _field("str", "Sequence part the position falls in.",
                               required=True, allowed_values=list(COLS_SEQ_PARTS),
                               example="tmd"),
            COL_DELTA_CPP: _field("float", "Summed absolute (L1) feature-space "
                                  "magnitude of the mutation.", required=True,
                                  range=[0, None], example=2.4),
            COL_SHIFT_SCORE: _field("float", "Signed shift toward the test-class "
                                    "profile.", required=True, example=-0.8),
            COL_DELTA_PRED: _field("float", "Change of the model prediction score "
                                   "(percentage points) the mutation induces; present "
                                   "only when a fitted model is bound to SeqMut.",
                                   required=False, example=12.5),
            COL_WT_PRED: _field("float", "Wild-type prediction score (%) for the target "
                                "class (per-entry constant); present only with a model.",
                                required=False, range=[0, 100], example=41.0),
            COL_WT_PRED_STD: _field("float", "Standard deviation of the wild-type "
                                    "prediction score (%); NaN when the model gives no "
                                    "std. Present only with a model.", required=False,
                                    nullable=True, range=[0, 100], example=49.2),
        },
    },
    "df_seqmut_variant": {
        "description": (
            "SeqMut.combine combined-variant table (one row per multi-mutation variant; "
            "all of a variant's point mutations are applied to the same sequence)."),
        "columns": {
            COL_ENTRY: _field("str", "Protein/sequence identifier.", required=True,
                              example="P05067"),
            COL_VARIANT: _field("str", "Combined-variant label, '+'-joined single "
                                "mutations.", required=True, example="R20K+K27P"),
            COL_N_MUT: _field("int", "Number of point mutations in the variant.",
                              required=True, range=[1, None], example=2),
            COL_SEQ_MUT: _field("str", "Full sequence with all the variant's mutations "
                                "applied.", required=True, example="...K...P..."),
            COL_DELTA_CPP: _field("float", "Summed absolute (L1) feature-space "
                                  "magnitude of the combined variant.", required=True,
                                  range=[0, None], example=4.1),
            COL_SHIFT_SCORE: _field("float", "Signed shift toward the test-class "
                                    "profile for the combined variant.", required=True,
                                    example=1.3),
            COL_DELTA_PRED: _field("float", "Change of the model prediction score "
                                   "(percentage points) for the combined variant; "
                                   "present only with a model.", required=False,
                                   example=18.7),
        },
    },
    "df_seqmut_eval": {
        "description": "SeqMut.eval per-protein/region summary (one row per region).",
        "columns": {
            COL_ENTRY: _field("str", "Protein/sequence identifier.", required=True,
                              example="P05067"),
            COL_REGION: _field("str", "Sequence part.", required=True,
                               allowed_values=list(COLS_SEQ_PARTS), example="tmd"),
            COL_N_MUT: _field("int", "Number of scanned mutations.", required=True,
                              range=[0, None], example=20),
            COL_N_DISRUPTIVE: _field("int", "Number flagged disruptive.", required=True,
                                     range=[0, None], example=3),
            COL_FRAC_DISRUPTIVE: _field("float", "n_disruptive / n_mut.", required=True,
                                        range=[0, 1], example=0.15),
            COL_MEAN_DELTA_CPP: _field("float", "Mean delta_cpp (summed absolute "
                                       "feature-space magnitude) over scanned "
                                       "mutations.", required=True, range=[0, None],
                                       example=1.1),
        },
    },
    "df_pareto": {
        "description": (
            "SeqOpt.run Pareto front (one row per non-dominated variant of the single "
            "wild-type). One column per objective is inserted between 'sequence_mut' and "
            "'rank' at run time (e.g. delta_pred, n_mut), so the objective columns below "
            "are documented as optional/dynamic."),
        "columns": {
            COL_ENTRY: _field("str", "Wild-type protein/sequence identifier (constant).",
                              required=True, example="P05067"),
            COL_VARIANT: _field("str", "Variant label, '+'-joined single mutations.",
                                required=True, example="R20K+K27P"),
            COL_N_MUT: _field("int", "Number of point mutations in the variant.",
                              required=True, range=[0, None], example=2),
            COL_SEQ_MUT: _field("str", "Full sequence with all the variant's mutations "
                                "applied.", required=True, example="...K...P..."),
            COL_RANK: _field("int", "Non-dominated front index from fast non-dominated "
                             "sorting (0 = best/first front).", required=True,
                             range=[0, None], example=0),
            COL_CROWDING: _field("float", "NSGA-II crowding distance within the front "
                                 "(larger = more isolated = preferred); inf at the "
                                 "boundary points.", required=True, range=[0, None],
                                 nullable=True, example=1.7),
            COL_DELTA_PRED: _field("float", "Objective: change of the model prediction "
                                   "score (percentage points). Present when an objective "
                                   "uses it.", required=False, example=18.7),
            COL_DELTA_CPP: _field("float", "Objective: summed absolute (L1) "
                                  "feature-space magnitude. Present when an objective "
                                  "uses it.", required=False, range=[0, None],
                                  example=4.1),
            COL_SHIFT_SCORE: _field("float", "Objective: signed shift toward the "
                                    "test-class profile. Present when an objective uses "
                                    "it.", required=False, example=1.3),
        },
    },
    "df_seqopt_eval": {
        "description": (
            "SeqOpt.eval Pareto-quality summary (one row per run / front)."),
        "columns": {
            COL_HYPERVOLUME: _field("float", "Objective-space volume dominated by the "
                                    "first front relative to the reference point.",
                                    required=True, range=[0, None], example=0.62),
            COL_N_FRONT: _field("int", "Number of variants on the first (rank=0) front.",
                                required=True, range=[1, None], example=12),
            COL_SPREAD: _field("float", "Objective-space diversity / spread of the front "
                               "(0 = degenerate).", required=True, range=[0, None],
                               nullable=True, example=0.41),
        },
    },
    "df_annot": {
        "description": (
            "Canonical per-residue annotation table (AnnotationPreprocessor); one row "
            "per annotated residue/feature."),
        "columns": {
            COL_PROTEIN_ID: _field("str", "UniProt accession (mirrors 'entry').",
                                   required=True, example="P05067"),
            COL_START: _field("int", "1-based start (UniProt-canonical frame).",
                              required=True, range=[1, None], example=672),
            COL_STOP: _field("int", "1-based stop, inclusive (single residue: "
                             "start==end).", required=True, range=[1, None], example=714),
            COL_AA: _field("str", "Expected residue identity (encode-time guard).",
                           required=True, nullable=True, example="K"),
            COL_FEATURE_TYPE: _field("str", "Registry key (e.g. 'phospho', 'binding').",
                                     required=True, example="binding"),
            COL_CAT: _field("str", "Annotation category.", required=False, example="PTM"),
            COL_SOURCE: _field("str", "'UniProt' or a user source name.", required=True,
                               example="UniProt"),
            COL_EVIDENCE: _field("str", "ECO evidence code.", required=False,
                                 nullable=True, example="ECO:0000269"),
            COL_SCORE: _field("float", "Optional annotation score.", required=False,
                              nullable=True, range=[0, 1], example=0.9),
            COL_BOND_ID: _field("str", "Pairing id for DISULFID / CROSSLNK endpoints.",
                                required=False, nullable=True, example="b1"),
        },
    },
    "df_logo": {
        "description": (
            "AAlogo composition/probability matrix consumed by sequence-logo plots."),
        "matrix": {
            "index": "1-based sequence position",
            "columns": "amino acid (one-letter)",
            "values": "per-position composition / probability / information value",
            "dtype": "float",
        },
    },
    # ------------------------------------------------ non-DataFrame contracts
    "X": {
        "description": (
            "ML-ready numeric feature matrix produced by "
            "SequenceFeature.feature_matrix; consumed by TreeModel / sklearn."),
        "matrix": {
            "index": "sample (row order matches df_seq / labels)",
            "columns": "feature (column order matches df_feat['feature'])",
            "values": "scale value of the feature for the sample",
            "dtype": "float",
        },
    },
    "prediction": {
        "description": (
            "TreeModel.predict_proba output: two 1-D arrays of length n_samples, "
            "(pred, pred_std) -- the Monte-Carlo mean and standard deviation of the "
            "positive-class probability per sample."),
        "matrix": {
            "index": "sample (row order matches X)",
            "columns": "(pred, pred_std)",
            "values": "positive-class probability in [0, 1] (pred) and its std",
            "dtype": "float",
        },
    },
}


def _format_value(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def render_schemas_rst():
    """Render ``DICT_DF_SCHEMAS`` to the reStructuredText reference page (single source
    of truth for the docs; a drift test asserts the committed page matches this output)."""
    out = []
    out.append(".. _df_schemas:")
    out.append("")
    out.append("Data Schemas")
    out.append("============")
    out.append("")
    out.append("This page documents the schemas (the data dictionary) for the key "
               "AAanalysis DataFrames: the documented, test-guarded contract for the "
               "columns each frame carries. It is generated from "
               "``aaanalysis.utils.DICT_DF_SCHEMAS`` and kept in sync by a contract "
               "test, so the documentation cannot drift from the code.")
    out.append("")
    out.append("Every column lists its dtype, whether it is required / nullable / unique, "
               "the allowed values or numeric range where applicable, a one-line meaning, "
               "and an example value.")
    out.append("")
    out.append("The :ref:`df_feat contract <df_feat_contract>` expands the ``df_feat`` "
               "entry below with the feature-id grammar, the per-residue ``positions`` "
               "encoding, and the stability policy.")
    out.append("")
    out.append(".. toctree::")
    out.append("   :maxdepth: 1")
    out.append("")
    out.append("   df_feat_contract")
    out.append("")
    for frame, spec in DICT_DF_SCHEMAS.items():
        out.append(f"``{frame}``")
        out.append("-" * (len(frame) + 4))
        out.append("")
        out.append(spec["description"])
        out.append("")
        if "formats" in spec:
            out.append("Accepted column formats (besides the always-required columns):")
            out.append("")
            for name, cols in spec["formats"].items():
                out.append(f"- **{name}**: {', '.join(f'``{c}``' for c in cols)}")
            out.append("")
        if "matrix" in spec:
            m = spec["matrix"]
            out.append("Matrix / array contract:")
            out.append("")
            out.append(f"- **index**: {m['index']}")
            out.append(f"- **columns**: {m['columns']}")
            out.append(f"- **values**: {m['values']} (dtype: {m['dtype']})")
            out.append("")
            continue
        if "dynamic_columns" in spec:
            d = spec["dynamic_columns"]
            out.append(f"Dynamic columns (dtype: {d['dtype']}, nullable: {d['nullable']}): "
                       f"{d['description']} Column names are drawn from the part "
                       f"vocabulary: {', '.join(f'``{c}``' for c in d['allowed_names'])}.")
            out.append("")
            continue
        # Column table.
        out.append(".. list-table::")
        out.append("   :header-rows: 1")
        out.append("   :widths: 18 8 8 8 8 34 18")
        out.append("")
        out.append("   * - Column\n     - Type\n     - Required\n     - Nullable\n"
                   "     - Unique\n     - Description\n     - Allowed / range / example")
        for col, rec in spec["columns"].items():
            extra = []
            if "allowed_values" in rec:
                vals = rec["allowed_values"]
                shown = ", ".join(str(x) for x in vals[:6]) + (", ..." if len(vals) > 6 else "")
                extra.append(f"allowed: {shown}")
            if "range" in rec:
                lo, hi = rec["range"]
                extra.append(f"range: [{lo if lo is not None else '-inf'}, "
                             f"{hi if hi is not None else 'inf'}]")
            if "example" in rec:
                extra.append(f"e.g. {_format_value(rec['example'])}")
            extra_str = "; ".join(extra) if extra else ""
            out.append(
                f"   * - ``{col}``\n     - {rec['dtype']}\n"
                f"     - {'yes' if rec['required'] else 'no'}\n"
                f"     - {'yes' if rec['nullable'] else 'no'}\n"
                f"     - {'yes' if rec['unique'] else 'no'}\n"
                f"     - {rec['description']}\n     - {extra_str}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
