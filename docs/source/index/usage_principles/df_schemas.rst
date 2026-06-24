.. _df_schemas:

DataFrame Schemas (Data Dictionary)
===================================

This page is the data dictionary for the key AAanalysis DataFrames: the documented, test-guarded contract for the columns each frame carries. It is generated from ``aaanalysis.utils.DICT_DF_SCHEMAS`` and kept in sync by a contract test, so the documentation cannot drift from the code.

Every column lists its dtype, whether it is required / nullable / unique, the allowed values or numeric range where applicable, a one-line meaning, and an example value.

``df_seq``
----------

Sequence-level input table; one row per protein/sequence. Accepts four column formats (besides the always-required 'entry'): position-based ['sequence', 'tmd_start', 'tmd_stop'], part-based ['jmd_n', 'tmd', 'jmd_c'], sequence-based ['sequence'], or sequence-TMD-based ['sequence', 'tmd'].

Accepted column formats (besides the always-required columns):

- **position_based**: ``sequence``, ``tmd_start``, ``tmd_stop``
- **part_based**: ``jmd_n``, ``tmd``, ``jmd_c``
- **sequence_based**: ``sequence``
- **sequence_tmd_based**: ``sequence``, ``tmd``

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``entry``
     - str
     - yes
     - no
     - yes
     - Unique protein/sequence identifier.
     - e.g. P05067
   * - ``sequence``
     - str
     - no
     - no
     - no
     - Amino acid sequence (one-letter code).
     - e.g. MLPGLALLLL...
   * - ``tmd_start``
     - int
     - no
     - no
     - no
     - 1-based start position of the TMD in 'sequence' (position-based format).
     - range: [1, inf]; e.g. 37
   * - ``tmd_stop``
     - int
     - no
     - no
     - no
     - 1-based stop position of the TMD in 'sequence', inclusive (position-based format).
     - range: [1, inf]; e.g. 59
   * - ``jmd_n``
     - str
     - no
     - no
     - no
     - N-terminal juxtamembrane domain subsequence (part-based format).
     - e.g. NSPFYYDWHS
   * - ``tmd``
     - str
     - no
     - no
     - no
     - Transmembrane domain subsequence (part-/sequence-TMD formats).
     - e.g. LQVGGLICAGVL
   * - ``jmd_c``
     - str
     - no
     - no
     - no
     - C-terminal juxtamembrane domain subsequence (part-based format).
     - e.g. KCKCKFGQKS
   * - ``label``
     - int
     - no
     - no
     - no
     - Class/group label. Binary contrasts by convention (reference=0, test=1); multi-class via the SequenceFeature.get_labels_* helpers.
     - e.g. 1
   * - ``name``
     - str
     - no
     - yes
     - no
     - Human-readable protein name.
     - e.g. APP

``df_parts``
------------

Sequence parts table consumed by CPP / SequenceFeature; one row per sequence. Columns are DYNAMIC: one column per selected sequence part, named by the part vocabulary; each value is the part's amino acid subsequence.

Dynamic columns (dtype: str, nullable: False): Amino acid subsequence of the named sequence part. Column names are drawn from the part vocabulary: ``tmd``, ``tmd_e``, ``tmd_n``, ``tmd_c``, ``jmd_n``, ``jmd_c``, ``ext_c``, ``ext_n``, ``tmd_jmd``, ``jmd_n_tmd_n``, ``tmd_c_jmd_c``, ``ext_n_tmd_n``, ``tmd_c_ext_c``.

``df_scales``
-------------

Amino acid scale matrix (AAontology / AAindex-derived). Loaded via load_scales; consumed by CPP / SequenceFeature.

Matrix / array contract:

- **index**: amino acid (one-letter; 20 canonical)
- **columns**: scale id (e.g. 'KLEP840101')
- **values**: min-max normalised scale value in [0, 1] (dtype: float)

``df_cat``
----------

AAontology scale metadata (one row per scale id). Loaded via load_scales(name='scales_cat').

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``scale_id``
     - str
     - yes
     - no
     - yes
     - Amino acid scale identifier; the join key to df_scales columns.
     - e.g. KLEP840101
   * - ``category``
     - str
     - yes
     - no
     - no
     - AAontology main category.
     - allowed: ASA/Volume, Composition, Conformation, Energy, Others, Polarity, ...; e.g. Energy
   * - ``subcategory``
     - str
     - yes
     - no
     - no
     - AAontology subcategory.
     - e.g. Charge
   * - ``scale_name``
     - str
     - yes
     - no
     - no
     - Human-readable scale name.
     - e.g. Charge
   * - ``scale_description``
     - str
     - yes
     - no
     - no
     - One-sentence scale description.
     - e.g. Net charge (Klein et al., 1984).

``df_subcat``
-------------

AAontology subcategory overview (one row per subcategory). Loaded via load_scales(name='subcat').

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``category``
     - str
     - yes
     - no
     - no
     - AAontology main category.
     - allowed: ASA/Volume, Composition, Conformation, Energy, Others, Polarity, ...; e.g. Energy
   * - ``subcategory``
     - str
     - yes
     - no
     - yes
     - AAontology subcategory.
     - e.g. Charge
   * - ``cluster``
     - str
     - yes
     - no
     - no
     - Cluster the subcategory belongs to.
     - e.g. Electrostatics
   * - ``interpret_grade``
     - int
     - yes
     - no
     - no
     - Interpretability grade, 1-10 (1 = best).
     - range: [1, 10]; e.g. 2
   * - ``top_explain``
     - float
     - yes
     - yes
     - no
     - Interpretability-tier threshold; NaN for unclassified subcategories.
     - e.g. 0.5
   * - ``n_scales``
     - int
     - yes
     - no
     - no
     - Number of scales in the subcategory.
     - range: [1, inf]; e.g. 12
   * - ``n_scales_aaindex``
     - int
     - yes
     - no
     - no
     - Number of those scales originating from AAindex.
     - range: [0, inf]; e.g. 10
   * - ``subcategory_description``
     - str
     - yes
     - no
     - no
     - Subcategory description.
     - e.g. Net charge of the residue.
   * - ``key_references``
     - str
     - yes
     - yes
     - no
     - Key literature references.
     - e.g. Klein84

``df_feat``
-----------

CPP feature table (the primary downstream contract; see also the simple DICT_DF_FEAT). One row per feature. Columns 1-13 are the canonical lower bound (LIST_COLS_FEAT); optional/dynamic columns follow in stable order.

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``feature``
     - str
     - yes
     - no
     - yes
     - Opaque PART-SPLIT-SCALE feature id; split with split_feat_id, never parse by hand.
     - e.g. TMD_C_JMD_C-Segment(3,4)-KLEP840101
   * - ``category``
     - str
     - yes
     - no
     - no
     - AAontology category of the feature's scale.
     - allowed: ASA/Volume, Composition, Conformation, Energy, Others, Polarity, ...; e.g. Energy
   * - ``subcategory``
     - str
     - yes
     - no
     - no
     - AAontology subcategory.
     - e.g. Charge
   * - ``scale_name``
     - str
     - yes
     - no
     - no
     - Human-readable scale name.
     - e.g. Charge
   * - ``scale_description``
     - str
     - yes
     - no
     - no
     - One-sentence scale description.
     - e.g. Net charge (Klein et al., 1984).
   * - ``abs_auc``
     - float
     - yes
     - no
     - no
     - Absolute adjusted AUC; primary ranking statistic.
     - range: [-0.5, 0.5]; e.g. 0.244
   * - ``abs_mean_dif``
     - float
     - yes
     - no
     - no
     - Absolute mean difference between test and reference group.
     - range: [0, 1]; e.g. 0.104
   * - ``mean_dif``
     - float
     - yes
     - no
     - no
     - Signed mean difference (test - reference); the sign gives the direction.
     - range: [-1, 1]; e.g. 0.104
   * - ``std_test``
     - float
     - yes
     - no
     - no
     - Standard deviation in the test group.
     - range: [0, inf]; e.g. 0.107
   * - ``std_ref``
     - float
     - yes
     - no
     - no
     - Standard deviation in the reference group.
     - range: [0, inf]; e.g. 0.11
   * - ``p_val_mann_whitney``
     - float
     - yes
     - no
     - no
     - Mann-Whitney U p-value (default). Named 'p_val_ttest_indep' when parametric=True.
     - range: [0, 1]; e.g. 0.0
   * - ``p_val_fdr_bh``
     - float
     - yes
     - no
     - no
     - Benjamini-Hochberg FDR-corrected p-value.
     - range: [0, 1]; e.g. 0.0
   * - ``positions``
     - str
     - yes
     - no
     - no
     - Comma-separated 1-based residue positions the feature spans.
     - e.g. 31,32,33,34,35
   * - ``p_val_ttest_indep``
     - float
     - no
     - no
     - no
     - Independent t-test p-value; replaces 'p_val_mann_whitney' when parametric=True.
     - range: [0, 1]; e.g. 0.01
   * - ``amino_acids_test``
     - str
     - no
     - no
     - no
     - Amino acids at the feature positions in the test group (diagnostic).
     - e.g. K,R,K
   * - ``amino_acids_ref``
     - str
     - no
     - no
     - no
     - Amino acids at the feature positions in the reference group (diagnostic).
     - e.g. A,L,G
   * - ``feature_description``
     - str
     - no
     - yes
     - no
     - Optional readable one-sentence feature description.
     - e.g. Charge of the C-terminal TMD segment.
   * - ``feat_importance``
     - float
     - no
     - no
     - no
     - Feature importance from TreeModel.fit (post-fit).
     - range: [0, inf]; e.g. 0.97
   * - ``feat_importance_std``
     - float
     - no
     - no
     - no
     - Std of the feature importance across CV rounds (post-fit).
     - range: [0, inf]; e.g. 1.44
   * - ``feat_impact``
     - float
     - no
     - no
     - no
     - SHAP-based signed feature impact from ShapModel (post-fit, pro).
     - e.g. 0.12
   * - ``feat_impact_std``
     - float
     - no
     - no
     - no
     - Std of the feature impact (post-fit, pro).
     - range: [0, inf]; e.g. 0.03

``df_eval``
-----------

Evaluation table; one row per evaluated set. Columns depend on the evaluating class (AAclust vs dPULearn); 'name' labels the row when present.

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``name``
     - str
     - no
     - no
     - no
     - Label of the evaluated set.
     - e.g. Set 1
   * - ``n_clusters``
     - int
     - no
     - no
     - no
     - AAclust: number of clusters.
     - range: [1, inf]; e.g. 8
   * - ``BIC``
     - float
     - no
     - no
     - no
     - AAclust: Bayesian Information Criterion.
     - e.g. -1234.5
   * - ``CH``
     - float
     - no
     - no
     - no
     - AAclust: Calinski-Harabasz score.
     - range: [0, inf]; e.g. 42.1
   * - ``SC``
     - float
     - no
     - no
     - no
     - AAclust: Silhouette Coefficient.
     - range: [-1, 1]; e.g. 0.34
   * - ``n_rel_neg``
     - int
     - no
     - no
     - no
     - dPULearn: number of reliable negatives identified.
     - range: [0, inf]; e.g. 50
   * - ``avg_STD``
     - float
     - no
     - no
     - no
     - dPULearn: average within-group standard deviation.
     - range: [0, inf]; e.g. 0.21
   * - ``avg_KLD_pos``
     - float
     - no
     - no
     - no
     - dPULearn: average KL divergence vs the positive set.
     - e.g. 0.12

``df_mut``
----------

AAMut per-scale mutation effects (one row per scale for a substitution).

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``from_aa``
     - str
     - yes
     - no
     - no
     - Substituted-from amino acid (one letter).
     - allowed: A, C, D, E, F, G, ...; e.g. M
   * - ``to_aa``
     - str
     - yes
     - no
     - no
     - Substituted-to amino acid (one letter).
     - allowed: A, C, D, E, F, G, ...; e.g. V
   * - ``scale_id``
     - str
     - yes
     - no
     - no
     - Amino acid scale identifier.
     - e.g. KLEP840101
   * - ``category``
     - str
     - yes
     - no
     - no
     - AAontology category.
     - allowed: ASA/Volume, Composition, Conformation, Energy, Others, Polarity, ...; e.g. Energy
   * - ``subcategory``
     - str
     - yes
     - no
     - no
     - AAontology subcategory.
     - e.g. Charge
   * - ``delta``
     - float
     - yes
     - no
     - no
     - Signed per-scale substitution delta (to_aa - from_aa).
     - e.g. -0.31
   * - ``abs_delta``
     - float
     - yes
     - no
     - no
     - Magnitude of the per-scale delta.
     - range: [0, inf]; e.g. 0.31

``df_seqmut_scan``
------------------

SeqMut.scan per-mutation scan (one row per scanned single mutation).

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``entry``
     - str
     - yes
     - no
     - no
     - Protein/sequence identifier.
     - e.g. P05067
   * - ``pos``
     - int
     - yes
     - no
     - no
     - 1-based mutated position.
     - range: [1, inf]; e.g. 123
   * - ``from_aa``
     - str
     - yes
     - no
     - no
     - Original amino acid.
     - allowed: A, C, D, E, F, G, ...; e.g. M
   * - ``to_aa``
     - str
     - yes
     - no
     - no
     - Substituted amino acid.
     - allowed: A, C, D, E, F, G, ...; e.g. V
   * - ``mutation``
     - str
     - yes
     - no
     - no
     - HGVS-like label '<from><pos><to>'.
     - e.g. M123V
   * - ``region``
     - str
     - yes
     - no
     - no
     - Sequence part the position falls in.
     - allowed: jmd_n, tmd, jmd_c; e.g. tmd
   * - ``delta_cpp``
     - float
     - yes
     - no
     - no
     - Sum|dX| feature-space magnitude of the mutation.
     - range: [0, inf]; e.g. 2.4
   * - ``shift_score``
     - float
     - yes
     - no
     - no
     - Signed shift toward the test-class profile.
     - e.g. -0.8
   * - ``delta_pred``
     - float
     - no
     - no
     - no
     - Change of the model prediction score (percentage points) the mutation induces; present only when a fitted model is bound to SeqMut.
     - e.g. 12.5
   * - ``wt_pred``
     - float
     - no
     - no
     - no
     - Wild-type prediction score (%) for the target class (per-entry constant); present only with a model.
     - range: [0, 100]; e.g. 41.0
   * - ``wt_pred_std``
     - float
     - no
     - yes
     - no
     - Standard deviation of the wild-type prediction score (%); NaN when the model gives no std. Present only with a model.
     - range: [0, 100]; e.g. 49.2

``df_seqmut_variant``
---------------------

SeqMut.combine combined-variant table (one row per multi-mutation variant; all of a variant's point mutations are applied to the same sequence).

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``entry``
     - str
     - yes
     - no
     - no
     - Protein/sequence identifier.
     - e.g. P05067
   * - ``variant``
     - str
     - yes
     - no
     - no
     - Combined-variant label, '+'-joined single mutations.
     - e.g. R20K+K27P
   * - ``n_mut``
     - int
     - yes
     - no
     - no
     - Number of point mutations in the variant.
     - range: [1, inf]; e.g. 2
   * - ``sequence_mut``
     - str
     - yes
     - no
     - no
     - Full sequence with all the variant's mutations applied.
     - e.g. ...K...P...
   * - ``delta_cpp``
     - float
     - yes
     - no
     - no
     - Sum|dX| feature-space magnitude of the combined variant.
     - range: [0, inf]; e.g. 4.1
   * - ``shift_score``
     - float
     - yes
     - no
     - no
     - Signed shift toward the test-class profile for the combined variant.
     - e.g. 1.3
   * - ``delta_pred``
     - float
     - no
     - no
     - no
     - Change of the model prediction score (percentage points) for the combined variant; present only with a model.
     - e.g. 18.7

``df_seqmut_eval``
------------------

SeqMut.eval per-protein/region summary (one row per region).

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``entry``
     - str
     - yes
     - no
     - no
     - Protein/sequence identifier.
     - e.g. P05067
   * - ``region``
     - str
     - yes
     - no
     - no
     - Sequence part.
     - allowed: jmd_n, tmd, jmd_c; e.g. tmd
   * - ``n_mut``
     - int
     - yes
     - no
     - no
     - Number of scanned mutations.
     - range: [0, inf]; e.g. 20
   * - ``n_disruptive``
     - int
     - yes
     - no
     - no
     - Number flagged disruptive.
     - range: [0, inf]; e.g. 3
   * - ``frac_disruptive``
     - float
     - yes
     - no
     - no
     - n_disruptive / n_mut.
     - range: [0, 1]; e.g. 0.15
   * - ``mean_delta_cpp``
     - float
     - yes
     - no
     - no
     - Mean |delta_cpp| over scanned mutations.
     - range: [0, inf]; e.g. 1.1

``df_annot``
------------

Canonical per-residue annotation table (AnnotationPreprocessor); one row per annotated residue/feature.

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 8 8 34 18

   * - Column
     - Type
     - Required
     - Nullable
     - Unique
     - Description
     - Allowed / range / example
   * - ``protein_id``
     - str
     - yes
     - no
     - no
     - UniProt accession (mirrors 'entry').
     - e.g. P05067
   * - ``start``
     - int
     - yes
     - no
     - no
     - 1-based start (UniProt-canonical frame).
     - range: [1, inf]; e.g. 672
   * - ``end``
     - int
     - yes
     - no
     - no
     - 1-based stop, inclusive (single residue: start==end).
     - range: [1, inf]; e.g. 714
   * - ``aa``
     - str
     - yes
     - yes
     - no
     - Expected residue identity (encode-time guard).
     - e.g. K
   * - ``feature_type``
     - str
     - yes
     - no
     - no
     - Registry key (e.g. 'phospho', 'binding').
     - e.g. binding
   * - ``category``
     - str
     - no
     - no
     - no
     - Annotation category.
     - e.g. PTM
   * - ``source``
     - str
     - yes
     - no
     - no
     - 'UniProt' or a user source name.
     - e.g. UniProt
   * - ``evidence``
     - str
     - no
     - yes
     - no
     - ECO evidence code.
     - e.g. ECO:0000269
   * - ``score``
     - float
     - no
     - yes
     - no
     - Optional annotation score.
     - range: [0, 1]; e.g. 0.9
   * - ``bond_id``
     - str
     - no
     - yes
     - no
     - Pairing id for DISULFID / CROSSLNK endpoints.
     - e.g. b1

``df_logo``
-----------

AAlogo composition/probability matrix consumed by sequence-logo plots.

Matrix / array contract:

- **index**: 1-based sequence position
- **columns**: amino acid (one-letter)
- **values**: per-position composition / probability / information value (dtype: float)

``X``
-----

ML-ready numeric feature matrix produced by SequenceFeature.feature_matrix; consumed by TreeModel / sklearn.

Matrix / array contract:

- **index**: sample (row order matches df_seq / labels)
- **columns**: feature (column order matches df_feat['feature'])
- **values**: scale value of the feature for the sample (dtype: float)

``prediction``
--------------

TreeModel.predict_proba output: two 1-D arrays of length n_samples, (pred, pred_std) -- the Monte-Carlo mean and standard deviation of the positive-class probability per sample.

Matrix / array contract:

- **index**: sample (row order matches X)
- **columns**: (pred, pred_std)
- **values**: positive-class probability in [0, 1] (pred) and its std (dtype: float)
