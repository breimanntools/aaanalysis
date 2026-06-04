# aaanalysis

Python framework for interpretable, sequence-based protein prediction. This glossary captures domain terms used across the package, sharpened over time as conversations resolve ambiguities. Definitions are short and opinionated; aliases are explicitly listed so the canonical term wins on touch.

## Language

### Core sequence vocabulary

**df_seq**:
A pandas DataFrame whose rows describe protein sequences; must contain an `entry` column (unique protein identifier) and a `sequence` column (full amino-acid sequence). Method-specific columns (`pos column`, `aa_context_col`) layer on top of this base schema.
_Avoid_: sequence frame, protein df, input df.

**entry**:
A unique protein identifier; conventionally a UniProt accession.
_Avoid_: id, accession, protein_id.

**canonical amino acids**:
The 20 standard amino acids in alphabetical order, `ACDEFGHIKLMNPQRSTVWY`, exposed as `ut.LIST_CANONICAL_AA`.
_Avoid_: standard AAs, "the 20 AAs" without specifying order.

**pos column**:
A `df_seq` column whose cells hold per-row 1-based integer positions of interest (typically labeled positives); each cell is a `list[int]`, a single `int`, or empty (`None` / `NaN` / empty list / empty string). Default column name `"pos"`, referenced as `ut.COL_POS`. Two consumers share this column as **P1 anchors**: `AAWindowSampler` explodes it into sampled windows, and `SequenceFeature.get_df_parts` / `NumericalFeature.get_parts` (the **anchor input mode**) explode it into one 3-part row per anchor (TMD placed by `ut.get_window_offsets`, then the `jmd_n_len`/`jmd_c_len` extension), ided by **entry_win**.
_Avoid_: positions column (collides with `df_feat`'s `COL_POSITION`, which is a CSV string of feature positions).

### Window sampling vocabulary

**window**:
A fixed-length contiguous slice of a protein sequence, anchored at a 1-based P1 position.
_Avoid_: subsequence, fragment, region.

**P1 anchor / source position**:
The 1-based residue immediately N-terminal to a notional scissile bond (Schechter–Berger cleavage convention); the anchor coordinate for a window. For window length L, the window covers `(L-1)//2` residues upstream of the anchor, the anchor itself, and `L//2` residues downstream — right-heavy for even L.
_Avoid_: center, midpoint (ambiguous for even window sizes).

**test window**:
A window extracted at a position listed in the **pos column**; treated as a known-positive reference for the identity filter.
_Avoid_: positive window, target window.

**reference window**:
The umbrella term for any sampled window that is not a test window — covers `Negative`, `Unlabeled`, and `Control` rows.
_Avoid_: negative window (too narrow — only one of the three roles), drawn window.

**control window**:
A reference window produced by `sample_synthetic`; carries `entry=""` and has no source protein coordinates.
_Avoid_: simulated window, fake window.

**role**:
A categorical tag stored in the output's `role` column describing what a sampled row represents in a workflow. Standard values: `Test`, `Negative`, `Unlabeled`, `Control`. Users may pass custom strings. Defaults are opinionated and assume PU-learning / hard-negative-mining workflows.
_Avoid_: class, label_kind, category.

**strategy**:
A tag stored in the output's `strategy` column identifying which sampling method produced the row. Values: `same_protein`, `different_protein`, `motif_matched`, `synthetic:<generator>`.
_Avoid_: method, source, origin.

**entry_win**:
A row's window identifier, formatted `<entry>_<start_pos>-<end_pos>` with 1-based inclusive coordinates for protein-sourced windows, or `synth_{i}` (per-call counter) for synthetic windows. Identical biological windows across calls share the same `entry_win`, making `drop_duplicates(subset="entry_win")` the natural dedupe primitive — except for synthetic outputs, where the per-call counter is not call-stable.
_Avoid_: window_id, row_id.

**candidate pool**:
The set of eligible windows from which a sampling method draws, defined per strategy. `same_protein`: positions whose distance to the nearest positive on the same protein lies in the `(min_distance_to_pos, max_distance_to_pos)` **distance band**. `different_protein` / `motif_matched`: any window on a protein with no listed positives. `synthetic`: drawn fresh from the generator distribution.
_Avoid_: candidates, eligible set.

**distance band** (`min_distance_to_pos`, `max_distance_to_pos`):
A pair of optional residue-distance bounds used by `sample_same_protein` to filter candidate P1 anchors by their L1 distance to the *nearest* positive on the same protein. `min_distance_to_pos` is the lower bound (or `None` for no lower bound); `max_distance_to_pos` is the upper bound (or `None` for no upper bound). Both default to `None`, in which case every fully-fitting window on a positive-containing protein is admissible — so sampled "Negative" windows may overlap positive windows. For non-overlapping hard-negatives, set `min_distance_to_pos=window_size`; for windows targeted near positives, pair with a finite `max_distance_to_pos`.
_Avoid_: distance-to-positive (singular — misses the band).

### Synthetic generation vocabulary

**generator**:
The recipe by which `sample_synthetic` produces a window — a string (built-in mode like `"uniform"` / `"global_freq"` / `"position_specific"` / `"scrambled"`, or an AAontology preset name), a list of preset names (multiplicative mix), or a `dict[str, float]` (custom-alphabet frequency table).
_Avoid_: mode (historical name; renamed for clarity), prior (inaccurate for `scrambled` and `position_specific`).

**AAontology preset**:
A named generator backed by a curated AAontology scale loaded via `aa.load_scales` and sum-normalized into a probability distribution over the 20 canonical AAs. Composition presets are true AA-frequency distributions; conformation presets are normalized propensities used as physicochemically-biased priors.
_Avoid_: scale-based mode.

**custom-alphabet generator**:
A `dict[str, float]` generator mapping single-character symbols to non-negative probabilities summing to 1; the only sampling path that produces non-amino-acid windows.
_Avoid_: custom freq, custom dist (too generic).

### Scoring vocabulary

**PWM (position-weight matrix)**:
A `pd.DataFrame` of shape `(window_size, 20)` representing per-position residue scores over the 20 canonical AAs. Columns are the canonical AA letters in any order and are reindexed internally to `ut.LIST_CANONICAL_AA` (alphabetical, `ACDEFGHIKLMNPQRSTVWY`). `np.ndarray` PWMs are rejected — wrap with `pd.DataFrame(arr, columns=ut.LIST_CANONICAL_AA)` if you only have an array.
_Avoid_: scoring matrix (too generic), motif matrix.

**motif filter**:
The pair `(motif_pwm, motif_score_threshold)` used by `sample_same_protein` and `sample_different_protein` to keep (`motif_match="in"`) or drop (`motif_match="out"`) windows whose PWM score crosses the threshold. Optional on these two methods; required and `"in"`-only on `sample_motif_matched` (where it defines the candidate pool, not an overlay filter).
_Avoid_: motif gate, PWM filter.

**identity filter**:
A pair of filters based on per-position residue identity between fixed-length, aligned windows. `max_similarity_to_test` drops sampled windows too similar to any test window (anti-leakage); `max_similarity_within_ref` drops sampled windows too similar to a previously kept sampled window (redundancy reduction).
_Avoid_: similarity filter (overloaded), redundancy filter (too narrow — covers only the second).

### Embedding-based feature engineering vocabulary

**EmbeddingPreprocessor**:
Public core class in `aaanalysis/data_handling/` for protein-language-model (PLM) embeddings, instance-based (`ep = EmbeddingPreprocessor()`). Its primary method `encode` turns raw per-residue embeddings into a `[0, 1]`-normalized [[dict_num]] for `CPP.run_num`; the secondary `build_scales` / `build_cat` pair derives [[pseudo-scale]]s / [[pseudo-category]]s for the scale-based `CPP.run` path. Mirrors the family shape of `StructurePreprocessor` / `AnnotationPreprocessor` (all three: `encode*` → dict_num, then `build_scales` / `build_cat`).
_Avoid_: PLMPreprocessor, ESMPreprocessor (too narrow).

**encode** (embedding):
`EmbeddingPreprocessor.encode(df_seq, embeddings, method='minmax'|'quantile'|'sigmoid', return_df=False) → dict_num`. Fits one per-embedding-dimension normalizer over the whole corpus (all residues of all proteins) and maps every entry's `(L, D)` tensor into `[0, 1]`, the range `CPP.run_num` expects. AAanalysis does **not** run the PLM — the user supplies `embeddings` as `{entry: (L, D)}`. The fitted parameters are kept on the instance as `norm_params_`. This is the sanctioned raw-embeddings → dict_num step (raw PLM values are unbounded and not directly usable by `run_num`).
_Avoid_: normalize_embeddings, to_dict_num.

**pseudo-scale**:
A (20,)-shaped vector representing one PLM embedding dimension's per-AA average, computed by context-free averaging of the dimension's per-residue values over occurrences of each canonical AA in a reference corpus (typically the user's `df_seq`). Dataset-dependent — pseudo-scales for the same PLM differ across input corpora. Used only to derive pseudo-categories and to name dimensions in `df_scales`; never used as a residue-value source for feature aggregation in `CPP.run_num` (the per-residue [[dict_num]] tensor is consumed directly when supplied).
_Avoid_: dimension scale, AA average, embedding scale.

**pseudo-category**:
A cluster label assigned to a pseudo-scale by AAclust correlation-based clustering. Carried in `df_cat`'s `cat` (coarser threshold) and `subcat` (finer threshold) columns, mirroring the AAontology two-level hierarchy. Cluster IDs are deterministic given `(pseudo_scales, thresholds, random_state)` but inherit the dataset-dependence of pseudo-scales.
_Avoid_: PLM cluster, embedding group.

### Structure-based feature engineering vocabulary

**dict_dssp**:
A `Dict[entry, np.ndarray (L, D_dssp)]` of per-residue DSSP-derived numerical features (secondary structure one-hot, ASA, dihedrals). Produced by `StructurePreprocessor().encode_dssp(df_seq, pdb_folder, features=[...])`.
_Avoid_: ss_dict, dssp_tensor.

**dict_pdb**:
A `Dict[entry, np.ndarray (L, D_pdb)]` of per-residue features extracted directly from PDB ATOM records (mean B-factor, residue depth). Produced by `StructurePreprocessor().encode_pdb(df_seq, pdb_folder, features=[...])`.
_Avoid_: pdb_tensor, raw_pdb_dict.

**feature key**:
A canonical string identifier in the `StructurePreprocessor` registry that maps to a fixed `(num_dims, dim_names, category, subcategory, normalization recipe)` tuple. Used in the `features=[...]` parameter of `encode_dssp` / `encode_pdb` / `encode_pae` and in `build_scales(features=[...])` / `build_cat(features=[...])`. Feature-set **rev 1.1** keys (the StructurePreprocessor's own feature-registry revision — *not* the package version; the class as a whole debuts in package v1.1.0): `ss3`, `ss8`, `rasa`, `phi_psi_sincos`, `bfactor`, `depth`, `hse` (plus the rev-1.1 AlphaFold additions `plddt`, `plddt_disorder`, `plddt_tier`, `chi1_sincos`, `chi2_sincos`, `ca_centroid_dist`, `ca_centroid_dist_norm`, `contact_count_8A`, `contact_count_12A`, `pae_row_mean`, `pae_row_min`, `pae_row_max`, `pae_local_mean`, `pae_distal_mean`, `pae_asymmetry`, `pae_band_means`). Rev 1's `asa` (absolute) and `phi_psi` (raw degrees) are **removed** in rev 1.1.
_Avoid_: feature_id (collides with the `df_feat.feature` column), dim_key.

**StructurePreprocessor**:
Public class in `aaanalysis/data_handling_pro/` that converts PDB / CIF / AlphaFold files (and AF PAE sidecars) into [[dict_num]]-shape per-residue numerical tensors for `CPP.run_num`. Mirrors `EmbeddingPreprocessor`'s instance-based pattern (`stp = StructurePreprocessor()`). Nine public methods: `fetch_alphafold` (bulk download of AF-DB model + PAE files into a folder — see [[fetch_alphafold]]), `get_dssp` (raw DSSP list output), `encode_dssp` (DSSP → dict_num), `encode_pdb` (raw PDB → dict_num, includes AF model-file features), `encode_pae` (AF PAE sidecar → dict_num), `get_domains` (raw domain segmentation via afragmenter/chainsaw), `encode_domains` (domain files → dict_num), `build_scales` (corpus-derived per-AA-mean df_scales), `build_cat` (corpus-free df_cat metadata). The four `encode_*` methods return a **bare `dict_num` by default**; pass `return_df=True` for the `(dict_num, df_seq_out)` form, where `df_seq_out` echoes `df_seq` with a per-row `*_ok` status column. All encoder outputs are normalized to `[0, 1]` per the registry's `NORMALIZATION_RECIPES`; the inverse formulas are documented in the class docstring. `verbose` is constructor-only (no per-call override — see [[preprocessor verb taxonomy]]). Pro-extra gated (biopython; `requests` for `fetch_alphafold`); `msms` is a runtime check inside `encode_pdb(features=['depth'])`.
_Avoid_: PDBPreprocessor, DSSPPreprocessor (too narrow).

**fetch_alphafold**:
`StructurePreprocessor().fetch_alphafold(df_seq, out_folder, file_format='pdb'|'cif', timeout=30.0, skip_existing=True, on_failure='nan', return_df=False)`. Bulk-downloads each `entry`'s AlphaFold-DB model file (`AF-<entry>-F1-model_v4.{pdb,cif}`) and PAE sidecar (`AF-<entry>-F1-predicted_aligned_error_v4.json`) from `https://alphafold.ebi.ac.uk`, saving them under the filenames the `resolve_structure_path` / `resolve_pae_path` resolvers already find — so one call populates the `pdb_folder` / `pae_folder` that `encode_pdb` / `encode_pae` / `get_dssp` consume, with no glue. Returns a per-entry status DataFrame (`entry, model_ok, pae_ok, alphafold_ok, skipped, model_path, pae_path`); `return_df=True` also appends an `alphafold_ok` column to `df_seq`. A 404 (accession not in AF-DB, or fragmented `F2+`-only proteins) is the soft failure governed by `on_failure`; other network errors (timeout, 5xx) raise `RuntimeError` and abort. It is the `fetch_` (web) acquisition verb — the structure-side analog of `AnnotationPreprocessor.fetch_uniprot`. See ADR-0017.
_Avoid_: download_alphafold, get_alphafold (web acquisition is `fetch_`, not `get_`), fetch_pae (PAE ships with the model, not separately).

**preprocessor verb taxonomy**:
The shared method-naming rule across the Family-B preprocessors (`Embedding` / `Structure` / `Annotation`), recorded in ADR-0017. **Acquisition** verbs do I/O and yield a raw artifact: `fetch_*` pulls from a **web** resource (`fetch_uniprot`, `fetch_alphafold`), `get_*` runs a **local** tool / reads local files (`get_dssp`, `get_domains`). An acquisition getter exists **only where the raw output is an independently-useful, inspectable / curatable artifact** (a DSSP list, a domain segmentation, a `df_annot`) — not for a thin numeric read (`encode_pdb`'s ATOM-field extraction and `encode_pae`'s matrix collapse have no `get_` twin by design, not omission). **Transform** verbs are pure (no I/O): `encode` / `encode_*` map a source into a `[0, 1]` `dict_num`, with **one `encode_*` per distinct raw source** (`Structure` has four file sources → `encode_dssp` / `encode_pdb` / `encode_pae` / `encode_domains`) and a **bare `encode`** when the class has a single input (`Embedding`'s `embeddings` dict) or a single canonical intermediate (`Annotation`'s `df_annot`); `build_scales` / `build_cat` derive the secondary AA-scale-path metadata. `ingest` (user table → `df_annot`), `register_feature` (open-vocabulary registration), and `to_df_seq` (annotation → `AAWindowSampler` anchors) are `Annotation`-only by design. `verbose` is **constructor-only** family-wide (no per-call override on any method).
_Avoid_: acquire/load/read as method prefixes (use `get_`/`fetch_`); collapsing the four `encode_*` into one `encode(source=...)` dispatcher.

**combine_dict_nums**:
Top-level `aa.combine_dict_nums(dict_nums: List[Dict[entry, ndarray]]) → Dict[entry, ndarray]` that concatenates multiple per-residue tensors along the D axis. Source-agnostic — works with `dict_dssp`, `dict_pdb`, [[dict_pae]], `dict_embeddings`, or any user-supplied dict matching the shape contract. Validates same entry set + same L per entry across all inputs.
_Avoid_: merge_dict_num, stack_dict_nums.

**dict_pae**:
A `Dict[entry, np.ndarray (L, D_pae)]` of per-residue summaries derived from an AlphaFold PAE sidecar (`AF-{uniprot}-F1-predicted_aligned_error_v4.json`). Produced by `StructurePreprocessor().encode_pae(df_seq, pae_folder, features=[...])`. The L×L matrix is collapsed to per-residue summaries — row-mean / row-min / row-max / local-mean (±`local_window`) / distal-mean / asymmetry / band-means. All values normalized to `[0, 1]` by dividing by AlphaFold's PAE saturation cap (31.75 Å).
_Avoid_: pae_dict, dict_alignment_error.

**Feature-category colors** (locked v1.1 palette; source in `ut.DICT_COLOR_CAT`):
- `Structure` → `#2E6E5E` (deep teal-green) — all `StructurePreprocessor` outputs (DSSP / PDB / PAE / AF features).
- `Embeddings` → `#6B4FB5` (indigo-violet) — all `EmbeddingPreprocessor` outputs.
- `PTMs` → `#B36BCB` (lilac-magenta) — closed-vocabulary UniProt PTM/Processing outputs of `AnnotationPreprocessor`.
- `Functional sites` → `#2C6E9E` (deep ocean-blue) — open-vocabulary functional-site outputs of `AnnotationPreprocessor` (BINDING/ACT_SITE/DNA_BIND seeds + user/predictor keys). Deliberately not `#6B4FB5` (that is `Embeddings`).
- Plus the 8 AAontology categories (`ASA/Volume`, `Composition`, `Conformation`, `Energy`, `Others`, `Polarity`, `Shape`, `Structure-Activity`).
The redundancy filter's `check_cat=True` arm groups features by these top-level buckets; fine-grained semantic splits (e.g. `'Secondary structure (3-state)'` vs `'B-factor (CA mean)'` vs `'AlphaFold pLDDT (raw)'`) live in `subcategory` — these follow the AAontology naming convention (descriptive name with source / detail in parentheses) so the `CPPPlot.feature_map` y-axis reads cleanly.

### Annotation-based feature engineering vocabulary

**AnnotationPreprocessor**:
Public pro-extra class in `aaanalysis/data_handling_pro/` that fetches per-residue PTM / functional-site annotations from UniProt (or ingests user/predictor labels), maps them into the canonical [[df_annot]] schema, and encodes them into `[0, 1]`-normalized per-residue [[dict_num]] tensors for `CPP.run_num`. Mirrors `StructurePreprocessor`'s instance-based pattern (`ap = AnnotationPreprocessor()`): `fetch_uniprot` (UniProt JSON → df_annot), `ingest` (user table → df_annot, open-vocabulary auto-register), `register_feature` (explicit open-vocabulary registration / override), `encode` (df_annot → dict_num; bare `dict_num` by default, `return_df=True` adds a `(dict_num, df_seq_out)` status echo with an `encode_ok` column — mirroring `StructurePreprocessor.encode_*`), `build_scales` (corpus per-AA-mean df_scales), `build_cat` (corpus-free df_cat), and `to_df_seq` (df_annot → df_seq with a `pos` column + an `aa_context` eligibility mask for residue-type-matched `AAWindowSampler` negatives — the seq-mode window-split path). Pro-gated (`requests`).
_Avoid_: PTMPreprocessor (too narrow — also handles functional sites), UniProtPreprocessor (also ingests non-UniProt user labels).

**df_annot**:
The canonical per-residue annotation schema (one row per annotated residue): `protein_id, start, end, aa, feature_type, category, source, evidence, score, bond_id` (columns in `ut.COLS_ANNOT`). Positions are 1-based in the **UniProt-canonical frame**; every mapped row is single-residue (`start == end`). `aa` is the expected residue identity, checked against the target `df_seq[sequence]` at encode time (`on_mismatch='raise'` by default — the off-by-isoform guard). `score` is a nullable float in `[0, 1]` (presence = `1.0`; predictor confidence otherwise). `bond_id` pairs the two endpoints of a disulfide / cross-link.
_Avoid_: annotation table, df_ptm (too narrow), df_sites.

**feature_type**:
The registry key naming one annotation channel (one `dict_num` dimension), e.g. `phospho`, `glyco_n`, `disulfide`, `signal_cleavage` (category `PTMs`) or `binding`, `act_site`, `dna_bind`, or a user key like `hotspot` (category `Functional sites`). Closed PTM vocabulary + functional seeds are built-in; user `feature_type`s auto-register as `Functional sites` at `ingest`.
_Avoid_: ptm_type, label, annotation_type.

**PTM (broad)**:
The `PTMs` category in the broad UniProt "PTM/Processing" sense — modified residues, glycosylation, lipidation, disulfide, cross-link, **and** the signal/propeptide/transit/SITE cleavage sources. A closed vocabulary, one source (UniProt). Disulfide stays in PTMs.
_Avoid_: post-translational-modification-only reading (here it includes cleavage/processing).

**Functional sites**:
The `Functional sites` category — an **open** vocabulary. UniProt BINDING/ACT_SITE/DNA_BIND ship as built-in seeds; user/predictor per-residue labels (RFdiffusion hotspots, BindCraft interface residues, custom) plug in as the extensibility point.
_Avoid_: active sites (too narrow — one seed of many), binding sites.

**evidence allow-set**:
The ECO-code filter applied by `fetch_uniprot(evidence=...)`. `'manual'` (default) keeps `ECO:0000269` (experimental) **and** `ECO:0007744` (combinatorial, manual); `'experimental'` keeps only `ECO:0000269`; `'all'` disables filtering. By-similarity (`ECO:0000250`) is never in the default positives. The raw ECO code is retained in the `evidence` column regardless.
_Avoid_: experimental-only (the default is broader), confidence (overloaded with `score`).

### CPP split vocabulary

**split**:
A rule that selects a subset of residue positions within a sequence **part**
(`jmd_n` / `tmd` / `jmd_c`), over which a scale's per-residue values are
averaged to produce one feature value. A feature ID is `PART-SPLIT-SCALE`
(e.g. `TMD-Segment(2,4)-ANDN920101`). Splits are residue-content-agnostic —
they map a part length to position indices, independent of the actual amino
acids.
_Avoid_: window (reserved for `AAWindowSampler`), segment (only one split type).

**split type**:
One of three split families, configured per-type via **split_kws**:
`Segment(i_th, n_split)` (the i-th of `n_split` contiguous chunks),
`Pattern(terminus, positions)` (fixed offsets from a terminus, bounded by
`len_max`), and `PeriodicPattern(terminus, step1/step2, start)` (alternating
periodic offsets). Exposed as `ut.LIST_SPLIT_TYPES`. The label generators
(`SplitRange.labels_*`) are **part-length independent** — they depend only on
`split_kws`, not on any part's length.
_Avoid_: split mode, split kind.

**empty split bucket**:
A `(split type, part)` pairing that produces zero splits. Because label
generation is part-length independent, this happens only when the split-type
*config itself* yields no labels — and in practice only for **Pattern**, when
`n_min * steps[0] > len_max` (the shortest pattern already overflows `len_max`;
Segment always yields ≥1, PeriodicPattern always yields splits once its two
steps validate). Such buckets are **silently dropped** from feature generation
(legacy CPP behavior, preserved for parity), so the run proceeds with the
remaining split types rather than erroring. `check_split_kws` emits a
`UserWarning` at validation time naming the offending Pattern config, since a
whole-type drop is almost always a user misconfiguration of `len_max` / `steps`.
_Avoid_: empty split, dropped feature.

### Numerical-mode CPP vocabulary

**dict_num**:
A `Dict[str, np.ndarray]` mapping `entry` to a per-residue numerical tensor of shape `(L, D)`. Generic value source for `CPP.run_num`: covers PLM embeddings, DSSP one-hots, PTM dummies, or any other per-residue numerical representation. Same shape contract as the `embeddings` argument of `EmbeddingPreprocessor.encode`; the rename to `dict_num` signals that the contents need not be PLM embeddings. When `dict_num` is supplied, the AA→scale lookup in `_filters/_assign.py:101` is bypassed; the per-protein tensor is sliced into parts and consumed directly. The accompanying `df_scales`/`df_cat` then *name* the D dimensions (e.g. `dim_0`, `DSSP_H`, `phospho_S`) for the redundancy filter and output columns. **All three preprocessor families emit `[0, 1]`-normalized values** — `StructurePreprocessor` / `AnnotationPreprocessor` encoders per the per-key `NORMALIZATION_RECIPES` (NaN for unresolved positions), and `EmbeddingPreprocessor.encode` per its `method=` normalizer. `CPP.run_num`'s default `max_std_test=0.2` pre-filter is calibrated for that `[0, 1]` convention; raw (unbounded) embeddings must be passed through `encode` first.
_Avoid_: embeddings (too narrow — covers only one source), num_tensor, per_residue_dict.

**CPP.run_num**:
A method of `CPP` whose per-residue value source is a pre-sliced numerical tensor (`dict_num_parts`) rather than an AA→scale lookup. **The whole `CPP` is constructor-bound** to `df_parts` + `df_scales` + `df_cat`; `run_num(dict_num_parts=, labels=, ...)` consumes the per-part NaN-padded tensors produced by `NumericalFeature.get_parts(df_seq, dict_num) → (df_parts, dict_num_parts)`. Hard invariant: `D == len(df_scales.columns) == len(df_cat)`; `df_scales`/`df_cat` *name* the D dims (the per-AA values are unused as a value source). `check_cat=True` (default) groups by `df_cat.category` in the redundancy filter. Same pipeline + output schema as `CPP.run`. _(Earlier drafts described run_num's value source as "per-call `df_seq` plus optional `dict_num`" — that is outdated; the value source is the constructor `df_parts` plus `dict_num_parts`.)_
_Avoid_: run_embed (misleading — also handles non-embedding inputs and pure sequences), run_v2.

**_filters/**:
Backend folder holding the canonical CPP pipeline (seq-mode AND numerical-mode). Per-residue values flow between stages as `dict[part] = (n_samples, L_part_max, D)` float32 tensors with NaN padding for short parts; downstream aggregation uses `np.nanmean`. Performance: split-position computation reused across D via numpy broadcasting (collapses the n_dims loop), and a *streaming pre-filter* keeps only the survivors of the `std_test` mask in memory so `add_stat` no longer recomputes feature values from scratch. Batching (`n_batches`) partitions over D, not scales/parts. The Cython acceleration lives in the sibling `_filters_c/` folder. Originally named `_filters_num/` during PR4-PR5 when a parallel legacy `_filters/` still existed; renamed to `_filters/` in PR6 after the legacy was removed.
_Avoid_: _embed_filters/ (too narrow), _filters_num/ (legacy PR5 name), _filters_v2/ (versioning is ephemeral).

### CPP sweep & diagnostics vocabulary

**CPPGrid**:
Public class that runs a **grid sweep** of `CPP` configurations in one call, orchestrating the full `get_df_parts` → `get_split_kws` → scales → `run`/`run_num` pipeline so the user skips the manual wiring. The dataset (`df_seq` + `labels`, plus `dict_num` for the numerical arm) is bound at construction; `run` takes four **stage-grouped param dicts**. Parallelizes **across** configurations (inner `run`/`run_num` stays serial); the default `backend="threads"` shares dataframes in-process and sidesteps the Py3.14/macOS spawn footgun. **Smart-sweeps:** configs differing only in `n_filter` run CPP **once at the max** and the rest are exact `head(n)` slices (greedy top-down redundancy filter ⇒ top-`n` invariant); `df_parts`/`split_kws` are built once per sub-config and reused.
_Avoid_: run_many (the rejected classmethod name), sweep (too generic), grid search (implies hyperparameter tuning with a model).

**stage-grouped param dicts** (`params_parts`, `params_split`, `params_scales`, `params_cpp`):
The four `CPPGrid.run` arguments, each feeding exactly one pipeline stage (`get_df_parts`/`get_parts`, `get_split_kws`, the list of `df_scales`, and `CPP.run`/`run_num`). Within a dict, a **`list` value is a swept axis** and a scalar is fixed; a list-valued knob (`steps_pattern`, `list_parts`) is swept by wrapping it in an outer list. The grid is the Cartesian product across all swept entries; `df_cat` is resolved internally from each `df_scales` ("df_scales is enough").
_Avoid_: param_ranges, param_grid, config list (earlier rejected shapes).

**df_params**:
The lightweight sweep-summary DataFrame returned by `CPPGrid.run` alongside `list_df_feat` (one row per configuration, `itertools.product` order). Scalar axes hold the literal value; object axes (`df_scales` and any list-valued knob) hold the **position index** into their candidate list — reconstructable from the inputs without storing heavy objects. Carries `n_warnings` and `n_errors` count columns. Naming parallel: `params_*` = what was *asked*; `df_params` = what was *run*.
_Avoid_: df_combos (earlier name), df_results (collides with `list_df_feat`), df_grid.

**last_filter_stats_**:
A plain dict set on a `CPP` instance after every `run`/`run_num` (also stashed in `df_feat.attrs["last_filter_stats"]`), recording the filter funnel: `n_candidates` (features generated before filtering), `n_after_prefilter`, `n_after_redundancy`, `n_final`. Exposed programmatically via `return_stats=True`. Two shortfalls are surfaced as mutually-exclusive warnings: a **sparse-config `UserWarning`** when `n_candidates < n_filter` (the config can't generate enough; small `n_jmd`/narrow `split_kws`), and a **filter-shortfall `RuntimeWarning`** when filtering removed too many.
_Avoid_: FilterStats (rejected typed record), filter_diagnostics.

**n_jobs contract**:
The unified parallelism convention: `1` = serial, `-1` = all cores, `N>1` = exactly N, `None` = optimized via `ut.resolve_n_jobs(n_jobs, n_work)`. The `options['n_jobs']` global (default `"off"`) overrides the per-call value when set. Inside `CPPGrid`, each configuration runs at `n_jobs=1` while the grid parallelizes across configurations.
_Avoid_: n_processes, n_workers (use `n_jobs` everywhere).

### Site-localization metrics vocabulary

**per-protein AP**:
Average precision computed per protein over its per-residue scores against that protein's positive positions, then aggregated across proteins (NaN-aware). The canonical site-localization ranking metric. A **tolerant** variant (`tolerance=±k`) counts a hit within `k` residues of a true positive as correct, for positional jitter. `aa.comp_per_protein_ap`.
_Avoid_: mAP (averaging convention differs), accuracy.

**detection metrics**:
Recall / precision / F1 / MCC at a fixed score threshold, pooled across all per-residue predictions — the "is the true site actually *called*?" question, distinct from ranking. `aa.comp_detection_metrics`.
_Avoid_: classification metrics (too generic), accuracy@threshold.

**bootstrap CI**:
A seeded percentile confidence interval over a per-protein metric vector, for honest small-N uncertainty reporting. `aa.comp_bootstrap_ci`.
_Avoid_: confidence band (plotting term), error bar.

**peak-preserving smoothing**:
NaN-aware triangular/gaussian smoothing of a per-residue score track that takes `max(smoothed, raw)` so true peaks are never attenuated — for windowed protease/PTM prediction where positional jitter is universal. Pure-numpy, seeded where stochastic. `aa.smooth_scores`.
_Avoid_: denoising, blurring (attenuates peaks).

**rank plot**:
A per-protein **max-score-vs-rank** scatter colored by group (substrate / hold-out / non-substrate) with optional threshold lines — the standard deployed-predictor sanity check. A standalone `aa.plot_rank` (deliberately not a `*Plot` method; pairs with the standalone `aa.metrics` functions).
_Avoid_: ranking plot (collides with `CPPPlot.ranking`, which ranks *features* from `df_feat`).

## Relationships

- A **df_seq** row contains one **entry** and one sequence; optionally a **pos column** cell of 1-based positions.
- A **test window** is extracted at a **P1 anchor** listed in the **pos column**.
- A **reference window** is sampled from a **candidate pool** and tagged with a **role** (workflow meaning) and a **strategy** (sampling provenance).
- A **control window** is a **reference window** produced by the **generator** in `sample_synthetic` (no source entry).
- Every row in a `segments`-mode output has an **entry_win**; for non-synthetic rows it is globally unique by construction across calls; for **control windows** it is unique per call only.
- The **identity filter** uses **test windows** as the anti-leakage reference; the **motif filter** uses a **PWM** as the gate.
- A CPP feature is one **split** applied to one **part** scored by one scale; an **empty split bucket** contributes no features and is silently dropped (with a validation-time warning for the Pattern case).

## Example dialogue

> **Dev:** "If I call `sample_different_protein`, do I get a row for every entry in `df_seq`?"
> **Domain expert:** "No — only entries with no **pos column** entries form the **candidate pool**. Entries with positives are excluded from sampling but still contribute their **test windows** to the **identity filter**, so they affect *which* candidates survive."
>
> **Dev:** "So a `Negative` row from `sample_same_protein` and an `Unlabeled` row from `sample_different_protein` carry different **role** tags, but I can tell which method produced them from the **strategy** column?"
> **Domain expert:** "Right. **Role** is workflow-level — what the row *means* in your downstream task; override the defaults if your workflow differs. **Strategy** is provenance — which sampling method produced it. Two rows can share a strategy but carry different roles, or vice versa."
>
> **Dev:** "What's a **control window** for?"
> **Domain expert:** "A synthetic window with no source protein. Useful as a null baseline, for composition-bias controls, or for benchmarking. It shares the segments-mode schema with the other outputs but its `entry_win` is only unique within one `sample_synthetic` call — dedupe across calls on the `window` string instead."
>
> **Dev:** "Why is `motif_pwm` DataFrame-only — can't I just pass an ndarray?"
> **Domain expert:** "Ndarray columns are implicitly alphabetical, and a wrong-order array silently gives wrong scores. We rejected that path. Pass `pd.DataFrame(arr, columns=ut.LIST_CANONICAL_AA)` if you only have an array; the column order then can't be wrong because pandas reindexes internally."

## Flagged ambiguities

- `mode` was used for both `sample_synthetic`'s polymorphic generator parameter AND for `output_mode`. **Resolved**: the synthetic parameter is now **generator**; `output_mode` retains its name (different axis — schema vs. recipe).
- `negative` was used informally for both labeled-negative rows and any non-test sampled row. **Resolved**: **reference window** is the umbrella term; **role** holds the workflow-specific meaning (`Negative`, `Unlabeled`, `Control`, …).
- `center` is used in backend code for the 0-based window-center index; in user-facing language and outputs we use **P1 anchor** / **source position** (1-based). **Resolved**: backend stays 0-based internally; frontend / output is 1-based throughout.
- `positions` was a column constant for `df_feat` (CSV string of feature positions); `pos` is the column constant for `df_seq`'s positive positions (list of ints). **Resolved**: distinct constants — `COL_POSITION` (`df_feat`, `str`) and `COL_POS` (`df_seq`, `list[int]`).
- `label-neutral` was a claim in the `AAWindowSampler` class docstring; in practice the API ships opinionated **role** and `label_ref` defaults. **Resolved**: framing is dropped; class docstring now states defaults explicitly assume PU-learning / hard-negative-mining workflows.
- `v1.1` / `v1.2` were used for **three different axes**: (a) the **package release version** (`pyproject.toml` / PyPI), (b) the **StructurePreprocessor feature-set revision** (rev 1 = DSSP+PDB; rev 1.1 = +AlphaFold — see ADR-0002 and the **feature key** entry), and (c) **git branch names** (`feat/structure-preprocessor-v1.2`). **Resolved**: the *only* authoritative version line is the **package version**, which follows semver — the line is `1.0.1 → 1.0.2 → 1.0.3 → 1.1.0` (the next, unreleased, minor; it ships the whole preprocessor family + CPPGrid + the site-localization metrics). The StructurePreprocessor's internal "rev 1 / rev 1.1" and any branch-name version are **not** package versions; never write a bare "v1.1" to mean a feature-set revision. See ADR-0010.
