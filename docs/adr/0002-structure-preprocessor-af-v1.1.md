# ADR-0002 — StructurePreprocessor v1.1: AlphaFold per-residue features

Status: **Accepted** (2026-05-25). Branch: `feat/structure-preprocessor-af`
(off `feat/structure-preprocessor` v1 tip `8ba5a292`). Three commits.

## Context

StructurePreprocessor v1 (commit `8ba5a292`) shipped DSSP + raw-PDB
encoders. Three pre-existing defects + an explicit user directive
("CPP works with any per-residue information. Please extend
StructurePreprocessor to work with any per-residue information we can get
from AlphaFold output") motivated this revision before v1 was released.

## Defect inventory (from grilling against CPP / _filters/ / CPPPlot)

1. **`build_scales` returned all-zero df_scales.** That broke
   `_redundancy_filter.filtering`'s `cor > max_cor` arm silently (every
   `df_scales.corr()` cell was NaN). The user would see a `df_feat` full
   of redundant features and discover only at downstream-model-training
   time. Verified live before the fix:
   `(stp.build_scales(features=[...])[0].values == 0).all() == True`.
2. **`CPPPlot.heatmap` raised on every StructurePreprocessor category.**
   `ut.DICT_COLOR_CAT` was hard-coded to the 8 AAontology categories;
   `check_match_dict_color_list_cat` raises on unknown categories. Every
   v1 category (`DSSP_SS`, `DSSP_ASA`, `Geometry`, `Flexibility`) was
   unknown to the dict. Same applied to EmbeddingPreprocessor's
   `PLM_cat_<k>`.
3. **Unbounded value ranges.** v1's encoders emitted `asa` in Å²,
   `bfactor` in Å², `phi`/`psi` in degrees — wildly outside the `[0, 1]`
   convention AAontology scales use. `CPP.run_num`'s default
   `max_std_test=0.2` pre-filter would silently reject most features.
   v1's integration test hid this with constant-per-protein synthetic
   values (zero within-class std).

## Decisions

### D1 — populate df_scales by per-AA averaging of the user corpus

Split v1's `build_scales` into:
- `build_pseudo_scales(df_seq, dict_num, features, return_std=False)` —
  per-AA mean of normalized per-residue values; mirrors
  `EmbeddingPreprocessor.build_pseudo_scales` recipe. Raises on missing
  corpus. Emits the same dataset-dependence `UserWarning`.
- `build_cat(features)` — corpus-free registry lookup, same shape as v1's
  second return.

Alternatives considered and rejected:
- Identity-shaped df_scales (1.0 on diagonal). Would make every dim
  orthogonal in corr space → cor gate never fires → same defect under a
  different mask.
- All-zero fallback when corpus missing. Reintroduces D1 silently. We
  raise instead.
- Single method with new required args. The split mirrors
  `EmbeddingPreprocessor`'s `build_pseudo_scales` + (separate) df_cat
  pattern and lets metadata-only consumers skip the corpus step.

### D2 — three locked category buckets in `ut.DICT_COLOR_CAT`

Locked palette (user-chosen, image-provided 2026-05-25):

```
Structure   '#2E6E5E'  deep teal-green   — every StructurePreprocessor output
Embeddings  '#6B4FB5'  indigo-violet     — every EmbeddingPreprocessor output
PTMs        '#B36BCB'  lilac-magenta     — reserved (no v1.1 implementation)
```

All StructurePreprocessor feature keys re-categorize under
`category='Structure'`; the fine-grained names (`DSSP_SS_3state`,
`AF_plddt_raw`, `Flexibility_bfactor`, `Geometry_centroid_dist`, …) move
into `subcategory`. EmbeddingPreprocessor's `cluster_pseudo_scales`
becomes `category='Embeddings'` with AAclust IDs combined into a
`Embeddings_cat<i>_subcat<j>` subcategory string.

The redundancy filter's `check_cat=True` arm now groups all Structure
features into one bucket; per-AA-mean df_scales (D1) makes the within-
bucket `cor > max_cor` gate active.

Alternatives considered and rejected:
- One color per fine-grained category (~8 buckets, my initial proposal):
  too noisy, user proposed the cleaner 3-bucket split via the palette
  image, locked. Future PTM work gets its own bucket without further
  palette negotiation.
- Keep PLM_cat_<k> categories: same CPPPlot ValueError, so they were
  already broken by the same defect.

### D3 — encoder-level min-max normalization to [0, 1]

Every feature gets a documented normalization recipe in
`feature_registry.NORMALIZATION_RECIPES`. The class docstring ships a
table of raw range → recipe → inverse so users can recover raw units.
Hard-coded saturation constants (no kwargs) keep the de-normalization
formula stable across sessions.

Consequence: v1's `asa` (absolute) and `phi_psi` (raw degrees) become
strictly redundant under min-max with rasa / phi_psi_sincos respectively.
v1.1 removes them from the registry.

### D4 — file-format support: `.pdb`, `.pdb.gz`, `.cif`, `.cif.gz`

Single resolver `_file_format.resolve_structure_path(folder, entry)`
tries the four extensions in order, decompresses `.gz` into a session
tempdir. PAE sidecars use a parallel resolver that also accepts the
AF-DB canonical filename `AF-<entry>-F1-predicted_aligned_error_v4.json`
(and its `.gz`).

### D5 — AF model-file feature set (commit 2)

9 new keys: `plddt`, `plddt_disorder`, `plddt_tier`, `chi1_sincos`,
`chi2_sincos`, `ca_centroid_dist`, `ca_centroid_dist_norm`,
`contact_count_8A`, `contact_count_12A`. `plddt` and `bfactor` are
intentionally separate keys (same B-factor-column read, different
subcategory labels); the corpus-derived df_scales lets the redundancy
filter spot the collision when both are requested.

### D6 — AF PAE sidecar feature set (commit 3)

7 new keys: `pae_row_mean`, `pae_row_min`, `pae_row_max`,
`pae_local_mean`, `pae_distal_mean`, `pae_asymmetry`, `pae_band_means`.
The `local_window` and `pae_band_edges` kwargs are at the method level
(consistent with the `feedback_preprocessor_api_shape` rule). New public
method `encode_pae(...)` keeps the "one source per encoder" contract
(separate file source = separate failure surface).

## Out of scope for v1.1

Explicitly deferred:
- HSE-up / HSE-down (orthogonal to depth; nice-to-have).
- DSSP H-bond donor/acceptor partner offsets.
- Disulfide-bond participation.
- Side-chain dihedrals chi3, chi4 (chi1+chi2 cover ≥85% of rotameric
  signal).
- Multi-chain PDBs (still v2).
- Promoting StructurePreprocessor defaults into `aa.options`.
- A built-in AF-DB bulk downloader (users still bring their own AF
  files).
- Global AF scores pTM / ipTM — not per-residue, don't fit `dict_num`.
- User-configurable normalization recipes (constants stay
  registry-locked).
- PTMs feature preprocessor — color slot reserved (`'#B36BCB'`) but no
  implementation in v1.1.

## Verification

```
pytest tests/unit/struct_analysis_pro_tests/ \
       tests/unit/data_handling_tests/ \
       tests/unit/cpp_tests/ \
       tests/unit/numerical_feature_tests/ \
       tests/unit/plotting_tests/test_dict_color_cat_categories.py -q
```

Parity suite remains green (no behavioral change to seq-mode /
numerical-mode CPP). Three-commit history on
`feat/structure-preprocessor-af`:

1. `685f6410` Fix v1 defects + file-format support for StructurePreprocessor v1.1
2. `71d1567c` Add 9 AlphaFold model-file features to encode_pdb
3. (this commit) Add encode_pae + ADR + example notebook + dev script
