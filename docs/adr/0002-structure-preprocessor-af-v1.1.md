# ADR-0002 — StructurePreprocessor v1.1: AlphaFold per-residue features

Status: Accepted — 2026-05-25

## Context

StructurePreprocessor v1 shipped DSSP + raw-PDB encoders. Three pre-existing
defects, plus a directive to "extend StructurePreprocessor to work with any
per-residue information we can get from AlphaFold output," motivated this
revision before v1 was released.

### Defect inventory (the *why* behind the API shape)

1. **`build_scales` returned all-zero df_scales**, silently breaking the
   redundancy filter's `cor > max_cor` arm (every `df_scales.corr()` cell was
   NaN). Surfaced only at downstream-model-training time.
2. **`CPPPlot.heatmap` raised on every StructurePreprocessor category.**
   `ut.DICT_COLOR_CAT` was hard-coded to the 8 AAontology categories;
   `check_match_dict_color_list_cat` raises on unknown categories. Every v1
   structure category (and EmbeddingPreprocessor's `PLM_cat_<k>`) was unknown.
3. **Unbounded value ranges.** v1 emitted `asa` in Å², `bfactor` in Å²,
   `phi`/`psi` in degrees — far outside the `[0, 1]` convention. `CPP.run_num`'s
   default `max_std_test=0.2` pre-filter would silently reject most features.

## Decision

**D1 — Populate df_scales by per-AA averaging of the user corpus.** Split v1's
`build_scales` into `build_pseudo_scales(df_seq, dict_num, features)` (per-AA
mean of normalized per-residue values; mirrors
`EmbeddingPreprocessor.build_pseudo_scales`; raises on missing corpus) and
`build_cat(features)` (corpus-free registry lookup).

**D2 — Locked category buckets in `ut.DICT_COLOR_CAT`** (user-chosen palette):
`Structure` `#2E6E5E` (every StructurePreprocessor output), `Embeddings`
`#6B4FB5` (every EmbeddingPreprocessor output), `PTMs` `#B36BCB` (reserved).
Fine-grained names (`Secondary structure (3-state)`, `AlphaFold pLDDT (raw)`,
…) move into `subcategory` following the AAontology convention so
`CPPPlot.feature_map` y-axis labels read cleanly. The redundancy filter's
`check_cat=True` arm groups all Structure features into one bucket; the per-AA-
mean df_scales (D1) makes the within-bucket `cor > max_cor` gate active.
*(Later extended to a fourth bucket, `Functional sites`, by ADR-0003.)*

**D3 — Encoder-level min-max normalization to `[0, 1]`.** Every feature gets a
documented recipe in `feature_registry.NORMALIZATION_RECIPES`; the class
docstring ships raw-range → recipe → inverse so users can recover raw units.
Saturation constants are hard-coded (no kwargs) to keep de-normalization stable.
Consequence: v1's `asa` (absolute) and `phi_psi` (raw degrees) become strictly
redundant under min-max with `rasa` / `phi_psi_sincos` and are **removed**.

**D4 — File-format support: `.pdb`, `.pdb.gz`, `.cif`, `.cif.gz`.** One resolver
tries the four in order, decompressing `.gz` into a session tempdir. PAE
sidecars use a parallel resolver that also accepts the AF-DB canonical filename.

**D5 — AF model-file feature set.** 9 keys: `plddt`, `plddt_disorder`,
`plddt_tier`, `chi1_sincos`, `chi2_sincos`, `ca_centroid_dist`,
`ca_centroid_dist_norm`, `contact_count_8A`, `contact_count_12A`. `plddt` and
`bfactor` stay separate keys (same column read, different subcategory); the
corpus-derived df_scales lets the redundancy filter spot the collision when both
are requested.

**D6 — AF PAE sidecar feature set.** 7 keys: `pae_row_mean / min / max`,
`pae_local_mean`, `pae_distal_mean`, `pae_asymmetry`, `pae_band_means`. A new
public `encode_pae(...)` keeps the "one source per encoder" contract.

## Rejected alternatives

- **Identity-shaped df_scales** (1.0 on diagonal): every dim orthogonal in corr
  space → cor gate never fires → defect 1 under a different mask.
- **All-zero fallback when corpus missing:** reintroduces defect 1 silently —
  we raise instead.
- **One color per fine-grained category** (~8 buckets): too noisy; the locked
  3-bucket palette is cleaner and leaves room for future categories.

## Out of scope for v1.1

HSE-up / HSE-down; DSSP H-bond partner offsets; disulfide participation;
side-chain χ3/χ4; multi-chain PDBs; promoting defaults into `aa.options`; a
bulk AF-DB downloader; global pTM / ipTM (not per-residue); user-configurable
normalization recipes.
