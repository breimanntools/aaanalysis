# ADR-0005 — Feature-preprocessor family: consolidate into `data_handling_pro`, unify the protocol

Status: Accepted — 2026-05-29 (partially superseded by ADR-0011, 2026-06-01)

> **Update (ADR-0011, 2026-06-01):** before the v1.1.0 release, two decisions
> below were revised. `EmbeddingPreprocessor` gained an `encode` method (raw
> embeddings → `[0, 1]` `dict_num`), so the "Embedding has no acquire/encode step"
> premise no longer holds. The builder methods were renamed for output-matching
> uniformity: `build_pseudo_scales` → `build_scales`, and Embedding's
> `cluster_pseudo_scales` → `build_cat` (reversing the "Force … `build_cat` —
> false symmetry" rejection below). See ADR-0011.

## Context

Four public `*Preprocessor` classes had drifted into two jobs and three shapes:

- **Input encoder (Family A):** `SequencePreprocessor` (seq → one-hot / integer
  / windows). Released in v1.0.0 — frozen.
- **Feature preprocessors (Family B):** `EmbeddingPreprocessor`,
  `StructurePreprocessor`, `AnnotationPreprocessor` — each turns a per-residue
  data source into a `dict_num` (+ `df_scales` / `df_cat`) for `CPP.run_num`.
  All three were **unreleased** (only on the feature branch; they first ship in
  package v1.1.0 — see ADR-0010), so reshaping them carried no semver cost.

Two problems: (1) the Family-B classes were scattered — `Embedding` in core
`data_handling`, `Structure` in `struct_analysis_pro`, `Annotation` in
`annotation_pro` (two single-class pro packages); (2) they didn't share a shape
(`Embedding` was a static namespace; `Structure.encode_*` returned a tuple while
`Annotation.encode` returned a bare `dict_num`; `Embedding.build_pseudo_scales`
took `embeddings=` vs the others' `dict_num=`).

## Decision

**D1 — Consolidate into `data_handling_pro`.** The two single-class pro packages
`struct_analysis_pro` + `annotation_pro` collapse into one new pro sibling
`aaanalysis/data_handling_pro/` (holding `StructurePreprocessor` +
`AnnotationPreprocessor`), mirroring the existing `data_handling` ↔
`data_handling_pro` core/pro split convention (`seq_analysis` ↔
`seq_analysis_pro`, etc.). `EmbeddingPreprocessor` stays in core `data_handling`
(no heavy deps). The public API is unchanged — `aa.StructurePreprocessor` /
`aa.AnnotationPreprocessor` are still re-exported top-level behind
`try/except ImportError` + `missing_feature_stub`.

**D2 — Keep the name `data_handling`, don't rename to `preprocessing`.**
`data_handling` already holds loaders (`load_dataset`) and FASTA I/O that don't
fit "preprocessing"; renaming would scatter those into yet another package. The
existing package already contains preprocessors (`Sequence`, `Embedding`), so a
`data_handling_pro` holding the pro ones is consistent, not novel.

**D3 — All three Family-B classes are instance-based** (`xp = XPreprocessor(verbose=...)`).
`EmbeddingPreprocessor` changed from a static namespace to instance-based to join
the family; `Embedding.build_pseudo_scales(embeddings=)` was renamed to
`dict_num=` to match the shared vocabulary.

**D4 — `encode_*` returns a bare `dict_num` by default; `return_df=True` opts
into `(dict_num, df_seq_out)`.** Applied to all four `StructurePreprocessor.encode_*`
and to `AnnotationPreprocessor.encode` (whose echo carries an `encode_ok` status
column). Honors the documented "one `dict_num` per `encode_*`" intent while
preserving the per-entry status DataFrame (the only failure surface for
`encode_pdb` / `encode_pae`, which have no `get_*` raw counterpart).

## Rejected alternatives

- **Merge `Embedding` + `Annotation` into one `FunctionPreprocessor`.** They
  have disjoint method surfaces (Embedding has no acquire/encode step; Annotation
  is fetch/ingest/encode) and opposite category strategies
  (`cluster_pseudo_scales` discovery vs `build_cat` registry), and they are
  distinct categories in the locked palette (`Embeddings` ≠ `Functional sites` /
  `PTMs`). One class would be two-headed, not simpler.
- **Rename `Annotation` → `Function`.** Over-claims: UniProt separates
  "PTM/Processing" from "Function", and this class handles both — "Annotation" is
  the honest umbrella.
- **Move the pro preprocessors into core `data_handling`.** Would inject
  biopython / requests gating into a pure-core package and break the
  `core ↔ *_pro` convention. `data_handling_pro` keeps the boundary clean.
- **Force `Embedding.cluster_pseudo_scales` to be named `build_cat`.** False
  symmetry — its signature (thresholds, `random_state`) differs fundamentally
  from the registry-lookup `build_cat`. Kept as a distinct name.

## Kept divergences (domain-justified, not cleaned up)

> **Update (ADR-0017, 2026-06-04):** these divergences are now written down as
> an explicit verb taxonomy (`fetch_`=web / `get_`=local acquisition;
> `encode`/`encode_*`=pure transform with one `encode_*` per raw source), and a
> web structure-acquisition method `StructurePreprocessor.fetch_alphafold` was
> added. Per-call `verbose` was also dropped family-wide (constructor-only).

- `get_` (local tool) vs `fetch_` (web) acquisition verbs.
- `Structure` fuses acquire+encode (`encode_dssp` runs DSSP); `Annotation`
  separates (`fetch_uniprot` → `encode`) because `df_annot` is a curatable
  intermediate while raw DSSP arrays are not.
