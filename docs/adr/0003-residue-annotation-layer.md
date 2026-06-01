# ADR-0003 — Residue-level annotation layer (AnnotationPreprocessor)

Status: Accepted — 2026-05-25

## Context

A per-residue PTM / functional-site annotation layer was requested: fetch
residue-level features from UniProt (and ingest user/predictor labels), map
everything into one canonical schema, and surface it to CPP. The original brief
framed this as "extend `SequencePreprocessor`" feeding the `AAWindowSampler`
test/reference window split; grilling against the codebase overturned that
framing and produced the decisions below.

## Decision

**D1 — A new pro class, not an extension of `SequencePreprocessor`.**
`SequencePreprocessor` is a stateless `@staticmethod` namespace. The instance-
based `encode_* → dict_num`, `features=[...]`, registry API is the
`StructurePreprocessor` / `EmbeddingPreprocessor` pattern. So this ships as
`AnnotationPreprocessor` in `aaanalysis/data_handling_pro/`, gated via
`missing_feature_stub` (trigger dep: `requests`, in `[pro]`).

**D2 — `dict_num` is the primary output, not the window split.** The deliverable
is a canonical per-residue `df_annot` + `encode → dict_num {entry:(L,D)}` for
`CPP.run_num`, stackable with DSSP / PAE / embedding tensors via
`combine_dict_nums`. Here a PTM is a *feature dimension*, not a window label;
test/reference labels come from the user's own `df_seq` (`run_num(labels=...)`).

**D3 — Value semantics: float in `[0, 1]`; absent = `0.0`; NaN = unresolved.**
Presence features emit `1.0` / `0.0`; predictor features carry their score in
`[0, 1]` (a nullable `score` column unifies binary and continuous). Non-
annotated in-coverage residues are `0.0`, not NaN — `run_num` aggregates with
`np.nanmean`, so NaN would zero out a window's signal. NaN is reserved for
genuinely unresolved positions.

**D4 — UniProt-canonical frame + residue-identity guard (raise).** `df_annot`
stores 1-based UniProt-canonical positions plus the expected residue `aa`. At
encode time the position maps into `df_seq[sequence]` and asserts
`sequence[pos-1] == aa`; on mismatch it **raises** by default (`on_mismatch`
toggle: `drop` / `warn`). Converts the off-by-isoform silent-corruption risk
into a loud failure. No fuzzy realignment. User-ingested rows with empty `aa`
skip the guard.

**D5 — Bond features expand to two endpoints + `bond_id`.** `DISULFID` /
`CROSSLNK` expand to two single-residue rows sharing a `bond_id`. Per-residue
profiling treats each Cys independently; `bond_id` keeps paired-window features
possible later without a schema migration.

**D6 — Cleavage from processing boundaries; SITE is description-routed.**
Signal / propeptide / transit cleavage P1 anchors come from the SIGNAL / PROPEP
/ TRANSIT span **end** (exact, no text parsing). The `SITE` grab-bag is routed
by description regex (`cleav`) — only matches become `cleavage_site`; other
SITEs are dropped, never blanket-dumped into PTM positives.

**D7 — Evidence default = {ECO:0000269, ECO:0007744}.** Default
`evidence='manual'` keeps experimental (0000269) AND combinatorial-manual
(0007744, the backbone of large-scale phosphoproteomics); it excludes
by-similarity (0000250). Toggles: `'experimental'` (0269 only), `'all'` (no
filter). Raw ECO is always retained in the `evidence` column.

**D8 — Two top-level categories; new color for Functional sites.** `'PTMs'`
keeps its reserved `#B36BCB`; a new top-level `'Functional sites'` is added at
`#2C6E9E` (deep ocean-blue) — explicitly **not** `#6B4FB5`, which is already
`Embeddings`. Both go in `ut.DICT_COLOR_CAT` and `LIST_CAT`, extending the
ADR-0002 palette. `run_num`'s `check_cat=True` default groups by these, so
phospho-type vs binding-type features land in distinct redundancy buckets.

**D9 — Window-split export via an `aa_context` eligibility mask.**
`to_df_seq(df_seq, df_annot, feature_type, ...)` projects annotations onto a
`df_seq` for the seq-mode `AAWindowSampler` path (where a PTM *is* the window
label). Since `AAWindowSampler` is not residue-type-aware, the export encodes
the reference-set rules into an `aa_context` per-residue mask (`'1'` = eligible
anchor) consumed via `context_in='1'`:

- `match_residue_type=True` (default) restricts eligible anchors to the residue
  types of the positives (e.g. {S,T,Y} for phospho) — the residue-type-matched
  negative. Set `False` for residue-type-agnostic classes (e.g. predictor
  hotspots): any non-annotated residue then qualifies.
- `exclude_other_annotations=True` (default) drops residues carrying any other
  `feature_type` from the reference pool (anti-contamination).
- Positives go in the `pos` column; terminal windows that don't fully fit are
  dropped by `AAWindowSampler` (no new edge logic).

## Rejected alternatives

- **Window-split-primary** (vs D2): defers the `dict_num` symmetry that makes
  annotations compose with structure / embeddings.
- **Both exporters in one PR:** largest surface, half-finished risk.
- **One color per fine-grained category** (vs D8): too noisy; the two-bucket
  split keeps the redundancy filter legible.

## Out of scope

- An AF-DB-style bulk UniProt downloader / caching layer.
- Coordinate realignment across isoforms (the D4 guard raises instead).
