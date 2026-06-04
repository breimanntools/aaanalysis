# ADR-0017 — `StructurePreprocessor.fetch_alphafold` + the preprocessor verb taxonomy

Status: Accepted — 2026-06-04

Relates to: ADR-0005 (feature-preprocessor family), ADR-0011 (encode/builder
naming). Amends ADR-0005's "Kept divergences" by writing the verb rules down.

## Context

A v1.1.0 API review of the four preprocessor classes (the unreleased Family-B
trio `Embedding` / `Structure` / `Annotation` plus the frozen Family-A
`Sequence`) surfaced four loose ends ADR-0005/0011 had left implicit:

- **A.** `Structure` exposes four `encode_*` while `Embedding` / `Annotation`
  expose a bare `encode`. The "why" was never stated as a rule.
- **B.** Example-notebook naming was inconsistent (`ep_*` / `sp_*` abbreviation
  prefixes for core vs full-name `structure_preprocessor*` for pro). *(Resolved
  separately: `stp_<method>` / `ap_<method>` under `examples/data_handling_pro/`.)*
- **C.** Only `Structure` / `Annotation` methods carried a per-call
  `verbose=None` override; `Embedding` and every other class in the package are
  constructor-only.
- **D.** The `get_*` raw-getter surface was uneven (`get_dssp` / `get_domains`
  exist; `encode_pdb` / `encode_pae` have no `get_` twin), and there was **no
  web-acquisition path for structures** at all — users had to download
  AlphaFold files by hand, a chore for agent-driven pipelines.

The family is still unreleased (first ships in v1.1.0, ADR-0010), so signature
changes carry no semver cost.

## Decision

**D1 — Add `StructurePreprocessor.fetch_alphafold`.** A bulk, web-acquisition
method that downloads each entry's AlphaFold-DB model file
(`AF-<entry>-F1-model_v4.{pdb,cif}`) **and** its PAE sidecar
(`AF-<entry>-F1-predicted_aligned_error_v4.json`) into one folder, saving them
under the names the existing `resolve_structure_path` / `resolve_pae_path`
resolvers already find — so a single call feeds `encode_pdb` / `encode_pae` /
`get_dssp` with no glue. It is the structure-side analog of
`AnnotationPreprocessor.fetch_uniprot`. Returns a per-entry status DataFrame;
`return_df=True` also appends an `alphafold_ok` column to `df_seq`. A 404 is the
soft "not in AF-DB" case governed by `on_failure={'nan','drop','raise'}`; other
network errors raise `RuntimeError`. `requests` is already a `[pro]` dependency,
so no new extra is added.

**D2 — Constructor-only `verbose`.** The per-call `verbose=None` override is
**removed** from all `Structure` / `Annotation` methods, matching `Embedding`
and the rest of the package (`self._verbose`).

**D3 — Write the verb taxonomy down** (in `CONTEXT.md` and below), so the
remaining divergences read as principled, not accidental:

| Verb | Role | I/O? | Rule |
|---|---|---|---|
| `fetch_*` | acquire | web | one per web resource (`fetch_uniprot`, `fetch_alphafold`) |
| `get_*` | acquire | local | exists **only** where the raw output is an independently-useful, curatable artifact (DSSP list, domain segmentation) |
| `encode` / `encode_*` | transform | none | **one `encode_*` per distinct raw source**; a bare `encode` when there is a single input or single canonical intermediate |
| `build_scales` / `build_cat` | transform | none | secondary AA-scale-path metadata |
| `ingest` / `register_feature` / `to_df_seq` | — | — | `Annotation`-only, by design |

This explains both A (the four `encode_*` are four distinct file sources;
`Embedding`'s `embeddings` dict and `Annotation`'s `df_annot` are single inputs)
and D (no `get_pdb` / `get_pae` because ATOM-field extraction and PAE collapse
produce no curatable intermediate; web retrieval is `fetch_`, never `get_`).

## Rejected alternatives

- **`get_pdb` / `get_pae` local-reader twins for symmetry.** They would be thin
  wrappers over `encode_*`'s own extraction with no curation value — symmetry
  for its own sake. The uneven getter surface is the rule (D3), not a gap.
- **A separate `fetch_pdb` (RCSB) in this pass.** Experimental PDBs have no PAE
  and are a distinct workflow; deferred. `fetch_alphafold` is the high-value,
  PAE-providing path.
- **`fetch_alphafold(with_pae=False)` / model-only default.** The PAE sidecar
  ships alongside the model in AF-DB and feeds `encode_pae`; bundling both in one
  call is what an agent wants. A standalone PAE fetch (`fetch_pae`) is rejected —
  PAE is not independently downloadable in a useful way.
- **Collapse the four `encode_*` into `encode(source=...)`.** Muddles which
  folder / feature set pairs with which source and loses clean per-source
  signatures.

## Consequences

- New public method on `StructurePreprocessor` (already top-level re-exported);
  no `__init__.py` or `pyproject.toml` change.
- New backend `data_handling_pro/_backend/struct_preproc/_alphafold.py`
  (mirrors `_uniprot.py`), with networkless unit tests
  (`test_alphafold_backend.py`, `test_structure_preprocessor_fetch_alphafold.py`).
- Per-call `verbose` removed across the pro classes; three existing tests
  updated to set constructor `verbose`.
- An `stp_fetch_alphafold` example notebook is pending with the rest of the
  v1.1.0 pro-notebook batch (its docstring `.. include::` resolves once authored).
- Known limit: only single-fragment (`F1`, `v4`) AlphaFold models are fetched;
  fragmented proteins are reported not-ok.
