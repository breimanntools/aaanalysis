# ADR-0004 — Domain segmentation: wrap ChainSaw + AFragmenter; Merizo stays BYO

Status: Accepted — 2026-05-25

## Context

`StructurePreprocessor.encode_domains` reads domain segmentations (Merizo /
ChainSaw / AFragmenter chopping strings) from per-entry files on disk. The
question: add a runtime wrapper so users don't have to run the tools by hand?

Three tools were surveyed (2026-05-25):

| Tool | pip-installable | Python API | Input | Wraps cleanly? |
|---|---|---|---|---|
| Merizo (PSIPRED) | No (git clone) | No (CLI-only) | PDB | ✗ awkward |
| ChainSaw | No (git clone + stride binary) | Importable modules | PDB / mmCIF | ✓ via subprocess |
| AFragmenter | Yes (`afragmenter`) | Yes | PAE matrix | ✓ via pip |

## Decision

**D1 — Wrap ChainSaw + AFragmenter; skip Merizo.** ChainSaw and Merizo are
co-equal SOTA on multi-domain segmentation; ChainSaw covers what Merizo covers.
ChainSaw operates on the PDB, AFragmenter on the PAE matrix — complementary
input modalities: `tool='chainsaw'` for experimental PDBs, `tool='afragmenter'`
for AlphaFold inputs.

**D2 — One `get_domains` method, `tool=` dispatch.** Per-tool kwargs
(`chainsaw_path`; `resolution` / `threshold`) are validated only when their tool
is selected. Mirrors the `get_dssp` → `encode_dssp` two-stage pattern: `get_*`
runs the tool inline and appends columns (`chopping` str, `domain_ok` bool);
`encode_*` consumes those columns, or runs the tool inline if they are absent.

**D3 — `encode_domains` accepts an in-memory `chopping` column.** If present
(typically from `get_domains`), the folder lookup is skipped and the string is
parsed per row. Same dual-mode pattern as `encode_dssp` / `encode_pdb`.

**D4 — AFragmenter ships in `[pro]`** as `afragmenter>=0.0.6`, lazy-imported;
absence raises a `RuntimeError` with an install hint. ChainSaw is **not** a
dependency — it is GPL-3 (vendoring into this BSD-3 package would violate the
license), so users clone it and pass `chainsaw_path=`; the wrapper validates the
path contains `get_predictions.py`, else raises a `RuntimeError` pointing at the
ChainSaw repo.

**D5 — Subprocess for ChainSaw, in-process for AFragmenter.** ChainSaw's stable
entry point is its CLI (`get_predictions.py`); subprocess via `sys.executable`
keeps the user's virtualenv intact and avoids in-process state pollution.
AFragmenter's `AFragmenter` class is published and pip-installed in the same
env, so in-process is the natural fit.

## Rejected alternatives

- **Wrap Merizo too.** Pins `torch==2.0.1` exactly (clashes with most modern
  downstream stacks) and is CLI-only — triples the install-pain surface for a
  capability ChainSaw already covers. Revisit if Merizo's install path improves.
- **A `[pro-domains]` sub-extra.** No repo precedent for sub-extras beyond
  `pro` / `docs` / `dev`; AFragmenter's ~10–15 MB footprint is not a different
  weight class from `[pro]`'s existing `shap` (~150 MB via numba + llvmlite).
- **Auto-detect the tool from folder contents**, or mix tools across rows.
  Explicit `tool=` is clearer; run `get_domains` twice and concat for a mixed
  corpus.

## Note — verify a PyPI package's identity before pinning

This wrapper first pinned `protein-domain-segmentation` as "AFragmenter's PyPI
name." That is an *unrelated* 185 MB torch-based segmenter (transitive deps
include `torch`, `MDAnalysis`); the real package is `afragmenter`. Lesson: check
wheel size + transitive deps + repo identity on PyPI before pinning a new dep.

## Out of scope

Merizo wrapper (see D1); parallel / batched tool invocation; caching tool
outputs across runs.
