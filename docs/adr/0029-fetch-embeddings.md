# ADR-0029 — EmbeddingPreprocessor.fetch_embeddings + fetch_* covers model-weight acquisition

Status: Accepted — 2026-06-12

Relates to: ADR-0017 (preprocessor verb taxonomy: `fetch_`/`get_`/`encode_`/`build_`),
ADR-0011 (`EmbeddingPreprocessor.encode` → `[0,1]` `dict_num`), ADR-0005 (preprocessor
family). Amends ADR-0017's `fetch_*` row to cover acquisition of a *model* (the Hugging
Face Hub), not only of data *files*.

## Context

`EmbeddingPreprocessor.encode` (ADR-0011) consumes raw per-residue PLM embeddings the
user must compute *externally* — its docstring says "AAanalysis does not run the model."
Users repeatedly asked for an in-package path to obtain ESM-2 / ProtT5 embeddings without
hand-rolling a `transformers` loop, both as a per-protein matrix (for clustering /
`TreeModel`) and as per-residue tensors (to feed `encode` → `CPP.run_num`). Two facts
shape the design:

1. **No clean per-accession precomputed-embedding endpoint exists.** UniProt publishes
   ProtT5 embeddings only as bulk per-proteome / Swiss-Prot HDF5 files plus a
   search-results bulk export — there is no `…/embeddings/<accession>` GET analogous to
   AlphaFold-DB's per-accession model URLs that `fetch_alphafold` (ADR-0017) relies on.
   A real direct fetch is therefore a bulk-cache workflow, ProtT5-only, covered-proteomes
   only. (Refs: https://www.uniprot.org/help/embeddings, https://www.uniprot.org/help/downloads.)
2. **The compute path needs `torch` + `transformers`** (+ `sentencepiece` for ProtT5,
   `huggingface_hub` for the weight cache) — none currently a dependency, and ~1–2 GB,
   far heavier than anything in `[pro]`.

The verb taxonomy (ADR-0017) defines `fetch_*` as "acquire / web / one per web resource."
Downloading model weights from the HF Hub *is* a web acquisition, so the verb fits — but
ADR-0017's examples were all data-file fetches, so this ADR records the broadened reading.

## Decision

**D1 — Add `EmbeddingPreprocessor.fetch_embeddings`.** A web-acquisition method that
downloads a curated PLM from the Hugging Face Hub and computes embeddings:
`mode="protein"` → a bare `(n, D)` ndarray row-aligned to `df_seq`; `mode="residue"` →
`{entry: (L, D)}` raw-float arrays that feed `encode` → `CPP.run_num`. It returns **raw**
(un-normalized) embeddings — normalization remains `encode`'s job (ADR-0011), preserving
`fetch`=acquire / `encode`=transform. `return_df=True` appends an `embeddings_ok` column;
`on_failure={'nan','drop','raise'}` mirrors the `fetch_alphafold` / `encode_*` sibling
contract.

**D2 — `fetch_*` covers model-weight acquisition.** ADR-0017's `fetch_*` row is read to
include "acquire a model from a hub", not only "download data files". This is the only
`fetch_*` whose web resource is executable weights; documented so it reads as principled,
not a category error.

**D3 — Pooling is explicit, with a reusable helper.** `mode="protein"` pools per-residue
embeddings to one vector via `pooling={'mean','cls','max'}` (default `'mean'`). The
residue→protein reduction is factored into the public helper
`EmbeddingPreprocessor.pool_embeddings(embeddings, pooling=...)` so a user who fetched
`mode="residue"` (or supplied their own) can pool explicitly. `'cls'` uses a model's
leading special token and is rejected for models without one (e.g. ProtT5). The richer
"pooling" — per-residue embeddings → `encode` → `CPP.run_num` — is CPP itself; the helper
is the simple statistical counterpart.

**D4 — Curated 8-model registry with hardware-grounded recommendation.** A backend
registry (ESM-2 t6_8M … t36_3B, ESM-1b, ProtT5-XL-U50 enc-half, ProstT5) carries
params / embedding-dim / RAM+VRAM footprint / input cap / license. `fetch_embeddings`
detects device + memory (`torch.cuda.mem_get_info` / `os.sysconf`; no `psutil`) and emits
a `RuntimeWarning` ("may crash") with a smaller-model recommendation when the chosen
model's estimated footprint exceeds available memory; `allow_oversized=True` bypasses.

**D5 — New `[embed]` extra; class stays in core, method lazy-imports.** `torch` /
`transformers` / `sentencepiece` / `huggingface_hub` go in a NEW `[embed]` extra (NOT
`[pro]`, which would ~50× its weight). `EmbeddingPreprocessor` stays in core
(`data_handling/`) — its three light methods (`encode`/`build_scales`/`build_cat`) must
import on a base install — and `fetch_embeddings` / `pool_embeddings` (when it needs
`torch`) lazy-import the heavy deps inside the method, raising the install hint
(`_EXTRA_MODULES["embed"]`) at call time. Follows the in-method degradation precedent of
`StructurePreprocessor`'s afragmenter / msms probes.

**D6 — Defer direct fetch of precomputed embeddings.** `source=` accepts
`{'auto','compute'}` now; `'uniprot'` is reserved and raises a `ValueError` until a
bulk-HDF5 cache path is designed. No API break when added.

## Rejected alternatives

- **A `fetch_alphafold`-style per-accession HTTP embedding fetch.** No such UniProt
  endpoint exists (Context #1); building one would be a bulk-HDF5 downloader in disguise,
  ProtT5-only — deferred (D6), not shipped as "fetch".
- **Put `torch`/`transformers` in `[pro]`.** ~50× the install weight for every existing
  pro user who doesn't want embeddings; isolate in `[embed]` (D5).
- **Relocate `EmbeddingPreprocessor` to `data_handling_pro`.** Breaks core import of the
  three dep-free methods for one heavy method; lazy in-method import is the sanctioned
  lighter tool (`pro-core-boundary.md`).
- **Return normalized embeddings from `fetch_embeddings`.** Conflates acquire with
  transform; normalization is corpus-dependent and owned by `encode` (ADR-0011).
- **A `seed` parameter.** Embedding inference is deterministic in eval mode; a no-op seed
  would mislead (`reproducibility.md` applies to functions that *use* randomness).

## Consequences

- New public methods on the already-exported core `EmbeddingPreprocessor`; no new
  top-level symbol, so no `__all__` change — but `_EXTRA_MODULES` gains an `"embed"` key
  (CONFIRM-FIRST: `__init__.py`) and `pyproject.toml` gains an `[embed]` extra
  (CONFIRM-FIRST: pyproject + new extra).
- New backend module `data_handling/_backend/embed_preproc/fetch.py` (`REGISTRY`,
  `recommend_model`, `detect_hardware`, `pool_residue_`, `compute_embeddings_`) with
  networkless / mocked unit tests.
- New `examples/data_handling/ep_fetch_embeddings.ipynb` example; CI runs it with a tiny
  mocked model (no real weight download), since nbmake / coverage cannot pull GB-scale
  weights — or the notebook is excluded from the nbmake glob.
- Known limits: ESM-1b 1022-residue cap (registry-enforced); footprint estimates are
  floors (peak activation memory scales with batch × length); `source='uniprot'` not yet
  implemented.
