# ADR-0004 — Domain segmentation: ChainSaw + AFragmenter wrapped; Merizo stays BYO

Status: **Accepted** (2026-05-25). Branch: `feat/structure-preprocessor-v1.2`
(off the v1.1 tip `f78acf00`). Adds `StructurePreprocessor.get_domains(...)`
in v1.2 commit 4 alongside the BYO file reader from commit 3 (`7b1b55cd`).

## Context

v1.2 commit 3 shipped `stp.encode_domains(df_seq, domain_folder, ...)` —
a BYO-segments reader that parses Merizo / ChainSaw / AFragmenter chopping
strings from per-entry files on disk. The user asked whether this is
user-friendly enough or whether we should add a runtime wrapper.

Three domain-segmentation tools were on the table. Comparative facts
(2026-05-25 survey):

| Tool | Citations | pip-installable | Python API | Input | Wraps cleanly? |
|---|---|---|---|---|---|
| **Merizo** (PSIPRED) | ~51 (Nat Commun 2023) | No (git clone) | No (CLI-only) | PDB | ✗ awkward |
| **ChainSaw** (Wells et al.) | ~51 (Bioinformatics 2024) | No (git clone + stride binary) | Importable Python modules | PDB / mmCIF | ✓ via subprocess |
| **AFragmenter** (Verwimp et al.) | New (Bioinformatics 2025) | **Yes** (`afragmenter`) | **Yes** (`AFragmenter.cluster(...)`) | PAE matrix | ✓ via pip |

## Decisions

### D1 — Wrap ChainSaw + AFragmenter; skip Merizo

ChainSaw and Merizo are co-equal SOTA on multi-domain segmentation
(both ~51 citations, both used as benchmarks in DPAM-AI / Foldclass /
SPAED). On the blind eval in the ChainSaw paper, ChainSaw is preferred
2:1 over UniDoc on AF models, with ~94% IoU vs. CATH. **ChainSaw
covers what Merizo covers**; wrapping both is redundant and triples our
install-pain surface (Merizo pins `torch==2.0.1` exactly, which clashes
with most modern downstream stacks).

AFragmenter is the only PyPI-installable tool with a Python API. It
operates on the PAE matrix from AlphaFold (already canonical in v1.1's
`encode_pae` story) rather than the PDB file, so the two wrapped tools
cover **complementary input modalities**:

- `tool='chainsaw'` for experimental PDBs (no PAE needed).
- `tool='afragmenter'` for AlphaFold inputs (uses PAE).

### D2 — Unified `get_domains` method with `tool=` dispatch

One method, two paths. Per-tool kwargs (`chainsaw_path` for ChainSaw,
`resolution` / `threshold` for AFragmenter) sit at the method level and
are validated only when their tool is selected. This mirrors the
`get_dssp` → `encode_dssp` two-stage pattern: `get_*` runs the tool
inline and appends list / string columns to `df_seq`; `encode_*`
consumes those columns OR runs the tool inline if they're missing.

The new method returns the input `df_seq` with two appended columns:
- `chopping` (str) — Merizo/ChainSaw common-format chopping string.
- `domain_ok` (bool) — `True` if the tool returned a non-empty chopping.

### D3 — `encode_domains` extended to accept in-memory `chopping`

When `df_seq` already has a `chopping` column (typically populated by
`get_domains`), `encode_domains` skips the folder lookup and parses
the in-memory string per row. Same dual-mode pattern as
`encode_dssp` / `encode_pdb` (use pre-computed list columns if present,
fall back to the file path otherwise).

### D4 — Dependency layering: AFragmenter ships in `[pro]`

AFragmenter is added to the existing `aaanalysis[pro]` extra as
`afragmenter>=0.0.6` (its actual PyPI distribution name). Lazy-imported
by the wrapper; absence raises a `RuntimeError` with a friendly install
hint when `tool='afragmenter'` is requested.

The added transitive footprint is ~10–15 MB (`python-igraph` ~10 MB
native binary, `rich` + `rich-click` pure-Python; `numpy`, `matplotlib`,
`biopython`, `requests` are already in core or `[pro]`). `[pro]` already
carries `shap` (numba + llvmlite, ~150 MB), so AFragmenter is not
categorically different in weight class and does not warrant its own
sub-extra. The repo has no precedent for sub-extras beyond
`pro` / `docs` / `dev`.

**Historical note — wrong-PyPI-name incident (2026-05-25).** v1.2 commit
4 originally introduced a sub-extra `[pro-domains]` pinning
`protein-domain-segmentation>=0.0.6` as "AFragmenter's PyPI name." PyPI
verification a day later showed that `protein-domain-segmentation` is
an *unrelated* 185 MB torch-based segmenter (transitive deps include
`torch`, `rotary_embedding_torch`, `einops`, `MDAnalysis`). The real
AFragmenter ships as `afragmenter`. v1.2 commit 7 fixes the name and
folds the dep into `[pro]` in the same commit. Lesson: when adding a
new PyPI dep, verify wheel size + transitive deps + repo identity on
PyPI before pinning.

ChainSaw is NOT a Python dependency. Users clone it themselves and
pass the path as `chainsaw_path=...`. The wrapper validates the path
exists and contains `get_predictions.py`; otherwise raises a
`RuntimeError` pointing at https://github.com/JudeWells/Chainsaw.
ChainSaw is GPL-3 licensed; vendoring its code into this BSD-3 package
would be a license violation.

### D5 — Subprocess for ChainSaw, in-process import for AFragmenter

ChainSaw's CLI (`get_predictions.py`) is the documented and stable
entry point; its internal Python API isn't published. Subprocess via
`sys.executable` keeps the user's virtualenv intact and avoids any
in-process state pollution. AFragmenter's `AFragmenter` class is
documented and pip-installed in the same env, so in-process is the
natural fit.

## Out of scope for v1.2 commit 4

- Merizo wrapper: see D1. If the install path becomes friendlier in a
  future Merizo release, revisit.
- Auto-detection of which tool to use based on what's present in the
  input folder. Explicit `tool=` is clearer.
- Parallel / batched tool invocation. v1.2 runs tools sequentially per
  entry; a corpus of 1000 proteins takes O(N × tool runtime) seconds.
- Caching of tool outputs across runs. Users can pre-run + dump to
  files for the BYO path if they want a persistent cache.
- Mixing tools across rows (some via ChainSaw, some via AFragmenter).
  Use two `get_domains` calls and concat the results yourself.

## Verification

```bash
pytest tests/unit/struct_analysis_pro_tests/test_structure_preprocessor_get_domains.py -q
pytest tests/unit/struct_analysis_pro_tests/ tests/unit/cpp_tests/ -q
```

Expected: 29 new tests pass (mocked both tools — no actual AFragmenter
install or ChainSaw clone required for CI). Full struct + parity suite
green.

## Commit history reference

- `7b1b55cd` (v1.2 commit 3) — BYO file reader `encode_domains`.
- (this commit, v1.2 commit 4) — `get_domains` wrapper + `encode_domains`
  in-memory chopping path.
