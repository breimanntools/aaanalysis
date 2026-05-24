# Design sketch: `CPP.run_num`

> **Status:** Design accepted in ADR-0001 (`docs/adr/0001-cpp-run-num.md`); not
> yet implemented. This file captures the *how*; the ADR captures the *why*
> and the rejected alternatives. The file is still named `run_embed.md` for
> git-history continuity — the design itself was renamed `run_num` during the
> grill-with-docs session on 2026-05-21.

## Motivation

`EmbeddingPreprocessor` produces a `(20, D)` `df_scales_emb` and a `(D, 5)`
`df_cat_emb` by **context-free averaging** of per-residue PLM embeddings.
Those tables plug into the existing `CPP.run` pipeline today (see the
end-to-end test `test_pipeline_into_cpp_run` in
`tests/unit/data_handling_tests/test_ep_cluster_pseudo_scales.py`), but the
averaging step throws away the very thing that makes PLM embeddings
interesting: **positional context**. Two leucines in different sequence
positions, with very different local neighbourhoods, get the same scale
value.

`CPP.run_num` consumes a per-residue numerical tensor directly, preserving
positional context all the way through feature extraction. The same path
also accepts pure sequences (no tensor → fall back to the constructor's
`df_scales` for the AA→scale lookup), which lets `CPP.run_num` and `CPP.run`
run head-to-head on the same input. The tensor is *generic*: PLM
embeddings, DSSP one-hots, PTM dummies, and any other per-residue numerical
representation share the same `Dict[entry, (L, D)]` shape contract — see
`CONTEXT.md` → "Numerical-mode CPP vocabulary" for the `dict_num` glossary
entry.

## Proposed signature

```python
class CPP(Tool):
    def run_num(
        self,
        df_seq: pd.DataFrame,                          # per-call (not constructor) — entry + sequence (+ COLS_SEQ_POS for variable TMDs)
        dict_num: Optional[Dict[str, np.ndarray]] = None,  # entry -> (L, D) per-residue tensor; None ⇒ AA→scale lookup
        df_scales: Optional[pd.DataFrame] = None,      # (20, D) when dict_num is set (names dims & drives max_cor); else uses self.df_scales
        df_cat: Optional[pd.DataFrame] = None,         # (D, 5) when dict_num is set (categorical filter); else uses self.df_cat
        labels: ut.ArrayLike1D = None,
        label_test: int = 1,
        label_ref: int = 0,
        # …all of CPP.run's existing filter / split / region args carry over…
        n_filter: int = 100,
        max_overlap: float = 0.5,
        max_cor: float = 0.5,
        start: int = 1,
        tmd_len: int = 20,
        jmd_n_len: int = 10,
        jmd_c_len: int = 10,
        n_jobs: Optional[int] = None,
        vectorized: bool = True,
        n_batches: Optional[int] = None,               # partitions over D, not scales
    ) -> pd.DataFrame:
        ...
```

Constructor expectations (unchanged from today):
- `CPP.__init__` still requires `df_parts` and `df_scales`; `run_num` does
  not bypass them. For head-to-head dev, build the CPP from a `df_seq`'s
  `get_df_parts` output and pass that same `df_seq` into `run_num`.
- When `dict_num` is supplied, the per-call `df_scales`/`df_cat` override
  the constructor's tables for naming the D dimensions and driving the
  redundancy / categorical filters. When `dict_num is None`, the
  constructor's tables are used.

Return contract: a `df_feat` DataFrame identical in shape and column meaning
to what `CPP.run` returns today, so downstream callers (`TreeModel`,
`CPPPlot`, …) don't need to change. In seq-only mode (`dict_num=None`), the
returned `df_feat` is **bit-identical** to `CPP.run`'s output — enforced by
a parity test fixture.

## Required backend changes

A new sibling folder
`aaanalysis/feature_engineering/_backend/cpp/_filters_num/` mirrors
`_filters/` stage-for-stage. The existing `_filters/` modules are not
touched — both paths coexist for parity / head-to-head profiling. Stage
map:

| Stage | `_filters/` (today) | `_filters_num/` (new) |
|---|---|---|
| Residue-value assignment | `_assign.py` — per-(scale, part) `(n, L+1)` float32 dict | `_assign.py` — `dict[part] = (n, L_part_max, D)` float32; NaN-padded |
| Pre-filter stats | `_stat_filter.py` — per-(scale, part, split) `nanmean`, stats only | `_stat_filter.py` — vectorized over D, **streams** per-sample feature values to the pre-filter |
| Pre-filter selection | `_pre_filter.py` — std_test mask + top-K by abs_mean_dif | `_pre_filter.py` — same logic; receives cached (n, n_pre_filter) matrix |
| Add stat (AUC, p, FDR) | `_add_stat.py` — calls `get_feature_matrix_` (full recompute) | `_add_stat.py` — uses cached matrix, no recompute |
| Redundancy filter | `_redundancy_filter.py` — greedy O(n²) Python loop | Ported verbatim. `# DEV:` note for future vectorization |
| Progress / MP | `_progress.py` | Reused as-is |

The core change in `_filters_num/_assign.py`:

1. Drop the per-AA mask sweep (`for aa, idx in aa_to_idx.items(): aa_idx_matrix[X_seq == aa] = idx`) and the per-(scale, part) loop.
2. When `dict_num is None`: build `aa_to_idx` once, look up `scale_matrix[aa_idx, :]` to produce `(n, L_part_max, D)` directly (D = n_scales).
3. When `dict_num is not None`: slice per-protein tensors `dict_num[entry]` into per-part `(n, L_part_max, D)` using the same `tmd_start/stop` / `jmd_n_len`/`jmd_c_len` semantics as `SequencePreprocessor.get_df_parts`.
4. Pad variable-length parts with NaN; downstream uses `np.nanmean`.

Split / pattern / segment grammar is **unchanged** — `_split.py` (and
`SplitRange`) are imported as-is. Only the value source per residue
position differs.

The dominant perf win comes from two changes:

- **Collapse the n_dims loop.** Split positions depend on `(part, split_type)`, not on the scale. Today they are recomputed per scale; with the `(n, L_part_max, D)` shape, they are computed once per part and broadcast across D via numpy fancy-indexing.
- **Streaming pre-filter.** Each part-slab produces `(n_samples, n_features_for_this_part)` per-sample values; the std_test mask is applied immediately and only survivors stay in memory. `add_stat_num` then takes a column slice of the cached matrix — no `get_feature_matrix_` recompute.

`n_batches` in `_filters_num/` partitions over the D axis (not scales);
this gives the same memory-tunable lever as today, with a single tensor
shape.

## Bench plan (A vs B)

`dev_scripts/bench_filters_num.py` (to be added) runs two variants of
`_filters_num/_assign.py` against the same CPP fixture and asserts
bit-identical `df_feat`:

- **A:** `dict[part] = (n, L_part_max_part, D)` — per-part tensor; preferred default.
- **B:** Single `(n, n_parts, L_part_max_global, D)` 4D tensor — padded across parts.

Bench reports timing and peak RSS for both, plus a baseline `CPP.run` call.
The loser is deleted after one PR cycle.

## Data flow

```
dict_num : Dict[entry, (L, D)]                          # supplied or built from df_seq + df_scales
  │
  ├── slice per part (tmd, jmd_n, jmd_c) ── (n, L_part_max_part, D)   # NaN-padded
  │       │
  │       └── for each (part, split): mean over residue axis
  │           → (n, n_splits, D)
  │              │
  │              └── stream into pre-filter; cache survivors
  │                  → (n, n_pre_filter) matrix in memory
  │
  └── df_cat drives the categorical / redundancy filter as today
      df_scales.corr() drives the max_cor filter as today
```

## Alignment with `EmbeddingPreprocessor`

- `build_pseudo_scales` stays useful even with `run_num`: it remains the
  input to `cluster_pseudo_scales`, which produces `df_cat` (the
  per-dimension category table needed by `run_num`'s redundancy filter).
- `df_scales_emb` becomes optional in the `dict_num` path: only the
  pseudo-categories are required for the categorical filter, but
  `df_scales_emb` is still useful as the `(20, D)` table that drives
  `max_cor` (its `.corr()` is the inter-dim correlation matrix).
- `return_std=True` from `build_pseudo_scales` gives `df_stds_emb`, which
  enables std-aware clustering — orthogonal to `run_num`.

## Open questions

1. ~~**Std-aware clustering.**~~ **Shipped** — see
   `EmbeddingPreprocessor.cluster_pseudo_scales(df_stds_emb=...)`.
2. **`dict_num` slicing alignment with `df_parts` strings.** Per-residue
   tensors live on the full sequence; `_filters_num/_assign.py` slices
   them into parts using `df_seq`'s `COLS_SEQ_POS` (`tmd_start/stop`) plus
   `jmd_n_len`/`jmd_c_len`. A parity test must assert the staged tensor's
   length-axis matches `get_df_parts(df_seq, …)` row-for-row.
3. **Per-position vs per-region aggregation.** v1 uses `mean` (matches
   current Segment). Expose richer aggregators (max, attention-weighted)
   behind a flag in a follow-up — out of scope for v1.
4. **AAclust k-optimization metric.** Unchanged — still Pearson; deeper
   AAclust refactor tracked separately.
5. ~~**Memory.**~~ **Resolved** via streaming pre-filter + D-axis batching
   (`n_batches`). Peak RSS bounded by `(n_samples × D_chunk × L_part_max)`
   during assign, then `(n_samples × n_pre_filter)` during add_stat.
6. **Caching.** Embedding computation is the expensive step in PLM
   workflows. Out of scope for v1; revisit if usage grows.

## Non-goals

- This sketch does **not** propose deprecating `CPP.run` — both paths
  coexist during the dev window. Long-term consolidation is a separate
  decision (and a separate ADR).
- It does **not** propose a new public extra (`embed`). PLM embeddings
  come from external libraries (transformers, esm, …); aaanalysis stays
  agnostic about *which* tool produced the `dict_num` contents.
- It does **not** change `df_cat`'s schema — `df_cat_emb` from
  `cluster_pseudo_scales` already conforms.

## Pointers

- Per-AA lookup that is *not* modified (legacy path):
  `aaanalysis/feature_engineering/_backend/cpp/_filters/_assign.py:56-101`
- New sibling folder (to be created):
  `aaanalysis/feature_engineering/_backend/cpp/_filters_num/`
- New frontend method (to be added):
  `aaanalysis/feature_engineering/_cpp.py` — `CPP.run_num`
- Parity test (to be added): in `tests/unit/cpp_tests/` — asserts
  `CPP.run(df_parts=…, df_scales=…)` and
  `CPP.run_num(df_seq=…, dict_num=None, df_scales=…)` produce bit-identical
  `df_feat` over the same fixture.
- Bench script (to be added): `dev_scripts/bench_filters_num.py` — A vs B
  staging shape head-to-head.
- Vocabulary: `CONTEXT.md` → "Numerical-mode CPP vocabulary"
  (`dict_num`, `CPP.run_num`, `_filters_num/`).
- ADR: `docs/adr/0001-cpp-run-num.md`.
