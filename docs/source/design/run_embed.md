# Design sketch: `CPP.run_embed`

> **Status:** Proposal only — no implementation. This file captures the design
> so the idea isn't lost; it does **not** record a finalised decision.
> Promote to an ADR (or delete) when an implementation actually lands.

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

A dedicated `CPP.run_embed` would consume the per-residue embeddings
directly, preserving positional context all the way through feature
extraction. Pseudo-categories (`df_cat_emb`) would still be derived via
`cluster_pseudo_scales` and would still drive the redundancy / categorical
filters — only the per-residue value lookup changes.

## Proposed signature

```python
class CPP(Tool):
    def run_embed(
        self,
        embeddings: Dict[str, np.ndarray],     # entry -> (L, D) per-residue PLM embedding
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
        n_batches: Optional[int] = None,
    ) -> pd.DataFrame:
        ...
```

Constructor expectations:
- `df_cat_emb` (the AAclust output) is passed via the existing `df_cat=`
  constructor argument — no schema change required.
- `df_scales` is **not** passed; the per-residue `embeddings` dict replaces
  it. CPP needs to know which `dim_i` the categorical filter refers to —
  it reads that from `df_cat_emb[scale_id]`.

Return contract: a `df_feat` DataFrame identical in shape and column meaning
to what `CPP.run` returns today, so downstream callers (`TreeModel`,
`CPPPlot`, …) don't need to change.

## Required backend changes

Today's per-residue value pipeline lives in
`aaanalysis/feature_engineering/_backend/cpp/_filters/_assign.py`:

1. `aa_to_idx` is built from `df_scales.index` (line 62).
2. A `scale_matrix` of shape `(n_aa + 1, n_scales)` is built (line 64-67).
3. Each residue in `X_seq` is mapped to its AA index (line 92-94).
4. Per-part values are pulled via `scale_matrix[aa_idx_matrix, s]` (line 101).

For `run_embed` this becomes a **(entry, position, dim) lookup** instead of
a **(aa_letter, scale_id) lookup**:

1. Drop the `aa_to_idx` table.
2. Replace `scale_matrix[aa_idx_matrix, s]` with positional gathering:
   for each part-slice covering `(start, stop)` on `entry`, gather
   `embeddings[entry][start:stop, d]` directly.
3. Aggregate over the part using the existing CPP split/segment logic
   (`Segment`, `Pattern`, `PeriodicPattern`) — these operate on `(n_samples,
   len_part)` arrays of *scale values*, so embeddings need the same
   `(n_samples, len_part)` shape per `(part, d)`. Pre-stage embeddings
   into a `dict_emb_part: {part: np.ndarray(n_samples, len_part_max, D)}`
   at constructor time.

The parallel-batch / `n_jobs` plumbing in `cpp_run_.py` can stay; the
inner loop's value source is what changes.

## Data flow

```
embeddings  (entry -> (L, D))
  │
  ├── pre-stage per part (tmd, jmd_n_tmd_n, …) -> (n_samples, len_part_max, D)
  │       │
  │       └── for each (part, d, split) generate a feature value via the
  │           existing aggregation (mean/std/min/…)
  │
  └── df_cat_emb drives the categorical / redundancy filter as today
```

The split / pattern / segment grammar is **unchanged** — only the value
source per residue position differs.

## Alignment with `EmbeddingPreprocessor`

- `build_pseudo_scales` stays useful even with `run_embed`: it remains the
  input to `cluster_pseudo_scales`, which produces `df_cat_emb`.
- `df_scales_emb` itself becomes optional in the `run_embed` path (only the
  pseudo-categories are required), but is still useful for the existing
  `CPP.run` fallback and for diagnostic plotting.
- `return_std=True` from `build_pseudo_scales` gives `df_stds_emb`, which
  enables the "std-aware clustering" idea below.

## Open questions

1. **Std-aware clustering — IMPLEMENTED.** Shipped via the optional
   `df_stds_emb=` parameter on
   `EmbeddingPreprocessor.cluster_pseudo_scales`. When supplied, each
   dimension is represented by the per-column z-scored concatenation of
   per-AA `(mean, std)` (shape `(D, 40)`), and AAclust then clusters by
   Pearson row-correlation on the concatenated descriptor. References in
   the method's Notes block: [MilliganCooper88] for standardization,
   [Eisen98] for the row-correlation precedent. **Not** shipped:
   true Bhattacharyya / symmetric-KL between per-AA Gaussians — these need
   a precomputed similarity matrix that AAclust does not currently accept
   (blocked on the AAclust refactor described in open question §4 below).
   Note that under the equal-variance assumption Bhattacharyya reduces to a
   function of `(μ₁ − μ₂)²` alone, so the shipped concat-z-scored recipe
   is **not** an approximation of Bhattacharyya — it is a separate moment-
   descriptor heuristic.
2. **Embedding alignment to JMD/TMD splits.** Per-residue embeddings exist
   on the *full sequence*; CPP splits operate on per-part slices. The
   pre-staging step needs to slice consistently with how `SequenceFeature.get_df_parts`
   defines parts. Need a parity test that staged embeddings line up with
   `df_parts` strings position-for-position.
3. **Per-position vs per-region aggregation.** Should embedding features
   be `mean(emb[start:stop, d])` (matches current Segment), or should
   `run_embed` expose richer aggregators (max, attention-weighted, etc.)?
   Default to mean for parity; expose richer aggregators behind a flag.
4. **AAclust k-optimization metric.** AAclust's k-optimization is
   hardcoded to Pearson correlation (`_aaclust.py:287-289`). The current
   `metric` parameter on `cluster_pseudo_scales` only affects the optional
   *merge* step, which is why the public signature is restricted to
   `{"correlation", "cosine"}`. A real cosine-driven k-optimization needs
   an AAclust refactor (track separately).
5. **Memory.** Pre-staging `(n_samples, len_part_max, D)` for D=1024 and
   n=10k samples is multiple GB. Decide on a chunked/streaming gather vs.
   eager staging at construction time.
6. **Caching.** Embedding computation is the expensive step in PLM
   workflows. Should `run_embed` (or `EmbeddingPreprocessor`) ship a
   tiny on-disk cache keyed on `(model_id, df_seq hash)`? Out of scope for
   v1; revisit if usage grows.

## Non-goals

- This sketch does **not** propose deprecating `CPP.run` or its per-AA
  scale model — that path stays as the AAontology consumer.
- It does **not** propose a new public extra (`embed`). PLM embeddings
  come from external libraries (transformers, esm, …); aaanalysis stays
  agnostic about *which* PLM produced them.
- It does **not** propose changes to `df_cat` schema — `df_cat_emb` from
  `cluster_pseudo_scales` already conforms.

## Pointers

- Per-AA lookup that needs to change:
  `aaanalysis/feature_engineering/_backend/cpp/_filters/_assign.py:56-101`
- Constructor-side scale staging that needs to be replaced:
  `aaanalysis/feature_engineering/_cpp.py:114-117`
- The end-to-end test that exercises the *averaging* path today (and that
  `run_embed` should match for `df_feat` shape parity):
  `tests/unit/data_handling_tests/test_ep_cluster_pseudo_scales.py::TestClusterPseudoScalesComplex::test_pipeline_into_cpp_run`
- Pseudo-scale / pseudo-category glossary entries:
  `CONTEXT.md` → "Embedding-based feature engineering vocabulary".
