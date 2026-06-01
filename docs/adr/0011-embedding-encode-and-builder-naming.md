# ADR-0011 — EmbeddingPreprocessor gains `encode`; unify builder names to `build_scales` / `build_cat`

Status: Accepted — 2026-06-01

Supersedes (in part): ADR-0005 (the `cluster_pseudo_scales` distinct-name decision
and the `EmbeddingPreprocessor.build_pseudo_scales(dict_num=)` naming).

## Context

The whole feature-preprocessor family is still **unreleased** (it first ships in
package v1.1.0 — ADR-0010), so its API shape can still change with no semver cost.
A v1.1.0 API review surfaced two problems that ADR-0005 had left standing:

1. **No raw-embeddings → `dict_num` path.** `CPP.run_num` requires per-residue
   values in `[0, 1]` (the `StructurePreprocessor` / `AnnotationPreprocessor`
   normalization convention), yet `EmbeddingPreprocessor` only *consumed* a
   `dict_num` (via `build_pseudo_scales`) and never produced one. Raw PLM
   embeddings (ESM/ProtT5) are unbounded floats, so they are **not** directly
   usable — but `CPP.run_num`'s docstring claimed "your PLM embeddings ARE the
   `dict_num`, no conversion needed", which was wrong. Structure and Annotation
   both have `encode_*` methods that emit a normalized `dict_num`; Embedding had
   no analog. This was the family's most important gap, not a cosmetic one.

2. **Builder names did not match their output, nor each other.** Across the three
   preprocessors the scale builder was `build_pseudo_scales` (→ `df_scales`) but
   the category builder was `cluster_pseudo_scales` in Embedding (→ `df_cat_emb`)
   vs `build_cat` in the two pro classes (→ `df_cat`). The worst offender,
   `cluster_pseudo_scales`, is named after *scales* but returns *categories*.

## Decision

**D1 — Add `EmbeddingPreprocessor.encode` as the primary method.**
`encode(df_seq, embeddings, method='minmax'|'quantile'|'sigmoid', clip=(1, 99),
return_df=False) → dict_num` fits one per-embedding-dimension normalizer over the
whole corpus and maps every entry's `(L, D)` tensor into `[0, 1]`. Mirrors
`AnnotationPreprocessor.encode` (bare `dict_num` by default; `return_df=True` →
`(dict_num, df_seq_out)`). The normalizer lives in a fresh core backend
(`data_handling/_backend/embed_preproc/encode.py`) because `normalize()` lives in
`data_handling_pro` and core cannot import from `*_pro`. `build_scales` / `build_cat`
become the *secondary* (AA-scale, `CPP.run`) path. `CPP.run_num`'s docstring is
corrected to point at `encode` and the `[0, 1]` requirement.

**D2 — Name builders after their output, uniformly across the family.**
`build_pseudo_scales` → **`build_scales`** (→ `df_scales`); Embedding's
`cluster_pseudo_scales` → **`build_cat`** (→ `df_cat`), matching the two pro
classes. Embedding's output suffixes `df_scales_emb` / `df_cat_emb` drop to
`df_scales` / `df_cat`, matching `CPP.run_num(df_scales=, df_cat=)` exactly. The
word "pseudo" survives only in docstrings/prose, not in identifiers. Rationale: a
builder/getter should be named after the object the pipeline consumes, and match
across its family — the same instinct that renamed `smooth_scores` →
`comp_smooth_scores` for metric-family uniformity.

## Why this reverses ADR-0005

ADR-0005 had **rejected** `Embedding.cluster_pseudo_scales → build_cat` as "false
symmetry" (signatures differ: thresholds/`random_state` vs registry lookup) and
asserted "Embedding has no acquire/encode step". Both premises are revised here:
the differing *signature* is an implementation detail of how categories are
formed (clustering vs registry), not a reason to misname the *output*; and adding
`encode` gives Embedding exactly the acquire/encode step ADR-0005 said it lacked,
completing the family symmetry. Because nothing has shipped, this carries no
deprecation cost.

## Consequences

- Public-method renames across all three preprocessors; swept through their
  backends, tests, the `cpp_run_num` / structure / annotation example notebooks,
  ADR-0005/0010, and CONTEXT.md.
- New `ep_encode` example notebook is the headline embedding example; new
  `ep_build_scales` / `ep_build_cat` cover the secondary path.
