# ADR-0053 — dPULearn.project: out-of-sample projection into the fitted PC space

Status: Accepted — 2026-07-06

## Context

The γ-secretase Use Case overlays experimentally-known non-substrates (held out of the PU fit) onto
the dPULearn PCA so all four canonical groups show in one view. The library exposed no way to project
held-out points into the **fitted** PC space, so the notebook hand-rolled it with a least-squares
affine reconstruction (`np.linalg.lstsq([X_pool | 1], Z_pool)` then applying the map to new points),
and `dPULearnPlot.pca` only accepted an already-transformed `df_pu` with labels `{0, 1, 2}` (three
groups), so overlaying a 4th group was manual. This was the last γ-secretase appendix helper without a
shipped API (#352, part of epic #305).

Two facts make this a design decision, not a copy-paste port:

1. dPULearn fits `PCA().fit(X.T)` (on the **transpose**) and stores `pca.components_.T` as `df_pu_`,
   so there is **no exact out-of-sample forward transform** for a new sample's feature vector — a new
   protein is a new *dimension* of that PCA, not a new observation. Any projection is a linear map
   *reconstructed* from the fit pairs `(X, df_pu_)`, exact on the fit pool and interpolating off it.
2. dPULearn is a negative-miner, not a dimensionality reducer, so a sklearn-style `.transform` would
   imply transformer semantics it does not really have.

## Decision

Add an **additive** projection method and plot overlay; existing defaults stay byte-identical.

- **`dPULearn.project(X, method='lstsq', alpha=1.0)`** — named `project`, not `transform`, to avoid
  implying dimensionality-reducer semantics on a negative-miner. Returns a DataFrame with the same
  `PCi` columns as `df_pu_`. `fit` now retains the fitted feature matrix (`self._X_fit`, rows aligned
  with `df_pu_`) so the map can be rebuilt on demand. Only valid after PCA-based identification
  (`metric=None`); distance-based fits have no PCs to project into.
- **Three projection methods**, all reconstructed from `(X_fit, df_pu_)` and therefore exact on the
  fit pool (when n_features >= n_samples), approximate for new samples:
  - `lstsq` (default) — affine least-squares map `[X | 1] -> df_pu_`, the minimum-norm solution.
    Reproduces the use-case hand-rolled projection **byte-for-byte** (regression-tested).
  - `components` — the exact PCA-geometry map: row-center each sample by its own feature mean, then
    the minimum-norm linear map, which equals the fitted PCA's `U @ inv(Sigma)` restricted to the
    stored components (proven equal to `pinv(X_centered) @ Z`).
  - `ridge` — L2-regularized affine map (`alpha`), stabilizing extrapolation when n_features >>
    n_samples; converges to `lstsq` on the fit pool as `alpha -> 0`.
- **`dPULearnPlot.pca(df_pu_add=, names_add=, colors_add=)`** — overlay one or more projected groups
  (each with its own name/color) on top of the three `df_pu` groups. `df_pu_add=None` (default)
  renders byte-identically to the previous three-group figure.

**Rejected alternative — refactor to a standard `PCA(X)`** whose `.transform` *is* an exact
out-of-sample map. It would change the stored `df_pu_` representation and the negative-mining
selection (different coordinates → different mined negatives), breaking the byte-identical-default
requirement, so it was not adopted. The honest position: the transposed-PCA has no exact out-of-sample
map, and `project` documents this (exact on the fitted samples, interpolation for new ones).

**ADR-0032 tier:** purely additive. `df_pu_` / `labels_` / `mask_neg_` and the default
`dPULearnPlot.pca` output are unchanged (165 existing dPULearn tests pass; new params default to
`None`). No new required dependency (uses numpy + the already-present scikit-learn).

## Consequences

- Projecting new proteins into the learned PC space and overlaying a 4-group PCA is now a two-call,
  library-native operation; the notebook's `lstsq` boilerplate is retired.
- `fit` retains `X` as private state (`_X_fit`) for PCA-based fits — a modest memory cost, no output
  change.
- Completes the γ-secretase appendix gap map for epic #305.
