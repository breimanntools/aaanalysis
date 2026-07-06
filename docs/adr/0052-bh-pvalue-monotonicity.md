# ADR-0052 — BH-adjusted p-value monotonicity (canonical Benjamini–Hochberg)

Status: Accepted — 2026-07-06

## Context

`_bh_corrected_pvalues` computed `sorted_pvals * n / ranks` and clipped to 1 but **omitted the
reverse cumulative-minimum**, the monotonicity step of the canonical Benjamini–Hochberg step-up
procedure. So `p_val_fdr_bh` could be non-monotone in p-value order and deviate from
`statsmodels.multipletests('fdr_bh')` in non-monotone regions (values **inflated / conservative**,
never anti-conservative). Part of #343 (defect #3).

The deviation affects only the **reported** `p_val_fdr_bh` column in `df_feat`: CPP's feature
selection and ranking use `abs_auc` / `abs_mean_dif`, not the BH p-value.

## Decision

Insert the reverse cumulative minimum before clipping:
`corrected = np.minimum.accumulate(corrected[::-1])[::-1]`, matching statsmodels. Applied directly
(the prior output was simply non-canonical; nothing worth preserving).

**ADR-0032 tier:** output-affecting for the reported column only — no selection/ranking change and
**no performance cost** (one extra O(n) vectorized pass over the already-sorted array, run once per
CPP run, dominated by the existing O(n log n) sort).

## Regolden / anchors

- **No regolden:** the CPP exact-value regression anchor (ADR-0015) freezes top-feature identity +
  `abs_auc`, not p-value columns — it passes unchanged with the fix (verified via
  `AAA_RUN_REGRESSION=1`). No test pinned `p_val_fdr_bh`.
- **New guard:** `test_cpp_bh_monotonicity.py` compares `_bh_corrected_pvalues` to an independent
  canonical reference (an explicit suffix-minimum, structurally different from
  `np.minimum.accumulate`) and asserts monotonicity + a known non-monotone case. Platform-independent
  (pure formula), so it gates on every runner.

## Consequences

- `p_val_fdr_bh` now matches canonical BH; values change only in the previously non-monotone regions.
  No selection, ranking, or performance change.
- Closes the last of #343's three output-changing defects (#1 shipped #351 / ADR-0050,
  #2 shipped #353 / ADR-0051).
