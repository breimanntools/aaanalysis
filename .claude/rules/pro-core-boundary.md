---
paths:
  - "aaanalysis/**/*.py"
  - "aaanalysis/__init__.py"
---

# Pro / core boundary

How to decide whether a feature lives in core or under a `*_pro` subpackage,
and the mechanics of gating pro features behind the `pro` extra. See also
`api-stability.md` for the public-API surface rules.

## Default to core

A new feature lives in core unless it depends on a package that is
**heavy** (e.g. shap, biopython, numba), **fragile to install**, or unloved
by users. In that case it lives in a `*_pro` sub-package and is gated by
the `pro` extra.

Core is allowed to have `try/except ImportError` for *trivial* enhancements
that gracefully degrade (e.g. nicer repr if `rich` is installed). Full
features always go in `*_pro`.

## New `*_pro` modules

1. Imported in top-level `aaanalysis/__init__.py` inside `try/except
   ImportError`.
2. Append to `__all__` only on success.
3. Failure case substitutes `missing_feature_stub("FeatureName", e,
   mode="pro")`.

## `missing_feature_stub` rules

- Maintain `_PRO_MODULES = {"shap", "Bio", "biopython", "upsetplot", ...}`.
- Decide using `e.name` (the standard `ImportError.name` attribute), never
  by substring-matching `str(e)`.
- If `e.name in _PRO_MODULES`: raise the friendly install hint.
- Otherwise: re-raise the original `ImportError` unchanged so real bugs
  surface with full traceback.

## In-core / in-pro parity pattern

When a pro wrapper around an external CLI mirrors a core method (e.g.
`aa.scan_motif` ↔ `AAWindowSampler.sample_motif_matched`), the wrapper
should return identical hits/results to the in-memory core: use the CLI as
a primitive (e.g. position scanner), then apply the user-facing
scoring/filtering in Python so both paths go through the same scoring
formula. Add a parity test that asserts set-equality of the returned
identifiers and numeric closeness of scores.

## New extras require approval

New extras beyond `pro` / `docs` / `dev` require user approval (CONFIRM-FIRST
list in the root CLAUDE.md).
