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

Implemented in `aaanalysis/__init__.py` as `_EXTRA_MODULES` + `_raise_missing_feature`.

- Maintain `_EXTRA_MODULES`, keyed by install extra:
  `{"pro": {"shap", "Bio", "biopython", "upsetplot", "UpSetPlot", "requests",
  "afragmenter"}, "dev": {"IPython"}}`. Add a module's **import name** here when a
  new `*_pro` (or dev) feature gates on it.
- Decide using `e.name` (the standard `ImportError.name` attribute), never
  by substring-matching `str(e)`.
- If `e.name in _EXTRA_MODULES[mode]`: raise the friendly install hint
  (chained `from e`).
- Otherwise (including `e.name is None`): re-raise the original `ImportError`
  unchanged so real bugs surface with full traceback.
- Covered by `tests/unit/api_tests/test_missing_feature_stub.py`.

## In-core / in-pro complementary pattern (not parity)

When a pro wrapper around an external CLI sits next to a related core method,
it must earn the external dependency by producing a **genuinely different**
result — never a re-scored mimic of the core (that makes the binary pure
redundancy; see ADR-0021).

The live example is `aa.scan_motif` (pro, FIMO) alongside
`AAWindowSampler.sample_motif_matched` (core, pure-Python). Both mine
motif-matched windows for training data, but by **different selection
criteria**:

- `sample_motif_matched` (core): keeps windows whose raw per-position **PWM-sum
  ≥ `motif_score_threshold`**; `motif_score` is that sum.
- `scan_motif` (pro): lets FIMO do probabilistic matching against the
  background model and keeps windows with **match p-value < `pvalue_threshold`**;
  `motif_score` is FIMO's log-odds score and an extra `p_value` column is added.

They share the output schema (so hits compose), but select different windows —
test the *difference* (an overlapping window carries different `motif_score`;
only the pro path reports `p_value`), not parity. An ex-CLI feature whose output
is forced to equal the core's belongs in core, not pro.

## New extras require approval

New extras beyond `pro` / `docs` / `dev` require user approval (CONFIRM-FIRST
list in the root CLAUDE.md).
