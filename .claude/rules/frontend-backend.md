---
paths:
  - "aaanalysis/**/*.py"
---

# Frontend vs backend asymmetry

The codebase deliberately treats frontend
(`aaanalysis/<subpkg>/_<feature>.py`) and backend
(`aaanalysis/<subpkg>/_backend/...`) differently. New code must respect this
asymmetry. See also `errors-warnings-logging.md` for the full raise/warn/log
policy.

## Signatures

Frontend (public methods) and backend functions both use
positional-or-keyword parameters with sklearn-style defaults. Pass kwargs
explicitly at call sites (`func(name=value)`) for readability — enforced by
code review, not by `*` separators in the signature.

## Validation

Every public method opens with a Validate block before any work:

```python
def some_method(self, df_seq=None, n=100, mode="global_freq"):
    """..."""
    # Validate
    ut.check_df_seq(df_seq=df_seq)
    ut.check_number_range(name="n", val=n, min_val=1, just_int=True)
    ut.check_str_options(name="mode", val=mode, list_str_options=LIST_MODES)
    # Build pool
    ...
    # Build output
    ...
```

Rules:
- Use kwargs explicitly when calling `ut.check_*` — always pass `name=` and
  `val=`. Forgetting `name=` is the highest-frequency defect class in this
  codebase.
- Joint constraints go in `check_match_<a>_<b>(a=..., b=...)` helpers,
  defined locally in the file under `# I Helper Functions`.
- **Backend modules do NOT validate raw user input.** They trust the
  frontend. However, backends MAY (and should) raise on **derived
  invariants** computed inside the backend — e.g. "no candidates after
  filtering", "scale sums to zero in normalization", "shape mismatch
  between intermediate arrays". Use `ValueError` for invariants traceable
  to user input, `RuntimeError` for invariants that indicate an internal
  bug.

## Backend module ownership (no cross-class imports)

Backend code is organized as **dedicated subpackages** (`_backend/<subpkg>/`,
private to one class — e.g. `_backend/num_feat/` is NumericalFeature's,
`_backend/aaclust/` is AAclust's) plus **shared** modules at the top level of
`_backend/` (`check_feature.py`, `cpp_run.py`, `feature_filter.py`, …) and the
deliberately shared `_backend/cpp/` package.

Rule: **a frontend imports backend helpers only from a shared module or from its
own dedicated subpackage — never from another class's dedicated subpackage.** If
two frontends need the same helper, it is shared by definition: move it to a
common top-level `_backend/*.py` and have both import from there. (Concrete
example: `filter_correlation_` / `filter_variance_` live in
`_backend/feature_filter.py`, imported by both NumericalFeature and
SequenceFeature; neither reaches into the other's subpackage.)

Enforced by `tests/unit/api_tests/test_backend_import_hygiene.py` — extend its
`DEDICATED_OWNERS` map when adding a new dedicated backend subpackage.
