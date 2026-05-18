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
