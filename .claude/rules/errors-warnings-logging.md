---
paths:
  - "aaanalysis/**/*.py"
  - "tests/**/*.py"
---

# Errors, warnings, logging

## Errors

- Validation → `ValueError`.
- Runtime / backend failure → `RuntimeError`.
- Soft failure → `warnings.warn(...)` (see below).
- **No custom error hierarchy.** Existing `BackendProcessingError` and
  `ClusteringConvergenceException` in `_utils/decorators.py` are legacy and
  remain non-rooted; do not introduce more bespoke exception classes. Do not
  create `AAanalysisError` — pandas/numpy don't, and we follow that style.
- `ValueError` messages must use the format
  `"'<name>' (<got>) should be <expected>"`.

## Warnings

Every `warnings.warn(...)` call passes an **explicit category**:

| Situation | Category |
|---|---|
| Clustering / convergence shortfall | `sklearn.exceptions.ConvergenceWarning` |
| Algorithmic shortfall (e.g. fewer samples kept than requested) | `RuntimeWarning` |
| Input-shaped issue the user can fix (missing positions, etc.) | `UserWarning` |
| Numerical edge (divide, undefined metric) | `RuntimeWarning` (or sklearn's `UndefinedMetricWarning` when wrapping sklearn) |

Bare `warnings.warn(msg)` (no category) is drift to fix on touch.

## Logging

- `ut.print_out(...)` is the sanctioned user-facing convenience. Library code
  never calls `print()`.
- Internally, `print_out` routes through `logging.getLogger("aaanalysis")` so
  power users can attach handlers, capture in pytest's `caplog`, or redirect
  to a file.
- The `verbose` flag controls whether the logger emits at INFO level
  (`verbose=True`) or stays at WARNING (`verbose=False`). Users may also call
  `logging.getLogger("aaanalysis").setLevel(...)` directly.
- Do not import `logging` from inside scattered modules; use `print_out`.
