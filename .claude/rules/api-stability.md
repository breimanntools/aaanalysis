---
paths:
  - "aaanalysis/__init__.py"
---

# Public API stability

Editing `aaanalysis/__init__.py` is **CONFIRM-FIRST** (see CLAUDE.md §10).

- **Strict semver** from v1.x onward.
- Adding new public symbols → minor bump.
- Renaming or removing any symbol in `aaanalysis.__all__` requires:
  1. one minor release with a `DeprecationWarning` shim,
  2. then removal in the next major.
- Behavior changes that could break downstream code → major bump.
- Use a `deprecated(reason, version_removed)` decorator helper (to be added)
  for shims.
