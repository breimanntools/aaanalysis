---
paths:
  - "aaanalysis/**/*.py"
---

# Code conventions

Cross-cutting style rules for source files. Applies to both frontend
(`aaanalysis/<subpkg>/_<feature>.py`) and backend
(`aaanalysis/<subpkg>/_backend/...`). See also `docstrings.md` (docstring
format), `frontend-backend.md` (validation, signatures), and
`errors-warnings-logging.md` (raise/warn/log policy).

## Module skeleton

Every module, frontend or backend, follows the same skeleton:

```python
"""
This is a script for the <frontend|backend> of the <Class or feature> ...
"""
from typing import ...
import numpy as np
import pandas as pd
import aaanalysis.utils as ut
from ._backend.<topic>... import ...   # frontends only

# I Helper Functions
def _private_helper(...): ...
def check_<thing>(...): ...

# II Main Functions
class <PublicClass>:    # or top-level public functions
    ...
```

Mandatory:
- Module docstring opens with `"""This is a script for ..."""` — keep that
  voice exactly.
- The `# I Helper Functions` and `# II Main Functions` markers (capital F) are
  required in every module file.
- Frontend file names start with `_` (`_aa_window_sampler.py`); the public
  name is exposed only via `__init__.py`.

Files with lowercase markers (`# I Helper functions`) are drift to fix on
touch.

## Naming

- **Modules:** `_lower_snake.py` for private/frontend; `__init__.py`
  re-exports.
- **Public classes:** `PascalCase`. Plot pair: `<Class>Plot`.
- **Functions / methods:** `lower_snake_case`. Top-level functions in
  `seq_analysis_pro/` follow short verb-noun: `comp_seq_sim`, `filter_seq`,
  `scan_motif`.
- **Constants:** `UPPER_SNAKE`. Domain bundles use plural prefix:
  - `COL_*` (single column name), `COLS_*` (list of columns), `LIST_*` (other
    lists), `DICT_*`, `COLOR_*`, `MODE_*`, `ROLE_*`, `STRATEGY_*`, `STR_*`
    (short symbolic strings).
- **DataFrame column names** are constants only; never hardcode `"entry"` —
  use `ut.COL_ENTRY`.
- **Trailing underscore on attributes** (`labels_`, `centers_`, `is_medoid_`)
  follows the sklearn meaning: "set during `.fit`". Keep that.
- **Trailing underscore on backend helper functions** (`get_features_`,
  `add_stat_`, `single_logo_`, `extend_alphabet_`) marks the function as an
  *internal* backend helper — imported across backend modules but never
  appearing in the public API. "Main" backend functions whose name mirrors
  a frontend method (`get_aa_window`, `compute_centers`, `sample_same_protein`,
  `build_segments_output`) use plain names. **Leading** underscore
  (`_get_cluster_names`) remains the marker for file-private helpers used
  only inside their defining file.
- **Frontend functions** never carry a trailing underscore.

## Domain-bundle constants live in `utils.py`

Any constant matching the bundle prefixes above (`COL_*`, `COLS_*`, `LIST_*`,
`DICT_*`, `COLOR_*`, `MODE_*`, `ROLE_*`, `STRATEGY_*`, `STR_*`) is **defined
in `aaanalysis/utils.py`** under a topical block and accessed from
subpackages as `ut.X` (with `import aaanalysis.utils as ut`).

The only exceptions are configuration **data structures** that aren't simple
constant bundles — e.g. registries with structured values like the `PRESETS`
dict in `_backend/aa_window_sampler/sample_synthetic.py`, which holds
biological scale IDs and citations.

Backend code MUST go through `ut`; never reach into `aaanalysis._utils.*`
directly.

## Type hints

- Annotate every public parameter and return type. Use `Optional[...]` **only**
  for params that genuinely accept `None` (the Validate block passes
  `accept_none=True` or `None` carries a real meaning). `Optional[T] = None` on
  an argument the check then *rejects* is a lie — it advertises `None` as valid
  when the call raises.
- **Required args carry no default and no `Optional`.** Never write
  `df_feat: pd.DataFrame = None` for an argument that must be given (the
  Validate block would reject `None` anyway — the `Optional[...] = None` then
  lies). Make it required by removing the default: `df_feat: pd.DataFrame`.
  Required args that lead the signature stay **positional-or-keyword** (no `*`
  separator — see `frontend-backend.md`); positional remains allowed so dropping
  a stale `= None` is non-breaking. Place required args before any defaulted arg
  (Python forbids a non-default after a default). When a required arg must sit
  **after** a defaulted one and can't be reordered without breaking positional
  callers, resolve it in priority order: (1) if a canonical default exists, give
  it one and keep `Optional[T] = None` honestly (e.g. `df_cat` defaults to
  `ut.load_default_scales(scale_cat=True)`); else (2) make it **keyword-only**
  with a `*` before it — `def feature(self, feature, feat_rank=1, *, df_seq,
  labels, ...)`. This `*` for an otherwise-stranded required arg is the **only**
  sanctioned `*` use; never add `*` merely to force keywords on args that could
  lead.
- Use `ut.ArrayLike1D` / `ut.ArrayLike2D` for numpy/list duck types.
- **No type checker runs in CI.** Local IDE-side type checking (Pylance in
  VS Code) is up to the contributor.

## Blank lines between methods (hard rule, test-enforced)

Two consecutive methods in a class body MUST be separated by at least one blank
line (PEP 8 E301). A `return` glued directly to the next `def` is a defect, not a
style preference. Because readable code review has missed this, it is enforced
programmatically by `tests/unit/api_tests/test_style_method_spacing.py` (AST
scan over the whole package). The check is narrow on purpose — it flags only
method-after-method spacing, not the blank line between a class docstring and the
first method.

## Linting and Python floor

- Stack: `black` (88), `isort` (black profile), `flake8` (88). No
  `pre-commit` hook; style is enforced at PR review. `ruff` migration is
  targeted for v2.
- **Floor: Python 3.11.** Don't use 3.12+ syntax (PEP 695 generics, `type`
  statement, `@override`, `typing.deprecated`).
- Updating the Python floor requires editing `pyproject.toml`,
  `classifiers`, and the CI matrix — covered by the CONFIRM-FIRST list in
  the root CLAUDE.md.
