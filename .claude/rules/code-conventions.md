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

- Annotate every public parameter and return type. Use `Optional[...]` for
  `None`-defaults.
- Avoid `pd.DataFrame = None` for required args. Either accept `None`
  honestly (and `check_df_seq(accept_none=True)`) or make the arg required
  by removing the default.
- Use `ut.ArrayLike1D` / `ut.ArrayLike2D` for numpy/list duck types.
- **No type checker runs in CI.** Local IDE-side type checking (Pylance in
  VS Code) is up to the contributor.

## Linting and Python floor

- Stack: `black` (88), `isort` (black profile), `flake8` (88). No
  `pre-commit` hook; style is enforced at PR review. `ruff` migration is
  targeted for v2.
- **Floor: Python 3.11.** Don't use 3.12+ syntax (PEP 695 generics, `type`
  statement, `@override`, `typing.deprecated`).
- Updating the Python floor requires editing `pyproject.toml`,
  `classifiers`, and the CI matrix — covered by the CONFIRM-FIRST list in
  the root CLAUDE.md.
