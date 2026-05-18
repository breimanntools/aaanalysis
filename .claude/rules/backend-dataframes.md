---
paths:
  - "aaanalysis/**/_backend/**/*.py"
---

# DataFrame contracts (backend)

- Backend constructs row-shaped intermediate data as **positional lists**:
  ```python
  rows.append([entry, seq, window, c + 1])
  ```
  The column order is centralized in a `COLS_*` constant in the file's local
  module-level (e.g. `COLS_SEGMENTS` in `build_output.py`). When adding a
  column, every `rows.append([...])` call must be updated and verified
  in the same PR.
- Do not introduce `NamedTuple` / `@dataclass` typed row records — backend
  stays positional-list-based.
- Public methods return `pd.DataFrame` with columns named by `ut.COL_*`
  constants.
