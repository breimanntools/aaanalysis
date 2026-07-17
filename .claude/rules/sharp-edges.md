# Known sharp edges

Not path-scoped. Referenced from the root CLAUDE.md. Accepted architectural
limitations — do not "fix" them opportunistically.

- **`utils.py` is a thinning import barrel.** The bulk of the constants now live
  in the sibling `aaanalysis/_constants.py` and are re-exported via
  `from ._constants import *`, so `utils.py` is ~600 lines (validators are already
  split under `_utils/check_*.py`; the check functions are re-exported too). Add a
  new **constant** to the appropriate topical block in `_constants.py` (it imports
  only stdlib + numpy — never `aaanalysis.utils`, to stay circular-free); add a new
  **helper/validator function** to `utils.py` or the relevant `_utils/*.py`. Every
  `ut.X` access still goes through `import aaanalysis.utils as ut` — the barrel keeps
  the access point stable. Extracting the remaining inline functions into
  `_plotting_glue.py` / `_data_io.py` is the next (optional) slice. The barrel
  invariants are guarded by `tests/unit/api_tests/test_utils_barrel.py`.
- **`config.py` is documented as untested.** Adding unit tests when
  touching it is welcomed.
- **No pre-commit, no ruff** — out of scope until v2. **Type checking is NOT deferred:**
  the package ships `py.typed` and `pyright` runs as a **no-regression ratchet** in CI
  (`pyright.yml`), public-API-first (`_backend` excluded for now); mypy is not used
  (the `mypy aaanalysis/__init__.py --follow-imports=skip` step was removed from CI as
  toothless). See ADR-0036. The pyright version is **pinned** in the workflow so the
  committed high-water mark is reproducible; bumping pyright is a deliberate re-baseline.
  The diagnostic count is being driven down in small, per-subpackage steps against that
  mark in `.github/pyright_baseline.txt` (`.github/scripts/check_pyright_budget.py`
  reports count + delta and **exits non-zero when the count exceeds the baseline** —
  the ratchet step gates the job, blocking *new* diagnostics while allowing the existing
  backlog to merge). It is **not** a clean/strict gate: merging at or below the baseline
  is fine. When a burn-down PR clears diagnostics, **lower** that number to the new count
  in the same PR; never raise it. Prefer honest signatures over runtime `assert`s, and a
  narrow `# pyright: ignore[<rule>]` (with an inline reason) only for a genuine stub false
  positive.
- **`pyproject.toml` carries both `[project]` and `[tool.poetry]` blocks.**
  Edit `[project]` only.
- **MEME format alphabet quirk** (relevant only for `aa.scan_motif`): the
  package PWM uses `ut.LIST_CANONICAL_AA` order; MEME requires alphabetic
  protein order (`ACDEFGHIKLMNPQRSTVWY`). `_pwm_to_meme` permutes columns
  before serialization.
- **Don't introduce typed row records** (NamedTuple / dataclass) until the
  v2 migration; backend stays positional-list-based.
- **Don't add a `SECURITY.md`, ruff config, pre-commit config, or mypy
  config** — these are explicitly out of scope for now.
- **Don't create `AAanalysisError`** or any custom exception base — bare
  `ValueError` / `RuntimeError` is the rule.
- **Don't add Pydantic / pandera or any schema-validation framework** — not in
  v1 and not planned for v2. Input validation is the hand-rolled `ut.check_*`
  family (sklearn-style Validate block); the `df_feat` output contract is a
  documented, test-guarded **data dictionary** (`ut.LIST_COLS_FEAT` /
  `sort_cols_feat`), not a typed model. Agent-facing typed I/O contracts
  (Pydantic `CPPRequest`/`CPPResult`, JSON/MCP tool schemas) live downstream in
  **ProtXplain**, never in this package. See ADR-0038 (the single border ADR;
  D12 carries this rule).
- **Don't re-attempt the rejected performance optimizations.** A whole-library
  perf audit already tried and dropped many candidates — FASTA `iterrows`,
  TreeModel CV parallelization, `encode_pae` loops, AAclust binary-search `k`
  (non-monotonic `min_cor(k)`), ShapModel rolling-mean (memory-only), sparse
  one-hot, and more — each for a documented reason (no measured gain, or
  output-changing). The full table (accepted **and** rejected, with evidence)
  is **ADR-0033**; the tolerance policy for any output-affecting optimization is
  **ADR-0032**. Consult both before optimizing, and benchmark every new
  candidate in isolation first.
