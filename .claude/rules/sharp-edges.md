# Known sharp edges

Not path-scoped. Referenced from the root CLAUDE.md. Accepted architectural
limitations — do not "fix" them opportunistically.

- **`utils.py` is a god module (~1500 lines).** Add new constants/validators
  to the appropriate topical block (search for the nearest `# `-style
  section header). Splitting into `_constants.py`, `_checks/`,
  `_plotting_glue.py`, `_data_io.py` is a v2 refactor.
- **`config.py` is documented as untested.** Adding unit tests when
  touching it is welcomed.
- **No pre-commit, no ruff** — out of scope until v2. **Type checking is NOT deferred:**
  the package ships `py.typed` and `pyright` runs **non-blocking (advisory)** in CI,
  public-API-first (`_backend` excluded for now); mypy is not used. See ADR-0036.
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
  **ProtXplain**, never in this package. See ADR-0035.
- **Don't re-attempt the rejected performance optimizations.** A whole-library
  perf audit already tried and dropped many candidates — FASTA `iterrows`,
  TreeModel CV parallelization, `encode_pae` loops, AAclust binary-search `k`
  (non-monotonic `min_cor(k)`), ShapModel rolling-mean (memory-only), sparse
  one-hot, and more — each for a documented reason (no measured gain, or
  output-changing). The full table (accepted **and** rejected, with evidence)
  is **ADR-0033**; the tolerance policy for any output-affecting optimization is
  **ADR-0032**. Consult both before optimizing, and benchmark every new
  candidate in isolation first.
