# Known sharp edges

Not path-scoped. Referenced from the root CLAUDE.md. Accepted architectural
limitations — do not "fix" them opportunistically.

- **`utils.py` is a god module (~1500 lines).** Add new constants/validators
  to the appropriate topical block (search for the nearest `# `-style
  section header). Splitting into `_constants.py`, `_checks/`,
  `_plotting_glue.py`, `_data_io.py` is a v2 refactor.
- **`config.py` is documented as untested.** Adding unit tests when
  touching it is welcomed.
- **No type checker, no pre-commit, no ruff.** Out of scope until v2.
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
- **Don't re-attempt the rejected performance optimizations.** A whole-library
  perf audit already tried and dropped many candidates — FASTA `iterrows`,
  TreeModel CV parallelization, `encode_pae` loops, AAclust binary-search `k`
  (non-monotonic `min_cor(k)`), ShapModel rolling-mean (memory-only), sparse
  one-hot, and more — each for a documented reason (no measured gain, or
  output-changing). The full table (accepted **and** rejected, with evidence)
  is **ADR-0033**; the tolerance policy for any output-affecting optimization is
  **ADR-0032**. Consult both before optimizing, and benchmark every new
  candidate in isolation first.
