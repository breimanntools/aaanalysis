# ADR-0001 — CPP backend architecture: unified numerical pipeline + Cython acceleration

Status: Accepted — 2026-05-25

## Context

`CPP` is the core feature-engineering algorithm in `aaanalysis` — it scores per-residue amino-acid scales over user-defined sequence parts and ranks the resulting features by discriminative power between two label groups. Two pressures shaped the current backend design:

1. **Numerical input support.** Users want CPP to operate on per-residue numerical representations (PLM embeddings, DSSP one-hots, PTM dummies) — not only on amino-acid sequences mediated by a `(20, n_scales)` AA-letter lookup. The natural change is to keep the scale lookup as ONE source of per-residue values and add per-residue tensors as a parallel value source consumed by the same downstream pipeline.

2. **Performance.** The original pipeline ran `np.mean(list)` per (sample, feature) in Python, and recomputed survivor feature values from scratch in pass-2 — both Python-overhead-bound. A streaming pre-filter (keep per-sample feature values for `std_test`-survivors and reuse in `add_stat`) plus a Cython kernel for the per-(sample, feature) reduction collapses both costs. The Cython path is bit-exact with numpy's pairwise summation (verified against `np.mean(arr[i, positions])` on every row of the parity fixture).

The third concern — install ergonomics — was solved by `cibuildwheel`: wheels published to PyPI ship the precompiled `_inner.cpython-XYZ.so` next to the `.pyx`, so end users `pip install aaanalysis` (or `uv add aaanalysis`) and the fast path works with zero ceremony. Users on unsupported platforms fall back to the sdist and build via the setuptools backend (or use the pure-Python kernel, which is bit-exact but slower).

## Decision

### Public API

`CPP` exposes three methods. The constructor takes the analysis configuration (`df_parts`, `df_scales`, `df_cat`, `split_kws`, `accept_gaps`, `verbose`, `random_state`):

```python
cpp.run(labels, ...)                                    # seq-mode
cpp.run_num(dict_num_parts, labels, ...)                # numerical mode
cpp.eval(list_df_feat, labels, ...)                     # feature-set evaluation (unchanged)
```

`run` and `run_num` differ by exactly one positional slot (`dict_num_parts`). All analysis params (`label_test`, `n_filter`, `max_std_test`, `max_overlap`, `max_cor`, `parametric`), position params (`tmd_len`, `jmd_n_len`, `jmd_c_len`, `start`), and execution params (`n_jobs`, `vectorized`, `n_batches`) are identical between the two.

`run_num(dict_num_parts=None)` raises `ValueError` — use `cpp.run()` for seq-mode.

### Preprocessing helper: `NumericalFeature.get_parts`

`run_num` requires `dict_num_parts: Dict[part_name, np.ndarray (n_samples, L_part_max, D)]` aligned to `self.df_parts`. The user produces it (along with the matching `df_parts`) by calling:

```python
df_parts, dict_num_parts = aa.NumericalFeature().get_parts(
    df_seq, dict_num, jmd_n_len=10, jmd_c_len=10,
)
```

`get_parts` slices both the sequence strings (matching `SequenceFeature.get_df_parts`) and the per-residue tensors with shared `(start, end)` boundaries — so the user never has to pass `df_seq + jmd_n_len + jmd_c_len` to two separate helpers. The `(df_seq ↔ dict_num)` pairing is enforced exactly once, at preprocessing time.

### Backend structure

```
aaanalysis/feature_engineering/_backend/
├── cpp_run.py                      # orchestrator (cpp_run_single, cpp_run_batch)
└── cpp/
    ├── _filters/                   # canonical pipeline (Python + per-part 3D tensors)
    │   ├── _assign.py              # AA→scale OR dict_num → per-part values
    │   ├── _stat_filter.py         # streaming pre-filter stats
    │   ├── _pre_filter.py          # threshold + top-K
    │   ├── _add_stat.py            # AUC, Mann-Whitney, FDR — on cached survivor matrix
    │   ├── _recompute.py           # survivor matrix rebuild (numerical mode)
    │   ├── _redundancy_filter.py   # greedy descending-AUC selection
    │   ├── _progress.py            # shared multiprocessing/progress helpers
    │   └── _get_feature_matrix_fast.py  # Python kernel for pass-2 recompute (seq-mode)
    └── _filters_c/                 # Cython kernel for the per-(sample, feature) reduction
        ├── _inner.pyx              # bit-exact hand-rolled pairwise summation
        └── _get_feature_matrix_c.py
```

`cpp_run._pick_feature_matrix_builder()` picks the Cython builder when the compiled `.so` is importable, else the Phase-C Python fallback. Both `cpp.run` and `cpp.run_num` route through this — there is no user-facing backend switch.

### Build + distribution

- Build backend: `setuptools` (declared in `pyproject.toml [build-system]`).
- `setup.py` at the project root declares the Cython extension via `cythonize(...)`.
- `cibuildwheel` (configured in `pyproject.toml [tool.cibuildwheel]`) builds wheels for cp310–cp314 × {Linux x86_64+aarch64, macOS x86_64+arm64, Windows AMD64} on release. Each wheel runs `pytest tests/unit/cpp_tests/test_get_feature_matrix_c_parity.py` as an install-time sanity check so bit-exact parity is verified per (Python × OS × arch) before publication.
- Optional `[pro]` extra includes `shap`, `biopython`, `UpSetPlot` (no `numba` — see "Considered alternatives" below).

### Bit-exact parity contract

`cpp.run` output is bit-identical to numpy `np.mean(arr[i, positions])`-based reduction for every per-(sample, feature) value: the Cython kernel replicates numpy's pairwise summation (8-way unrolled tree, `np.round(_, 5)` boundary). The Python fallback uses `np.mean` directly. Mann-Whitney p-values (rank-based, ULP-sensitive) match across both paths.

Verified by the parity suite in `tests/unit/cpp_tests/`:
- `test_get_feature_matrix_c_parity.py` — Cython kernel vs `np.mean` reference
- `test_get_feature_matrix_fast_parity.py` — Python kernel vs `np.mean` reference
- `test_run_num_parity.py` — end-to-end determinism + Layer-2 validation
- `tests/unit/numerical_feature_tests/test_nf_get_parts.py` — `NumericalFeature.get_parts` contract

## Considered alternatives

- **Numba-based acceleration as a parallel backend.** Tried, removed. Performance was within 1.01–1.27× of Cython depending on n (after porting Mann-Whitney + AUC + gather amortization into `@njit`). Dependency posture is the disqualifier: numba lags NumPy major releases (NumPy 2.0 broke shap-via-numba for ~3 months until numba 0.60), pins Python versions strictly (Python 3.14 + numba 0.65 had a `cache=True` segfault), has unstable ARM/Mac wheels, and conflicts with Colab's bundled numba. SHAP — the largest numba consumer in the scientific-Python stack — is actively planning to remove its numba dependency for the same reasons. Cython has none of this churn (its compiled `.so` depends only on the stable Python and NumPy C ABIs).
- **Hatchling build backend + hatch-cython plugin.** Targeted for v2 (per `.claude/rules/dependencies-and-pyproject.md`). Setuptools is the well-trodden Cython path today; hatchling can be revisited with a smaller diff when the rest of v2's migration happens.
- **Poetry as build backend (with a `build.py` hook).** Rejected — poetry-core doesn't support compiled extensions natively, and Poetry's build-hook story wraps setuptools anyway.
- **Migrate dev workflow from Poetry to uv in the same PR.** Independent concern, handled separately (now done — see Followups). End-user `pip install aaanalysis` and `uv add aaanalysis` both work today regardless of which dev tool the project uses internally.

## Consequences

- **End users `pip install aaanalysis` get a fast CPP path with zero ceremony** on the supported wheel matrix (cp310–cp314 × Linux/macOS/Windows). Unsupported platforms fall back to sdist + setuptools compile (the standard scipy/numpy pattern) or to the bit-exact pure-Python kernel.
- **`pro` extras stay focused on explainability + sequence-analysis features** (shap, biopython, UpSetPlot). The shap-pulls-numba transitive dep persists until SHAP's planned numba removal; aaanalysis users on `[pro]` get a leaner install then.
- **CI release cost** is ~15–25 min wall-clock per release tag for the cibuildwheel matrix. Per-PR CI is unaffected.
- **The dual-method API (`run` vs `run_num`)** is preserved instead of auto-dispatching inside a single `run`. The two have genuinely different mental models — `run` operates on AA-letter parts via the constructor's scales, `run_num` operates on per-residue numerical tensors per-call. Hiding the difference behind an `auto-dispatch` would lose that conceptual clarity. Bit-exact parity between them in the round-trip case (build `dict_num` from `df_scales` lookup) is verified to within ULP/Mann-Whitney rank tolerance.

- **Empty split buckets are silently dropped, not errored.** A `(split type, part)` bucket can yield zero splits (only Pattern, when `n_min * steps[0] > len_max`; the labels are part-length independent so it is a config-level, not a short-part, condition). Legacy CPP dropped these silently and the vectorized rewrite preserves that for parity: `iter_scale_chunks` early-returns on a zero-width split axis (shared by both the pass-1 streaming pre-filter in `_stat_filter.py` and the pass-2 `recompute_feature_matrix`, so one guard covers both). The earlier rewrite regressed into a `ZeroDivisionError` here (`per_scale_bytes == 0`) until the guard was restored. To keep silent-drop from masking misconfiguration, `check_split_kws` emits a `UserWarning` at validation time when a Pattern config is empty — chosen over a hard `ValueError`, which would have broken existing workflows that relied on the legacy silent behavior. Verified by `tests/unit/cpp_tests/test_recompute_zero_splits.py` and `test_cpp_run.py::TestCPPRunComplex::test_empty_pattern_bucket_silently_dropped`.

## Followups

- **V2-1 (done)**: Dev workflow migrated from Poetry to uv — `poetry.lock`
  dropped, `uv.lock` committed, CI and `CONTRIBUTING.rst` install instructions
  on uv. Independent of this ADR. The remaining build-tooling migration (drop
  `[tool.poetry]`, hatchling, ruff, pre-commit, type checker) is tracked in
  `.claude/rules/sharp-edges.md`.
