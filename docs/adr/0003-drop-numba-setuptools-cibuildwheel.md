---
status: accepted
supersedes: ADR-0002
---

# Drop Numba; switch build backend to setuptools; ship prebuilt Cython wheels via `cibuildwheel`

## Context

ADR-0002 introduced `CPP.run_nb` as a Numba-accelerated, `pro`-gated drop-in for the Cython `CPP.run_c`. PR3 landed Paths 1 (Numba mass Mann-Whitney + AUC) and 3 (gather amortization) on top of the Stage 1 verbatim port — bit-exact, but only **1.01–1.27×** faster than Cython depending on n.

Two follow-up observations forced a re-evaluation:

1. **The performance win is marginal and shrinks with n.** Profiling showed the kernel layer is only ~6.5% of `run_nb` time at n=200, and the larger non-kernel hotspots (`pre_filtering_info_num` already in vectorized numpy, scipy-backed `rankdata` in `add_stat`) only Numba-accelerate cleanly inside `add_stat`. Once `add_stat` is ported, further Numba work has poor ROI.
2. **Numba's dependency posture is a long-term liability.** Per the maintainer discussion the user surfaced: SHAP (the largest consumer of numba in the scientific-Python stack) is actively planning to **remove its numba dependency** because numba lags new NumPy releases (NumPy 2.0 broke shap for ~3 months until numba 0.60), pins Python versions strictly (we hit a `cache=True` segfault on numba 0.65.1 + Python 3.14 in PR3), has ARM/Mac issues, and conflicts with Colab's bundled numba. If SHAP drops numba, the `pro` extra would still pull numba via aaanalysis alone — making aaanalysis the sole reason for the dep.

Cython, by contrast, has *no* runtime dependency on a fast-moving library: the compiled `.so` depends only on the Python C ABI and NumPy C ABI at build time, both stable APIs. The only remaining user-facing cost was the **manual `python setup_inner.py build_ext --inplace`** step. `cibuildwheel` solves that by building per-platform wheels in CI, so `pip install aaanalysis` ships the compiled extension to end users with zero install ceremony.

## Decision

Remove the Numba backend entirely. Keep Cython as the sole acceleration path. Make the Cython build *automatic at wheel-build time* via `cibuildwheel`. End-user experience: `pip install aaanalysis` (or `uv add aaanalysis`) downloads a wheel with `_inner.cpython-XYZ.so` already inside; `CPP.run_c` just works.

Specifically:

- **Delete** `aaanalysis/feature_engineering/_backend/cpp/_filters_num_nb/` (the entire Numba backend), the `test_get_feature_matrix_nb_parity.py` parity test, the `dev_run_nb.py` smoke harness, and the `bench_stage1_numba.py` bench. Also delete `aaanalysis/feature_engineering/_backend/cpp/_filters_num_c/setup_inner.py` (the standalone manual-build script is obsolete once setuptools handles the build).
- **Drop** `numba>=0.60` from `[project.optional-dependencies].pro`. SHAP transitively pulls numba today, but that's SHAP's problem to manage.
- **Drop** `CPP.run_nb` from the public API and the `_HAS_NUMBA_INNER` / `add_stat_func` / `filtering_func` plumbing from `cpp_run_num.py`. Final public surface: `CPP.run`, `CPP.run_num`, `CPP.run_c`, `CPP.eval`.
- **Switch the build backend** in `pyproject.toml` from `poetry-core` to `setuptools` — poetry-core can't build compiled extensions. Add a `setup.py` at the project root that declares the Cython extension via `cythonize()`. Poetry remains the dev-workflow tool (`poetry install`, `poetry.lock`) until a separate uv migration; only the *build backend* changes here.
- **Add `[tool.cibuildwheel]`** config to `pyproject.toml`: build matrix for cp310–cp314 on Linux (x86_64+aarch64), macOS (x86_64+arm64), Windows (AMD64). Skip PyPy and 32-bit platforms. Each built wheel runs `pytest tests/unit/cpp_tests/test_get_feature_matrix_c_parity.py -x -q` as the install-time sanity check so bit-exact parity is verified per (Python × OS × arch) wheel before publication.
- **Bump** `.github/workflows/build_wheels.yml` to `cibuildwheel 2.21.3`, replace the placeholder matrix-generation step with a working `cibuildwheel --print-build-identifiers | jq` pipeline.

## Considered alternatives

- **Keep both `run_c` and `run_nb`** (PR3 status quo). Rejected: two acceleration code paths with ~equivalent performance, double the maintenance, and `[pro]` users would still get the numba dep-hell exposure.
- **Drop Cython, keep Numba.** Rejected for the dep-hell argument above. The "no compile step" UX win for Numba is real but undone by recurring NumPy / Python / ARM / Colab compatibility issues — and the SHAP team's documented intention to drop numba is the clearest signal that the ecosystem trend is *away from* numba.
- **Keep `poetry-core` as build backend and use a `build.py` hook.** Rejected: Poetry's build-hook story for compiled extensions is fragile and effectively wraps setuptools anyway. Switching the backend cleanly is less moving-parts.
- **Switch to `hatchling` (with `hatch-cython` plugin) now.** Rejected for *now*: the rule file (`dependencies-and-pyproject.md`) targets hatchling for v2 and warns against premature migration. Setuptools is the well-trodden Cython path; hatchling can be revisited at v2 with a smaller diff.
- **Migrate dev workflow from Poetry to uv in the same PR.** Rejected: independent concern. Tracked as a separate GitHub Issue (linked from this ADR). End-user `pip install` and `uv add` both work regardless of which dev tool the project uses internally.
- **Re-architect the vectorized redundancy filter (Path 2 from PR3) to stream per-row instead of eagerly precomputing `(n_pre_filter, n_pre_filter)` matrices.** Possible but out-of-scope; the legacy filter is already fast enough at default settings (`n_filter=100 << n_pre_filter`).

## Consequences

- **Public surface shrinks** from 4 `run*` methods to 3 (`run`, `run_num`, `run_c`). Future work (separate ADR) may unify these into `run` with a backend `auto-dispatch` fallback chain.
- **User experience: `pip install aaanalysis` gets a fast CPP path with zero ceremony** on Linux / macOS / Windows + Python 3.10–3.14 (the wheel matrix). Users on unsupported platforms (e.g. 32-bit Linux, ARM Windows) fall back to the sdist and build from source via setuptools — the standard scipy/numpy pattern.
- **`pro` extras stay focused on the explainability + sequence-analysis features** (shap, biopython, UpSetPlot). The shap-pulls-numba transitive dep remains *until* SHAP's planned numba removal; once that ships, aaanalysis users on `[pro]` get a leaner install.
- **CI release cost rises** by ~15–25 min per release (cibuildwheel matrix). Acceptable — it runs on release tags only, not per PR.
- **Build-backend switch invalidates `poetry.lock` for build deps but not for runtime/dev deps.** Poetry still manages `poetry install` correctly because PEP 621 metadata is its source of truth.
- **The `_filters_num_c/_inner.pyx` source is unchanged.** The bit-exact pairwise-summation contract from PR3 / Phase D remains; only the build mechanism around it changes.
- **ADR-0002 is superseded but kept in `docs/adr/` as historical record.** Future readers tracing why `CPP.run_nb` doesn't exist will find the ADR and its supersession link.

## Followups (not in this PR)

Tracked in `docs/v2-followups.md`:

- **V2-1**: Migrate dev workflow from Poetry to uv. Independent of this ADR; end-user `pip install` and `uv add` already work today.
- ~~**V2-2**: Possibly unify `CPP.run` / `run_num` / `run_c` into a single `run` with auto-dispatch.~~ **Resolved in PR5 + PR6** (see amendments below).

(Remaining followups will be promoted to GitHub Issues when the v2 milestone starts.)

## Amendment (PR5): API consolidation + Cython auto-dispatch + `NumericalFeature.get_parts`

After ADR-0003 (PR4) landed, PR5 rewired `CPP.run` to route through the same Cython kernel `CPP.run_c` used (with pure-Python fallback when the compiled `.so` isn't available). `CPP.run_c` was then deleted from the public surface — its kernel lives on internally, selected by `cpp_run_num._pick_feature_matrix_builder()`. `CPP.run_num` was repurposed: it now REQUIRES `dict_num_parts` (call `CPP.run` for seq-mode); the only difference between the two public methods is the leading `dict_num_parts` slot.

The "preprocessing is multi-step" pain was solved by adding `NumericalFeature.get_parts(df_seq, dict_num, ...)`, which slices BOTH sequence strings and per-residue tensors with shared boundaries in one call. Eliminates the duplication of `df_seq + jmd_n_len + jmd_c_len` between `sf.get_df_parts` and a separate dict_num slicer.

Public surface after PR5: `CPP.run`, `CPP.run_num`, `CPP.eval` (three methods, down from PR4's four). All bit-exact with legacy `CPP.run` from the pre-PR5 state — verified by the `test_run_num_parity.py` suite.

## Amendment (PR6): remove legacy `_filters/` + rename `_filters_num/` → `_filters/`

After PR5 made the legacy `_filters/` unreachable from any public method, PR6 removed it entirely and collapsed the now-meaningless `_num` suffix from the surviving folder/module/function names. Three phases:

- **Phase A**: lifted `pre_filtering`, `filtering` / `filtering_info_`, and the multiprocessing helpers (`_FloatBox`, `_resolve_shared`, `_reset_progress`, `_get_mp_shared`, `_cleanup_mp_manager`, `_is_main_process`, defaults) from `_filters/*` into the corresponding `_filters_num/*` files as their canonical home. Replaced the thin re-exports.

- **Phase B**: deleted 9 files (~700 LoC of dead code):
  - `aaanalysis/feature_engineering/_backend/cpp_run.py` (legacy seq-mode orchestrator)
  - `aaanalysis/feature_engineering/_backend/cpp/cpp_run_.py` (legacy bridge)
  - All 7 files in `aaanalysis/feature_engineering/_backend/cpp/_filters/`

- **Phase C**: pure namespace operation — renamed folders/modules/functions to drop the `_num` suffix (which only existed as a disambiguator from the now-removed legacy `_filters/`):
  - `_filters_num/` → `_filters/`
  - `_filters_num_c/` → `_filters_c/`
  - `cpp_run_num.py` → `cpp_run.py`
  - `cpp_run_num_single` → `cpp_run_single` (and `_batch`, `_sample_batched`)
  - `assign_scale_values_to_seq_num` → `assign_scale_values_to_seq`
  - `pre_filtering_info_num` → `pre_filtering_info`
  - `add_stat_num` → `add_stat`
  - `recompute_feature_matrix_num` → `recompute_feature_matrix`

  **Kept** (the suffix is semantic): `CPP.run_num`, `NumericalFeature`, `assign_dict_num_to_parts`, `dict_num_parts`, `dict_part_vals`.

Net diff: ~+200 LoC lifted, ~−700 LoC deleted, ~150 LoC namespace renames = ~−500 LoC of removed dead code + a cleaner namespace.

Bit-exact verification: the 63-test parity suite (`test_run_num_parity.py` + `test_get_feature_matrix_c_parity.py` + `test_get_feature_matrix_fast_parity.py` + `test_nf_get_parts.py` + `test_cpp_run.py`) passes identically before and after each phase. Broader CPP/NF/SF suite (223 tests) also clean.
