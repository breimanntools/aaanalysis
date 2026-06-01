# Handoff — aaanalysis v1.2 session (CPPGrid + smart-sweep, plot_rank, batch feature_matrix, CI fixes)

## State
- Branch: `feat/structure-preprocessor-v1.2`. **All work committed AND pushed to `origin/master`** (fast-forward).
  `HEAD == origin/master == 9d8d4586`, working tree clean (aside from this `HANDOFF.md`).
- **master CI is green** (Unit Tests across py 3.11–3.14 × Linux/Windows, Test Coverage, CodeQL) at `3cf53035`;
  the docs-only `8a8041dc` and deps-only `9d8d4586` on top are no-ops for code tests.
- Full local unit suite last run: **2282 passed, 1 skipped, 0 warnings** (run with `-c tests/pytest.ini`).
- Dependabot lock bump done this session as `9d8d4586` (see REMAINING for residual).

## Authoritative artifacts (do NOT duplicate — read these)
- Approved design plan: `~/.claude/plans/magical-toasting-ripple.md`
- Live progress + gotchas memory: `~/.claude/projects/-Users-stephanbreimann-Programming-1Packages-aaanalysis/memory/project_feedback_integration_plan.md`
- ADRs written this session: `docs/adr/0006`–`0009` (rank-plot / cppgrid / n-jobs / discard-cppstate)
- Glossary: `CONTEXT.md` (new entries: CPPGrid, df_params, params_*, last_filter_stats_, n_jobs contract, the 4 site-localization metrics, rank plot; extended `pos column`)
- Source feedback file `aaanalysis_feedback.md` was DELETED this session (was untracked; fully addressed — see below)

## Newly implemented this session (public API)

### Class
- **`CPPGrid`** (`aaanalysis/feature_engineering/_cpp_grid.py`; exported `aa.CPPGrid`). Tool-style grid sweep over CPP configs.
  - `CPPGrid.__init__(df_seq, labels, *, dict_num=None, accept_gaps=False, verbose=True, random_state=None, n_jobs=-1, backend="threads")`
  - `CPPGrid.run(params_parts=None, params_split=None, params_scales=None, params_cpp=None) -> (list_df_feat, df_params)`
  - `CPPGrid._build_parts(...)`, `CPPGrid._run_base(...)` (internal workers)
  - Module helpers: `_as_candidates`, `_is_scalar_axis`, `_expand_with_records`, `_scales_candidates`,
    `_resolve_df_cat`, `_n_warnings_member`, `_combo_key`, `_err_record`, `_JOBLIB_BACKEND`
  - **Smart-sweep (key feature):** configs differing only in `n_filter` run CPP ONCE at max(n_filter), the rest are
    exact `head(n)` slices (verified byte-identical; ~14× on a 15-value sweep). `df_parts`/`split_kws` cached per sub-config.
    Threads-default across configs (loky opt-in), inner `n_jobs=1`. Lightweight `df_params` (object axes as index, +`n_warnings`/`n_errors`).

### Functions
- **`aa.plot_rank`** (`aaanalysis/plotting/_plot_rank.py`) — standalone per-protein rank scatter (max-score-vs-rank, group
  coloring, threshold lines). Helpers `_resolve_group_colors`, `_DEFAULT_GROUP_COLORS`. Full type-validation block.
- **`ut.get_window_offsets(window_size)`** (`aaanalysis/utils.py`) — canonical right-heavy P1-anchor geometry.

### New method / new method-mode
- **`SequenceFeature.feature_matrix(..., batch=False)`** — added `batch` param: `batch=True` takes a **list** of `df_parts`,
  concatenates → one Cython build → splits back → list of X (exact vs per-call, ~7× on small batches).
  (Earlier added as a separate `feature_matrix_batch` then **refactored into the `batch=` flag** per user request — no standalone method.)
- **Anchor (`pos`) input mode** added to `SequenceFeature.get_df_parts(..., tmd_len=None)` and
  `NumericalFeature.get_parts(..., tmd_len=None)`: a `(sequence + pos)` df_seq explodes each 1-based anchor into one 3-part
  row (ided by `entry_win`). Backend helpers `expand_pos_anchors_`, `_parse_anchor_cell` in
  `aaanalysis/feature_engineering/_backend/check_feature.py`.
- **`aa.__version__`** exposed in `aaanalysis/__init__.py` via `importlib.metadata`.

### Behavior/warnings (no new symbols)
- D5b low-feature `UserWarning` (sparse-config `n_candidates < n_filter`) + D7 shortfall `RuntimeWarning`, mutually
  exclusive, in `_backend/cpp_run.py:_attach_filter_stats`.
- `CPP.run` docstring Notes: O(n_scales × n_parts × n_splits) complexity note + classifier-head↔metric guidance + the
  Py3.14/macOS `__main__`-guard `n_jobs` footgun `.. warning::`.
- `CPP.run_num` docstring Notes: "PLM embeddings ARE the dict_num" + three-arms explanation.

### Tests added (all green; one positive + one negative per parameter, n·2 convention)
- `tests/unit/version_tests/test_version.py`
- `tests/unit/cpp_grid_tests/test_cpp_grid.py` (incl. `TestRunShortcuts` for the n_filter collapse)
- `tests/unit/plotting_tests/test_plot_rank.py` (incl. wrong-type negatives for cosmetic params)
- `tests/unit/sequence_feature_tests/test_sf_get_df_parts_anchor.py`, `test_sf_feature_matrix_batch.py`
- `tests/unit/numerical_feature_tests/test_nf_get_parts_anchor.py`
- extended `tests/unit/cpp_tests/test_filter_stats.py` (D5b)

### Docs
- 3 notebooks: `examples/plotting/plot_rank.ipynb`, `examples/feature_engineering/cpp_run_num.ipynb`,
  `tutorials/tutorial6_comparison_harness.ipynb` (all exec-clean; `nbmake` not installable locally — verify by exec'ing cells).
- `docs/source/api.rst` restructured to match `aa.__all__` exactly (StructurePreprocessor/AnnotationPreprocessor under
  **Data Handling**; standalone Structure/Annotation sections removed; CPPGrid + the new metrics/plot_rank added).

## Full public-API delta since v1.0.3 (the entire v1.1 + v1.2 branch, not just this session)
Diff of `aa.__all__` at tag `v1.0.3` vs `HEAD`. **(★ = added in THIS session; the rest landed in earlier v1.1/v1.2 branch commits.)**
Already present in v1.0.3 (NOT new): SequencePreprocessor, NumericalFeature, AAlogo, AAlogoPlot, comp_seq_sim, filter_seq.

### Preprocessing classes
- **`EmbeddingPreprocessor`** (core, `aaanalysis/data_handling/`) — instance-based. Methods:
  `build_pseudo_scales`, `cluster_pseudo_scales`.
- **`StructurePreprocessor`** (pro, `aaanalysis/data_handling_pro/`) — Methods:
  `get_dssp`, `encode_dssp`, `encode_pdb`, `encode_pae`, `get_domains`, `encode_domains`,
  `build_pseudo_scales`, `build_cat`.
- **`AnnotationPreprocessor`** (pro, `aaanalysis/data_handling_pro/`) — Methods:
  `fetch_uniprot`, `ingest`, `register_feature`, `encode`, `build_pseudo_scales`, `build_cat`, `to_df_seq`.
- (related top-level fn) **`combine_dict_nums`** (core) — concatenate per-residue `dict_num` tensors along D.

### Sampling / sequence-analysis
- **`AAWindowSampler`** (core, `aaanalysis/seq_analysis/`) — Methods:
  `sample_same_protein`, `sample_different_protein`, `sample_motif_matched`, `sample_synthetic`.
- **`scan_motif`** (pro function, `seq_analysis_pro`) — MEME/FIMO motif scan.

### Feature engineering
- ★ **`CPPGrid`** (see this-session section above) — `.run` + smart n_filter sweep.
- ★ **`SequenceFeature.feature_matrix(..., batch=)`** (new batch mode) and ★ `pos`-anchor input mode on
  `get_df_parts` / `NumericalFeature.get_parts` (`tmd_len=`).

### Metrics / plotting / misc top-level functions
- ★ **`comp_per_protein_ap`**, ★ **`comp_detection_metrics`**, ★ **`comp_bootstrap_ci`**, ★ **`smooth_scores`** (`aa.metrics`).
- ★ **`plot_rank`** (`aa.plotting`).
- **`display_df`** (dev-extra; `show_html`).
- ★ **`aa.__version__`** attribute.

Method lists above were introspected from the installed classes (pro extra present). The data_handling_pro
consolidation + the preprocessor unified protocol are recorded in `docs/adr/0005-feature-preprocessor-family.md`
and memory `data-handling-pro-move-pending` / `preprocessor-api-shape`.

## CI fixes landed this session (root causes worth remembering)
1. **Windows Cython buffer dtype mismatch** — kernels in `_inner.pyx` declare `long[::1]` (C long = 32-bit on Windows) but
   the glue passed int64. Fixed in `_backend/cpp/_filters_c/_get_feature_matrix_c.py` by casting seq_lens/positions/list_pos
   to numpy typecode `'l'` (C long, matches on every platform). Commit `76b48c3f`.
2. **Hardcoded `/tmp` in a test** (doesn't exist on Windows) — `test_structure_preprocessor_get_dssp.py::test_invalid_no_mkdssp_binary`
   switched to the `tmp_path` fixture. Commit `3cf53035`. (Two more `/tmp` uses in `test_…encode_dssp.py` PASS on Windows
   because they assert bare `ValueError`; left as-is, but worth tidying.)
- **aaclust eval numpy fp warning** silenced at source with `np.errstate(over/invalid/divide="ignore")` in
  `_backend/aaclust/aaclust_eval.py:_evaluate_clustering` (NaN/inf results already handled by warn_ch/warn_sc).

## Environment gotchas (cost real time)
- `coverage`/`pytest-cov` CRASHES locally on this machine (numpy reimport) on BOTH system py3.14 AND `.venv` py3.13
  → cannot measure coverage % locally; rely on the CI Test Coverage job (it's green).
- Run tests with `-c tests/pytest.ini` so the D5b/D7 advisory warnings are filtered (already configured there).
- `.venv` is py3.13; bare `python3` is system py3.14. CPP Cython kernel IS compiled/active locally.
- `aa.load_dataset(name="DOM_GSEC", n=N)` returns **2N** rows (n per class); use `df_seq["label"].to_list()`.
- Pushing to **master** triggers 3 workflows; feature-branch pushes trigger none (CI gated to master push/PR).

## REMAINING / next steps
- **Dependabot lock bump — DONE** (`9d8d4586`, pushed): dulwich/urllib3/mistune/idna upgraded in `uv.lock`,
  and `urllib3` floor raised to `>=2.7.0` in `pyproject.toml`. Residual: the **2 mistune "no patch" alerts**
  can't be resolved by upgrading — leave them until upstream ships a patch.
- D9a calibration example was intentionally SKIPPED by the user (decision: no `aa.calibrate`; sklearn covers it).

## Suggested skills for next session
- No `grill-with-docs` (planning complete).
- For dependabot: just `uv lock` + verify; `code-review`/`simplify` before any commit.
- NEVER push/commit without explicit per-action approval (hard rule); stage precisely.
