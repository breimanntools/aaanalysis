# AAanalysis test suite

## Philosophy

Tests exercise the package **only through the public API** — `import aaanalysis as aa`
and the symbols re-exported by `aaanalysis/__init__.py`. Private backend functions are
never imported or called directly; a backend branch is covered by feeding the
public method an input that routes there. This keeps the tests honest about what users
can actually reach, and keeps the suite stable across internal refactors.

House conventions live in `.claude/rules/testing.md`. In short: one file per public
method, `Test<Method>` / `Test<Method>Complex` classes, negative tests with
`pytest.raises(..., match=...)`, property tests with `@given` + `@settings(deadline=None)`,
and golden-value checks for the scientific core.

## Test tiers (unit · integration · e2e)

Three tiers with **distinct jobs** (full taxonomy + rationale in
`.claude/rules/testing.md` and **ADR-0031**). The rule that keeps them from
overlapping: *don't re-test at a higher tier what a lower tier already covers.*

| Tier | Directory | Job | Negatives are… | Hypothesis is… |
|---|---|---|---|---|
| **unit** | `tests/unit/` | one public method, in isolation | per-parameter invalid-input sweep | per-argument strategies |
| **integration** | `tests/integration/` | a **seam** between two/three real classes (no mocks) | **composition failures only** | pipeline invariants / metamorphic |
| **e2e** | `tests/e2e/` | a full **workflow** mirroring a `protocols/*.ipynb` | degenerate dataset → clear error | reproducibility (seed→same artifact) |

- **Integration** (`test_feature_pipeline`, `test_prediction_pipeline`,
  `test_seq_pipeline`, `test_data_pipeline`) asserts that one component's output is a
  valid input to the next — e.g. `CPP.run` → `feature_matrix` → `TreeModel.fit`,
  `dPULearn` carve → `TreeModel`, `AAWindowSampler` → `AAlogo`, `CPPGrid` sweep + `eval`,
  fasta round-trip, encoder → consumer. Its negatives are failures that only emerge *when
  components meet* (shape/label-set mismatch, PU dataset with no unlabeled rows, a sampler
  that returns no windows) — never a re-run of unit-level input validation.
- **e2e** (`test_protocols.py`) runs one tiny, seeded, assertion-bearing analogue of each
  `protocol1`–`10` notebook (the notebooks themselves assert nothing). Core-only: where a
  protocol uses a `pro` feature it substitutes the core path.
- Both tiers assert **structural/range** artifacts (shapes, schema columns, metrics in
  `[0, 1]`, finiteness, monotonic ranking) and **never frozen exact values** — exact-value
  freezing is the CPP regression anchor's job (ADR-0015). Their shared seeded spine lives
  in `tests/_pipeline.py`; integration exposes it as session fixtures in
  `tests/integration/conftest.py`, and e2e imports the builders directly. Count is bounded
  by *seams × workflows*, not by unit-test volume.

`tests/integration/test_online_fetches.py` is a separate, `network`-marked live-endpoint
tier (deselected by default), not part of the cross-component integration tier.

## Running

```bash
# full suite (parallel)
pytest tests -m "not regression" -n auto -c tests/pytest.ini

# just the cross-component + workflow tiers
pytest tests/integration tests/e2e -m "not regression" -n auto -c tests/pytest.ini

# branch + line coverage gate (matches CI: line >=88, branch >=80)
COVERAGE_CORE=sysmon pytest tests -m "not regression" \
    --cov=aaanalysis --cov-branch --cov-report=xml -n auto -c tests/pytest.ini
python .github/scripts/check_branch_coverage.py
```

Always pass `-c tests/pytest.ini` (it registers the `regression`/`slow`/`integration`/`e2e`
markers and the `filterwarnings` for the deliberate CPP advisories). Coverage config is
`/.coveragerc`.

**In CI** the `integration`/`e2e` tiers run in their own **`Integration & E2E Tests`**
workflow (`.github/workflows/integration_e2e.yml`) and are excluded from the `Unit Tests`
matrix (`main.yml` runs `-m "not regression and not integration and not e2e"`) so they
appear as their own check and never double-run.

## Coverage policy

Current: **line ~98%, branch ~95%** (gates 88% / 80%, ratcheting per ADR-0016).

Branch coverage is pursued **only via the public API**. The arcs that remain uncovered
are not gaps in the tests — they are **unreachable from the public surface by design**,
and fall into four categories:

1. **Defensive backend guards.** The frontend validates/normalizes input *before* the
   backend runs, so the backend's own guard arm can never fire from a public call
   (e.g. `check_feature.py` "should be already checked by interface"; backend
   `if x is None` where the frontend always passes a concrete value).
2. **Dead code.** Functions defined but never called — see the inventory below.
3. **Heavier-fixture pro paths.** A few `data_handling_pro` (structure) arcs need a
   multi-chain PDB, a `.cif.gz`, or a network fetch (UniProt/AlphaFold) — out of scope
   for offline, public-API tests.
4. **Subprocess-attributed arcs.** Lines that execute inside joblib/loky worker
   processes, which `coverage.py` does not measure across the process boundary.

Categories 1, 3 and 4 are left uncovered intentionally. They would only be "closeable"
with `# pragma: no cover` (source edits) or heavier fixtures — not with more public-API
tests.

## Dead-function inventory

Found while broadening branch coverage. Each is **private** (a frontend module-level
helper not in `aaanalysis.__all__`, or a backend helper) and has **zero call sites** in
both `aaanalysis/` and `tests/` (verified by `grep "<name>("` excluding the definition).
They are removal candidates; source was **not** modified (deletions need maintainer
sign-off).

| Function | Location | Note |
|---|---|---|
| `check_sample_in_df_seq` | `feature_engineering/_cpp.py:75` | validator with no caller |
| `_check_dict_num` | `feature_engineering/_cpp.py:129` | superseded by `_check_dict_num_parts` (which *is* used at `_cpp.py:833`) |
| `_check_dict_num_df_scales_match` | `feature_engineering/_cpp.py:163` | validator with no caller |
| `check_match_df_seq_df_parts` | `feature_engineering/_cpp.py:173` | validator with no caller |
| `check_match_df_eval_names` | `feature_engineering/_aaclust_plot.py:20` | validator with no caller |
| `_get_starts` | `feature_engineering/_backend/cpp/_utils_cpp_plot_positions.py:179` | helper method with no caller |
| `set_title_` | `feature_engineering/_backend/cpp/_utils_cpp_plot_elements.py:45` | helper with no caller |
| `gather_means_chunked` | `feature_engineering/_backend/cpp/_filters/_recompute.py:179` | unused sibling of `recompute_feature_matrix` (which *is* used) |

Verify before removing:

```bash
grep -rn "check_match_df_eval_names(" aaanalysis/ tests/ | grep -v "def "
```

> Note on the orphan module `seq_analysis_pro/_comp_seq_cons.py`: it is unreachable from
> the public API entirely (no `__init__` imports it) and is excluded from coverage in
> `/.coveragerc` rather than tested. It is a separate "wire-it-up or remove" decision.
