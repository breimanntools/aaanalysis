# ADR-0015 — Exact-value CPP regression anchor, pinned to one canonical CI cell

Status: Accepted — 2026-06-03

## Context

The unit suite freezes core numeric behaviour only *locally* — the
Cython/pure-Python parity tests (`test_get_feature_matrix_c_parity.py`,
`test_run_num_parity.py`) assert the two builders agree, and
`test_run_num_structural.py` asserts output shape/columns. Nothing guards the
*end-to-end scientific result* of a `CPP` run against silent drift during
refactoring (the published-pipeline regression the package's Nature
Communications usage cares about).

A frozen "mini-pipeline" regression — load a tiny bundled dataset, run `CPP`,
assert the top feature and AUC do not change — fills that gap. But exact-value
freezing (`auc.round(3) == 0.912`) is only reproducible on a *fixed*
environment: the CI matrix spans py 3.11–3.14 × Linux/Windows, and this package
has already hit platform-specific numeric divergence (Windows C-`long` width in
the Cython kernel; numpy fp warnings in aaclust eval). A 3-dp assertion run on
every matrix cell would flake whenever a build shifts the third decimal.

## Decision

**D1 — Add one exact-value regression anchor.** A single test (
`tests/unit/cpp_tests/test_cpp_regression.py`) runs a seeded `CPP` pipeline over
a small fixed `aa.load_dataset("DOM_GSEC", …)` slice with fixed scales and
asserts (a) the top-ranked feature **identity** is stable and (b) the AUC equals
a frozen `round(3)` value. Frozen values are captured once and re-frozen only on
an intentional, reviewed change — never auto-updated to chase drift.

**D2 — Run it in the non-gating nightly on one canonical cell; deselect it from
the blocking matrix.** The test carries `@pytest.mark.regression` and
`@pytest.mark.skipif(not <canonical env>)` (Linux + floor Python 3.11;
`AAA_RUN_REGRESSION=1` forces it locally). The blocking workflows
(`main.yml`, `test_coverage.yml`) run `pytest tests -m "not regression"`, so the
exact-value assertion never gates a fail-fast matrix cell — its frozen values
are canonical-cell-specific and a 3rd-decimal drift must not block merges. It
runs (and re-verifies/re-freezes) in `mutation_nightly.yml` on Linux/py3.11.
This matches the intent: a regression guard on a reference environment,
per-nightly, not on every cell.

**D3 — It extends the unit-level parity precedent, not the integration tier.**
The anchor lives in `tests/unit/` and reuses the same `DOM_GSEC` fixture pattern
as the parity tests. It is therefore *not* the deferred `tests/integration/` /
`tests/e2e/` work (still deferred to v2 per `sharp-edges.md`); it is one frozen
unit-level guard.

## Rejected alternatives

- **Tolerance-based assertion (AUC within ±band, identity only).** Robust across
  the full matrix with no pin, but a looser signal — it would miss a real
  scientific regression that stays inside the band. Rejected in favour of the
  sharper exact-value guard plus the canonical-cell pin.
- **Exact values on the full matrix.** Maximal signal but periodic red builds
  and manual re-freezes whenever a platform shifts a decimal. Rejected per D2.
- **No pipeline-level regression (rely on parity tests).** The parity tests
  prove the two builders *agree with each other*, not that the result *has not
  changed* — they would both drift together silently. Rejected.

## Consequences

- New markers `regression` and `slow` are registered in `tests/pytest.ini`.
- The frozen feature/AUC values become a reviewed artifact; changing CPP scoring
  intentionally requires re-freezing them in the same PR.
- Off-canonical matrix cells report the anchor as skipped, not failed.
