# ADR-0016 — Measure coverage on the package only; ratchet module gates

Status: Accepted — 2026-06-03

## Context

The Test Coverage workflow runs `pytest tests --cov=./`, so the coverage
denominator includes the **test files themselves** (~30,700 lines, 29,137
covered → 94.9% on Codecov). Measured against the package alone
(`--cov=aaanalysis`), the honest figure is ~90% (≈12,600 statements). The
headline therefore overstates package coverage by ~5 points, and any
*per-module* target is meaningless while test files dilute the denominator.

There is also no enforced floor today: the coverage workflow has no
`--cov-fail-under`, and there is no `codecov.yml` — Codecov is report-only. So
gates are being chosen from zero, not tightened from an existing one.

A blanket high target is brittle here: some core lines are *provably
unreachable* defensive branches (e.g. `encode_dssp.py:37` — every
`DICT_DSSP_3STATE` value except the pre-caught `'-'` already maps into
`{H,E,C}`). A hard 100% with no policy forces `# pragma: no cover` gaming. (Note:
several lines that *look* dead are live — the `_annot_preproc` `sequence` guards
are reached by a part-based `df_seq`, which `check_df_seq` accepts; those are
covered by tests, not excluded.)

## Decision

**D1 — Measure the package, not the tests.** Add `codecov.yml` with
`ignore: ["tests/**", "setup.py"]` (equivalently, `--cov=aaanalysis` in CI) so
the reported number reflects the package. This is a one-time, fully explainable
drop of the public badge from 94.9% to the honest ~90%.

**D2 — Ratchet the gates; do not flip absolutes on at today's level.** Enable
now, all green at ~90%: Codecov `project: {target: auto}` (block any drop vs
base) + `patch: {target: 90%}` (new code), plus a CI backstop
`--cov-fail-under=88`. The aspirational targets — `project 95%`, a `cpp_core`
component (the `_filters/**` numeric pipeline) at 100%, `patch 95%` — are the
documented climb; their hard numbers are turned on per-component only once the
tests actually reach them, and the CI floor is raised in tracked steps.

**D3 — 100% core is honest, not gamed.** The `cpp_core` 100% target is reached
by writing real tests and by removing only *proven-dead* lines (prove no input
satisfies the upstream validator while failing the guard); reachable guards get
tests that also assert their tailored error message. `# pragma: no cover` is
allowed only with an inline justification.

## Rejected alternatives

- **Keep `--cov=./` (flattering 94.9%).** Avoids the visible drop but the number
  lies about package coverage and makes per-module gating impossible. Rejected.
- **Measure honestly locally, leave CI inflated.** Divergent internal/external
  numbers; the public badge still misleads. Rejected.
- **Enable 95% / core-100% / 95% immediately.** Red builds at today's 90% until
  the work lands, blocking all merges. Rejected per D2.
- **Lint / type-check CI gate** (from the same feedback). Out of scope:
  `sharp-edges.md` defers ruff/mypy/pre-commit to v2; that decision stands.

## Consequences

- Public Codecov badge drops ~5 points in one commit — **expected and
  intentional; do not "fix" it back** to `--cov=./`.
- A `codecov.yml` and a CI `--cov-fail-under` floor exist for the first time.
- The `cpp_core` Codecov component and the absolute targets are added/raised
  incrementally as coverage climbs, not in one switch.

## Addendum — Branch + parameter coverage (2026-06, issue #84)

Line coverage alone has two blind spots the ratchet above does not catch: an
untested *branch* (only one arm of a conditional runs) and an unexercised public
*parameter* (a kwarg added, documented, never called with a non-default value).
Both pass the line gate green. Two checks close them, in the same package-only,
ratcheted spirit as D1–D3.

**D4 — Gate branch coverage with a separate, honest number.** The coverage
pytest step adds `--cov-branch`. Crucially, `--cov-fail-under` is *not* used for
this: once branch is on, that flag measures the *combined* line+branch number,
which would silently re-define the historical line floor. Instead a post-pytest
step (`.github/scripts/check_branch_coverage.py`) parses the cobertura `coverage.xml`
root attributes `line-rate` and `branch-rate` and fails if either drops below its
own committed gate — keeping an independent line floor (88, unchanged) and branch
floor. The branch gate is set conservatively at-or-below the measured baseline and
ratcheted up in tracked steps, never flipped to an aspirational number.

**D5 — Enforce parameter coverage with a meta-test.** `tests/unit/api_tests/
test_param_coverage.py` enumerates every public parameter of every symbol in
`aaanalysis.__all__` (for classes: `__init__` + every public method; properties,
`self`/`cls` and `*args/**kwargs` excluded) and fails if a parameter is neither
referenced as a keyword argument anywhere under `tests/` nor on an explicit
`ALLOWLIST` with a reason. Two deliberate trade-offs:

- *Detection is global and name-based.* A parameter counts as covered if its name
  appears as a kwarg at any call site, not necessarily a call to the owning
  symbol. Per-(symbol, method, param) attribution would need call-site type
  resolution; the global approach is simple and robust to test-layout drift, at
  the cost of a known false-positive surface for ambient names (`verbose`,
  `random_state`, `n_jobs`, `df_seq`, ...). The complementary branch gate (D4)
  still catches genuinely untested code paths those names mask.
- *Pro symbols are skipped when their extra is absent.* When an optional
  dependency is missing the public symbol is a `missing_feature_stub` lambda with
  no real signature; the meta-test detects that and skips it with a recorded
  reason, so the check is green in a core-only (`[dev]`) environment and enforces
  the pro surface only where `[pro]` is installed.

The `ALLOWLIST` is the registry of justified, intentionally-untested params —
visual-only styling passthroughs (`plot_legend`, `AALogoPlot` logomaker kwargs)
and whole-class gaps awaiting a dedicated suite (`AAMut`/`SeqMut`). It is kept
small; the test reports a *test-covered* percentage (allowlist excluded) that must
stay ≥95%, so allowlist growth is visible drift.

**Rejected for D4/D5:** folding branch into `--cov-fail-under` (re-defines the
line floor); a `dev_scripts/` script for D5 (cannot gate CI); allowlisting
behavioural params to reach green (only visual-only passthroughs and whole-class
gaps are allowlisted — behavioural params get a real test).
