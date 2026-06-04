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
