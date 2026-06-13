# ADR-0030 — Strict-semver deprecation policy, `deprecated` decorator, and a two-file changelog

Status: Accepted — 2026-06-13

Relates to: the `api-stability.md` and `changelog.md` path-scoped rules; the
`Versioning and Deprecation Policy` section of `CONTRIBUTING.rst`.

## Context

AAanalysis is Production/Stable on PyPI and declares itself semver-strict from
v1.x onward, but had no mechanism to honor that promise: nothing marked a public
symbol as deprecated, and there was no root `CHANGELOG.md` (the format
`.claude/rules/changelog.md` already anticipated). A narrative changelog existed
only as `docs/source/index/release_notes.rst`, which is RTD-rendered prose — not
the terse, machine-greppable file developers and tooling expect at the repo root.
Separately, an external user (the antibody/nanobody project) ran `1.0.3`, which
predates the CPP performance work, hit hour-long low-CPU runs, and blamed the
algorithm — there was no changelog line telling them a newer release fixes it.

The Python floor is 3.11, so `typing.deprecated` / `warnings.deprecated`
(PEP 702, 3.13+) is unavailable.

## Decision

- **D1 — Strict-semver deprecation window.** Any rename/removal of a symbol in
  `aaanalysis.__all__` ships at least one **minor** release carrying a
  `DeprecationWarning` before the symbol is removed. PATCH never breaks the public
  API; MINOR may introduce deprecations; MAJOR may complete removals.
- **D2 — Hand-rolled `deprecated(reason, version_removed)` decorator** in
  `aaanalysis/_utils/decorators.py`, exposed through the `ut` barrel as
  `ut.deprecated`. It wraps a function/method (warn on call) or class (warn on
  instantiation) with a `DeprecationWarning`, preserves the signature via
  `functools.wraps`, and prepends a `.. admonition:: Deprecated` note so the
  deprecation shows both at call time and in the rendered API docs. (An
  `.. admonition::`, not the `.. deprecated::` directive: the latter requires a
  "deprecated since" version argument we do not track, and an argument-less
  `.. deprecated::` is a Sphinx build error.)
- **D3 — `deprecated` is an internal helper, not public API.** It is reachable as
  `ut.deprecated` for maintainers but is not added to `aaanalysis.__all__`.
- **D4 — Two-file changelog, terse + narrative.** A root `CHANGELOG.md`
  (Keep a Changelog) is the terse, one-line-per-change developer index;
  `release_notes.rst` stays as the narrative RTD release notes. A user-visible
  change updates **both** in the same PR, and each file cross-links the other.
- **D5 — Record the CPP-performance release line.** The changelog explicitly
  notes that the Cython feature-matrix kernel, threaded `n_jobs`, caching, and
  batching land post-`1.0.3`, and that `≤1.0.3` users should upgrade.

## Rejected alternatives

- **`typing.deprecated` / `warnings.deprecated` (PEP 702).** Cleaner and
  type-checker-aware, but requires Python 3.13; the floor is 3.11. Revisit when
  the floor rises.
- **Single changelog only.** Either keep just `release_notes.rst` (no root file
  that tooling/users expect, and the rule already anticipates one) or replace it
  with `CHANGELOG.md` (discards the RTD-rendered narrative and its cross-links).
  Keeping both, with terse one-liners in the root file, makes the ongoing
  maintenance cost of the second file negligible while satisfying both audiences.
- **Making `deprecated` public API.** It is a release-engineering tool, not part
  of the analysis surface; exposing it would enlarge the stability contract for no
  user benefit.
- **Auto-generating one changelog from the other.** No generator exists and the
  two serve different audiences (terse vs narrative); a generator would couple
  their formats and add build machinery for little gain.

## Consequences

- Future public-API renames/removals must go through `ut.deprecated` for ≥1 minor
  release; the `Removed` changelog subsection records the eventual removal.
- Contributors update `CHANGELOG.md` + `release_notes.rst` in the same PR
  (now stated in `CONTRIBUTING.rst` and its RTD copy).
