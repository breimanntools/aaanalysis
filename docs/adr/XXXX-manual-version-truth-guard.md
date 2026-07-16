# ADR-XXXX — Version truth: a manual version kept ahead of PyPI, enforced by a CI divergence guard

Status: Accepted — 2026-07-16

## Context

`pyproject.toml` on `master` declared `version = "1.0.3"` — byte-identical to the PyPI 1.0.3 release
(Apr 2026) — while `master` carried substantial unreleased work (AAPred, SeqOpt, ReliabilityModel,
the golden pipelines, and more). Both trees therefore answered `1.0.3` to
`importlib.metadata.version("aaanalysis")`, so a development checkout and a released install were
**indistinguishable from the outside**. That breaks bug triage (a reported "1.0.3" could be either
tree), cached environments, coding agents reasoning about capabilities, and scientific
reproducibility — the version string is the one identifier a published analysis records. It also
made the "Production/Stable" classifier premature, and it is the state ADR-0060 worked around when
it noted "v1.1 unreleased (`pyproject` at `1.0.3`)".

Nothing detected the divergence: the version is a hand-edited constant touched only at release time,
and no check compared it against what was actually published.

## Decision

**D1 — The declared version is manual and must always name the *next, unreleased* number.** It stays
a plain hand-edited `[project] version` string in `pyproject.toml`. The invariant is
`pyproject_version > latest_published_release`, held at all times on `master`, not just at release
time. `master` is bumped to `1.1.0`, matching the `v1.1.0 (Unreleased)` section the release notes
already declared.

**D2 — A dedicated `version_guard.yml` workflow enforces the invariant.**
`.github/scripts/check_version_ahead.py` parses `[project] version`, resolves the latest published
release, and exits non-zero unless the declared version is strictly greater. It is its own workflow
rather than a step bolted onto an existing job so the failure is legible on its own status line, and
so it can be made a required check independently. It is **code-gated** (the same
`paths-ignore: docs/**, **/*.md, **/*.rst` as the other code-gated workflows): a docs-only change
cannot alter the declared version, and `pyproject.toml` is not a docs path, so every version edit
still runs it.

**D3 — PyPI is the source of truth for "published", with git tags as an offline fallback.** The
guard reads PyPI's `info.version` (its own latest-release pointer, so yanked and pre-release uploads
need no hand-filtering). When the API is unreachable it falls back to the highest `vX.Y.Z` git tag,
which keeps the guard usable locally and offline; CI checks out with `fetch-depth: 0` so tags are
present. Tags are compared by *parsed version*, not by name, so `v1.0.10` beats `v1.0.9`.

**D4 — An unresolvable "published" version reports loudly and passes.** If PyPI is unreachable *and*
no git tags exist, the guard prints `INCONCLUSIVE` and exits 0. A PyPI outage is an infrastructure
failure, not a version error, and must not block every merge in the repo; the real defect this guard
exists to catch (a forgotten bump) is always detectable when either source resolves.

**D5 — Release tags are `vX.Y.Z`; the guard tolerates the legacy unprefixed spelling.** The tag regex
accepts an optional leading `v`, so the pre-1.0 `0.1.1` tag (which predates the convention) still
parses and counts instead of being silently skipped. This is the "set the tag regex the guard reads"
half of the tag-hygiene requirement: it makes the guard correct today without a tag rename, which
would mean deleting a published remote tag. Renaming `0.1.1` → `v0.1.1` is left to the maintainer as
cosmetic cleanup, not a correctness fix.

## Rejected alternatives

- **Derive the version from git tags (setuptools-scm / hatch-vcs).** The obvious "single source of
  truth" answer, and it makes divergence structurally impossible. Rejected because it attaches
  `.devN` / `+g<sha>` suffixes to every non-tagged commit: every dev install reports a distinct
  noisy string, the version stops being a stable thing a human can quote in an issue, and the build
  gains a VCS dependency (a `pip install` from a tarball without `.git` has no version at all). For
  a package with infrequent, deliberate, manual releases, the cost outweighs the benefit — the same
  guarantee is obtainable from one CI check over a hand-edited constant.
- **`.devN` / pre-release suffixes on `master` (e.g. `1.1.0.dev0`).** Truthful, and conventional in
  many projects. Rejected for the proliferation it invites (which commit is `dev7`?) and because it
  buys nothing the strict `>` comparison does not already give: `1.1.0` on `master` is *already*
  distinguishable from the published `1.0.3`.
- **Fail the build when the published version cannot be resolved.** Strictly safer in theory, but it
  converts every PyPI outage into a repo-wide merge freeze. Rejected; see D4.
- **Fold the check into an existing workflow (e.g. the packaging or unit-test job).** Fewer
  workflows, but the failure would be buried inside an unrelated job's log and could not be required
  on its own. Rejected; see D2.
- **Compare against the newest git tag only, never touching the network.** Simpler and hermetic, but
  a tag is a *local claim* about a release while PyPI is the *actual* publication record; the two can
  drift (a tag pushed without a publish, or a publish from an untagged commit). Rejected as the
  primary source, kept as the fallback.

## Consequences

- `master` now reports `1.1.0`; `python -c "import aaanalysis; print(aaanalysis.__version__)"` no
  longer collides with the published release.
- The release procedure gains a closing step: after publishing `X.Y.Z` and tagging `vX.Y.Z`,
  immediately bump `master` to the next unreleased number. Until that lands the guard fails — by
  design, as the reminder that the bump is owed. Documented under *Version truth* in
  `CONTRIBUTING.rst` and its RTD port.
- The comparison logic is pinned by unit tests with fixtures (`test_check_version_ahead.py`),
  including the exact regression (declared == published must fail) and the dual
  `[project]` / `[tool.poetry]` block sharp edge.

## Out of scope

- **Docs-version coherence** (`conf.py` `version` / `release`, RTD stable-vs-latest, the dev banner)
  — tracked separately; this ADR covers only the package version string.
- **Renaming the `0.1.1` tag** to `v0.1.1` — maintainer cleanup; the guard already handles it (D5).
- **Trusted publishing / release-job hardening / a single canonical release path** — separate
  release-maturity work.
