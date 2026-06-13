---
paths:
  - "CHANGELOG.md"
  - "CHANGELOG"
---

# CHANGELOG

> **Status:** active — `CHANGELOG.md` exists at the repo root (added with the
> strict-semver deprecation policy, ADR-0030).

- `CHANGELOG.md` lives at repo root, in
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
- Sections per release: `Added / Changed / Deprecated / Removed / Fixed`.
- An `Unreleased` section is always present at the top.
- **Claude maintains the `Unreleased` section.** Whenever a PR makes a
  user-visible change (new public symbol, signature change, behavior change,
  deprecation, important bug fix), Claude appends a one-line entry under the
  appropriate subsection in the same PR.
- `CHANGELOG.md` is the **terse** index; the narrative, RTD-rendered notes live
  in `docs/source/index/release_notes.rst`. Update **both** in the same PR — the
  terse one-liner here, the prose entry there — and keep their `Unreleased`
  sections in sync. Each file cross-links the other.
- **Deprecation policy (semver-strict, ADR-0030).** Renaming/removing a public
  symbol (one in `aaanalysis.__all__`) ships ≥1 minor release decorated with
  `ut.deprecated(reason=..., version_removed=...)` (a `DeprecationWarning` +
  a docstring deprecation note) before removal. Record the eventual removal
  under `Removed`. The decorator is internal (`ut.deprecated`), not public API.
- No `SECURITY.md` for now (documented gap).
