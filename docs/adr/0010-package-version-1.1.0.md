# ADR-0010 — Next package release is v1.1.0, not v1.0.4

Status: Accepted — 2026-06-01

## Context

The package on PyPI is at **v1.0.3**. A large body of work accumulated on the
`feat/structure-preprocessor-v1.2` branch and is **unreleased**: the
feature-preprocessor family (`EmbeddingPreprocessor`, `StructurePreprocessor`,
`AnnotationPreprocessor`, `combine_dict_nums`), numerical-mode CPP
(`CPP.run_num`), `AAWindowSampler`, `scan_motif`, `CPPGrid`, `plot_rank`, the
site-localization metrics (`comp_per_protein_ap`, `comp_detection_metrics`,
`comp_bootstrap_ci`, `comp_smooth_scores`), new method modes
(`SequenceFeature.feature_matrix(batch=)`, the `pos`-anchor input mode on
`get_df_parts` / `get_parts`), and `aa.__version__`.

The version label was ambiguous. Commit messages and ADR-0002 carry the strings
"v1.1" and "v1.2", and the branch is named `…-v1.2`. None of these are the
package version: "v1.1"/"v1.2" there denote the **StructurePreprocessor
feature-set revision** (rev 1 = DSSP+PDB; rev 1.1 = +AlphaFold) and **git branch
names**. The package version line is `1.0.1 → 1.0.2 → 1.0.3 → ?`, and a naïve
read suggested the next tag might be a patch (`1.0.4`).

The root `CLAUDE.md` mandates **semver-strict from v1.x onward**. This release
adds new public API, which under semver is a *minor* increment, not a *patch*.

## Decision

**D1 — The next package release is `v1.1.0`** (minor bump from `v1.0.3`). New
public symbols (classes, functions, methods) require a minor under semver-strict;
a patch (`1.0.4`) would violate the rule and is rejected.

**D2 — All unreleased branch work ships in this single `v1.1.0` release.** It is
not split into separate `v1.1`/`v1.2` package releases — that split exists only
in commit history and reflects internal milestones, not shipped versions.

**D3 — The package version is the *only* authoritative version line, and it
follows semver.** "v1.1"/"v1.2" in commit messages, branch names, and ADR-0002
refer to StructurePreprocessor feature-set revisions or branches, never package
versions. A bare "v1.1" must not be written to mean a feature-set revision (use
"feature-set rev 1.1"); see CONTEXT.md "Flagged ambiguities".

**D4 — Documentation is updated to match.** `release_notes.rst` gains a
`v1.1.0 (Unreleased)` section grouped by module; docstrings across the public
API are backfilled with `.. versionadded::` (true first-release version) and
`.. versionchanged:: 1.1.0` (for the new method modes). `pyproject.toml` stays at
`1.0.3` until the actual release — the version pin is bumped at tag time, not as
part of this documentation pass.

**D5 — Pre-release API-shape fixes (2026-06-01).** Because the whole surface is
unreleased, several shapes were finalized before tagging (no semver cost):
`EmbeddingPreprocessor` gained `encode` and the builder methods were unified to
`build_scales` / `build_cat` (ADR-0011); `CPPGrid` became a `Tool` with `eval`
(ADR-0007 update); `smooth_scores` was renamed to `comp_smooth_scores` for
metric-family uniformity; and `comp_bootstrap_ci` now returns a dict
(`{'mean', 'ci_low', 'ci_high'}`) instead of a bare tuple.

## Rejected alternatives

- **`v1.0.4` (patch).** Matches the naïve increment but ships new public API,
  breaking the semver-strict contract. Rejected.
- **Two releases (`v1.1.0` then `v1.2.0`).** Reifies an internal milestone split
  that never shipped; nothing was tagged between them and they release together.
  Rejected — one `v1.1.0`.
