# ADR-0060 — Packaging gate: ship the sdist sources, build wheel + sdist, install from both, import the public API

Status: Accepted — 2026-07-14

## Context

PyPI advertises Python 3.10–3.14 support, but no pre-merge check proved the **built distributions
actually install and import on a clean environment**. Every test matrix runs against the source tree
(editable/dev install), which always carries every module and data file — so a missing
`[tool.setuptools.package-data]` glob, a bad build-backend include, a broken `__init__` re-export, or
a sdist that omits the Cython sources would only surface *after* a user `pip install`s the release,
the most expensive moment to find it. A narrow `py.typed`-in-wheel guard already runs inside the
release `cibuildwheel` `test-command`, but there was no general build → install → public-API-import
gate, and it ran only on `release: published`, never on a PR.

Two defects sat in that blind spot. First, no PR-time gate at all (the original scope of the gate).
Second, a latent sdist defect: the sdist omitted the Cython source `_filters_c/_inner.pyx` (there was
no `MANIFEST.in` and no `*.pyx` in package-data), so the default `python -m build` (which builds the
wheel *from the sdist*) and any source install (`pip install <sdist>`, `pip install aaanalysis
--no-binary`) failed to cythonize. With v1.1 unreleased (`pyproject` at `1.0.3`), both were fixed
together before the next publish.

## Decision

**D1 — A dedicated `packaging.yml` gate on push + PR to `master`.** It is separate from the
release-only `build_wheels.yml` (triggered on `release: published` / `workflow_dispatch`), which
never runs pre-merge. The gate matrixes the supported brackets **py3.10 + py3.14 on Linux**, reusing
the min+max convention of the unit matrix; multi-OS is out of scope (Windows min+max is already
bracketed in `main.yml`). No `paths-ignore`, so it can be made a required status check without
GitHub's "skipped-but-required blocks the merge" trap (matching `test_coverage.yml`).

**D2 — Ship the Cython sources in the sdist via a `MANIFEST.in`.** A single
`recursive-include aaanalysis *.pyx *.pxd` adds `_filters_c/_inner.pyx` to the sdist. The generated
`_inner.c` already rides along automatically (setuptools includes the cythonized extension source),
and `build-system.requires` pins `Cython>=3.0`, so an isolated build regenerates the extension from
the `.pyx`. This is the minimal fix: the wheel ships the compiled `.so` and needs none of these, so
nothing is added to `[tool.setuptools.package-data]` and the CONFIRM-FIRST `pyproject.toml` is left
untouched — `MANIFEST.in` targets the sdist, which is exactly (and only) where the sources are needed.

**D3 — Each matrix leg builds both distributions with the default `python -m build`.** The default
builds the sdist, then builds the wheel *from that sdist* — the exact chain a `pip install aaanalysis`
release relies on, and the one that failed before D2 because the sdist could not cythonize. Now that
D2 ships the sources, the default build is the right thing to exercise; the leg asserts **exactly one**
sdist and **one** wheel are produced.

**D4 — Both the wheel and the sdist are installed into fresh, base-deps-only venvs, then a standalone
check script runs against each.** `tests/_check_public_api_packaged.py` (a sibling of
`_check_py_typed_packaged.py`) is **not** a pytest test: run as `python <file>` from a neutral working
directory, `import aaanalysis` resolves to the installed distribution, never the checkout — and a
source-tree guard fails loudly if it resolved to source anyway. It asserts (a) every
`aaanalysis.__all__` symbol imports; (b) each `pro`/`dev` optional-dependency symbol degrades to a
`missing_feature_stub` (an `ImportError` with an install hint on use) when its extra is absent; and
(c) representative `load_scales()` / `load_dataset(...)` calls read bundled `_data/*.tsv`, proving
package data shipped. The **wheel** install proves the shipped binary artifact; the **sdist** install
(a separate venv, forcing a real source build) proves the `pip install <sdist>` path D2 unblocks.

**D5 — The wheel is built per Python version, not once and reused.** The compiled Cython extension
makes the wheel ABI-specific (`cp310-*` vs `cp314-*`), so a single wheel cannot install across the
matrix; each leg builds and installs its own. The sdist is portable, but is built + installed per leg
too so the source build runs on both interpreters.

## Rejected alternatives

- **Add `*.pyx` / `*.c` to `[tool.setuptools.package-data]` instead of a `MANIFEST.in`** (D2): edits
  the CONFIRM-FIRST `pyproject.toml` and drags the sources into the *wheel*, where they are dead weight
  next to the compiled `.so`; `MANIFEST.in` targets the sdist only.
- **Track the generated `_inner.c` and add a Cython-free `setup.py` fallback** (D2): more moving parts
  (un-gitignore a 1.3 MB generated file, a `USE_CYTHON` toggle) for the `--no-build-isolation` edge
  case; the default isolated build always has Cython via `build-system.requires`, and the `.c` already
  ships as the extension source.
- **Build both from source with `--sdist --wheel` instead of the default `python -m build`** (D3):
  skips the wheel-from-sdist chain, the very path a plain release install uses — the defect D2 fixes
  would go unexercised.
- **Fold the gate into `build_wheels.yml`** (D1): that workflow is release-triggered, so it would
  never run on a PR; the gate must block *before* merge.
- **Build one wheel and install it across the version matrix** (D5): impossible — the wheel is
  ABI-specific to its build interpreter.
- **A pytest test instead of a standalone script** (D4): under the editable dev matrix
  `import aaanalysis` resolves to the source tree, where package data always exists, so the drift the
  gate exists to catch would pass unnoticed.

## Out of scope

- **Making the check a *required* status context** is a repo branch-protection setting, not part of
  the workflow file.
- **Publishing / release automation** stays in `build_wheels.yml`.
