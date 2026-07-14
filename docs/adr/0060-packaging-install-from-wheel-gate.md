# ADR-0060 — Packaging gate: build wheel/sdist, install from the wheel, import the public API

Status: Accepted — 2026-07-14

## Context

PyPI advertises Python 3.10–3.14 support, but no pre-merge check proved the **built wheel actually
installs and imports on a clean environment**. Every test matrix runs against the source tree
(editable/dev install), which always carries every module and data file — so a missing
`[tool.setuptools.package-data]` glob, a bad build-backend include, or a broken `__init__`
re-export would only surface *after* a user `pip install`s the release, the most expensive moment
to find it. A narrow `py.typed`-in-wheel guard already runs inside the release `cibuildwheel`
`test-command`, but there was no general build → install-from-wheel → public-API-import gate, and it
ran only on `release: published`, never on a PR. With v1.1 unreleased (`pyproject` at `1.0.3`), this
was the missing safety net before the next publish.

## Decision

**D1 — A dedicated `packaging.yml` gate on push + PR to `master`.** It is separate from the
release-only `build_wheels.yml` (triggered on `release: published` / `workflow_dispatch`), which
never runs pre-merge. The gate matrixes the supported brackets **py3.10 + py3.14 on Linux**, reusing
the min+max convention of the unit matrix; multi-OS is out of scope (Windows min+max is already
bracketed in `main.yml`).

**D2 — Each matrix leg builds *both* the sdist and the wheel from source with
`python -m build --sdist --wheel`, not the default `python -m build`.** The default builds an sdist
and then the wheel *from that sdist*; that path currently **fails**, because the sdist omits the
Cython source `_filters_c/_inner.pyx` (there is no `MANIFEST.in` and no `*.pyx` in package-data), so
the wheel-from-sdist step cannot cythonize. Building both from source sidesteps that, matches the
release pipeline (`cibuildwheel` builds the wheel from the source tree), and keeps the wheel — the
shipped artifact — as the thing under test. The leg still asserts **exactly one** sdist and **one**
wheel are produced.

**D3 — The wheel is installed into a fresh, base-deps-only venv, then a standalone check script runs
against it.** `tests/_check_public_api_packaged.py` (a sibling of `_check_py_typed_packaged.py`) is
**not** a pytest test: run as `python <file>` from a neutral working directory, `import aaanalysis`
resolves to the installed wheel, never the checkout — and a source-tree guard fails loudly if it
resolved to source anyway. It asserts (a) every `aaanalysis.__all__` symbol imports; (b) each
`pro`/`dev` optional-dependency symbol degrades to a `missing_feature_stub` (an `ImportError` with an
install hint on use) when its extra is absent; and (c) representative `load_scales()` /
`load_dataset(...)` calls read bundled `_data/*.tsv`, proving package data shipped.

**D4 — The wheel is built per Python version, not once and reused.** The compiled Cython extension
makes the wheel ABI-specific (`cp310-*` vs `cp314-*`), so a single wheel cannot install across the
matrix; each leg builds and installs its own.

## Rejected alternatives

- **Default `python -m build` (sdist → wheel-from-sdist)** (D2): fails today because the sdist omits
  `_inner.pyx`, and it tests the *sdist* install path, which is out of scope — the gate tests the
  wheel.
- **Fold the gate into `build_wheels.yml`** (D1): that workflow is release-triggered, so it would
  never run on a PR; the gate must block *before* merge.
- **Build one wheel and install it across the version matrix** (D4): impossible — the wheel is
  ABI-specific to its build interpreter.
- **A pytest test instead of a standalone script** (D3): under the editable dev matrix
  `import aaanalysis` resolves to the source tree, where package data always exists, so the drift the
  gate exists to catch would pass unnoticed.

## Out of scope

- **The sdist itself is incomplete** — it omits `_inner.pyx`, so `pip install <sdist>` (a source
  install) would fail to build the extension. This gate tests the wheel, not the sdist; shipping the
  Cython sources in the sdist (a `MANIFEST.in` / `*.pyx` + `*.c` package-data addition to the
  CONFIRM-FIRST `pyproject.toml`) is a separate follow-up.
- **Making the check a *required* status context** is a repo branch-protection setting, not part of
  the workflow file.
- **Publishing / release automation** stays in `build_wheels.yml`.
