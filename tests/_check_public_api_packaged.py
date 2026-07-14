"""Guard: the built wheel and sdist install clean and the full public API imports.

This is **not** a pytest test — it is a standalone script run against a freshly
built distribution (wheel **or** sdist) installed into a clean, **base-deps-only**
venv (no ``[dev]`` / no ``[pro]``). The packaging CI gate
(``.github/workflows/packaging.yml``) builds the wheel + sdist with
``python -m build`` and runs this against **each** install — wheel and sdist — on
the min + max supported Python. Its whole reason to exist is that the editable dev
matrix imports the *source tree* (where every module and data file always exists),
so a missing ``[tool.setuptools.package-data]`` entry, a bad build-backend include,
a broken ``__init__`` re-export, or a sdist that omits the Cython sources only
surfaces *after* a user ``pip install``s the release — the most expensive moment to
find it. This script catches it first.

It is a sibling of ``tests/_check_py_typed_packaged.py`` (the narrower
``py.typed``-in-wheel guard the cibuildwheel ``test-command`` runs); this one
covers the general build -> install (wheel or sdist) -> public-API-import contract.

Run as ``python <this-file>`` **from a directory that is not the repo root** so
``import aaanalysis`` resolves to the installed distribution, never the checkout —
the source-tree guard below fails loudly if it did resolve to source anyway. It
checks three things:

1. every name in :data:`aaanalysis.__all__` is importable from the install;
2. each ``pro`` / ``dev`` optional-dependency symbol degrades to a
   ``missing_feature_stub`` (raising ``ImportError`` with an install hint when
   called) instead of breaking the import, whenever its extra is absent;
3. bundled ``_data`` resources load — a representative ``load_scales()`` (top
   level ``_data/*.tsv``) and ``load_dataset(...)`` (``_data/benchmarks/*.tsv``)
   succeed, proving the package data shipped in the distribution.
"""
import importlib.util
import sys
from importlib.resources import files
from pathlib import Path

import aaanalysis as aa  # resolves to the installed distribution; see module docstring


def _fail(msg):
    sys.exit(f"FAIL: {msg}")


# --- 0. Resolve to the installed distribution, never the source / sdist tree --
# An installed distribution lives under site-packages: its parent holds no
# pyproject.toml. The repo checkout (and an unpacked sdist) does — importing from
# there would let a missing package-data entry pass unnoticed, defeating the gate.
pkg_dir = Path(str(files("aaanalysis")))
if (pkg_dir.parent / "pyproject.toml").is_file():
    _fail(f"aaanalysis resolved to a source/sdist tree, not an install: {pkg_dir}")

# --- 1. Every public __all__ symbol imports from the wheel --------------------
missing = [name for name in aa.__all__ if not hasattr(aa, name)]
if missing:
    _fail(f"names in aaanalysis.__all__ are not importable from the install: {missing}")
print(f"OK: all {len(aa.__all__)} aaanalysis.__all__ symbols import from the installed distribution")

# --- 2. Optional-dependency symbols degrade to missing_feature_stub -----------
# When an extra is absent these are not in __all__; __init__ swaps each for a
# missing_feature_stub that raises ImportError with an install hint *on use*.
# Assert that graceful degradation only when the extra is genuinely absent (so
# the check still passes if someone runs it in a [pro]/[dev] env).
_GATED = {
    "ShapModel": "shap",
    "comp_seq_sim": "Bio",
    "filter_seq": "Bio",
    "scan_motif": "Bio",
    "StructurePreprocessor": "Bio",
    "AnnotationPreprocessor": "Bio",
    "CPPStructurePlot": "py3Dmol",
    "display_df": "IPython",
}
for name, probe in _GATED.items():
    obj = getattr(aa, name, None)
    if obj is None:
        _fail(f"optional-dependency symbol {name!r} is missing entirely (expected a stub)")
    if importlib.util.find_spec(probe) is not None:
        continue  # the extra is installed here -> real object, nothing to degrade
    try:
        obj()
    except ImportError:
        pass  # graceful degradation confirmed
    except Exception as exc:  # noqa: BLE001 - report the wrong error type explicitly
        _fail(f"stub {name!r} raised {type(exc).__name__}, expected ImportError with an install hint")
    else:
        _fail(f"stub {name!r} did not raise when called with its extra {probe!r} absent")
print(f"OK: {len(_GATED)} pro/dev symbols degrade to missing_feature_stub when their extra is absent")

# --- 3. Bundled _data resources load from the wheel ---------------------------
# A missing package-data glob surfaces here as a FileNotFoundError from the loader;
# wrap it so the gate reports "package data missing from the wheel", not a raw
# pandas traceback, and name the exact glob that failed to ship.
def _load(call, what, glob):
    try:
        df = call()
    except FileNotFoundError as exc:
        _fail(f"{what} could not read bundled package data ({glob} missing from the wheel): {exc}")
    if df.empty:
        _fail(f"{what} returned an empty frame - bundled {glob} missing from the wheel")
    return df

df_scales = _load(aa.load_scales, "load_scales()", "_data/*.tsv")
df_seq = _load(lambda: aa.load_dataset(name="DOM_GSEC", n=2), "load_dataset('DOM_GSEC')",
               "_data/benchmarks/*.tsv")
print(f"OK: bundled _data loads (scales {df_scales.shape}, DOM_GSEC {df_seq.shape})")

print("PASS: built distribution installs clean and its public API imports")
