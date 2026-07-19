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
checks four things:

1. every name in :data:`aaanalysis.__all__` is importable from the install;
2. each ``pro`` / ``dev`` optional-dependency symbol degrades to a
   ``missing_feature_stub`` (raising ``ImportError`` with an install hint when
   called) instead of breaking the import, whenever its extra is absent;
3. bundled ``_data`` resources load — a representative ``load_scales()`` (top
   level ``_data/*.tsv``) and ``load_dataset(...)`` (``_data/benchmarks/*.tsv``)
   succeed, proving the package data shipped in the distribution;
4. the golden-pipeline **error surface** holds from the install — a couple of
   core failure paths raise the documented bare ``ValueError``, and (base-deps
   only) ``ap.explain_features`` degrades to an install-hint ``ImportError`` — so
   the failure contract is part of the packaging smoke, not only the editable
   dev tree. The full condition matrix lives in
   ``tests/integration/test_failure_contracts.py``.
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

# --- 1. Every public __all__ symbol imports from the install ------------------
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

# --- 3. Bundled _data resources load from the install -------------------------
# A missing package-data glob surfaces here as a FileNotFoundError from the loader;
# wrap it so the gate reports "package data missing from the distribution", not a raw
# pandas traceback, and name the exact glob that failed to ship.
def _load(call, what, glob):
    try:
        df = call()
    except FileNotFoundError as exc:
        _fail(f"{what} could not read bundled package data ({glob} missing from the distribution): {exc}")
    if df.empty:
        _fail(f"{what} returned an empty frame - bundled {glob} missing from the distribution")
    return df

df_scales = _load(aa.load_scales, "load_scales()", "_data/*.tsv")
df_seq = _load(lambda: aa.load_dataset(name="DOM_GSEC", n=2), "load_dataset('DOM_GSEC')",
               "_data/benchmarks/*.tsv")
print(f"OK: bundled _data loads (scales {df_scales.shape}, DOM_GSEC {df_seq.shape})")

# --- 4. Golden-pipeline error surface holds from the installed distribution ----
# A coding agent meets the public API on the unhappy path too. Assert a few failure paths raise
# the documented bare error *from the install*, so a broken re-export or a message regression is
# caught by the packaging gate, not only by the editable dev suite. Pick fail-fast validation
# (no model fitting). find_features / predict_samples are core; explain_features gates on [pro].
import pandas as pd
from aaanalysis import pipe as ap


def _expect_valueerror(call, needle, what):
    try:
        call()
    except ValueError as exc:
        if type(exc) is not ValueError:
            _fail(f"{what}: expected a bare ValueError, got {type(exc).__name__}")
        if needle not in str(exc):
            _fail(f"{what}: ValueError {str(exc)!r} does not name the offending input ({needle!r})")
    except Exception as exc:  # noqa: BLE001 - report the wrong error type explicitly
        _fail(f"{what}: expected ValueError, got {type(exc).__name__}: {exc}")
    else:
        _fail(f"{what}: expected a ValueError from the install, none raised")


_expect_valueerror(lambda: aa.CPP(df_parts=pd.DataFrame()),
                   "'df_parts'", "CPP(empty df_parts)")
_expect_valueerror(lambda: ap.predict_samples(list_df_feat=[], df_seq=df_seq,
                                              labels=df_seq["label"].to_list()),
                   "'list_df_feat'", "predict_samples(empty list_df_feat)")
print("OK: core golden-pipeline validation raises bare, self-explaining ValueErrors from the install")

# explain_features gates on [pro] (SHAP). In the base-deps packaging venv it is a
# missing_feature_stub: calling it must raise ImportError naming the extra to install.
if importlib.util.find_spec("shap") is None:
    try:
        ap.explain_features(df_feat=df_seq, df_seq=df_seq, labels=[0, 1])
    except ImportError as exc:
        if "aaanalysis[pro]" not in str(exc):
            _fail(f"ap.explain_features stub ImportError lacks the install hint: {exc!r}")
    except Exception as exc:  # noqa: BLE001 - report the wrong error type explicitly
        _fail(f"ap.explain_features without [pro] raised {type(exc).__name__}, expected ImportError")
    else:
        _fail("ap.explain_features without [pro] did not raise (expected an install-hint ImportError)")
    print("OK: ap.explain_features degrades to an install-hint ImportError without the [pro] extra")

print("PASS: built distribution installs clean and its public API imports")
