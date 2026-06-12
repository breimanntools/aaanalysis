"""This is a script to test the pro/dev import-stub *wiring* in aaanalysis/__init__.py.

``test_missing_feature_stub.py`` covers the ``missing_feature_stub`` factory in
isolation. The complementary gap is the module-level ``try: from .<extra> import X /
except ImportError as e: globals()[X] = missing_feature_stub(...)`` blocks — they only
execute when an optional dependency is genuinely absent, which never happens in CI
(every job installs ``pro``/``dev``). Here we mask the optional deps, reload the package,
and assert each missing feature degrades to a friendly-erroring stub that is dropped from
``__all__`` — the real user-facing behaviour of a ``pip install aaanalysis`` (no extras).

The masking is fully restored in a ``finally`` (sys.modules snapshot + a final reload), so
no global state leaks into sibling tests.
"""
import importlib
import sys
from contextlib import contextmanager

import pytest

import aaanalysis

# Optional deps imported at module load by each extra. Masking these top-level names makes
# every *_pro / show_html import fail with an ImportError whose ``.name`` is in the extra's
# set, so __init__ installs the stub. Core is free of these imports, so it still loads.
_BLOCKED = {"shap", "Bio", "biopython", "requests", "IPython"}

# (public name, install extra) for every gated feature in __init__.py.
_PRO_FEATURES = [
    "ShapModel",
    "comp_seq_sim",
    "filter_seq",
    "scan_motif",
    "StructurePreprocessor",
    "AnnotationPreprocessor",
]
_DEV_FEATURES = ["display_df"]


class _BlockFinder:
    """A meta-path finder that raises ModuleNotFoundError for masked top-level packages."""

    def __init__(self, blocked):
        self.blocked = blocked

    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in self.blocked:
            raise ModuleNotFoundError(f"No module named '{top}'", name=top)
        return None  # defer to the real finders for everything else


@contextmanager
def _extras_masked(blocked):
    """Reload aaanalysis with ``blocked`` optional deps unavailable, then fully restore.

    Restore re-binds the *original* ``__dict__`` (a final ``reload`` would recreate
    every function/class with a new identity and break sibling tests that hold an
    ``from aaanalysis import X`` reference).
    """
    saved_modules = dict(sys.modules)
    saved_meta = list(sys.meta_path)
    saved_dict = dict(aaanalysis.__dict__)
    try:
        # Evict the masked deps and all aaanalysis submodules so the reload re-imports them
        # (a cached pro submodule would skip its `import shap`/`from Bio ...` and never fail).
        for name in list(sys.modules):
            top = name.split(".")[0]
            if top in blocked or (name.startswith("aaanalysis.")):
                del sys.modules[name]
        sys.meta_path.insert(0, _BlockFinder(blocked))
        importlib.reload(aaanalysis)
        yield aaanalysis
    finally:
        sys.meta_path[:] = saved_meta
        sys.modules.clear()
        sys.modules.update(saved_modules)
        # Restore the original module namespace in place, preserving object identities.
        aaanalysis.__dict__.clear()
        aaanalysis.__dict__.update(saved_dict)


@pytest.fixture(scope="module")
def masked_state():
    """Reload once with extras masked; capture __all__ + the installed stubs, then restore."""
    with _extras_masked(_BLOCKED) as aamod:
        names = _PRO_FEATURES + _DEV_FEATURES
        state = {
            "all": list(aamod.__all__),
            "stubs": {name: getattr(aamod, name) for name in names},
        }
    return state


class TestMissingExtrasWiring:
    """Each gated feature degrades to a stub and leaves __all__ when its extra is absent."""

    @pytest.mark.parametrize("name", _PRO_FEATURES)
    def test_pro_feature_becomes_install_hint_stub(self, masked_state, name):
        stub = masked_state["stubs"][name]
        assert callable(stub)
        with pytest.raises(ImportError, match=r"aaanalysis\[pro\]"):
            stub()

    @pytest.mark.parametrize("name", _DEV_FEATURES)
    def test_dev_feature_becomes_install_hint_stub(self, masked_state, name):
        stub = masked_state["stubs"][name]
        assert callable(stub)
        with pytest.raises(ImportError, match=r"aaanalysis\[dev\]"):
            stub()

    @pytest.mark.parametrize("name", _PRO_FEATURES + _DEV_FEATURES)
    def test_missing_feature_dropped_from_all(self, masked_state, name):
        assert name not in masked_state["all"]

    def test_core_still_exported_when_extras_missing(self, masked_state):
        # Core public symbols are unaffected by missing optional deps.
        for core in ("load_dataset", "CPP", "AAclust", "TreeModel", "options"):
            assert core in masked_state["all"]


class TestMaskingRestored:
    """After the masked reload, the real features must be back for the rest of the suite."""

    def test_real_features_restored(self, masked_state):
        # masked_state is built and torn down before this runs; aaanalysis is clean again.
        import aaanalysis as aa
        assert "ShapModel" in aa.__all__
        assert "display_df" in aa.__all__
        assert callable(aa.load_dataset)
