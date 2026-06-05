"""This is a script to test aaanalysis.missing_feature_stub().

Covers the pro/dev import-stub mechanism in ``aaanalysis/__init__.py``: a genuinely
missing optional dependency (``error.name`` in the extra's module set) must surface a
friendly ``pip install`` hint, while any other ImportError (a real bug inside an extra
module, or an error with no ``name``) must be re-raised unchanged so the original
traceback survives.
"""
import pytest

import aaanalysis as aa
from aaanalysis import missing_feature_stub, _EXTRA_MODULES


def _mnfe(name):
    """Build a ModuleNotFoundError with its standard ``.name`` set, as the import system does."""
    e = ModuleNotFoundError(f"No module named '{name}'")
    e.name = name
    return e


class TestMissingFeatureStub:
    """Normal cases: one branch / one input per test."""

    @pytest.mark.parametrize("name", sorted(_EXTRA_MODULES["pro"]))
    def test_pro_known_dep_raises_install_hint(self, name):
        stub = missing_feature_stub("ShapModel", _mnfe(name), mode="pro")
        with pytest.raises(ImportError, match=r"aaanalysis\[pro\]"):
            stub()

    @pytest.mark.parametrize("name", sorted(_EXTRA_MODULES["dev"]))
    def test_dev_known_dep_raises_install_hint(self, name):
        stub = missing_feature_stub("display_df", _mnfe(name), mode="dev")
        with pytest.raises(ImportError, match=r"aaanalysis\[dev\]"):
            stub()

    def test_pro_unknown_name_reraises_original(self):
        # A real bug inside a pro module: a broken internal import, NOT a missing optional dep.
        err = _mnfe("aaanalysis.seq_analysis_pro._typo")
        stub = missing_feature_stub("scan_motif", err, mode="pro")
        with pytest.raises(ImportError) as exc:
            stub()
        assert exc.value is err  # original object re-raised, traceback preserved
        assert "aaanalysis[pro]" not in str(exc.value)

    def test_dev_unknown_name_reraises_original(self):
        err = _mnfe("aaanalysis.show_html._broken")
        stub = missing_feature_stub("display_df", err, mode="dev")
        with pytest.raises(ImportError) as exc:
            stub()
        assert exc.value is err
        assert "aaanalysis[dev]" not in str(exc.value)

    def test_name_none_reraises_original(self):
        # ImportError without a .name (e.g. "cannot import name X") must not be guessed at.
        err = ImportError("cannot import name 'foo'")
        assert err.name is None
        stub = missing_feature_stub("ShapModel", err, mode="pro")
        with pytest.raises(ImportError) as exc:
            stub()
        assert exc.value is err

    def test_install_hint_chains_original_as_cause(self):
        err = _mnfe("shap")
        stub = missing_feature_stub("ShapModel", err, mode="pro")
        with pytest.raises(ImportError) as exc:
            stub()
        assert exc.value.__cause__ is err  # raised `from error`

    def test_returns_callable_and_defers_error(self):
        # Building the stub must NOT raise; only calling it does.
        stub = missing_feature_stub("ShapModel", _mnfe("shap"), mode="pro")
        assert callable(stub)

    def test_stub_accepts_and_ignores_call_args(self):
        stub = missing_feature_stub("ShapModel", _mnfe("shap"), mode="pro")
        with pytest.raises(ImportError):
            stub(1, 2, key="value")

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be 'pro' or 'dev'"):
            missing_feature_stub("X", _mnfe("shap"), mode="bogus")

    def test_public_symbol_accessible(self):
        # missing_feature_stub is reachable on the package (used by __init__'s try/except).
        assert aa.missing_feature_stub is missing_feature_stub
