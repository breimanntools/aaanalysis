"""This is a script to test the CPP Python-kernel fallback notice in
aaanalysis.feature_engineering._backend.cpp_run._pick_feature_matrix_builder.

The notice was promoted from an INFO ``print_out`` to a one-time ``UserWarning``
so it surfaces even with ``aa.options['verbose'] = False`` (issue #74).
"""
import warnings

import pytest

import aaanalysis.feature_engineering._backend.cpp_run as cpp_run


@pytest.fixture
def reset_fallback_guard():
    """Restore the module-level cython flag + one-time guard after each test."""
    orig_has = cpp_run._HAS_CYTHON_INNER
    orig_notified = cpp_run._PYTHON_FALLBACK_NOTIFIED
    yield
    cpp_run._HAS_CYTHON_INNER = orig_has
    cpp_run._PYTHON_FALLBACK_NOTIFIED = orig_notified


class TestFallbackNotice:
    """The fallback notice is a one-time UserWarning, not an INFO print."""

    def test_warns_userwarning_when_no_cython(self, reset_fallback_guard):
        cpp_run._HAS_CYTHON_INNER = False
        cpp_run._PYTHON_FALLBACK_NOTIFIED = False
        with pytest.warns(UserWarning, match="Python kernel fallback"):
            builder = cpp_run._pick_feature_matrix_builder()
        # Falls back to the pure-Python builder.
        assert builder is cpp_run.get_feature_matrix_fast_

    def test_warns_only_once(self, reset_fallback_guard):
        cpp_run._HAS_CYTHON_INNER = False
        cpp_run._PYTHON_FALLBACK_NOTIFIED = False
        with pytest.warns(UserWarning, match="Python kernel fallback"):
            cpp_run._pick_feature_matrix_builder()
        # Guard is set; a second call must be silent.
        assert cpp_run._PYTHON_FALLBACK_NOTIFIED is True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cpp_run._pick_feature_matrix_builder()

    def test_no_warning_when_cython_available(self, reset_fallback_guard):
        if cpp_run.get_feature_matrix_c_ is None:
            pytest.skip("compiled Cython extension not built in this environment")
        cpp_run._HAS_CYTHON_INNER = True
        cpp_run._PYTHON_FALLBACK_NOTIFIED = False
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder = cpp_run._pick_feature_matrix_builder()
        assert builder is cpp_run.get_feature_matrix_c_
