"""This is a script to test the warning/exception decorators in
aaanalysis._utils.decorators (exposed via aaanalysis.utils): the backend-error
wrapper, the RuntimeWarning catcher, the invalid-division catcher, and the
UndefinedMetricWarning catcher.
"""
import warnings

import pytest
from sklearn.exceptions import UndefinedMetricWarning

import aaanalysis.utils as ut
from aaanalysis._utils.decorators import BackendProcessingError, InvalidDivisionException


class TestCatchBackendProcessingError:
    """catch_backend_processing_error wraps any raised exception."""

    def test_wraps_exception_with_cause(self):
        @ut.catch_backend_processing_error()
        def boom():
            raise ValueError("inner failure")
        with pytest.raises(BackendProcessingError) as exc_info:
            boom()
        # __str__ includes the original cause
        assert "inner failure" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, ValueError)

    def test_passes_through_on_success(self):
        @ut.catch_backend_processing_error()
        def ok():
            return 7
        assert ok() == 7

    def test_str_without_cause(self):
        err = BackendProcessingError("just a message")
        assert str(err) == "just a message"


class TestCatchRuntimeWarnings:
    """catch_runtime_warnings collects RuntimeWarnings and re-issues others."""

    def test_runtime_warning_resummarized(self):
        @ut.catch_runtime_warnings(suppress=False)
        def warns():
            warnings.warn("overflow happened", RuntimeWarning)
            return 1
        with pytest.warns(RuntimeWarning, match="RuntimeWarnings' were caught"):
            assert warns() == 1

    def test_runtime_warning_suppressed(self):
        @ut.catch_runtime_warnings(suppress=True)
        def warns():
            warnings.warn("overflow happened", RuntimeWarning)
            return 1
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert warns() == 1
        # suppress=True means no summary RuntimeWarning escapes the decorator
        assert not any("were caught" in str(w.message) for w in rec)

    def test_non_runtime_warning_reissued(self):
        @ut.catch_runtime_warnings(suppress=False)
        def warns_user():
            warnings.warn("user-facing", UserWarning)
            return 2
        with pytest.warns(UserWarning, match="user-facing"):
            assert warns_user() == 2

    def test_no_warning_no_emit(self):
        @ut.catch_runtime_warnings(suppress=False)
        def quiet():
            return 3
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert quiet() == 3


class TestCatchInvalidDivideWarning:
    """catch_invalid_divide_warning turns a RuntimeWarning into an exception."""

    def test_runtime_warning_raises_invalid_division(self):
        @ut.catch_invalid_divide_warning()
        def divide():
            warnings.warn("invalid value encountered in divide", RuntimeWarning)
            return 0
        with pytest.raises(InvalidDivisionException):
            divide()

    def test_no_warning_passes_through(self):
        @ut.catch_invalid_divide_warning()
        def ok():
            return 4
        assert ok() == 4


class TestCatchUndefinedMetricWarning:
    """catch_undefined_metric_warning aggregates UndefinedMetricWarnings."""

    def test_undefined_metric_warning_resummarized(self):
        @ut.catch_undefined_metric_warning()
        def warns():
            warnings.warn("Precision is ill-defined", UndefinedMetricWarning)
            return 1
        with pytest.warns(UndefinedMetricWarning, match="was caught"):
            assert warns() == 1

    def test_non_undefined_metric_warning_reissued(self):
        @ut.catch_undefined_metric_warning()
        def warns_user():
            warnings.warn("something else", UserWarning)
            return 2
        with pytest.warns(UserWarning, match="something else"):
            assert warns_user() == 2

    def test_no_warning_passes_through(self):
        @ut.catch_undefined_metric_warning()
        def quiet():
            return 5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert quiet() == 5
