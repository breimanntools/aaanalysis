"""This is a script to test the warning/exception decorators in
aaanalysis._utils.decorators (exposed via aaanalysis.utils): the backend-error
wrapper, the RuntimeWarning catcher, the invalid-division catcher, the
UndefinedMetricWarning catcher, and the ``deprecated`` decorator.
"""
import inspect
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


class TestDeprecated:
    """deprecated() emits a DeprecationWarning and annotates the docstring."""

    def test_function_warns_on_call(self):
        @ut.deprecated(reason="Use 'new_fn' instead.", version_removed="1.2.0")
        def old_fn():
            return 42
        with pytest.warns(DeprecationWarning) as record:
            assert old_fn() == 42
        msg = str(record[0].message)
        assert "'old_fn' is deprecated" in msg
        assert "1.2.0" in msg
        assert "Use 'new_fn' instead." in msg

    def test_function_return_value_and_args_pass_through(self):
        @ut.deprecated()
        def add(a, b=1):
            return a + b
        with pytest.warns(DeprecationWarning):
            assert add(2, b=3) == 5

    def test_signature_and_metadata_preserved(self):
        @ut.deprecated(reason="gone soon")
        def f(a, b=2):
            """Original summary."""
            return a
        assert f.__name__ == "f"
        assert list(inspect.signature(f).parameters) == ["a", "b"]
        # The original docstring is kept below the deprecation note.
        assert "Original summary." in f.__doc__
        assert ".. admonition:: Deprecated" in f.__doc__
        assert "gone soon" in f.__doc__

    def test_no_args_warns_without_reason_or_version(self):
        @ut.deprecated()
        def g():
            return None
        with pytest.warns(DeprecationWarning, match="'g' is deprecated"):
            g()

    def test_class_warns_on_instantiation(self):
        @ut.deprecated(reason="Use NewCls.", version_removed="2.0.0")
        class OldCls:
            """A class on its way out."""
            def __init__(self, x):
                self.x = x
        with pytest.warns(DeprecationWarning, match="'OldCls' is deprecated"):
            obj = OldCls(7)
        # Instance is constructed correctly despite the warning.
        assert obj.x == 7
        assert ".. admonition:: Deprecated" in OldCls.__doc__

    def test_class_keeps_type_identity(self):
        @ut.deprecated()
        class C:
            def __init__(self):
                pass
        assert isinstance(C, type)
        with pytest.warns(DeprecationWarning):
            assert isinstance(C(), C)

    def test_multiline_reason_stays_indented_in_admonition(self):
        # A multi-line reason must keep every continuation line inside the
        # admonition body (3-space indent), or Sphinx drops the later lines.
        @ut.deprecated(reason="First line.\nSecond line.")
        def h():
            """Summary."""
            return None
        body = h.__doc__.split("Summary.")[0]
        content = [ln for ln in body.splitlines()
                   if ln and not ln.startswith(".. admonition")]
        assert content, "expected an indented admonition body"
        assert all(ln.startswith("   ") for ln in content)
        assert "   Second line." in h.__doc__
