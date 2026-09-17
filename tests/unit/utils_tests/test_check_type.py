"""This is a script to test the numeric validators in _utils/check_type.py
(check_number_val / check_number_range), exposed through ``ut``.

Focus: non-finite input. ``NaN`` satisfies no range comparison (``nan < 0`` and
``nan > 1`` are each ``False``), so before the shared guard it passed every range
check and surfaced much later as a ``NaN`` result. Doubles as user-facing
error-message coverage.
"""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

LIST_NON_FINITE = [float("nan"), float("inf"), float("-inf"),
                   np.float64("nan"), np.float64("inf"), np.float64("-inf"),
                   np.float32("nan"), np.float32("inf")]


class TestCheckNumberVal:
    """Normal cases, one parameter per test."""

    @settings(max_examples=5, deadline=None)
    @given(val=some.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    def test_valid_float(self, val):
        assert ut.check_number_val(name="v", val=val, just_int=False) is None

    @settings(max_examples=5, deadline=None)
    @given(val=some.integers(min_value=-1000, max_value=1000))
    def test_valid_int(self, val):
        assert ut.check_number_val(name="v", val=val, just_int=True) is None

    @pytest.mark.parametrize("val", LIST_NON_FINITE)
    def test_non_finite_raises(self, val):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_val(name="v", val=val, just_int=False)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_val(name="v", val=float("nan"), just_int=False)

    def test_pos_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_val(name="v", val=float("inf"), just_int=False)

    def test_neg_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_val(name="v", val=float("-inf"), just_int=False)

    def test_non_finite_message_names_parameter(self):
        with pytest.raises(ValueError,
                           match=r"'my_param' \(nan\) should be a finite float or an integer"):
            ut.check_number_val(name="my_param", val=float("nan"), just_int=False)

    def test_non_finite_just_int_reports_type_first(self):
        # An integer parameter rejects a non-finite float by type, before finiteness.
        with pytest.raises(ValueError, match="should be an integer"):
            ut.check_number_val(name="v", val=float("nan"), just_int=True)

    def test_none_accepted(self):
        assert ut.check_number_val(name="v", val=None, accept_none=True, just_int=False) is None

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="should not be None"):
            ut.check_number_val(name="v", val=None, just_int=False)

    def test_str_rejected(self):
        with pytest.raises(ValueError, match="should be a float or an integer"):
            ut.check_number_val(name="v", val="0.5", just_int=False)


class TestCheckNumberValComplex:
    """Combinations and edge interactions."""

    def test_non_finite_with_str_add(self):
        with pytest.raises(ValueError, match="extra hint"):
            ut.check_number_val(name="v", val=float("nan"), just_int=False,
                                str_add="extra hint")

    def test_accept_none_does_not_accept_nan(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_val(name="v", val=float("nan"), accept_none=True, just_int=False)

    def test_numpy_float_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_val(name="v", val=np.float64("nan"), just_int=False)

    def test_numpy_int_is_finite(self):
        assert ut.check_number_val(name="v", val=np.int64(3), just_int=True) is None

    def test_bool_is_not_non_finite(self):
        # bool is an int subclass; it is finite and must keep passing this validator.
        assert ut.check_number_val(name="v", val=True, just_int=True) is None

    def test_large_but_finite_accepted(self):
        assert ut.check_number_val(name="v", val=1e308, just_int=False) is None


class TestCheckNumberRange:
    """Normal cases, one parameter per test."""

    @settings(max_examples=5, deadline=None)
    @given(val=some.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    def test_valid_float_in_range(self, val):
        assert ut.check_number_range(name="v", val=val, min_val=0, max_val=1,
                                     just_int=False) is None

    @settings(max_examples=5, deadline=None)
    @given(val=some.integers(min_value=1, max_value=100))
    def test_valid_int_in_range(self, val):
        assert ut.check_number_range(name="v", val=val, min_val=1, just_int=True) is None

    @pytest.mark.parametrize("val", LIST_NON_FINITE)
    def test_non_finite_raises(self, val):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=val, min_val=0, max_val=1, just_int=False)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=float("nan"), min_val=0, max_val=1,
                                  just_int=False)

    def test_pos_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=float("inf"), min_val=0, max_val=1,
                                  just_int=False)

    def test_neg_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=float("-inf"), min_val=0, max_val=1,
                                  just_int=False)

    def test_non_finite_message_names_parameter(self):
        with pytest.raises(ValueError,
                           match=r"'my_param' \(inf\) should be a finite float or an integer"):
            ut.check_number_range(name="my_param", val=float("inf"), min_val=0, just_int=False)

    def test_out_of_range_still_reports_range(self):
        with pytest.raises(ValueError, match="0 <= n <= 1"):
            ut.check_number_range(name="v", val=2.0, min_val=0, max_val=1, just_int=False)

    def test_none_accepted(self):
        assert ut.check_number_range(name="v", val=None, accept_none=True,
                                     just_int=False) is None

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="should not be None"):
            ut.check_number_range(name="v", val=None, just_int=False)

    def test_str_rejected(self):
        with pytest.raises(ValueError, match="should be a float or an integer"):
            ut.check_number_range(name="v", val="0.5", min_val=0, max_val=1, just_int=False)


class TestCheckNumberRangeComplex:
    """Combinations and edge interactions."""

    @pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected_without_upper_bound(self, val):
        # Without 'max_val', '+inf' satisfied the lower bound and passed.
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=val, min_val=0, just_int=False)

    @pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected_with_exclusive_limits(self, val):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=val, min_val=0, max_val=1,
                                  exclusive_limits=True, just_int=False)

    def test_non_finite_with_str_add(self):
        with pytest.raises(ValueError, match="extra hint"):
            ut.check_number_range(name="v", val=float("nan"), min_val=0, just_int=False,
                                  str_add="extra hint")

    def test_accept_none_does_not_accept_nan(self):
        with pytest.raises(ValueError, match="finite"):
            ut.check_number_range(name="v", val=float("nan"), min_val=0, accept_none=True,
                                  just_int=False)

    def test_non_finite_just_int_reports_type_first(self):
        with pytest.raises(ValueError, match="should be an integer"):
            ut.check_number_range(name="v", val=float("nan"), min_val=0, just_int=True)

    def test_finite_boundary_values_accepted(self):
        assert ut.check_number_range(name="v", val=0.0, min_val=0, max_val=1,
                                     just_int=False) is None
        assert ut.check_number_range(name="v", val=1.0, min_val=0, max_val=1,
                                     just_int=False) is None

    def test_negative_range_accepts_finite_negative(self):
        assert ut.check_number_range(name="v", val=-5.0, min_val=-10, max_val=0,
                                     just_int=False) is None


class TestNonFinitePublicParameters:
    """Regression: a public numeric parameter no longer swallows a non-finite value."""

    @staticmethod
    def _X():
        return np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.9, 0.2],
                         [0.2, 0.7, 0.1, 0.6], [0.9, 0.3, 0.5, 0.1]])

    @pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
    def test_aaclust_fit_min_th_non_finite_raises(self, val):
        # 'min_th=nan' was accepted and clustered as if no threshold applied.
        with pytest.raises(ValueError, match="'min_th'"):
            aa.AAclust().fit(self._X(), min_th=val)

    def test_aaclust_fit_min_th_message_is_finite(self):
        with pytest.raises(ValueError, match="finite"):
            aa.AAclust().fit(self._X(), min_th=float("nan"))

    def test_aaclust_fit_min_th_valid_still_works(self):
        ac = aa.AAclust().fit(self._X(), min_th=0.5)
        assert ac.n_clusters >= 1
