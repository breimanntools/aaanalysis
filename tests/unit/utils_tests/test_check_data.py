"""This is a script to test the data-checking validators in _utils/check_data.py
(check_array_like / check_X / check_labels / check_match_X_labels / check_df /
check_warning_consecutive_index / check_file_path_exists).

Pure validators exposed via ``ut``; tested directly to reach their error /
warning branches. Doubles as user-facing error-message coverage.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis.utils as ut


class TestCheckArrayLike:
    def test_valid(self):
        out = ut.check_array_like(name="v", val=[[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(out, np.ndarray)

    def test_invalid_dtype_option(self):
        with pytest.raises(ValueError, match="dtype"):
            ut.check_array_like(name="v", val=[[1, 2]], dtype="bogus")

    def test_none_not_accepted(self):
        with pytest.raises(ValueError, match="should not be None"):
            ut.check_array_like(name="v", val=None)

    def test_convert_2d_from_1d(self):
        out = ut.check_array_like(name="v", val=[1.0, 2.0, 3.0], convert_2d=True)
        assert out.ndim == 2

    def test_convert_2d_ragged_nested_raises(self):
        with pytest.raises(ValueError):
            ut.check_array_like(name="v", val=[[1, 2], [3]], convert_2d=True,
                                dtype="float")


class TestCheckX:
    def test_valid(self):
        X = np.arange(12, dtype=float).reshape(4, 3)
        assert ut.check_X(X=X).shape == (4, 3)

    def test_inf_raises(self):
        # allow_nan=True lets inf past sklearn's finite check, so the explicit
        # np.isinf guard in check_X is what raises.
        X = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError, match="infinite"):
            ut.check_X(X=X, allow_nan=True)

    def test_min_unique_features_raises(self):
        # all columns constant -> 0 unique features < required
        X = np.ones((4, 3))
        with pytest.raises(ValueError, match="n_unique_features"):
            ut.check_X(X=X, min_n_unique_features=2)

    def test_none_not_accepted(self):
        with pytest.raises(ValueError, match="should not be None"):
            ut.check_X(X=None)


class TestCheckMatchXLabels:
    def test_valid(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        labels = [0, 1, 0, 1, 0, 1]
        assert ut.check_match_X_labels(X=X, labels=labels) is None

    def test_low_variance_raises(self):
        # one label group has identical rows -> zero variance
        X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        labels = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="too low"):
            ut.check_match_X_labels(X=X, labels=labels, check_variability=True)


class TestCheckDf:
    def _df(self):
        return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    def test_all_positive_raises(self):
        df = pd.DataFrame({"a": [1.0, -2.0]})
        with pytest.raises(ValueError, match="non-positive"):
            ut.check_df(name="df", df=df, check_all_positive=True)

    def test_cols_forbidden_raises(self):
        with pytest.raises(ValueError, match="forbidden"):
            ut.check_df(name="df", df=self._df(), cols_forbidden=["a"])

    def test_cols_nan_check_raises(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        with pytest.raises(ValueError, match="NaN"):
            ut.check_df(name="df", df=df, cols_nan_check=["a"])

    def test_cols_required_ok(self):
        assert ut.check_df(name="df", df=self._df(), cols_required=["a", "b"]) is None


class TestConsecutiveIndex:
    def test_unsorted_index_warns(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[2, 0, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ut.check_warning_consecutive_index(name="df", df=df)
        assert any("unsorted" in str(x.message) for x in w)

    def test_missing_values_index_warns(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ut.check_warning_consecutive_index(name="df", df=df)
        assert any("missing values" in str(x.message) for x in w)


class TestFilePath:
    def test_nonexistent_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            ut.check_file_path_exists(file_path="/no/such/__file__.txt")

    def test_existing_ok(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        assert ut.check_file_path_exists(file_path=str(f)) is None
