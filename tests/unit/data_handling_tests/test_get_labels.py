"""
This script tests the top-level get_labels() function (issue #308).

get_labels is the single-call form of the recurring
``(df[col] == positive_label).astype(int).to_numpy()`` expression that appears in 4+ places
of the gamma-secretase use case. It maps the positive value onto 1 and everything else onto 0.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa


# Helper functions
def _manual(df, positive_label, col="label"):
    return (df[col] == positive_label).astype(int).to_numpy()


# Normal Cases Test Class
class TestGetLabels:
    """Test get_labels() for each parameter individually."""

    def test_returns_int_numpy_array(self):
        df = pd.DataFrame({"entry": ["a", "b", "c"], "label": [1, 2, 1]})
        labels = aa.get_labels(df=df, positive_label=1)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype.kind == "i"
        assert labels.shape == (3,)

    def test_positive_label_default(self):
        df = pd.DataFrame({"label": [1, 0, 1, 0]})
        labels = aa.get_labels(df=df)
        assert np.array_equal(labels, np.array([1, 0, 1, 0]))

    def test_df_parameter(self):
        df = pd.DataFrame({"label": [2, 2, 1]})
        labels = aa.get_labels(df=df, positive_label=1)
        assert np.array_equal(labels, np.array([0, 0, 1]))

    def test_col_label_parameter(self):
        df = pd.DataFrame({"y": [1, 2, 1, 2]})
        labels = aa.get_labels(df=df, positive_label=2, col_label="y")
        assert np.array_equal(labels, np.array([0, 1, 0, 1]))


# Golden equivalence to the manual expression (KPI: >= 2 encodings)
class TestGetLabelsEquivalence:
    """Result equals the manual expression on multiple label encodings (KPI #308)."""

    def test_pu_encoding_1_2(self):
        # PU encoding: 1 = positive, 2 = unlabeled
        df = pd.DataFrame({"label": [1, 2, 1, 2, 2, 1]})
        assert np.array_equal(aa.get_labels(df=df, positive_label=1),
                              _manual(df, 1))

    def test_binary_encoding_0_1(self):
        # Standard {0, 1} encoding
        df = pd.DataFrame({"label": [0, 1, 1, 0]})
        assert np.array_equal(aa.get_labels(df=df, positive_label=1),
                              _manual(df, 1))

    def test_multiclass_encoding(self):
        # Multi-class: pick one class as positive
        df = pd.DataFrame({"label": [0, 1, 2, 0, 1, 2]})
        for pos in (0, 1, 2):
            assert np.array_equal(aa.get_labels(df=df, positive_label=pos),
                                  _manual(df, pos))

    def test_string_labels(self):
        df = pd.DataFrame({"label": ["sub", "non", "sub", "unl"]})
        assert np.array_equal(aa.get_labels(df=df, positive_label="sub"),
                              _manual(df, "sub"))

    def test_single_class_column_maps_all_ones(self):
        # Pure mapping: unlike dPULearn.fit, get_labels does not require >1 distinct value,
        # so an all-positive column maps to all ones rather than raising.
        df = pd.DataFrame({"label": [1, 1, 1]})
        assert np.array_equal(aa.get_labels(df=df, positive_label=1),
                              np.array([1, 1, 1]))

    def test_nan_maps_to_zero(self):
        # NaN never equals positive_label, so it becomes 0.
        df = pd.DataFrame({"label": [1.0, np.nan, 1.0]})
        assert np.array_equal(aa.get_labels(df=df, positive_label=1.0),
                              np.array([1, 0, 1]))


# Negative Cases Test Class
class TestGetLabelsNegative:
    """Invalid inputs must raise informative ValueErrors."""

    def test_df_none(self):
        with pytest.raises(ValueError):
            aa.get_labels(df=None, positive_label=1)

    def test_df_not_dataframe(self):
        with pytest.raises(ValueError):
            aa.get_labels(df=[1, 2, 3], positive_label=1)

    def test_missing_label_column(self):
        df = pd.DataFrame({"entry": ["a", "b"], "y": [1, 0]})
        with pytest.raises(ValueError):
            aa.get_labels(df=df, positive_label=1)

    def test_custom_col_missing(self):
        df = pd.DataFrame({"label": [1, 0]})
        with pytest.raises(ValueError):
            aa.get_labels(df=df, positive_label=1, col_label="missing")

    def test_positive_label_absent(self):
        df = pd.DataFrame({"label": [1, 2, 1]})
        with pytest.raises(ValueError):
            aa.get_labels(df=df, positive_label=9)
