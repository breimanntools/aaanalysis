"""This script tests the filter_coverage() method of the AAclust class."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import aaanalysis as aa  # Assuming AAclust is part of aaanalysis module
import warnings

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# Mock Utility Functions
def create_mock_df_cat(scale_ids, col_name):
    return pd.DataFrame({
        'scale_id': scale_ids,
        'category': ['cat']*len(scale_ids),
        'subcategory': ['subcat']*len(scale_ids),
        'scale_name': ['scale']*len(scale_ids),
    })


# Helper function to create valid inputs
def create_valid_inputs():
    df_scales = aa.load_scales()
    scale_ids = list(df_scales.columns)[:200]
    X = df_scales[scale_ids].T.values
    df_cat = aa.load_scales(name="scales_cat")
    names_ref = df_cat[df_cat["scale_id"].isin(scale_ids)]["subcategory"].tolist()
    return X, scale_ids, names_ref, df_cat


# Check invalid conditions function
def check_invalid_conditions(X, min_samples=3, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
        (n_unique_samples <= 2, "n_uniuqe_samples should be >= 3")
    ]
    if check_unique:
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False


class TestFilterCoverage:
    """Test filter_coverage() method for each parameter individually with positive test cases."""

    # Positive test cases
    @settings(deadline=100000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        aac = aa.AAclust()
        size, n_feat = X.shape
        if size > 2 and not check_invalid_conditions(X):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scale_ids = [f"scale_{i}" for i in range(size)]
                df_cat = create_mock_df_cat(scale_ids, "subcategory")
                names_ref = df_cat["subcategory"].to_list()
                selected_scale_ids = aac.filter_coverage(X, scale_ids, names_ref, 100, df_cat, "subcategory")
                assert isinstance(selected_scale_ids, list)
                assert all(isinstance(scale_id, str) for scale_id in selected_scale_ids)

    @settings(deadline=100000, max_examples=10)
    @given(scale_ids=st.lists(st.text(), min_size=2, max_size=10))
    def test_scale_ids_parameter(self, scale_ids):
        aac = aa.AAclust()
        size = len(scale_ids)
        X = np.random.rand(size, 5)
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        names_ref = df_cat["subcategory"].to_list()
        if not check_invalid_conditions(X):
            selected_scale_ids = aac.filter_coverage(X, scale_ids, names_ref, 100, df_cat, "subcategory")
            assert isinstance(selected_scale_ids, list)
            assert all(isinstance(scale_id, str) for scale_id in selected_scale_ids)

    @settings(deadline=100000, max_examples=10)
    @given(names_ref=st.lists(st.text(), min_size=2, max_size=10))
    def test_names_ref_parameter(self, names_ref):
        aac = aa.AAclust()
        size = len(names_ref)
        X = np.random.rand(size, 5)
        scale_ids = [f"scale_{i}" for i in range(size)]
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        df_cat["subcategory"] = names_ref
        if not check_invalid_conditions(X):
            selected_scale_ids = aac.filter_coverage(X, scale_ids, names_ref, 100, df_cat, "subcategory")
            assert isinstance(selected_scale_ids, list)
            assert all(isinstance(scale_id, str) for scale_id in selected_scale_ids)

    @settings(deadline=100000, max_examples=10)
    @given(min_coverage=st.integers(min_value=10, max_value=100))
    def test_min_coverage_parameter(self, min_coverage):
        aac = aa.AAclust()
        size = 10
        X = np.random.rand(size, 5)
        scale_ids = [f"scale_{i}" for i in range(size)]
        names_ref = [f"subcat_{i}" for i in range(size)]
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        df_cat["subcategory"] = names_ref
        if not check_invalid_conditions(X):
            selected_scale_ids = aac.filter_coverage(X, scale_ids, names_ref, min_coverage, df_cat, "subcategory")
            assert isinstance(selected_scale_ids, list)
            assert all(isinstance(scale_id, str) for scale_id in selected_scale_ids)

    @settings(deadline=100000, max_examples=10)
    @given(col_name=st.sampled_from(['category', 'subcategory', 'scale_name']))
    def test_col_name_parameter(self, col_name):
        aac = aa.AAclust()
        size = 10
        X = np.random.rand(size, 5)
        scale_ids = [f"scale_{i}" for i in range(size)]
        df_cat = create_mock_df_cat(scale_ids, col_name)
        names_ref = df_cat[col_name].to_list()
        if not check_invalid_conditions(X):
            selected_scale_ids = aac.filter_coverage(X, scale_ids, names_ref, 100, df_cat, col_name)
            assert isinstance(selected_scale_ids, list)
            assert all(isinstance(scale_id, str) for scale_id in selected_scale_ids)

    def test_df_cat_parameter(self):
        X, scale_ids, names_ref, df_cat = create_valid_inputs()
        aac = aa.AAclust()
        if not check_invalid_conditions(X):
            selected_scale_ids = aac.filter_coverage(X, scale_ids, names_ref, 100, df_cat, "subcategory")
            assert isinstance(selected_scale_ids, list)
            assert all(isinstance(scale_id, str) for scale_id in selected_scale_ids)

    # Negative test cases
    def test_invalid_X_parameter(self):
        """Test with invalid 'X' parameter."""
        aac = aa.AAclust()
        scale_ids = ["scale_1", "scale_2"]
        names_ref = ["subcat_1", "subcat_2"]
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        with pytest.raises(ValueError):
            aac.filter_coverage(X="invalid", scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=[], scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X={}, scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=np.array(["asdf", "asdf"]), scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat)

    def test_invalid_scale_ids_parameter(self):
        """Test with invalid 'scale_ids' parameter."""
        aac = aa.AAclust()
        X = np.random.rand(10, 5)
        names_ref = ["subcat_1", "subcat_2"]
        df_cat = create_mock_df_cat(["scale_1", "scale_2"], "subcategory")

        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids="invalid", names_ref=names_ref, df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=[], names_ref=names_ref, df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids={}, names_ref=names_ref, df_cat=df_cat)

    def test_invalid_names_ref_parameter(self):
        """Test with invalid 'names_ref' parameter."""
        aac = aa.AAclust()
        X = np.random.rand(10, 5)
        scale_ids = ["scale_1", "scale_2"]
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref="invalid", df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=[], df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref={}, df_cat=df_cat)

    def test_invalid_min_coverage_parameter(self):
        """Test with invalid 'min_coverage' parameter."""
        aac = aa.AAclust()
        X = np.random.rand(10, 5)
        scale_ids = ["scale_1", "scale_2"]
        names_ref = ["subcat_1", "subcat_2"]
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, min_coverage="invalid", df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, min_coverage=[], df_cat=df_cat)
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, min_coverage={}, df_cat=df_cat)

    def test_invalid_df_cat_parameter(self):
        """Test with invalid 'df_cat' parameter."""
        aac = aa.AAclust()
        X = np.random.rand(10, 5)
        scale_ids = ["scale_1", "scale_2"]
        names_ref = ["subcat_1", "subcat_2"]
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, df_cat="invalid")
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, df_cat=[])
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, df_cat={})

    def test_invalid_col_name_parameter(self):
        """Test with invalid 'col_name' parameter."""
        aac = aa.AAclust()
        X = np.random.rand(10, 5)
        scale_ids = ["scale_1", "scale_2"]
        names_ref = ["subcat_1", "subcat_2"]
        df_cat = create_mock_df_cat(scale_ids, "subcategory")
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat, col_name="invalid")
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat, col_name=[])
        with pytest.raises(ValueError):
            aac.filter_coverage(X=X, scale_ids=scale_ids, names_ref=names_ref, df_cat=df_cat, col_name={})
