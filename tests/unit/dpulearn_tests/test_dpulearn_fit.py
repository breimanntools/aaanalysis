"""
This script tests the dPULearn.fit() method.
"""
from hypothesis import given, example, settings
import hypothesis.strategies as some
import hypothesis.extra.numpy as npst
import numpy as np
import pandas as pd
import pytest
import aaanalysis as aa

# Helper functions
def create_labels(size):
    labels = np.array([1, 2] + list(np.random.choice([1, 2], size=size-2)))
    return labels


def check_invalid_conditions(X, labels, min_samples=3, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
        (len(labels) != n_samples, "Length of labels should match n_samples."),
    ]
    if check_unique:
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False


# Normal Cases Test Class
class TestdPULearnFit:
    """Test dPULearn.fit() method for each parameter individually."""

    # Positive tests
    @settings(deadline=350, max_examples=100)
    @given(X=npst.arrays(dtype=np.float64,
                         shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        """Test the 'X' parameter with valid inputs."""
        dpul = aa.dPULearn()
        size = X.shape[0]
        if size >= 2:
            labels = create_labels(X.shape[0])
            valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
            is_invalid =  check_invalid_conditions(X=X, labels=labels)
            if not is_invalid and valid_labels:
                df_pu = dpul.fit(X, labels).df_pu_
                assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=1000, max_examples=100)
    @given(labels=npst.arrays(dtype=np.int32, shape=(100,)))
    def test_labels_parameter(self, labels):
        """Test the 'labels' parameter with valid inputs."""
        X = np.random.rand(100, 5)  # Assuming 100 samples, 5 features
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if len(set(labels)) < 2 or 1 not in labels or 2 not in labels:
            valid_labels = False
        if not is_invalid and valid_labels:
            df_pu = dpul.fit(X, labels).df_pu_
            assert isinstance(df_pu, pd.DataFrame)

    def test_labels(self):
        """Test the 'labels' parameter with valid inputs."""
        X = np.random.rand(100, 5)  # Assuming 100 samples, 5 features
        labels = np.asarray([1]* 50 + [2]* 50)
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if not is_invalid and valid_labels:
            df_pu = dpul.fit(X, labels).df_pu_
            assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=1000, max_examples=100)
    @given(n_unl_to_neg=some.integers(min_value=1))
    def test_n_unl_to_neg_parameter(self, n_unl_to_neg):
        """Test the 'n_unl_to_neg' parameter with valid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        n_unl = sum([x == 2 for x in labels])
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if not is_invalid and valid_labels and n_unl > n_unl_to_neg :
            df_pu = dpul.fit(X, labels, n_unl_to_neg=n_unl_to_neg).df_pu_
            assert isinstance(df_pu, pd.DataFrame)
    
    @settings(deadline=1000, max_examples=4)
    @given(metric=some.none() | some.sampled_from(["euclidean", "manhattan", "cosine", None]))
    def test_metric_parameter(self, metric):
        """Test the 'metric' parameter."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        if not is_invalid:
            for n in [5, 25, 39]:
                df_pu = dpul.fit(X, labels, metric=metric, n_unl_to_neg=n).df_pu_
                assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=1000, max_examples=100)
    @given(n_components=some.one_of(some.floats(min_value=0.1, max_value=1.0), some.integers(min_value=1)))
    def test_n_components_parameter(self, n_components):
        """Test the 'n_components' parameter with valid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        n_samples, n_features = X.shape
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        for i in [5, 26, 39]:
            if not is_invalid:
                if n_components < min(n_features, n_samples) and n_components not in [0.0, 1.0]:
                    df_pu = dpul.fit(X, labels, n_components=n_components, n_unl_to_neg=i).df_pu_
                    assert isinstance(df_pu, pd.DataFrame)

    # Negative tests
    @settings(deadline=1000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(10,)))
    def test_X_invalid_shape(self, X):
        """Test the 'X' parameter with invalid shape."""
        dpul = aa.dPULearn()
        labels = create_labels(X.size)
        with pytest.raises(ValueError):
            dpul.fit(X, labels)

    @settings(deadline=1000, max_examples=10)
    @given(labels=npst.arrays(dtype=np.int32, shape=(99,)))
    def test_labels_invalid_shape(self, labels):
        """Test the 'labels' parameter with invalid shape."""
        X = np.random.rand(100, 5)
        dpul = aa.dPULearn()
        with pytest.raises(ValueError):
            dpul.fit(X, labels)

    def test_invalid_labels(self):
        """Test the 'labels' parameter with invalid shape."""
        X = np.random.rand(6, 5)
        dpul = aa.dPULearn()
        invalid_labels = [[1, 1, 2, 2, 3, 3],
                          [0, 0, 1, 1, 1, 1],
                          [0, 1, 2, 2, 2, 2],
                          ["A", 1, 1, 2, 2, 2],
                          [1, 1, 1, 1, 1, 1],
                          [2, 2, 2, 2, 2, 2],
                          [-1, -1, 2, 2, 2, 2],
                          [4, 1.0, 1, 2, 2, 2, 2],
                          [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]
        for labels in invalid_labels:
            with pytest.raises(ValueError):
                dpul.fit(X, labels)

    @settings(deadline=1000, max_examples=100)
    @given(n_unl_to_neg=some.integers(max_value=0))
    def test_n_unl_to_neg_invalid(self, n_unl_to_neg):
        """Test the 'n_unl_to_neg' parameter with invalid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        if is_invalid:
            with pytest.raises(ValueError):
                dpul.fit(X, labels, n_unl_to_neg=n_unl_to_neg)
    
    @settings(deadline=1000, max_examples=10)
    @given(metric=some.just("invalid_metric"))
    def test_invalid_metric(self, metric):
        """Test with an invalid 'metric' value."""
        valid_metrics = ["euclidean", "manhattan", "cosine", None]
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        if not is_invalid and not metric in valid_metrics:
            with pytest.raises(ValueError):
               dpul.fit(X, labels, metric=metric)

    
    @settings(deadline=1000, max_examples=10)
    @given(n_components=some.one_of(some.floats(max_value=0.0), some.integers(max_value=0)))
    def test_n_components_invalid(self, n_components):
        """Test the 'n_components' parameter with invalid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if is_invalid and valid_labels:
            with pytest.raises(ValueError):
                dpul.fit(X, labels, n_components=n_components)


class TestdPULearnFitComplex:
    """Test dPULearn.fit() method for combinations of parameters."""

    @settings(deadline=1000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64,
                         shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50)),
           labels=npst.arrays(dtype=np.int32, elements=some.sampled_from([1, 2]), shape=(20,)),
           n_unl_to_neg=some.integers(min_value=1, max_value=10),
           n_components=some.floats(min_value=0.5, max_value=1.0))
    def test_valid_combinations(self, X, labels, n_unl_to_neg, n_components):
        """Test valid combinations of parameters."""
        dpul = aa.dPULearn()
        n_samples, n_features = X.shape
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if not is_invalid and n_components < min(n_features, n_samples) and valid_labels:
            if X.shape[0] != labels.size:
                with pytest.raises(ValueError):
                    dpul.fit(X, labels, n_unl_to_neg, n_components)
            else:
                df_pu = dpul.fit(X, labels, n_unl_to_neg, n_components).df_pu_
                assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=1000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(10, 5)),
           labels=npst.arrays(dtype=np.int32, elements=some.integers(min_value=3, max_value=100), shape=(10,)),
           n_unl_to_neg=some.integers(max_value=0),
           n_components=some.floats(max_value=0.0))
    def test_invalid_combinations(self, X, labels, n_unl_to_neg, n_components):
        """Test invalid combinations of parameters."""
        dpul = aa.dPULearn()
        is_invalid =  check_invalid_conditions(X=X, labels=labels)
        if is_invalid:
            with pytest.raises(ValueError):
                dpul.fit(X, labels, n_unl_to_neg, n_components)
