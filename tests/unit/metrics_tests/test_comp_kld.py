"""
This is a script for testing aa.comp_auc_adjusted function
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as some
import hypothesis.extra.numpy as npst
import aaanalysis as aa
from sklearn.decomposition import PCA
import warnings

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# Helper function
def has_sufficient_variability(X=None, labels=None):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        group_X = X[labels == label]
        if not np.all(np.var(group_X, axis=0) != 0):
            return False
    return True

def is_not_lower_dimensional(X, explained_variance_threshold=0.6):
    pca = PCA()
    pca.fit(X)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    return np.sum(explained_variance < explained_variance_threshold) < len(explained_variance)


def create_labels(size, unique_values):
    labels = np.array([0, 1, 0, 1] + list(np.random.choice([0, 1], size=size - 4)))
    return labels


def check_invalid_conditions(X, labels, min_samples=3, check_unique=True, check_kdl=False,
                             variance_threshold=0.01, explained_variance_threshold=0.95):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress all warnings within this block
        n_samples, n_features = X.shape
        n_unique_samples = len(set(map(tuple, X)))
        conditions = [
            (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
            (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
            (n_features < 1, f"n_features={n_features} should be >= 2"),
            (len(labels) != n_samples, "Length of labels should match n_samples."),
        ]
        if check_unique:
            conditions.append((n_unique_samples < 3, "Feature matrix 'X' should not have all identical samples."))
        if check_kdl:
            conditions.extend([(not has_sufficient_variability(X, variance_threshold), "Insufficient variability in X's features."),
                               (is_not_lower_dimensional(X, explained_variance_threshold), "X lies in a lower-dimensional subspace.")])

        for condition, msg in conditions:
            if condition:
                print(msg)  # Optionally, print the message for debugging
                return True
        return False


class TestCompKld:
    # Test for parameter X
    @settings(deadline=350, max_examples=400)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_valid(self, X):
        X = np.asarray(X)
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(len(X), [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels, check_kdl=True)
            if not is_invalid:
                try:
                    assert isinstance(aa.comp_kld(X, labels), np.ndarray)
                except ValueError as e:
                    # Check if the error is the specific "lower-dimensional subspace" error
                    if "lower-dimensional subspace" in str(e):
                        pytest.skip("Specific lower-dimensional subspace error occurred, skipping this test.")
                    else:
                        raise

    @settings(deadline=350, max_examples=20)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=some.floats(min_value=-1e3, max_value=1e3)))
    def test_X_invalid(self, X):
        X = np.asarray(X)
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(len(X), [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels)
            if is_invalid:
                with pytest.raises(ValueError):
                    aa.comp_kld(X, labels)

    # Test for parameter labels
    @settings(deadline=350, max_examples=400)
    @given(labels=some.lists(some.integers(min_value=0, max_value=1), min_size=4))
    def test_labels_valid(self, labels):
        X = np.random.rand(len(labels), 3)
        is_invalid = check_invalid_conditions(X=X, labels=labels, check_kdl=True)
        valid_labels = sum([x == 0 for x in labels]) >= 2 and sum([x == 1 for x in labels]) >= 2
        if not is_invalid and valid_labels:
            try:
                assert isinstance(aa.comp_kld(X, labels), np.ndarray)
            except ValueError as e:
                # Check if the error is the specific "lower-dimensional subspace" error
                if "lower-dimensional subspace" in str(e):
                    pytest.skip("Specific lower-dimensional subspace error occurred, skipping this test.")
                else:
                    raise

    @settings(deadline=350, max_examples=20)
    @given(labels=some.lists(some.integers(), min_size=2))
    def test_labels_invalid(self, labels):
        X = np.random.rand(len(labels), 3)
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        if is_invalid:
            with pytest.raises(ValueError):
                aa.comp_kld(X, labels)

    def test_labels_invalid_short(self):
        X = np.random.rand(3, 3)
        labels = [0, 1, 1]
        with pytest.raises(ValueError):
            aa.comp_kld(X, labels)

    # Test for parameter label_test
    @settings(deadline=350, max_examples=5)
    @given(label_test=some.integers(min_value=0, max_value=1))
    def test_label_test_valid(self, label_test):
        X = np.random.rand(10, 3)
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(len(X), [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels, check_kdl=True)
            if not is_invalid:
                try:
                    assert isinstance(aa.comp_kld(X, labels, label_test=label_test), np.ndarray)
                except ValueError as e:
                    # Check if the error is the specific "lower-dimensional subspace" error
                    if "lower-dimensional subspace" in str(e):
                        pytest.skip("Specific lower-dimensional subspace error occurred, skipping this test.")
                    else:
                        raise

    @settings(deadline=350, max_examples=20)
    @given(label_test=some.integers())
    def test_label_test_invalid(self, label_test):
        X = np.random.rand(10, 3)
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(len(X), [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels)
            if is_invalid:
                with pytest.raises(ValueError):
                    aa.comp_kld(X, labels, label_test=label_test)

    # Test for parameter label_ref
    @settings(deadline=350, max_examples=5)
    @given(label_ref=some.integers(min_value=0, max_value=1))
    def test_label_ref_valid(self, label_ref):
        X = np.random.rand(10, 3)
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(10, [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels, check_kdl=True)
            if not is_invalid:
                try:
                    assert isinstance(aa.comp_kld(X, labels, label_ref=label_ref), np.ndarray)
                except ValueError as e:
                    # Check if the error is the specific "lower-dimensional subspace" error
                    if "lower-dimensional subspace" in str(e):
                        pytest.skip("Specific lower-dimensional subspace error occurred, skipping this test.")
                    else:
                        raise

    @settings(deadline=350, max_examples=20)
    @given(label_ref=some.integers())
    def test_label_ref_invalid(self, label_ref):
        X = np.random.rand(10, 3)
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(10, [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels)
            if is_invalid:
                with pytest.raises(ValueError):
                    aa.comp_kld(X, labels, label_ref=label_ref)

class TestCompKldComplex:
    # Test for combinations of valid inputs
    @settings(deadline=350, max_examples=400)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10),
                         elements=some.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
           label_test=some.integers(min_value=0, max_value=1),
           label_ref=some.integers(min_value=0, max_value=1))
    def test_valid_input_combinations(self, X, label_test, label_ref):
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(len(X), [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels, check_kdl=True)
            if not is_invalid:
                try:
                    assert isinstance(aa.comp_kld(X, labels, label_test, label_ref), np.ndarray)
                except ValueError as e:
                    # Check if the error is the specific "lower-dimensional subspace" error
                    if "lower-dimensional subspace" in str(e):
                        pytest.skip("Specific lower-dimensional subspace error occurred, skipping this test.")
                    else:
                        raise

    # Test for combinations of invalid inputs
    @settings(deadline=350, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10),
                         elements=some.floats(min_value=-1e3, max_value=1e3, )),
           label_test=some.integers(),
           label_ref=some.integers())
    def test_invalid_input_combinations(self, X, label_test, label_ref):
        size = X.shape[0]
        if size >= 4:
            labels = create_labels(len(X), [0, 1])
            is_invalid = check_invalid_conditions(X=X, labels=labels)
            if is_invalid:
                with pytest.raises(ValueError):
                    aa.comp_kld(X, labels, label_test, label_ref)
