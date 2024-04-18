"""This script tests the TreeModel.fit() method."""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import aaanalysis as aa
import hypothesis.extra.numpy as npst

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


aa.options["verbose"] = False

def create_labels(size):
    labels = np.array([1, 1, 0, 0] + list(np.random.choice([1, 0], size=size-4)))
    return labels


def check_invalid_conditions(X, min_samples=3, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_samples = len(set(map(tuple, X)))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
    ]
    if check_unique:
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False

N_ROUNDS = 2
ARGS = dict(use_rfe=False, n_cv=2, n_rounds=N_ROUNDS)


class TestTreeModelFit:
    """Test TreeModel.fit() method for each parameter individually with positive test cases."""

    # Positive test cases
    @settings(deadline=100000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        tree_model = aa.TreeModel()
        size, n_feat = X.shape
        if size > 3 and n_feat > 3:
            labels = create_labels(X.shape[0])
            if not check_invalid_conditions(X):
                tree_model.fit(X=X, labels=labels, **ARGS)
                assert tree_model.list_models_ is not None
                assert len(tree_model.is_selected_) == N_ROUNDS
                assert len(tree_model.feat_importance) == n_feat
                assert len(tree_model.feat_importance_std) == n_feat
                assert isinstance(tree_model.is_selected_, np.ndarray)
                assert isinstance(tree_model.feat_importance, np.ndarray)
                assert isinstance(tree_model.feat_importance_std, np.ndarray)

    @settings(max_examples=5, deadline=100000)
    @given(labels=st.lists(st.integers(0, 1), min_size=10, max_size=20))
    def test_labels_parameter(self, labels):
        """Test the 'labels' parameter with valid inputs."""
        X = np.random.rand(len(labels), 10)
        size, n_feat = X.shape
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        if min_class_count >= 2 and n_feat > 3 and len(set(labels)) == 2:
            if not check_invalid_conditions(X):
                tree_model = aa.TreeModel()
                tree_model.fit(X=X, labels=labels, **ARGS)
                assert tree_model.list_models_ is not None
                assert len(tree_model.is_selected_) == N_ROUNDS
                assert len(tree_model.feat_importance) == n_feat
                assert len(tree_model.feat_importance_std) == n_feat
                assert isinstance(tree_model.is_selected_, np.ndarray)
                assert isinstance(tree_model.feat_importance, np.ndarray)
                assert isinstance(tree_model.feat_importance_std, np.ndarray)

    @settings(max_examples=3, deadline=100000)
    @given(n_rounds=st.integers(min_value=1, max_value=3))
    def test_n_rounds_parameter(self, n_rounds):
        """Test the 'n_rounds' parameter with valid inputs."""
        X = np.random.rand(10, 5)
        size, n_feat = X.shape
        labels = create_labels(X.shape[0])
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        if min_class_count >= 2 and n_feat > 3 and not check_invalid_conditions(X):
            tree_model = aa.TreeModel()
            tree_model.fit(X=X, labels=labels, n_rounds=n_rounds, use_rfe=False, n_cv=2)
            assert len(tree_model.list_models_) == n_rounds
            assert tree_model.list_models_ is not None
            assert len(tree_model.is_selected_) == n_rounds
            assert len(tree_model.feat_importance) == n_feat
            assert len(tree_model.feat_importance_std) == n_feat
            assert isinstance(tree_model.is_selected_, np.ndarray)
            assert isinstance(tree_model.feat_importance, np.ndarray)
            assert isinstance(tree_model.feat_importance_std, np.ndarray)

    def test_use_rfe_parameter(self):
        """Test the 'use_rfe' parameter with valid inputs."""
        X = np.random.rand(10, 5)
        labels = create_labels(X.shape[0])
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        size, n_feat = X.shape
        if min_class_count >= 2 and n_feat > 3 and not check_invalid_conditions(X):
            tree_model = aa.TreeModel()
            tree_model.fit(X=X, labels=labels, n_feat_min=n_feat-2, n_feat_max=n_feat-1, n_cv=2, n_rounds=2)
            assert tree_model.list_models_ is not None
            assert len(tree_model.is_selected_) == N_ROUNDS
            assert len(tree_model.feat_importance) == n_feat
            assert len(tree_model.feat_importance_std) == n_feat
            assert isinstance(tree_model.is_selected_, np.ndarray)
            assert isinstance(tree_model.feat_importance, np.ndarray)
            assert isinstance(tree_model.feat_importance_std, np.ndarray)

    @settings(max_examples=10, deadline=100000)
    @given(n_cv=st.integers(min_value=2))
    def test_n_cv_parameter(self, n_cv):
        """Test the 'n_cv' parameter with valid inputs."""
        X = np.random.rand(10, 5)
        labels = create_labels(X.shape[0])
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        size, n_feat = X.shape
        if min_class_count >= n_cv and n_feat > 3 and not check_invalid_conditions(X):
            tree_model = aa.TreeModel()
            tree_model.fit(X=X, labels=labels, n_cv=n_cv, use_rfe=False, n_rounds=2)
            assert tree_model.list_models_ is not None
            assert len(tree_model.is_selected_) == N_ROUNDS
            assert len(tree_model.feat_importance) == n_feat
            assert len(tree_model.feat_importance_std) == n_feat
            assert isinstance(tree_model.is_selected_, np.ndarray)
            assert isinstance(tree_model.feat_importance, np.ndarray)
            assert isinstance(tree_model.feat_importance_std, np.ndarray)

    @settings(max_examples=5, deadline=100000)
    @given(n_feat_min=st.integers(min_value=1), n_feat_max=st.integers(min_value=2))
    def test_n_feat_min_max_parameter(self, n_feat_min, n_feat_max):
        """Test the 'n_feat_min' and 'n_feat_max' parameters with valid inputs."""
        X = np.random.rand(10, 5)
        labels = create_labels(X.shape[0])
        size, n_feat = X.shape
        if size > 2 and n_feat > 3 and not check_invalid_conditions(X):
            tree_model = aa.TreeModel()
            if n_feat_min >= n_feat_max:
                n_feat_max = n_feat_min + 1
            tree_model.fit(X=X, labels=labels, n_feat_min=n_feat_min, n_feat_max=n_feat_max, **ARGS)
            assert tree_model.list_models_ is not None
            assert len(tree_model.is_selected_) == N_ROUNDS
            assert len(tree_model.feat_importance) == n_feat
            assert len(tree_model.feat_importance_std) == n_feat
            assert isinstance(tree_model.is_selected_, np.ndarray)
            assert isinstance(tree_model.feat_importance, np.ndarray)
            assert isinstance(tree_model.feat_importance_std, np.ndarray)



    @settings(max_examples=3, deadline=100000)
    @given(metric=st.sampled_from(['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']))
    def test_metric_parameter(self, metric):
        X = np.random.rand(10, 5)
        labels = create_labels(X.shape[0])
        size, n_feat = X.shape
        if size > 2 and n_feat > 3 and not check_invalid_conditions(X):
            tree_model = aa.TreeModel()
            tree_model.fit(X=X, labels=labels, metric=metric, n_feat_min=n_feat-2, n_feat_max=n_feat-1, **ARGS)
            assert tree_model.list_models_ is not None
            assert len(tree_model.is_selected_) == N_ROUNDS
            assert len(tree_model.feat_importance) == n_feat
            assert len(tree_model.feat_importance_std) == n_feat
            assert isinstance(tree_model.is_selected_, np.ndarray)
            assert isinstance(tree_model.feat_importance, np.ndarray)
            assert isinstance(tree_model.feat_importance_std, np.ndarray)

    @settings(max_examples=3, deadline=100000)
    @given(step=st.one_of(st.integers(min_value=1, max_value=3), st.none()))
    def test_step_parameter(self, step):
        """Test the 'step' parameter with valid inputs."""
        X = np.random.rand(10, 10)
        labels = create_labels(X.shape[0])
        size, n_feat = X.shape
        if size > 2 and n_feat > 3 and not check_invalid_conditions(X) and (step is None or step < n_feat):
            tree_model = aa.TreeModel()
            tree_model.fit(X=X, labels=labels, step=step, n_feat_min=n_feat-2, n_feat_max=n_feat-1, n_cv=2, n_rounds=2)
            assert tree_model.list_models_ is not None
            assert len(tree_model.is_selected_) == N_ROUNDS
            assert len(tree_model.feat_importance) == n_feat
            assert len(tree_model.feat_importance_std) == n_feat
            assert isinstance(tree_model.is_selected_, np.ndarray)
            assert isinstance(tree_model.feat_importance, np.ndarray)
            assert isinstance(tree_model.feat_importance_std, np.ndarray)

    # Negative test cases
    def test_invalid_X_parameter(self):
        """Test with invalid 'X' parameter."""
        tree_model = aa.TreeModel()
        with pytest.raises(ValueError):
            tree_model.fit(X="invalid", labels=np.array([0, 1]))
        with pytest.raises(ValueError):
            tree_model.fit(X=[], labels=np.array([0, 1]))
        with pytest.raises(ValueError):
            tree_model.fit(X={}, labels=np.array([0, 1]))
        with pytest.raises(ValueError):
            tree_model.fit(X=np.array(["asdf", "asdf"]), labels=np.array([0, 1]))

    def test_invalid_labels_parameter(self):
        """Test with invalid 'labels' parameter."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        size, _ = X.shape
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels="invalid")
        labels = [1] * size
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels)
        labels[0] = "str"
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels)
        labels[0] = 0
        labels.append(1)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels)

    def test_invalid_n_rounds_parameter(self):
        """Test with invalid 'n_rounds' parameter."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        invalid_n_rounds = -1  # Negative integer
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_rounds=invalid_n_rounds)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_rounds="estr")
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_rounds=[12, 2])

    def test_invalid_n_cv_parameter(self):
        """Test with invalid 'n_cv' parameter."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        n_samples, n_feat = X.shape
        labels = np.random.randint(0, 2, 10)
        invalid_n_cv = 0  # Non-positive integer
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_cv=invalid_n_cv)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_cv=1000)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_cv=n_samples)

    def test_invalid_n_feat_min_max_parameter(self):
        """Test with invalid 'n_feat_min' and 'n_feat_max' parameters."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_feat_min=5, n_feat_max=3)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_feat_min=5, n_feat_max="str")
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_feat_min=5, n_feat_max=None)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, n_feat_min=-123, n_feat_max=123)

    def test_invalid_metric_parameter(self):
        """Test with invalid 'metric' parameter."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        invalid_metric = 'not_a_metric'  # Non-existent metric
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, metric=invalid_metric)

    def test_invalid_step_parameter(self):
        """Test with invalid 'step' parameter."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 10)
        labels = np.random.randint(0, 2, 10)
        invalid_step = 11  # Greater than number of features
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, step=invalid_step)

    def test_X_with_nan_or_inf(self):
        """Test 'X' parameter containing NaN or Inf."""
        tree_model = aa.TreeModel()
        X_with_nan = np.array([[np.nan, 1], [2, 3]])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            tree_model.fit(X=X_with_nan, labels=labels)

    def test_labels_not_matching_X(self):
        """Test when 'labels' length does not match 'X'."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        mismatched_labels = np.array([1, 0])  # Length not matching X's row count
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=mismatched_labels)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=[0, 12, "ser"])

    def test_invalid_use_rfe_parameter(self):
        """Test with invalid 'use_rfe' parameter."""
        tree_model = aa.TreeModel()
        X = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        invalid_use_rfe = 'not_boolean'  # Not a boolean value
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, use_rfe=invalid_use_rfe)
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, use_rfe=[])
        with pytest.raises(ValueError):
            tree_model.fit(X=X, labels=labels, use_rfe="asrter")