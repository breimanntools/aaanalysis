"""This script tests the ShapExplainer.fit() method."""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import aaanalysis as aa
import hypothesis.extra.numpy as npst

aa.options["verbose"] = False


# Helper functions
def check_invalid_conditions(X, min_samples=3, min_unique_features=2, check_unique=True):
    n_samples, n_features = X.shape
    # Check for a minimum number of unique values in each feature
    unique_features_count = sum([len(set(X[:, col])) > 1 for col in range(n_features)])
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 3, f"n_features={n_features} should be >= 3"),
        (unique_features_count < min_unique_features, f"Not enough unique features: found {unique_features_count}, require at least {min_unique_features}")
                  ]
    if check_unique:
        n_unique_samples = len(set(map(tuple, X)))
        conditions.append((n_unique_samples == 1, "Feature matrix 'X' should not have all identical samples."))
    for condition, msg in conditions:
        if condition:
            return True
    return False


def create_labels(size):
    labels = np.array([1, 1, 0, 0] + list(np.random.choice([1, 0], size=size-4)))
    return labels


def create_list_is_selected(n_features=None, n_rows=1, n_arrays=2, d1=True):
    if d1:
        list_is_selected = [np.random.choice([True, False], size=n_features) for _ in range(n_arrays)]
    else:
        list_is_selected = [np.random.choice([True, False], size=(n_rows, n_features)) for _ in range(n_arrays)]
    return list_is_selected


N_ROUNDS = 2
ARGS = dict(n_rounds=N_ROUNDS)

MODEL_KWARGS = dict(list_model_classes=[RandomForestClassifier, ExtraTreesClassifier])

# Create valid X
df_seq = aa.load_dataset(name="DOM_GSEC")
df_feat = aa.load_features()
valid_labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)


class TestShapExplainerFit:
    """
    Simple Positive Test Cases for ShapExplainer.fit() method.
    Each test focuses on one parameter.
    """

    # Positive test cases
    @settings(deadline=100000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        se = aa.ShapExplainer(**MODEL_KWARGS)
        size, n_feat = X.shape
        if size > 3 and n_feat > 3:
            labels = create_labels(X.shape[0])
            if not check_invalid_conditions(X):
                se.fit(X, labels=labels, **ARGS)
                assert se.shap_values is not None
                assert se.exp_value is not None

    @settings(max_examples=5, deadline=100000)
    @given(labels=st.lists(st.integers(0, 1), min_size=10, max_size=20))
    def test_labels_parameter(self, labels):
        X = np.random.rand(len(labels), 10)
        size, n_feat = X.shape
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        if min_class_count >= 2 and n_feat > 3 and len(set(labels)) == 2:
            if not check_invalid_conditions(X):
                se = aa.ShapExplainer(**MODEL_KWARGS)
                se.fit(X, labels=labels, **ARGS)
                assert se.shap_values is not None
                assert se.exp_value is not None

    def test_is_selected_parameter(self):
        for i in range(2):
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            list_is_selected = create_list_is_selected(n_features=n_feat)
            se = aa.ShapExplainer(**MODEL_KWARGS)
            se.fit(valid_X, labels=labels, is_selected=list_is_selected, **ARGS)
            assert se.shap_values is not None
            assert se.exp_value is not None

    @settings(max_examples=3, deadline=100000)
    @given(n_rounds=st.integers(min_value=1, max_value=3))
    def test_n_rounds_parameter(self, n_rounds):
        size, n_feat = valid_X.shape
        labels = create_labels(valid_X.shape[0])
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        if min_class_count >= 2 and n_feat > 3 and not check_invalid_conditions(valid_X):
            se = aa.ShapExplainer(**MODEL_KWARGS)
            se.fit(valid_X, labels=labels, n_rounds=n_rounds)
            assert se.shap_values is not None
            assert se.exp_value is not None

    def test_fuzzy_labeling_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS, verbose=False)
        for fuzzy_labeling in [True, False]:
            labels = create_labels(valid_X.shape[0])
            se.fit(valid_X, labels=labels, fuzzy_labeling=fuzzy_labeling, **ARGS)
            assert se.shap_values is not None
            assert se.exp_value is not None
        labels[0] = 0.5
        se.fit(valid_X, labels=labels, fuzzy_labeling=True)
        assert se.shap_values is not None
        assert se.exp_value is not None

    def test_class_index_parameter(self):
        for class_index in [0 ,1]:
            se = aa.ShapExplainer(**MODEL_KWARGS)
            labels = create_labels(valid_X.shape[0])
            se.fit(valid_X, labels=labels, class_index=class_index, **ARGS)
            assert se.shap_values is not None

    def test_n_background_data_parameter(self):
        for n_background_data in [None, 5, 45]:
            labels = create_labels(valid_X.shape[0])
            se = aa.ShapExplainer(**MODEL_KWARGS)
            se.fit(valid_X, labels=labels, n_background_data=n_background_data, **ARGS)
            assert se.shap_values is not None

    # Negative tests
    def test_invalid_X_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            se.fit(X="invalid", labels=create_labels(10))
        with pytest.raises(ValueError):
            se.fit(X=np.array([]), labels=create_labels(0))
        with pytest.raises(ValueError):
            se.fit(X=np.array([np.nan, np.nan, np.nan]).reshape(1, -1), labels=create_labels(1))

    def test_invalid_labels_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            se.fit(valid_X, labels="invalid")
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=np.array([2, -1, 3]))
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=np.array([1]))

    def test_invalid_is_selected_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS)
        labels = create_labels(valid_X.shape[0])
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=labels, is_selected="invalid")
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=labels, is_selected=np.array([True, False]))
        with pytest.raises(ValueError):
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            # Wrong dimension
            list_is_selected = create_list_is_selected(n_features=n_feat, d1=False)
            se = aa.ShapExplainer(**MODEL_KWARGS)
            se.fit(valid_X, labels=labels, is_selected=list_is_selected, **ARGS)

    def test_invalid_n_rounds_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_rounds="invalid")
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_rounds=-1)

    def test_invalid_fuzzy_labeling_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), fuzzy_labeling="invalid")
        with pytest.raises(ValueError):
            labels = valid_labels.copy()
            se = aa.ShapExplainer()
            labels[0] = 0.5
            se.fit(valid_X, labels=labels, fuzzy_labeling=False)

    def test_invalid_class_index_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), class_index="invalid")
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), class_index=-1)

    def test_invalid_n_background_data_parameter(self):
        se = aa.ShapExplainer(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_background_data="invalid")
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_background_data=-5)


class TestShapExplainerFitComplex:
    """
    Complex Test Cases for ShapExplainer.fit() method.
    Includes one positive and one negative test case, each combining multiple parameters.
    """

    # Complex positive test case
    def test_complex_valid_scenario(self):
        se = aa.ShapExplainer(**MODEL_KWARGS, verbose=False)
        size, n_feat = valid_X.shape
        labels = create_labels(size)
        list_is_selected = create_list_is_selected(n_features=n_feat)
        n_rounds = 2
        fuzzy_labeling = True
        class_index = 1
        n_background_data = 10
        # Execute with a combination of valid parameters
        se.fit(valid_X, labels=labels, is_selected=list_is_selected, n_rounds=n_rounds, fuzzy_labeling=fuzzy_labeling,
               class_index=class_index, n_background_data=n_background_data)
        # Assertions to ensure proper functionality
        assert se.shap_values is not None
        assert se.exp_value is not None
        assert len(se.shap_values) == size
        assert se.shap_values.shape[1] == n_feat
        assert isinstance(se.exp_value, float)

    # Complex negative test case
    def test_complex_invalid_scenario(self):
        se = aa.ShapExplainer(**MODEL_KWARGS, verbose=False)
        size, n_feat = valid_X.shape
        labels = create_labels(size)
        labels[0] = -1  # Invalid label
        list_is_selected = create_list_is_selected(n_features=n_feat)
        n_rounds = "invalid"  # Invalid type for n_rounds
        fuzzy_labeling = "maybe"  # Invalid type for fuzzy_labeling
        class_index = 3  # Invalid class_index for binary classification
        n_background_data = -10  # Invalid n_background_data
        # Execute with a combination of invalid parameters and expect a ValueError
        with pytest.raises(ValueError):
            se.fit(valid_X, labels=labels, is_selected=list_is_selected, n_rounds=n_rounds,
                   fuzzy_labeling=fuzzy_labeling, class_index=class_index, n_background_data=n_background_data)
