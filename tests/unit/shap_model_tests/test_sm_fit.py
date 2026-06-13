"""This script tests the ShapModel.fit() method."""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import aaanalysis as aa
import hypothesis.extra.numpy as npst

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


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


class TestShapModelFit:
    """
    Simple Positive Test Cases for ShapModel.fit() method.
    Each test focuses on one parameter.
    """

    # Positive test cases
    @settings(deadline=None, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        sm = aa.ShapModel(**MODEL_KWARGS)
        size, n_feat = X.shape
        if size > 3 and n_feat > 3:
            labels = create_labels(X.shape[0])
            if not check_invalid_conditions(X):
                sm.fit(X, labels=labels, **ARGS)
                assert sm.shap_values is not None
                assert sm.exp_value is not None

    @settings(max_examples=5, deadline=None)
    @given(labels=st.lists(st.integers(0, 1), min_size=10, max_size=20))
    def test_labels_parameter(self, labels):
        X = np.random.rand(len(labels), 10)
        size, n_feat = X.shape
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        if min_class_count >= 2 and n_feat > 3 and len(set(labels)) == 2:
            if not check_invalid_conditions(X):
                sm = aa.ShapModel(**MODEL_KWARGS)
                sm.fit(X, labels=labels, **ARGS)
                assert sm.shap_values is not None
                assert sm.exp_value is not None

    def test_is_selected_parameter(self):
        for i in range(2):
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            list_is_selected = create_list_is_selected(n_features=n_feat)
            sm = aa.ShapModel(**MODEL_KWARGS)
            sm.fit(valid_X, labels=labels, is_selected=list_is_selected, **ARGS)
            assert sm.shap_values is not None
            assert sm.exp_value is not None

    @settings(max_examples=3, deadline=None)
    @given(n_rounds=st.integers(min_value=1, max_value=3))
    def test_n_rounds_parameter(self, n_rounds):
        size, n_feat = valid_X.shape
        labels = create_labels(valid_X.shape[0])
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        if min_class_count >= 2 and n_feat > 3 and not check_invalid_conditions(valid_X):
            sm = aa.ShapModel(**MODEL_KWARGS)
            sm.fit(valid_X, labels=labels, n_rounds=n_rounds)
            assert sm.shap_values is not None
            assert sm.exp_value is not None

    def test_fuzzy_labeling_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        for fuzzy_labeling in [True, False]:
            labels = create_labels(valid_X.shape[0])
            sm.fit(valid_X, labels=labels, fuzzy_labeling=fuzzy_labeling, **ARGS)
            assert sm.shap_values is not None
            assert sm.exp_value is not None
        labels[0] = 0.5
        sm.fit(valid_X, labels=labels, fuzzy_labeling=True)
        assert sm.shap_values is not None
        assert sm.exp_value is not None

    def test_label_target_class_parameter(self):
        for label_target_class in [0 ,1]:
            sm = aa.ShapModel(**MODEL_KWARGS)
            labels = create_labels(valid_X.shape[0])
            sm.fit(valid_X, labels=labels, label_target_class=label_target_class, **ARGS)
            assert sm.shap_values is not None

    def test_n_background_data_parameter(self):
        for n_background_data in [None, 5, 45]:
            labels = create_labels(valid_X.shape[0])
            sm = aa.ShapModel(**MODEL_KWARGS)
            sm.fit(valid_X, labels=labels, n_background_data=n_background_data, **ARGS)
            assert sm.shap_values is not None

    # Negative tests
    def test_invalid_X_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            sm.fit(X="invalid", labels=create_labels(10))
        with pytest.raises(ValueError):
            sm.fit(X=np.array([]), labels=create_labels(0))
        with pytest.raises(ValueError):
            sm.fit(X=np.array([np.nan, np.nan, np.nan]).reshape(1, -1), labels=create_labels(1))

    def test_invalid_labels_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels="invalid")
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=np.array([2, -1, 3]))
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=np.array([1]))

    def test_invalid_is_selected_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS)
        labels = create_labels(valid_X.shape[0])
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=labels, is_selected="invalid")
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=labels, is_selected=np.array([True, False]))
        with pytest.raises(ValueError):
            size, n_feat = valid_X.shape
            labels = create_labels(valid_X.shape[0])
            # Wrong dimension
            list_is_selected = create_list_is_selected(n_features=n_feat, d1=False)
            sm = aa.ShapModel(**MODEL_KWARGS)
            sm.fit(valid_X, labels=labels, is_selected=list_is_selected, **ARGS)

    def test_invalid_n_rounds_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_rounds="invalid")
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_rounds=-1)

    def test_invalid_fuzzy_labeling_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), fuzzy_labeling="invalid")
        with pytest.raises(ValueError):
            labels = valid_labels.copy()
            sm = aa.ShapModel()
            labels[0] = 0.5
            sm.fit(valid_X, labels=labels, fuzzy_labeling=False)

    def test_invalid_label_target_class_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), label_target_class="invalid")
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), label_target_class=-1)

    def test_invalid_n_background_data_parameter(self):
        sm = aa.ShapModel(**MODEL_KWARGS)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_background_data="invalid")
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=create_labels(valid_X.shape[0]), n_background_data=-5)


class TestShapModelFitComplex:
    """
    Complex Test Cases for ShapModel.fit() method.
    Includes one positive and one negative test case, each combining multiple parameters.
    """

    # Complex positive test case
    def test_complex_valid_scenario(self):
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        size, n_feat = valid_X.shape
        labels = create_labels(size)
        list_is_selected = create_list_is_selected(n_features=n_feat)
        n_rounds = 2
        fuzzy_labeling = True
        label_target_class = 1
        n_background_data = 10
        # Execute with a combination of valid parameters
        sm.fit(valid_X, labels=labels, is_selected=list_is_selected, n_rounds=n_rounds, fuzzy_labeling=fuzzy_labeling,
               label_target_class=label_target_class, n_background_data=n_background_data)
        # Assertions to ensure proper functionality
        assert sm.shap_values is not None
        assert sm.exp_value is not None
        assert len(sm.shap_values) == size
        assert sm.shap_values.shape[1] == n_feat
        assert isinstance(sm.exp_value, float)

    # Complex negative test case
    def test_complex_invalid_scenario(self):
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        size, n_feat = valid_X.shape
        labels = create_labels(size)
        labels[0] = -1  # Invalid label
        list_is_selected = create_list_is_selected(n_features=n_feat)
        n_rounds = "invalid"  # Invalid type for n_rounds
        fuzzy_labeling = "maybe"  # Invalid type for fuzzy_labeling
        label_target_class = 3  # Invalid label_target_class for binary classification
        n_background_data = -10  # Invalid n_background_data
        # Execute with a combination of invalid parameters and expect a ValueError
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=labels, is_selected=list_is_selected, n_rounds=n_rounds,
                   fuzzy_labeling=fuzzy_labeling, label_target_class=label_target_class, n_background_data=n_background_data)


class TestShapModelFitExplainers:
    """Exercise non-default SHAP explainer classes (Kernel / Linear) on small data."""

    @staticmethod
    def _small_data(n_samples=14, n_feat=4, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.random((n_samples, n_feat))
        labels = np.array([1, 0] * (n_samples // 2))
        return X, labels

    def test_kernel_explainer_with_background_data(self):
        """KernelExplainer + n_background_data triggers the kmeans background path."""
        import shap
        from sklearn.ensemble import RandomForestClassifier
        X, labels = self._small_data()
        sm = aa.ShapModel(explainer_class=shap.KernelExplainer,
                          list_model_classes=[RandomForestClassifier], verbose=False)
        sm.fit(X, labels=labels, n_rounds=1, n_background_data=3)
        assert sm.shap_values.shape == X.shape

    def test_kernel_explainer_without_background_data(self):
        """KernelExplainer without n_background_data uses the full matrix as background."""
        import shap
        from sklearn.ensemble import RandomForestClassifier
        X, labels = self._small_data(seed=1)
        sm = aa.ShapModel(explainer_class=shap.KernelExplainer,
                          list_model_classes=[RandomForestClassifier], verbose=False)
        sm.fit(X, labels=labels, n_rounds=1)
        assert sm.shap_values.shape == X.shape

    def test_linear_explainer(self):
        """LinearExplainer routes through the model-only explainer branch."""
        import shap
        from sklearn.linear_model import LogisticRegression
        X, labels = self._small_data(seed=2)
        sm = aa.ShapModel(explainer_class=shap.LinearExplainer,
                          list_model_classes=[LogisticRegression], verbose=False)
        sm.fit(X, labels=labels, n_rounds=1)
        assert sm.shap_values.shape == X.shape


class TestShapModelFitFuzzyLabels:
    """Entry-keyed soft labels (``fuzzy_labels``) aligned to ``X`` via ``df_seq``."""

    # Positive tests
    def test_fuzzy_labels_valid(self):
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False, random_state=0)
        sm.fit(valid_X, labels=valid_labels, df_seq=df_seq, fuzzy_labels={entry: 0.6}, **ARGS)
        assert sm.shap_values is not None
        assert sm.exp_value is not None

    @settings(max_examples=5, deadline=None)
    @given(score=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    def test_fuzzy_labels_score_range(self, score):
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        sm.fit(valid_X, labels=valid_labels, df_seq=df_seq, fuzzy_labels={entry: score}, **ARGS)
        assert sm.shap_values is not None

    def test_fuzzy_labels_float_labels_base(self):
        # Base labels already float; fuzzy_labels overrides one entry's value
        entry = df_seq["entry"].iloc[1]
        y = [float(v) for v in valid_labels]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        sm.fit(valid_X, labels=y, df_seq=df_seq, fuzzy_labels={entry: 0.4}, **ARGS)
        assert sm.shap_values is not None

    def test_fuzzy_labels_numpy_value(self):
        # numpy float values in the dict are accepted (not just python float)
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        sm.fit(valid_X, labels=valid_labels, df_seq=df_seq, fuzzy_labels={entry: np.float64(0.6)}, **ARGS)
        assert sm.shap_values is not None

    def test_df_seq_only_consumed_with_fuzzy_labels(self):
        # df_seq passed without fuzzy_labels is harmless (binary path unchanged)
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        sm.fit(valid_X, labels=valid_labels, df_seq=df_seq, **ARGS)
        assert sm.shap_values is not None

    # Golden / regression: KPI #129 — keyed path == manual array mutation
    def test_fuzzy_labels_matches_manual_mutation(self):
        entry = df_seq["entry"].iloc[0]
        i = list(df_seq["entry"]).index(entry)
        y = [float(v) for v in valid_labels]
        y[i] = 0.6
        sm_manual = aa.ShapModel(**MODEL_KWARGS, verbose=False, random_state=42)
        sm_manual.fit(valid_X, labels=y, fuzzy_labeling=True, **ARGS)
        sm_keyed = aa.ShapModel(**MODEL_KWARGS, verbose=False, random_state=42)
        sm_keyed.fit(valid_X, labels=valid_labels, df_seq=df_seq, fuzzy_labels={entry: 0.6}, **ARGS)
        assert np.array_equal(sm_manual.shap_values, sm_keyed.shap_values)
        assert sm_manual.exp_value == sm_keyed.exp_value

    # Negative tests
    def test_fuzzy_labels_requires_df_seq(self):
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=valid_labels, fuzzy_labels={entry: 0.6}, **ARGS)

    def test_fuzzy_labels_invalid_key(self):
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=valid_labels, df_seq=df_seq, fuzzy_labels={"NOT_AN_ENTRY": 0.6}, **ARGS)

    def test_fuzzy_labels_invalid_value(self):
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        for bad in [1.5, -0.1, "x"]:
            with pytest.raises(ValueError):
                sm.fit(valid_X, labels=valid_labels, df_seq=df_seq, fuzzy_labels={entry: bad}, **ARGS)

    def test_fuzzy_labels_df_seq_length_mismatch(self):
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=valid_labels, df_seq=df_seq.head(3), fuzzy_labels={entry: 0.6}, **ARGS)

    def test_fuzzy_labels_df_seq_non_unique_entries(self):
        entry = df_seq["entry"].iloc[0]
        df_dup = df_seq.copy()
        df_dup["entry"] = entry  # collapse to a single repeated entry
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=valid_labels, df_seq=df_dup, fuzzy_labels={entry: 0.6}, **ARGS)

    def test_fuzzy_labels_invalid_df_seq_type(self):
        entry = df_seq["entry"].iloc[0]
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=valid_labels, df_seq="not_a_df", fuzzy_labels={entry: 0.6}, **ARGS)

    def test_fuzzy_labels_df_seq_missing_entry_column(self):
        entry = df_seq["entry"].iloc[0]
        df_bad = df_seq.rename(columns={"entry": "acc"})
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError):
            sm.fit(valid_X, labels=valid_labels, df_seq=df_bad, fuzzy_labels={entry: 0.6}, **ARGS)
