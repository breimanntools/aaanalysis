"""This script tests the ShapExplainer.add_sample_mean_dif method."""
import pandas as pd
import numpy as np
import pytest
import random
import aaanalysis as aa
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst

# Set default deadline from 200 to 400
aa.options["verbose"] = False
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


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


def create_df_feat(drop=True):
    df_feat = aa.load_features(name="DOM_GSEC").head(50)
    if drop:
        df_feat = df_feat[[x for x in list(df_feat) if "FEAT_IMPACT" not in x]]
    return df_feat


def create_labels(size):
    labels = np.array([1, 1, 0, 0] + list(np.random.choice([1, 0], size=size-4)))
    return labels

# Create valid X for testing
df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
_df_feat = aa.load_features().head(50)
valid_labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=_df_feat["feature"], df_parts=df_parts)

N_ROUNDS = 2
ARGS = dict(n_rounds=N_ROUNDS)


class TestAddSampleMeanDif:
    """ Test simple positive cases for each parameter of the add_sample_mean_dif method. """

    # Positive test cases
    @settings(deadline=10000, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        se = aa.ShapExplainer()
        size, n_feat = X.shape
        if size > 3 and n_feat > 3:
            labels = create_labels(X.shape[0])
            if not check_invalid_conditions(X):
                df_feat = create_df_feat()
                df_feat = se.add_sample_mean_dif(X, labels=labels, df_feat=df_feat)
                assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=5, deadline=1500)
    @given(labels=st.lists(st.integers(0, 1), min_size=10, max_size=20))
    def test_labels_parameter(self, labels):
        X = np.random.rand(len(labels), 10)
        size, n_feat = X.shape
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        df_feat = create_df_feat()
        if min_class_count >= 2 and n_feat > 3 and len(set(labels)) == 2:
            if not check_invalid_conditions(X):
                se = aa.ShapExplainer()
                df_feat_updated = se.add_sample_mean_dif(X, labels=labels, df_feat=df_feat)
                assert isinstance(df_feat_updated, pd.DataFrame)

    def test_label_ref_valid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        unique_labels = list(set(labels))
        for label_ref in unique_labels:
            df_feat_updated = se.add_sample_mean_dif(valid_X, labels=labels,
                                                     label_ref=label_ref, df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)

    def test_drop_valid(self):
        for drop in [True, False]:
            se = aa.ShapExplainer()
            df_feat = create_df_feat(drop=not drop)
            labels = create_labels(len(valid_X))
            df_feat_updated = se.add_sample_mean_dif(valid_X, labels=labels, drop=drop, df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)

    def test_sample_positions_valid(self):
        se = aa.ShapExplainer(verbose=False)
        sample_positions = random.sample(range(len(df_seq)), 3)
        labels = create_labels(valid_X.shape[0])
        for pos in sample_positions:
            se = aa.ShapExplainer(verbose=False)
            df_feat = create_df_feat()
            df_feat_updated = se.add_sample_mean_dif(valid_X, labels=labels,
                                                     sample_positions=pos,
                                                     df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)
        df_feat = create_df_feat()
        df_feat = se.add_sample_mean_dif(valid_X, labels=labels,
                                                 sample_positions=list(range(0, len(df_seq)-1)), df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = create_df_feat()
        df_feat = se.add_sample_mean_dif(valid_X, labels=labels,
                                                 sample_positions=sample_positions, df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)

    def test_names_valid(self):
        names = [f"P{i}" for i in range(len(df_seq))]
        se = aa.ShapExplainer(verbose=False)
        df_feat = create_df_feat()
        labels = create_labels(valid_X.shape[0])
        df_feat = se.add_sample_mean_dif(valid_X, labels=labels, df_feat=df_feat, names=names)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = se.add_sample_mean_dif(valid_X, labels=labels, df_feat=df_feat, names="test", sample_positions=0)
        assert isinstance(df_feat, pd.DataFrame)

    def test_group_average_valid(self):
        se = aa.ShapExplainer(verbose=False)
        labels = create_labels(len(valid_X))
        for group_average in [True, False]:
            df_feat = create_df_feat()
            df_feat_updated = se.add_sample_mean_dif(valid_X, labels=labels, group_average=group_average,
                                                     df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat_updated = se.add_sample_mean_dif(valid_X, labels=labels, group_average=group_average,
                                                 df_feat=df_feat, sample_positions=[1, 2, 3])
        assert isinstance(df_feat_updated, pd.DataFrame)

    # Negative tests
    def test_X_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(10)
        # Test with various invalid X inputs
        invalid_X_inputs = [np.array([]), np.array([np.nan, np.nan]), np.array([np.inf, -np.inf]), 'invalid']
        for X in invalid_X_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(X, labels=labels, df_feat=df_feat)

    def test_labels_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        # Test with various invalid labels inputs
        invalid_labels_inputs = [np.array([]), np.array(['a', 'b']), 123, 'invalid']
        for labels in invalid_labels_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(valid_X, labels=labels, df_feat=df_feat)

    def test_label_ref_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid label_ref inputs
        invalid_label_ref_inputs = ['invalid', np.nan, 2.5, []]
        for label_ref in invalid_label_ref_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(valid_X, labels=labels, label_ref=label_ref, df_feat=df_feat)

    def test_drop_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid drop inputs
        invalid_drop_inputs = ['invalid', 123, None]
        for drop in invalid_drop_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(valid_X, labels=labels, drop=drop, df_feat=df_feat)

    def test_sample_positions_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid sample_positions inputs
        invalid_sample_positions_inputs = ['invalid', -1, 100, [100, -1]]
        for sample_positions in invalid_sample_positions_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(valid_X, labels=labels, sample_positions=sample_positions, df_feat=df_feat)

    def test_names_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid names inputs
        invalid_names_inputs = [123, False, [123, 'valid']]
        for names in invalid_names_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(valid_X, labels=labels, names=names, df_feat=df_feat)

    def test_group_average_invalid(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid group_average inputs
        invalid_group_average_inputs = ['invalid', 123, None]
        for group_average in invalid_group_average_inputs:
            with pytest.raises(ValueError):
                se.add_sample_mean_dif(valid_X, labels=labels, group_average=group_average, df_feat=df_feat)

class TestAddSampleMeanDifComplex:
    """Complex test cases for add_sample_mean_dif method."""

    def test_complex_positive(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        label_ref = 0
        drop = True
        sample_positions = [0, 1, 2]
        names = ["Sample1", "Sample2", "Sample3"]
        df_feat_updated = se.add_sample_mean_dif(valid_X, labels=labels, label_ref=label_ref,
                                                 df_feat=df_feat, drop=drop,
                                                 sample_positions=sample_positions, names=names,
                                                 group_average=False)
        assert isinstance(df_feat_updated, pd.DataFrame)

    def test_complex_negative(self):
        se = aa.ShapExplainer()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))  # Assume this generates a valid label array
        label_ref = "invalid_ref"  # Invalid label reference
        drop = "invalid_drop"  # Invalid drop value
        sample_positions = "invalid_position"  # Invalid sample positions
        names = 123  # Invalid names
        group_average = "invalid_average"  # Invalid group average
        with pytest.raises(ValueError):
            se.add_sample_mean_dif(valid_X, labels=labels, label_ref=label_ref,
                                   df_feat=df_feat, drop=drop,
                                   sample_positions=sample_positions, names=names,
                                   group_average=group_average)