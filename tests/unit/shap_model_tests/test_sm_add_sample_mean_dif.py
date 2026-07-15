"""This script tests the ShapModel.add_sample_mean_dif method."""
import pandas as pd
import numpy as np
import pytest
import random
import aaanalysis as aa
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst

# Set default deadline from 200 to 400
aa.options["verbose"] = False
settings.register_profile("ci", deadline=None)
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
    @settings(deadline=None, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2),
                         elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_X_parameter(self, X):
        sm = aa.ShapModel()
        size, n_feat = X.shape
        if size > 3 and n_feat > 3:
            labels = create_labels(X.shape[0])
            if not check_invalid_conditions(X):
                df_feat = create_df_feat()
                df_feat = sm.add_sample_mean_dif(X, labels=labels, df_feat=df_feat)
                assert isinstance(df_feat, pd.DataFrame)

    @settings(max_examples=5, deadline=None)
    @given(labels=st.lists(st.integers(0, 1), min_size=10, max_size=20))
    def test_labels_parameter(self, labels):
        X = np.random.rand(len(labels), 10)
        size, n_feat = X.shape
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)
        df_feat = create_df_feat()
        if min_class_count >= 2 and n_feat > 3 and len(set(labels)) == 2:
            if not check_invalid_conditions(X):
                sm = aa.ShapModel()
                df_feat_updated = sm.add_sample_mean_dif(X, labels=labels, df_feat=df_feat)
                assert isinstance(df_feat_updated, pd.DataFrame)

    def test_label_ref_valid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        unique_labels = list(set(labels))
        for label_ref in unique_labels:
            df_feat_updated = sm.add_sample_mean_dif(valid_X, labels=labels,
                                                     label_ref=label_ref, df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)

    def test_drop_valid(self):
        for drop in [True, False]:
            sm = aa.ShapModel()
            df_feat = create_df_feat(drop=not drop)
            labels = create_labels(len(valid_X))
            df_feat_updated = sm.add_sample_mean_dif(valid_X, labels=labels, drop=drop, df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)

    def test_sample_positions_valid(self):
        sm = aa.ShapModel(verbose=False)
        sample_positions = random.sample(range(len(df_seq)), 3)
        labels = create_labels(valid_X.shape[0])
        for pos in sample_positions:
            sm = aa.ShapModel(verbose=False)
            df_feat = create_df_feat()
            df_feat_updated = sm.add_sample_mean_dif(valid_X, labels=labels,
                                                     samples=pos,
                                                     df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)
        df_feat = create_df_feat()
        df_feat = sm.add_sample_mean_dif(valid_X, labels=labels,
                                                 samples=list(range(0, len(df_seq)-1)), df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = create_df_feat()
        df_feat = sm.add_sample_mean_dif(valid_X, labels=labels,
                                                 samples=sample_positions, df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)

    def test_names_valid(self):
        names = [f"P{i}" for i in range(len(df_seq))]
        sm = aa.ShapModel(verbose=False)
        df_feat = create_df_feat()
        labels = create_labels(valid_X.shape[0])
        df_feat = sm.add_sample_mean_dif(valid_X, labels=labels, df_feat=df_feat, names=names)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = sm.add_sample_mean_dif(valid_X, labels=labels, df_feat=df_feat, names="test", samples=0)
        assert isinstance(df_feat, pd.DataFrame)

    def test_group_average_valid(self):
        sm = aa.ShapModel(verbose=False)
        labels = create_labels(len(valid_X))
        for group_average in [True, False]:
            df_feat = create_df_feat()
            df_feat_updated = sm.add_sample_mean_dif(valid_X, labels=labels, group_average=group_average,
                                                     df_feat=df_feat)
            assert isinstance(df_feat_updated, pd.DataFrame)
        sm.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat_updated = sm.add_sample_mean_dif(valid_X, labels=labels, group_average=group_average,
                                                 df_feat=df_feat, samples=[1, 2, 3])
        assert isinstance(df_feat_updated, pd.DataFrame)

    # Negative tests
    def test_X_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(10)
        # Test with various invalid X inputs
        invalid_X_inputs = [np.array([]), np.array([np.nan, np.nan]), np.array([np.inf, -np.inf]), 'invalid']
        for X in invalid_X_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(X, labels=labels, df_feat=df_feat)

    def test_labels_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        # Test with various invalid labels inputs
        invalid_labels_inputs = [np.array([]), np.array(['a', 'b']), 123, 'invalid']
        for labels in invalid_labels_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(valid_X, labels=labels, df_feat=df_feat)

    def test_label_ref_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid label_ref inputs
        invalid_label_ref_inputs = ['invalid', np.nan, 2.5, []]
        for label_ref in invalid_label_ref_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(valid_X, labels=labels, label_ref=label_ref, df_feat=df_feat)

    def test_drop_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid drop inputs
        invalid_drop_inputs = ['invalid', 123, None]
        for drop in invalid_drop_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(valid_X, labels=labels, drop=drop, df_feat=df_feat)

    def test_sample_positions_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid sample_positions inputs
        invalid_sample_positions_inputs = ['invalid', -1, 100, [100, -1]]
        for sample_positions in invalid_sample_positions_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(valid_X, labels=labels, samples=sample_positions, df_feat=df_feat)

    def test_names_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid names inputs
        invalid_names_inputs = [123, False, [123, 'valid']]
        for names in invalid_names_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(valid_X, labels=labels, names=names, df_feat=df_feat)

    def test_group_average_invalid(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        # Test with invalid group_average inputs
        invalid_group_average_inputs = ['invalid', 123, None]
        for group_average in invalid_group_average_inputs:
            with pytest.raises(ValueError):
                sm.add_sample_mean_dif(valid_X, labels=labels, group_average=group_average, df_feat=df_feat)

class TestAddSampleMeanDifComplex:
    """Complex test cases for add_sample_mean_dif method."""

    def test_complex_positive(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))
        label_ref = 0
        drop = True
        sample_positions = [0, 1, 2]
        names = ["Sample1", "Sample2", "Sample3"]
        df_feat_updated = sm.add_sample_mean_dif(valid_X, labels=labels, label_ref=label_ref,
                                                 df_feat=df_feat, drop=drop,
                                                 samples=sample_positions, names=names,
                                                 group_average=False)
        assert isinstance(df_feat_updated, pd.DataFrame)

    def test_complex_negative(self):
        sm = aa.ShapModel()
        df_feat = create_df_feat()
        labels = create_labels(len(valid_X))  # Assume this generates a valid label array
        label_ref = "invalid_ref"  # Invalid label reference
        drop = "invalid_drop"  # Invalid drop value
        sample_positions = "invalid_position"  # Invalid sample positions
        names = 123  # Invalid names
        group_average = "invalid_average"  # Invalid group average
        with pytest.raises(ValueError):
            sm.add_sample_mean_dif(valid_X, labels=labels, label_ref=label_ref,
                                   df_feat=df_feat, drop=drop,
                                   samples=sample_positions, names=names,
                                   group_average=group_average)


class TestAddSampleMeanDifDfSeq:
    """Accession-based sample selection: entry-name ``sample_positions`` resolved via ``df_seq``."""

    # Positive tests
    def test_sample_positions_entry_name(self):
        entry = df_seq["entry"].iloc[0]
        df_feat = aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                                   df_seq=df_seq, samples=entry)
        assert f"mean_dif_{entry}" in df_feat.columns

    def test_entry_name_matches_int_position(self):
        entry = df_seq["entry"].iloc[2]
        i = list(df_seq["entry"]).index(entry)
        df_a = aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                                df_seq=df_seq, samples=entry)
        df_b = aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                                samples=i, names=entry)
        assert df_a[f"mean_dif_{entry}"].equals(df_b[f"mean_dif_{entry}"])

    def test_sample_positions_entry_list(self):
        entries = df_seq["entry"].iloc[:3].to_list()
        df_feat = aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                                   df_seq=df_seq, samples=entries)
        for e in entries:
            assert f"mean_dif_{e}" in df_feat.columns

    # Negative tests
    def test_entry_name_requires_df_seq(self):
        entry = df_seq["entry"].iloc[0]
        with pytest.raises(ValueError):
            aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                             samples=entry)

    def test_entry_not_in_df_seq(self):
        with pytest.raises(ValueError):
            aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                             df_seq=df_seq, samples="NOT_AN_ENTRY")


class TestAddSampleMeanDifXRef:
    """External reference matrix via ``X_ref`` (explain each sample against a separate population)."""

    def test_x_ref_matches_vstack_reference(self):
        # KPI: X_ref=X_others reproduces the old "vstack samples + Others, label_ref=0" result
        # byte-for-byte, without the manual concatenation / synthetic labels.
        X_others = valid_X[15:20]
        positions = [0, 2, 4]
        names = ["A", "B", "C"]
        entries = [df_seq["entry"].iloc[p] for p in positions]
        _Xc = np.vstack([valid_X[positions], X_others])
        _lc = np.array([1] * len(positions) + [0] * len(X_others))
        df_old = aa.ShapModel.add_sample_mean_dif(_Xc, labels=_lc, label_ref=0, df_feat=create_df_feat(),
                                                  samples=list(range(len(positions))), names=names)
        df_new = aa.ShapModel.add_sample_mean_dif(valid_X, df_feat=create_df_feat(), X_ref=X_others,
                                                  samples=entries, names=names, df_seq=df_seq)
        for name in names:
            assert np.array_equal(df_old[f"mean_dif_{name}"].to_numpy(),
                                  df_new[f"mean_dif_{name}"].to_numpy())

    def test_x_ref_no_labels_needed(self):
        entry = df_seq["entry"].iloc[0]
        df_feat = aa.ShapModel.add_sample_mean_dif(valid_X, df_feat=create_df_feat(), X_ref=valid_X[15:20],
                                                   samples=entry, df_seq=df_seq)
        assert f"mean_dif_{entry}" in df_feat.columns

    def test_x_ref_positional_samples(self):
        df_feat = aa.ShapModel.add_sample_mean_dif(valid_X, df_feat=create_df_feat(), X_ref=valid_X[15:20],
                                                   samples=[0, 1], names=["A", "B"])
        assert "mean_dif_A" in df_feat.columns and "mean_dif_B" in df_feat.columns

    def test_x_ref_and_labels_mutually_exclusive_raises(self):
        with pytest.raises(ValueError):
            aa.ShapModel.add_sample_mean_dif(valid_X, labels=valid_labels, df_feat=create_df_feat(),
                                             X_ref=valid_X[15:20], samples=0, names="A")

    def test_x_ref_feature_mismatch_raises(self):
        with pytest.raises(ValueError):
            aa.ShapModel.add_sample_mean_dif(valid_X, df_feat=create_df_feat(), X_ref=valid_X[15:20, :10],
                                             samples=0, names="A")

    def test_neither_x_ref_nor_labels_raises(self):
        with pytest.raises(ValueError):
            aa.ShapModel.add_sample_mean_dif(valid_X, df_feat=create_df_feat(), samples=0, names="A")