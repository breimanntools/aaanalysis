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
import warnings

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helper functions
def create_labels(size):
    labels = np.array([1, 2] + list(np.random.choice([1, 2], size=size-2)))
    return labels


def check_invalid_conditions(X, labels, min_samples=3, check_unique=True):
    n_samples, n_features = X.shape
    n_unique_labels = len(set(labels))
    n_unique_samples = len(set(map(tuple, X)))
    conditions = [
        (np.any(np.isinf(X)) or np.any(np.isnan(X)), "X contains NaN or Inf"),
        (n_samples < min_samples, f"n_samples={n_samples} should be >= {min_samples}"),
        (n_features < 2, f"n_features={n_features} should be >= 2"),
        (len(labels) != n_samples, "Length of labels should match n_samples."),
        (n_unique_labels < 2, f"n_unique_labels={n_unique_samples} should be >= 2")
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
    @settings(deadline=None, max_examples=100)
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
            is_invalid = check_invalid_conditions(X=X, labels=labels)
            if not is_invalid and valid_labels:
                df_pu = dpul.fit(X, labels).df_pu_
                assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=None, max_examples=100)
    @given(labels=npst.arrays(dtype=np.int32, shape=(100,)))
    def test_labels_parameter(self, labels):
        """Test the 'labels' parameter with valid inputs."""
        X = np.random.rand(100, 5)  # Assuming 100 samples, 5 features
        dpul = aa.dPULearn()
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if len(set(labels)) < 2 or 1 not in labels or 2 not in labels:
            valid_labels = False
        if not is_invalid and valid_labels:
            df_pu = dpul.fit(X, labels).df_pu_
            assert isinstance(df_pu, pd.DataFrame)

    def test_labels(self):
        """Test the 'labels' parameter with valid inputs."""
        X = np.random.rand(100, 5)  # Assuming 100 samples, 5 features
        labels = np.asarray([1] * 50 + [2] * 50)
        dpul = aa.dPULearn()
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if not is_invalid and valid_labels:
            df_pu = dpul.fit(X, labels).df_pu_
            assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=None, max_examples=100)
    @given(n_unl_to_neg=some.integers(min_value=1))
    def test_n_unl_to_neg_parameter(self, n_unl_to_neg):
        """Test the 'n_unl_to_neg' parameter with valid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        n_unl = sum([x == 2 for x in labels])
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if not is_invalid and valid_labels and n_unl > n_unl_to_neg :
            df_pu = dpul.fit(X, labels, n_unl_to_neg=n_unl_to_neg).df_pu_
            assert isinstance(df_pu, pd.DataFrame)
    
    @settings(deadline=None, max_examples=4)
    @given(metric=some.none() | some.sampled_from(["euclidean", "manhattan", "cosine", None]))
    def test_metric_parameter(self, metric):
        """Test the 'metric' parameter."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        if not is_invalid:
            for n in [5, 25, 39]:
                df_pu = dpul.fit(X, labels, metric=metric, n_unl_to_neg=n).df_pu_
                assert isinstance(df_pu, pd.DataFrame)

    @settings(deadline=None, max_examples=100)
    @given(n_components=some.one_of(some.floats(min_value=0.1, max_value=1.0), some.integers(min_value=1)))
    def test_n_components_parameter(self, n_components):
        """Test the 'n_components' parameter with valid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        n_samples, n_features = X.shape
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        for i in [5, 26, 39]:
            if not is_invalid:
                if n_components < min(n_features, n_samples) and n_components not in [0.0, 1.0]:
                    df_pu = dpul.fit(X, labels, n_components=n_components, n_unl_to_neg=i).df_pu_
                    assert isinstance(df_pu, pd.DataFrame)

    # Negative tests
    @settings(deadline=None, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(10,)))
    def test_X_invalid_shape(self, X):
        """Test the 'X' parameter with invalid shape."""
        dpul = aa.dPULearn()
        labels = create_labels(X.size)
        with pytest.raises(ValueError):
            dpul.fit(X, labels)

    @settings(deadline=None, max_examples=10)
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

    @settings(deadline=None, max_examples=100)
    @given(n_unl_to_neg=some.integers(max_value=0))
    def test_n_unl_to_neg_invalid(self, n_unl_to_neg):
        """Test the 'n_unl_to_neg' parameter with invalid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        if is_invalid:
            with pytest.raises(ValueError):
                dpul.fit(X, labels, n_unl_to_neg=n_unl_to_neg)

    def test_n_neg_none_message(self):
        """Omitting both count args requires exactly one of n_neg / n_unl_to_neg."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        with pytest.raises(ValueError, match="exactly one"):
            dpul.fit(X, labels)

    def test_labels_wrong_encoding_message(self):
        """Standard {0, 1} labels under the default 1/2 encoding point the user to label_unl=0."""
        X = np.random.rand(100, 5)
        labels = np.array([0, 1] * 50)
        dpul = aa.dPULearn()
        with pytest.raises(ValueError, match="label_unl=0"):
            dpul.fit(X, labels, n_neg=5)
    
    @settings(deadline=None, max_examples=10)
    @given(metric=some.just("invalid_metric"))
    def test_invalid_metric(self, metric):
        """Test with an invalid 'metric' value."""
        valid_metrics = ["euclidean", "manhattan", "cosine", None]
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        if not is_invalid and not metric in valid_metrics:
            with pytest.raises(ValueError):
               dpul.fit(X, labels, metric=metric)

    
    @settings(deadline=None, max_examples=10)
    @given(n_components=some.one_of(some.floats(max_value=0.0), some.integers(max_value=0)))
    def test_n_components_invalid(self, n_components):
        """Test the 'n_components' parameter with invalid inputs."""
        X = np.random.rand(100, 5)
        labels = create_labels(100)
        dpul = aa.dPULearn()
        is_invalid = check_invalid_conditions(X=X, labels=labels)
        valid_labels = sum([x for x in set(labels) if x in [1, 2]]) == 2
        if is_invalid and valid_labels:
            with pytest.raises(ValueError):
                dpul.fit(X, labels, n_components=n_components)


class TestdPULearnFitLabelEncoding:
    """Test the label-marker ergonomics (label_pos/label_unl/label_neg) + the n_neg / n_unl_to_neg count options."""

    @staticmethod
    def _det_data(n=60, n_features=5, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.random((n, n_features))
        labels_pu = np.array([1, 2] * (n // 2))      # 1 = positive, 2 = unlabeled
        return X, labels_pu

    # Positive tests (exercise each new parameter by name)
    def test_n_neg_parameter(self):
        """'n_neg' is the primary count name."""
        X, labels = self._det_data()
        dpul = aa.dPULearn(random_state=42)
        df_pu = dpul.fit(X, labels, n_neg=5).df_pu_
        assert isinstance(df_pu, pd.DataFrame)
        assert np.sum(dpul.labels_ == 0) == 5

    def test_label_unl_zero_one_encoding(self):
        """Standard {0, 1} labels work directly when label_unl=0 (0 = unlabeled, 1 = positive)."""
        X, labels_pu = self._det_data()
        labels_01 = np.where(labels_pu == 2, 0, 1)   # 2 -> 0 (unlabeled), 1 stays positive
        dpul = aa.dPULearn(random_state=42)
        df_pu = dpul.fit(X, labels_01, label_unl=0, n_neg=5).df_pu_
        assert isinstance(df_pu, pd.DataFrame)
        assert np.sum(dpul.labels_ == 0) == 5

    def test_label_pos_custom_encoding(self):
        """A fully custom positive/unlabeled marker pair is normalized internally."""
        X, labels_pu = self._det_data()
        # Encode positives as 7, unlabeled as 9
        labels_custom = np.where(labels_pu == 1, 7, 9)
        dpul = aa.dPULearn(random_state=42)
        df_pu = dpul.fit(X, labels_custom, label_pos=7, label_unl=9, n_neg=5).df_pu_
        assert isinstance(df_pu, pd.DataFrame)
        assert np.sum(dpul.labels_ == 0) == 5

    def test_equivalence_1_0_vs_1_2(self):
        """1/0 + n_neg yields byte-identical labels_/df_pu_ to the 1/2 + n_neg path."""
        X, labels_pu = self._det_data()
        labels_01 = np.where(labels_pu == 2, 0, 1)
        dpul_pu = aa.dPULearn(random_state=42).fit(X, labels_pu, n_neg=5)
        dpul_01 = aa.dPULearn(random_state=42).fit(X, labels_01, label_unl=0, n_neg=5)
        assert np.array_equal(dpul_pu.labels_, dpul_01.labels_)
        pd.testing.assert_frame_equal(dpul_pu.df_pu_, dpul_01.df_pu_)

    def test_n_unl_to_neg_equals_n_neg_without_pre_neg(self):
        """Without pre-labeled negatives, n_unl_to_neg (direct) == n_neg (total)."""
        X, labels_pu = self._det_data()
        dpul_total = aa.dPULearn(random_state=42).fit(X, labels_pu, n_neg=5)
        dpul_direct = aa.dPULearn(random_state=42).fit(X, labels_pu, n_unl_to_neg=5)
        assert np.array_equal(dpul_total.labels_, dpul_direct.labels_)
        pd.testing.assert_frame_equal(dpul_total.df_pu_, dpul_direct.df_pu_)

    # Count options: exactly one of n_neg / n_unl_to_neg
    def test_both_counts_raises(self):
        """Passing both 'n_neg' and 'n_unl_to_neg' raises (exactly one allowed)."""
        X, labels_pu = self._det_data()
        dpul = aa.dPULearn(random_state=42)
        with pytest.raises(ValueError, match="exactly one"):
            dpul.fit(X, labels_pu, n_neg=5, n_unl_to_neg=7)

    def test_neither_count_raises(self):
        """Passing neither 'n_neg' nor 'n_unl_to_neg' raises (exactly one required)."""
        X, labels_pu = self._det_data()
        dpul = aa.dPULearn(random_state=42)
        with pytest.raises(ValueError, match="exactly one"):
            dpul.fit(X, labels_pu)

    # Pre-labeled negatives (label_neg) + total-n_neg semantics
    @staticmethod
    def _data_with_pre_neg(n=60, n_pre_neg=4, seed=42):
        X, labels_pu = TestdPULearnFitLabelEncoding._det_data(n=n, seed=seed)
        labels = labels_pu.copy()
        unl_idx = np.where(labels == 2)[0][:n_pre_neg]  # mark some unlabeled as pre-labeled negatives
        labels[unl_idx] = 0
        return X, labels, unl_idx

    def test_label_neg_pre_labeled_negatives(self):
        """n_neg is the TOTAL wanted; pre-labeled negatives are kept and the rest identified."""
        X, labels, pre_idx = self._data_with_pre_neg(n=60, n_pre_neg=4)
        dpul = aa.dPULearn(random_state=42)
        dpul.fit(X, labels, label_neg=0, n_neg=10)  # 4 pre-labeled + 6 newly identified
        assert np.sum(dpul.labels_ == 0) == 10
        # the pre-labeled negatives are preserved (never re-selected)
        assert all(dpul.labels_[i] == 0 for i in pre_idx)

    def test_n_neg_not_exceeding_pre_labeled_raises(self):
        """n_neg must exceed the number of pre-labeled negatives by at least 1."""
        X, labels, _ = self._data_with_pre_neg(n=60, n_pre_neg=4)
        dpul = aa.dPULearn(random_state=42)
        with pytest.raises(ValueError, match="pre-labeled"):
            dpul.fit(X, labels, label_neg=0, n_neg=4)  # 4 total but 4 already pre-labeled

    def test_n_unl_to_neg_direct_control_with_pre_neg(self):
        """n_unl_to_neg is direct: final negatives = pre-labeled + n_unl_to_neg (not a total)."""
        X, labels, pre_idx = self._data_with_pre_neg(n=60, n_pre_neg=4)
        dpul = aa.dPULearn(random_state=42)
        dpul.fit(X, labels, label_neg=0, n_unl_to_neg=6)  # 4 pre-labeled + 6 identified = 10
        assert np.sum(dpul.labels_ == 0) == 4 + 6
        assert all(dpul.labels_[i] == 0 for i in pre_idx)

    def test_n_neg_total_vs_n_unl_to_neg_direct_differ_with_pre_neg(self):
        """With pre-labeled negatives the two count options yield different totals."""
        X, labels, _ = self._data_with_pre_neg(n=60, n_pre_neg=4)
        dpul_total = aa.dPULearn(random_state=42).fit(X, labels, label_neg=0, n_neg=10)
        dpul_direct = aa.dPULearn(random_state=42).fit(X, labels, label_neg=0, n_unl_to_neg=10)
        assert np.sum(dpul_total.labels_ == 0) == 10        # total wanted
        assert np.sum(dpul_direct.labels_ == 0) == 4 + 10   # pre-labeled + identified

    # Negative tests
    def test_label_pos_equals_label_unl_raises(self):
        """Label markers must be distinct."""
        X, labels_pu = self._det_data()
        dpul = aa.dPULearn(random_state=42)
        with pytest.raises(ValueError, match="distinct"):
            dpul.fit(X, labels_pu, label_pos=1, label_unl=1, n_neg=5)

    def test_value_outside_markers_raises(self):
        """A label value matching none of the markers is rejected."""
        X, labels_pu = self._det_data(n=60)
        labels = labels_pu.copy()
        labels[np.where(labels == 2)[0][:3]] = 0  # 0 present but label_neg not declared
        dpul = aa.dPULearn(random_state=42)
        with pytest.raises(ValueError):
            dpul.fit(X, labels, n_neg=5)  # label_neg=None -> 0 is an unexpected value


class TestdPULearnFitComplex:
    """Test dPULearn.fit() method for combinations of parameters."""

    @settings(deadline=None, max_examples=10)
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

    @settings(deadline=None, max_examples=10)
    @given(X=npst.arrays(dtype=np.float64, shape=(10, 5)),
           labels=npst.arrays(dtype=np.int32, elements=some.integers(min_value=3, max_value=100), shape=(10,)),
           n_unl_to_neg=some.integers(max_value=0),
           n_components=some.floats(max_value=0.0))
    def test_invalid_combinations(self, X, labels, n_unl_to_neg, n_components):
        """Test invalid combinations of parameters."""
        dpul = aa.dPULearn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            is_invalid = check_invalid_conditions(X=X, labels=labels)
            if is_invalid:
                with pytest.raises(ValueError):
                    dpul.fit(X, labels, n_unl_to_neg, n_components)


class TestdPULearnFitReproducibility:
    """Same random_state -> identical identified negatives (reproducibility.md)."""

    def test_same_random_state_same_labels(self):
        rng = np.random.default_rng(0)
        X = rng.random((40, 6))
        labels = np.array([1, 2] * 20)  # positives (1) + unlabeled (2)
        a = aa.dPULearn(random_state=0); a.fit(X, labels=labels, n_unl_to_neg=10)
        b = aa.dPULearn(random_state=0); b.fit(X, labels=labels, n_unl_to_neg=10)
        assert np.array_equal(a.labels_, b.labels_)
