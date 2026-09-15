"""This is a script to test ModelEvaluator.learning_curve()."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# I Helper Functions
def _data(n_per_class=20, n_feat=4, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0.8, 1.0, size=(n_per_class, n_feat)),
                   rng.normal(-0.8, 1.0, size=(n_per_class, n_feat))])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


def _me(models="log_reg", random_state=0, **kwargs):
    return aa.ModelEvaluator(models=models, random_state=random_state, verbose=False, **kwargs)


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Toy classifier that logs the sample ids (column 0) it is trained and tested on."""
    log = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.train_ids_ = np.asarray(X[:, 0], dtype=int)
        self.train_labels_ = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        _RecordingClassifier.log.append((self.train_ids_, self.train_labels_,
                                         np.asarray(X[:, 0], dtype=int)))
        return (X[:, 1] > 0).astype(int)


def _id_data(n_per_class=20):
    """Feature matrix whose column 0 is a unique sample id and column 1 carries the class signal."""
    ids = np.arange(2 * n_per_class, dtype=float)
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    signal = np.where(labels == 1, 1.0, -1.0)
    return np.column_stack([ids, signal]), labels


# II Main Functions
class TestLearningCurve:
    """Normal cases: one parameter per test (positive and negative)."""

    # Positive tests
    @settings(max_examples=3, deadline=None)
    @given(n_feat=some.integers(min_value=2, max_value=6))
    def test_X(self, n_feat):
        X, labels = _data(n_feat=n_feat)
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"])
        assert list(df_curve.columns) == ut.COLS_CURVE_MODELEVAL
        assert np.isfinite(df_curve[ut.COL_SCORE]).all()

    @settings(max_examples=3, deadline=None)
    @given(n_per_class=some.integers(min_value=10, max_value=25))
    def test_labels(self, n_per_class):
        X, labels = _data(n_per_class=n_per_class)
        df_curve = _me().learning_curve(X, list(labels), train_sizes=[0.5, 1.0], metrics=["accuracy"])
        assert len(df_curve) == 2

    @settings(max_examples=5, deadline=None)
    @given(fracs=some.lists(some.floats(min_value=0.1, max_value=1.0), min_size=2, max_size=5, unique=True))
    def test_train_sizes_fractions(self, fracs):
        X, labels = _data()  # smallest training fold: 32 samples
        # Fractions resolve against that fold, floored at 2 samples, capped at 32, de-duplicated.
        sizes = sorted({min(32, max(2, int(np.floor(f * 32)))) for f in fracs})
        if len(sizes) < 2:
            with pytest.raises(ValueError, match="at least 2 distinct sizes"):
                _me().learning_curve(X, labels, train_sizes=fracs, metrics=["mcc"])
            return
        df_curve = _me().learning_curve(X, labels, train_sizes=fracs, metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == sizes

    def test_train_sizes_fractions_floor_and_dedup(self):
        X, labels = _data(n_per_class=5)  # n_cv=5 -> smallest training fold: 8 samples
        # 0.1 * 8 = 0.8 is floored to the 2-sample minimum instead of being rejected.
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.1, 0.2, 1.0], n_cv=5, metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == [2, 8]

    @settings(max_examples=5, deadline=None)
    @given(sizes=some.lists(some.integers(min_value=2, max_value=32), min_size=2, max_size=5, unique=True))
    def test_train_sizes_ints(self, sizes):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, train_sizes=sizes, metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == sorted(sizes)

    def test_train_sizes_default(self):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == [6, 12, 19, 25, 32]
        assert df_curve[ut.COL_TRAIN_SIZE].nunique() >= 4

    def test_train_sizes_array_like(self):
        X, labels = _data()
        for train_sizes in [np.array([0.25, 0.5, 1.0]), (8, 16, 32), pd.Series([4, 8])]:
            df_curve = _me().learning_curve(X, labels, train_sizes=train_sizes, metrics=["mcc"])
            assert df_curve[ut.COL_TRAIN_SIZE].is_monotonic_increasing

    @settings(max_examples=4, deadline=None)
    @given(n_cv=some.integers(min_value=2, max_value=5))
    def test_n_cv(self, n_cv):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], n_cv=n_cv, metrics=["mcc"])
        assert (df_curve[ut.COL_N_SCORES] == n_cv).all()

    @settings(max_examples=3, deadline=None)
    @given(n_rounds=some.integers(min_value=1, max_value=3))
    def test_n_rounds(self, n_rounds):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], n_cv=3,
                                        n_rounds=n_rounds, metrics=["mcc"])
        assert (df_curve[ut.COL_N_SCORES] == 3 * n_rounds).all()

    @settings(max_examples=5, deadline=None)
    @given(metrics=some.lists(some.sampled_from(ut.LIST_METRICS_MODELEVAL), min_size=1, max_size=3, unique=True))
    def test_metrics(self, metrics):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=metrics)
        assert list(dict.fromkeys(df_curve[ut.COL_METRIC])) == metrics
        assert len(df_curve) == 2 * len(metrics)

    def test_metrics_default_from_constructor(self):
        X, labels = _data()
        me = _me(list_metrics=["f1", "recall"])
        df_curve = me.learning_curve(X, labels, train_sizes=[0.5, 1.0])
        assert set(df_curve[ut.COL_METRIC]) == {"f1", "recall"}

    @settings(max_examples=4, deadline=None)
    @given(ci=some.floats(min_value=0.5, max_value=0.99))
    def test_ci(self, ci):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["accuracy"], ci=ci)
        assert (df_curve[ut.COL_CI_LOW] <= df_curve[ut.COL_SCORE] + 1e-9).all()
        assert (df_curve[ut.COL_CI_HIGH] >= df_curve[ut.COL_SCORE] - 1e-9).all()

    def test_ci_none(self):
        X, labels = _data()
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"], ci=None)
        assert df_curve[ut.COL_CI_LOW].isna().all() and df_curve[ut.COL_CI_HIGH].isna().all()
        assert df_curve[ut.COL_SCORE].notna().all()

    @settings(max_examples=3, deadline=None)
    @given(random_state=some.integers(min_value=0, max_value=1000))
    def test_random_state(self, random_state):
        X, labels = _data()
        me = _me(random_state=None)
        df1 = me.learning_curve(X, labels, train_sizes=[0.25, 1.0], metrics=["mcc"], random_state=random_state)
        df2 = me.learning_curve(X, labels, train_sizes=[0.25, 1.0], metrics=["mcc"], random_state=random_state)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    # Negative tests
    def test_invalid_X(self):
        _, labels = _data()
        with pytest.raises(ValueError, match="should not be None"):
            _me().learning_curve(None, labels)
        for X in ["X", [[1, 2], [3]], np.full((40, 3), np.nan)]:
            with pytest.raises(ValueError, match="should be array-like"):
                _me().learning_curve(X, labels)

    def test_invalid_labels(self):
        X, _ = _data()
        with pytest.raises(ValueError, match="should not be None"):
            _me().learning_curve(X, None)
        with pytest.raises(ValueError, match="more than one different value"):
            _me().learning_curve(X, [0] * 40)
        for labels in [[1, 2] * 20, [0, 1, 2, 3] * 10]:
            with pytest.raises(ValueError, match="exactly the two classes 0 and 1"):
                _me().learning_curve(X, labels)

    def test_invalid_X_labels_mismatch(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="n_samples does not match"):
            _me().learning_curve(X, labels[:-2])

    def test_invalid_train_sizes_type(self):
        X, labels = _data()
        for train_sizes in ["0.5", 0.5, 8]:
            with pytest.raises(ValueError, match="should be one of"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)
        for train_sizes in [["a", "b"], [True, False]]:
            with pytest.raises(ValueError, match="should be numbers"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)
        with pytest.raises(ValueError, match="not a mix"):
            _me().learning_curve(X, labels, train_sizes=[0.5, 8])
        with pytest.raises(ValueError, match="should not contain 'None'"):
            _me().learning_curve(X, labels, train_sizes=[0.5, None])

    def test_invalid_train_sizes_fraction_range(self):
        X, labels = _data()
        for train_sizes in [[0.0, 0.5], [-0.1, 0.5], [0.5, 1.5]]:
            with pytest.raises(ValueError, match="should be fractions in"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)

    def test_invalid_train_sizes_int_range(self):
        X, labels = _data()
        for train_sizes in [[1, 8], [0, 8], [-4, 8]]:
            with pytest.raises(ValueError, match="sample counts of at least 2"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)
        for train_sizes in [[8, 33], [8, 100]]:
            with pytest.raises(ValueError, match="smallest training fold"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)

    def test_invalid_train_sizes_too_short(self):
        X, labels = _data()
        for train_sizes in [[], [0.5], [16]]:
            with pytest.raises(ValueError, match="at least 2 elements"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)

    def test_invalid_train_sizes_duplicates(self):
        X, labels = _data()
        for train_sizes in [[8, 8], [4, 8, 4]]:
            with pytest.raises(ValueError, match="distinct sample counts"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)

    def test_invalid_train_sizes_fractions_collapse(self):
        X, labels = _data()  # 0.5 and 0.51 both floor to 16 samples of the 32-sample fold
        for train_sizes in [[0.5, 0.51], [0.02, 0.03, 0.04]]:
            with pytest.raises(ValueError, match="at least 2 distinct sizes"):
                _me().learning_curve(X, labels, train_sizes=train_sizes)

    def test_invalid_n_cv(self):
        X, labels = _data(n_per_class=6)
        with pytest.raises(ValueError, match="should not be None"):
            _me().learning_curve(X, labels, n_cv=None)
        for n_cv in [1, 0, 2.5, "5"]:
            with pytest.raises(ValueError, match="'n_cv' should be an integer"):
                _me().learning_curve(X, labels, n_cv=n_cv)
        with pytest.raises(ValueError, match="smallest class count"):
            _me().learning_curve(X, labels, n_cv=7)

    def test_invalid_n_rounds(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="should not be None"):
            _me().learning_curve(X, labels, n_rounds=None)
        for n_rounds in [0, -1, 1.5, "2"]:
            with pytest.raises(ValueError, match="'n_rounds' should be an integer"):
                _me().learning_curve(X, labels, n_rounds=n_rounds)

    def test_invalid_metrics(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="at least 1 elements"):
            _me().learning_curve(X, labels, metrics=[])
        for metrics in [["not_a_metric"], ["mcc", "auc"]]:
            with pytest.raises(ValueError, match="should each be one of"):
                _me().learning_curve(X, labels, metrics=metrics)
        with pytest.raises(ValueError, match="should be one of"):
            _me().learning_curve(X, labels, metrics=5)

    def test_invalid_ci(self):
        X, labels = _data()
        for ci in [0, 1, 1.5, -0.2, "0.95"]:
            with pytest.raises(ValueError, match="'ci' should be a float or an integer"):
                _me().learning_curve(X, labels, ci=ci)

    def test_invalid_random_state(self):
        X, labels = _data()
        for random_state in [-1, 1.5, "seed"]:
            with pytest.raises(ValueError, match="'random_state' should be an integer"):
                _me().learning_curve(X, labels, random_state=random_state)


class TestLearningCurveComplex:
    """Combinations and edge interactions."""

    # Positive tests
    def test_models_metrics_sizes_shape_and_order(self):
        X, labels = _data()
        me = _me(models=["log_reg", "rf"])
        df_curve = me.learning_curve(X, labels, train_sizes=[16, 4, 32], metrics=["mcc", "accuracy"])
        assert len(df_curve) == 2 * 3 * 2
        assert list(dict.fromkeys(df_curve[ut.COL_MODEL])) == ["log_reg", "rf"]
        for _, sub in df_curve.groupby(ut.COL_MODEL):
            assert sub[ut.COL_TRAIN_SIZE].to_list() == [4, 4, 16, 16, 32, 32]

    def test_does_not_change_run_state(self):
        X, labels = _data()
        me = _me(models=["log_reg", "svm"])
        me.learning_curve(X, labels, train_sizes=[0.5, 1.0])
        assert me.df_scores_ is None and me.df_eval_ is None
        me.run(X, labels)
        df_scores, df_eval = me.df_scores_.copy(), me.df_eval_.copy()
        me.learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["f1"])
        pd.testing.assert_frame_equal(me.df_scores_, df_scores)
        pd.testing.assert_frame_equal(me.df_eval_, df_eval)
        assert len(me.eval(metric="mcc")) == 1

    def test_constructor_seed_equals_per_call_seed(self):
        X, labels = _data()
        df1 = _me(random_state=11).learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"])
        df2 = _me(random_state=None).learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"],
                                                    random_state=11)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    def test_n_cv_rounds_and_ci_levels(self):
        X, labels = _data()
        me = _me()
        kws = dict(train_sizes=[0.5, 1.0], n_cv=4, n_rounds=3, metrics=["accuracy"], random_state=1)
        df_wide = me.learning_curve(X, labels, ci=0.99, **kws)
        df_narrow = me.learning_curve(X, labels, ci=0.5, **kws)
        assert (df_wide[ut.COL_N_SCORES] == 12).all()
        width_wide = df_wide[ut.COL_CI_HIGH] - df_wide[ut.COL_CI_LOW]
        width_narrow = df_narrow[ut.COL_CI_HIGH] - df_narrow[ut.COL_CI_LOW]
        assert (width_wide >= width_narrow - 1e-12).all()
        pd.testing.assert_series_equal(df_wide[ut.COL_SCORE], df_narrow[ut.COL_SCORE])

    def test_uneven_folds_use_smallest_training_fold(self):
        X, labels = _data(n_per_class=21)  # 42 samples, n_cv=5 -> training folds of 33 or 34
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == [16, 33]
        with pytest.raises(ValueError, match="smallest training fold"):
            _me().learning_curve(X, labels, train_sizes=[16, 34])

    def test_small_data_default_train_sizes(self):
        # Regression: 5 samples per class with n_cv=5 (smallest training fold: 8 samples). The
        # default fraction grid must resolve, not raise, and every size must carry a bootstrap CI.
        X, labels = _data(n_per_class=5)
        df_curve = _me().learning_curve(X, labels, n_cv=5, metrics=["accuracy"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == [2, 3, 4, 6, 8]
        assert df_curve[ut.COL_TRAIN_SIZE].nunique() == 5
        assert np.isfinite(df_curve[ut.COL_CI_LOW]).all() and np.isfinite(df_curve[ut.COL_CI_HIGH]).all()

    def test_proba_metric_with_proba_models(self):
        X, labels = _data()
        me = _me(models=["log_reg", SVC(kernel="linear", probability=True)])
        df_curve = me.learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["roc_auc"])
        assert df_curve[ut.COL_SCORE].between(0, 1).all()

    # Negative tests
    def test_invalid_int_sizes_exceed_fold_for_large_n_cv(self):
        X, labels = _data(n_per_class=10)  # n_cv=2 -> 10 training samples, n_cv=10 -> 18
        _me().learning_curve(X, labels, train_sizes=[4, 18], n_cv=10, metrics=["mcc"])
        with pytest.raises(ValueError, match="smallest training fold"):
            _me().learning_curve(X, labels, train_sizes=[4, 18], n_cv=2, metrics=["mcc"])

    def test_invalid_default_sizes_on_tiny_data(self):
        X, labels = _data(n_per_class=2)  # n_cv=2 -> 2 training samples, every fraction floors to 2
        with pytest.raises(ValueError, match="at least 2 distinct sizes"):
            _me().learning_curve(X, labels, n_cv=2, metrics=["accuracy"])

    def test_invalid_n_cv_checked_before_train_sizes(self):
        X, labels = _data(n_per_class=4)
        with pytest.raises(ValueError, match="should not be greater than the smallest class count"):
            _me().learning_curve(X, labels, train_sizes=[2, 3], n_cv=5)

    def test_invalid_proba_metric_without_predict_proba(self):
        X, labels = _data()
        me = _me(models=["log_reg", SVC(kernel="linear")])
        with pytest.raises(ValueError, match="predict_proba"):
            me.learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc", "roc_auc"])

    def test_invalid_mixed_train_sizes_and_ci(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="not a mix"):
            _me().learning_curve(X, labels, train_sizes=[0.5, 16], ci=0.9)
        with pytest.raises(ValueError, match="'ci' should be a float or an integer"):
            _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], ci=1.0)


class TestLearningCurveGoldenValues:
    """Hand-computed sizes, exact agreement with run, leakage and reproducibility properties."""

    def test_resolved_sizes_hand_computed(self):
        X, labels = _data()  # 40 samples, n_cv=5 -> 32 training samples per fold
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.25, 0.5, 0.75, 1.0], metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == [8, 16, 24, 32]

    def test_full_size_reproduces_run(self):
        # Same resolved random_state: the training-fold size reproduces run row for row, including
        # score_std and the bootstrap CI (both paths share one aggregation helper).
        X, labels = _data()
        me = _me(models=["rf", "log_reg"], random_state=3)
        df_eval = me.run(X, labels, n_rounds=2, metrics=["mcc", "accuracy"])
        df_curve = me.learning_curve(X, labels, train_sizes=[8, 32], n_rounds=2, metrics=["mcc", "accuracy"])
        df_full = df_curve[df_curve[ut.COL_TRAIN_SIZE] == 32].drop(columns=ut.COL_TRAIN_SIZE)
        pd.testing.assert_frame_equal(df_full.reset_index(drop=True), df_eval, check_exact=True)

    def test_small_data_sizes_hand_computed(self):
        X, labels = _data(n_per_class=5)  # n_cv=5 -> 8 training samples
        # floor(0.2*8)=1 -> floored to 2, floor(0.4*8)=3, floor(0.6*8)=4, floor(0.8*8)=6, 1.0*8=8
        df_curve = _me().learning_curve(X, labels, n_cv=5, metrics=["mcc"])
        assert df_curve[ut.COL_TRAIN_SIZE].to_list() == [2, 3, 4, 6, 8]

    def test_separable_data_scores_one(self):
        X, labels = _data(seed=2)
        X = X + labels.reshape(-1, 1) * 8
        df_curve = _me().learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["accuracy", "mcc"])
        assert np.allclose(df_curve[ut.COL_SCORE], 1.0)
        assert (df_curve[ut.COL_SCORE_STD] == 0).all()

    @settings(max_examples=3, deadline=None)
    @given(random_state=some.integers(min_value=0, max_value=100),
           n_cv=some.integers(min_value=2, max_value=5))
    def test_no_test_fold_leakage_nested_stratified(self, random_state, n_cv):
        X, labels = _id_data(n_per_class=20)
        train_sizes = [4, 10, 20]
        _RecordingClassifier.log = []
        me = aa.ModelEvaluator(models=[_RecordingClassifier()], verbose=False)
        df_curve = me.learning_curve(X, labels, train_sizes=train_sizes, n_cv=n_cv, n_rounds=2,
                                     metrics=["accuracy"], random_state=random_state)
        log = _RecordingClassifier.log
        assert len(log) == 2 * n_cv * len(train_sizes)
        for i in range(0, len(log), len(train_sizes)):
            group = log[i:i + len(train_sizes)]
            test_ids = group[0][2]
            prev = set()
            for (train_ids, train_labels, ids_test), size in zip(group, train_sizes):
                assert np.array_equal(ids_test, test_ids)  # the test fold never changes
                assert len(train_ids) == len(set(train_ids)) == size
                assert not set(train_ids) & set(ids_test)  # no test-fold leakage
                assert prev <= set(train_ids)  # nested subsets within a fold
                prev = set(train_ids)
                # Balanced classes stay balanced in every subset (stratification)
                assert abs(int(train_labels.sum()) - size / 2) <= 1
        assert (df_curve[ut.COL_SCORE] == 1.0).all()

    def test_reproducible_byte_identical(self):
        X, labels = _data()
        kws = dict(train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], n_rounds=2, metrics=["mcc", "roc_auc"])
        df1 = _me(models=["rf", "log_reg"], random_state=42).learning_curve(X, labels, **kws)
        df2 = _me(models=["rf", "log_reg"], random_state=42).learning_curve(X, labels, **kws)
        assert df1.equals(df2)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    def test_different_seeds_differ(self):
        X, labels = _data()
        kws = dict(train_sizes=[4, 8, 32], n_rounds=2, metrics=["mcc"])
        df1 = _me(random_state=0).learning_curve(X, labels, **kws)
        df2 = _me(random_state=1).learning_curve(X, labels, **kws)
        assert not df1.equals(df2)
