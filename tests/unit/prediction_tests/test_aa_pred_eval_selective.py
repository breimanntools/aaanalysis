"""This is a script to test AAPred.eval_selective()."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _data(n_per_class=15, n_feat=6, seed=0):
    """Separable-ish two-class data; 2 * n_per_class rows."""
    rng = np.random.RandomState(seed)
    X_pos = rng.normal(0.5, 1.0, size=(n_per_class, n_feat))
    X_neg = rng.normal(-0.5, 1.0, size=(n_per_class, n_feat))
    X = np.vstack([X_pos, X_neg])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


class _GradedClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic rule classifier whose probability is a graded function of the first feature.

    ``predict_proba`` returns ``clip(0.5 + x0 / 2, 0, 1)`` for the positive class and ignores the
    training data, so its out-of-fold scores are known up front no matter how the folds are cut.
    That makes both the confidence ranking and the metric on every retained subset a
    hand-computable, oracle-free expectation for :meth:`AAPred.eval_selective`.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > 0).astype(int)

    def predict_proba(self, X):
        p = np.clip(0.5 + np.asarray(X)[:, 0] / 2, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _graded_data():
    """Eight samples whose out-of-fold probabilities are 1.0, 0.9, 0.0, 0.1, 0.6, 0.4, 0.7, 0.3.

    Margins are 1.0, 0.8, 1.0, 0.8, 0.2, 0.2, 0.4, 0.4, and the two least-confident samples
    (probabilities 0.6 and 0.4) are the only two the rule gets wrong.
    """
    x0 = np.array([1.0, 0.8, -1.0, -0.8, 0.2, -0.2, 0.4, -0.4])
    X = np.column_stack([x0, np.zeros_like(x0)])
    labels = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    return X, labels


def _aap_graded():
    return aa.AAPred(models=_GradedClassifier(), random_state=42)


# I Positive and negative tests, one parameter at a time
class TestEvalSelective:
    """Normal cases: one parameter per test, positive and negative."""

    # Positive tests
    def test_returns_expected_columns(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(X, labels)
        assert list(df_eval_selective) == list(ut.COLS_EVAL_SELECTIVE)
        assert isinstance(df_eval_selective, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(n_per_class=some.integers(min_value=6, max_value=12))
    def test_X_various_sizes(self, n_per_class):
        X, labels = _data(n_per_class=n_per_class)
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(X, labels)
        assert len(df_eval_selective) > 0
        assert df_eval_selective[ut.COL_N_RETAINED].max() == len(labels)

    @settings(max_examples=3, deadline=None)
    @given(n_feat=some.integers(min_value=2, max_value=8))
    def test_labels_are_row_aligned(self, n_feat):
        X, labels = _data(n_feat=n_feat)
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(X, list(labels))
        assert df_eval_selective[ut.COL_N_RETAINED].max() == len(labels)

    def test_confidence_custom_ranking(self):
        X, labels = _data()
        confidence = np.linspace(0, 1, len(labels))
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, confidence=confidence, metrics=["accuracy"])
        assert list(df_eval_selective) == list(ut.COLS_EVAL_SELECTIVE)
        assert df_eval_selective[ut.COL_N_RETAINED].is_monotonic_increasing

    @settings(max_examples=3, deadline=None)
    @given(metric=some.sampled_from(["accuracy", "balanced_accuracy", "f1", "mcc"]))
    def test_metrics_subset(self, metric):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=[metric])
        assert set(df_eval_selective[ut.COL_METRIC]) == {metric}

    def test_coverages_custom_grid(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.5, 1.0])
        assert df_eval_selective[ut.COL_COVERAGE].to_list() == [0.5, 1.0]

    @settings(max_examples=3, deadline=None)
    @given(n_cv=some.integers(min_value=2, max_value=5))
    def test_n_cv_valid(self, n_cv):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"], n_cv=n_cv)
        assert len(df_eval_selective) == len(ut.LIST_COVERAGES)

    def test_label_pos_zero(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"], label_pos=0)
        assert df_eval_selective[ut.COL_SCORE].notna().all()

    def test_default_coverage_grid(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"])
        assert df_eval_selective[ut.COL_COVERAGE].to_list() == list(ut.LIST_COVERAGES)

    def test_full_coverage_retains_all_samples(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"])
        row = df_eval_selective[df_eval_selective[ut.COL_COVERAGE] == 1.0]
        assert int(row[ut.COL_N_RETAINED].iloc[0]) == len(labels)

    def test_reproducible(self):
        X, labels = _data()
        aap = aa.AAPred(models="rf", random_state=42)
        first = aap.eval_selective(X, labels, metrics=["accuracy"])
        second = aap.eval_selective(X, labels, metrics=["accuracy"])
        pd.testing.assert_frame_equal(first, second)

    def test_does_not_fit_deployment_models(self):
        X, labels = _data()
        aap = aa.AAPred(models="rf", random_state=42)
        aap.eval_selective(X, labels, metrics=["accuracy"])
        assert aap.list_models_ is None

    # Negative tests
    @pytest.mark.parametrize("X", [None, "invalid", 1, [1, 2, 3]])
    def test_X_invalid_raises(self, X):
        _, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels)

    def test_labels_length_mismatch_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels[:-1])

    def test_labels_non_binary_raises(self):
        X, labels = _data()
        labels = np.array(labels)
        labels[:3] = 2
        with pytest.raises(ValueError, match="exactly two classes"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels)

    def test_confidence_length_mismatch_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="'confidence' n_samples"):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, confidence=np.ones(len(labels) - 1))

    def test_confidence_with_nan_raises(self):
        X, labels = _data()
        confidence = np.ones(len(labels))
        confidence[0] = np.nan
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, confidence=confidence)

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_confidence_with_infinity_raises(self, value):
        X, labels = _data()
        confidence = np.ones(len(labels))
        confidence[0] = value
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, confidence=confidence)

    def test_confidence_non_numeric_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, confidence=["a"] * len(labels))

    @pytest.mark.parametrize("metrics", [["unknown"], ["accuracy", "nonsense"], ["ACCURACY"]])
    def test_metrics_invalid_raises(self, metrics):
        X, labels = _data()
        with pytest.raises(ValueError, match="'metrics'"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, metrics=metrics)

    def test_metrics_empty_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, metrics=[])

    def test_coverages_zero_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="greater than 0"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, coverages=[0, 1.0])

    @pytest.mark.parametrize("coverages", [[1.5], [0.5, 2.0], [-0.5, 1.0]])
    def test_coverages_out_of_range_raises(self, coverages):
        X, labels = _data()
        with pytest.raises(ValueError, match="coverages"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, coverages=coverages)

    @pytest.mark.parametrize("coverages", [[1.0, 0.5], [0.5, 0.5], [0.2, 0.8, 0.4]])
    def test_coverages_not_increasing_raises(self, coverages):
        X, labels = _data()
        with pytest.raises(ValueError, match="strictly increasing"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, coverages=coverages)

    def test_coverages_empty_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, coverages=[])

    def test_coverages_nan_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, coverages=[float("nan"), 1.0])

    def test_n_cv_too_large_raises(self):
        X, labels = _data(n_per_class=5)
        with pytest.raises(ValueError, match="smallest class count"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, n_cv=6)

    @pytest.mark.parametrize("n_cv", [1, 0, -2])
    def test_n_cv_too_small_raises(self, n_cv):
        X, labels = _data()
        with pytest.raises(ValueError, match="n_cv"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, n_cv=n_cv)

    def test_label_pos_absent_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError, match="'label_pos'"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, label_pos=7)

    @pytest.mark.parametrize("label_pos", [0.5, "1", None])
    def test_label_pos_invalid_type_raises(self, label_pos):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, label_pos=label_pos)


# II Combinations and edge interactions
class TestEvalSelectiveComplex:
    """Combinations and edge interactions across parameters."""

    # Positive tests
    def test_metrics_times_coverages_shape(self):
        X, labels = _data()
        metrics = ["accuracy", "f1", "mcc"]
        coverages = [0.25, 0.5, 0.75, 1.0]
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=metrics, coverages=coverages)
        assert len(df_eval_selective) == len(metrics) * len(coverages)
        assert df_eval_selective[ut.COL_METRIC].to_list() == [m for m in metrics for _ in coverages]

    def test_n_retained_is_non_decreasing_per_metric(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy", "f1"], coverages=[0.3, 0.6, 1.0])
        for _, df_metric in df_eval_selective.groupby(ut.COL_METRIC):
            assert df_metric[ut.COL_N_RETAINED].is_monotonic_increasing

    def test_custom_confidence_changes_the_ranking(self):
        X, labels = _graded_data()
        aap = _aap_graded()
        default = aap.eval_selective(X, labels, metrics=["accuracy"], coverages=[0.25, 1.0], n_cv=2)
        # Reversed ranking: the two samples the rule gets wrong become the most confident ones.
        confidence = np.array([0.0, 0.1, 0.0, 0.1, 1.0, 1.0, 0.5, 0.5])
        reversed_ = aap.eval_selective(X, labels, confidence=confidence, metrics=["accuracy"],
                                       coverages=[0.25, 1.0], n_cv=2)
        assert default[ut.COL_SCORE].iloc[0] == 1.0
        assert reversed_[ut.COL_SCORE].iloc[0] == 0.0

    def test_single_level_grid_has_nan_area(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"], coverages=[1.0])
        assert len(df_eval_selective) == 1
        assert np.isnan(df_eval_selective[ut.COL_SCORE_AURC].iloc[0])

    def test_single_class_subset_gives_nan_for_class_dependent_metrics(self):
        X, labels = _graded_data()
        # The most confident two samples (probabilities 1.0 and 0.0) carry one label each, so a
        # ranking that puts both positives first leaves roc_auc a single-class subset.
        confidence = np.where(labels == 1, 1.0, 0.0)
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, confidence=confidence, metrics=["roc_auc", "balanced_accuracy"],
            coverages=[0.25, 1.0], n_cv=2)
        first_rows = df_eval_selective[df_eval_selective[ut.COL_COVERAGE] == 0.25]
        assert first_rows[ut.COL_SCORE].isna().all()
        assert df_eval_selective[ut.COL_SCORE_AURC].isna().all()

    def test_accuracy_is_non_decreasing_when_confidence_is_informative(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 0.5, 0.75, 1.0], n_cv=2)
        scores = df_eval_selective[ut.COL_SCORE].to_numpy()
        assert np.all(np.diff(scores) <= 0)

    def test_area_lies_between_the_curve_extremes(self):
        X, labels = _data()
        df_eval_selective = aa.AAPred(models="rf", random_state=42).eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 0.5, 0.75, 1.0])
        scores = df_eval_selective[ut.COL_SCORE].to_numpy()
        area = df_eval_selective[ut.COL_SCORE_AURC].iloc[0]
        assert scores.min() <= area <= scores.max()

    # Negative tests
    def test_estimator_without_predict_proba_raises(self):
        X, labels = _data()
        aap = aa.AAPred(models=SVC(probability=False), random_state=42)
        with pytest.raises(ValueError, match="eval_selective"):
            aap.eval_selective(X, labels, metrics=["accuracy"])

    def test_confidence_two_dimensional_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, confidence=np.ones((len(labels), 2)))

    def test_coverages_as_string_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, labels, coverages="1.0")

    def test_metrics_as_dict_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, metrics={"accuracy": 1})

    def test_n_cv_too_large_with_custom_coverages_raises(self):
        X, labels = _data(n_per_class=4)
        with pytest.raises(ValueError, match="smallest class count"):
            aa.AAPred(models="rf", random_state=42).eval_selective(
                X, labels, coverages=[0.5, 1.0], n_cv=5)

    def test_single_class_labels_raise(self):
        X, _ = _data()
        with pytest.raises(ValueError, match="more than one different value"):
            aa.AAPred(models="rf", random_state=42).eval_selective(X, np.ones(len(X), dtype=int))


# III Hand-computed expectations
class TestEvalSelectiveGoldenValues:
    """Hand-computed values on the graded eight-sample set.

    Out-of-fold probabilities are 1.0, 0.9, 0.0, 0.1, 0.6, 0.4, 0.7, 0.3, so the margin ranking
    is 0, 2 (margin 1.0), then 1, 3 (0.8), then 6, 7 (0.4), then 4, 5 (0.2). The rule is right
    on every sample except the last two, giving accuracy 1.0 at coverages 0.25, 0.5 and 0.75 and
    6/8 = 0.75 at full coverage.
    """

    def test_n_retained_is_the_ceiling_of_the_fraction(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.2, 0.25, 0.5, 1.0], n_cv=2)
        assert df_eval_selective[ut.COL_N_RETAINED].to_list() == [2, 2, 4, 8]

    def test_accuracy_curve(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 0.5, 0.75, 1.0], n_cv=2)
        assert df_eval_selective[ut.COL_SCORE].to_list() == [1.0, 1.0, 1.0, 0.75]

    def test_area_under_the_accuracy_curve(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 0.5, 0.75, 1.0], n_cv=2)
        # (0.25*1 + 0.25*1 + 0.25*0.875) / 0.75
        assert df_eval_selective[ut.COL_SCORE_AURC].iloc[0] == pytest.approx(0.9583333333, abs=1e-9)

    def test_coverage_and_area_use_actual_retained_fraction(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.2, 0.5, 1.0], n_cv=2)
        assert df_eval_selective[ut.COL_COVERAGE].to_list() == [0.25, 0.5, 1.0]
        # (0.25*1 + 0.5*0.875) / 0.75
        assert df_eval_selective[ut.COL_SCORE_AURC].iloc[0] == pytest.approx(0.9166666667, abs=1e-9)

    def test_area_is_repeated_on_every_row_of_the_metric(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 0.5, 1.0], n_cv=2)
        assert df_eval_selective[ut.COL_SCORE_AURC].nunique() == 1

    def test_full_coverage_equals_the_plain_accuracy(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.5, 1.0], n_cv=2)
        aap = _aap_graded()
        df_pred = aap.predict_oof(X, labels, n_cv=2)
        expected = accuracy_score(labels, (df_pred[ut.COL_SCORE].to_numpy() >= 0.5).astype(int))
        full = df_eval_selective[df_eval_selective[ut.COL_COVERAGE] == 1.0][ut.COL_SCORE].iloc[0]
        assert full == pytest.approx(expected, abs=1e-9)

    def test_full_coverage_balanced_accuracy_equals_sklearn(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["balanced_accuracy"], coverages=[0.5, 1.0], n_cv=2)
        df_pred = _aap_graded().predict_oof(X, labels, n_cv=2)
        expected = balanced_accuracy_score(
            labels, (df_pred[ut.COL_SCORE].to_numpy() >= 0.5).astype(int))
        full = df_eval_selective[df_eval_selective[ut.COL_COVERAGE] == 1.0][ut.COL_SCORE].iloc[0]
        assert full == pytest.approx(expected, abs=1e-9)

    def test_flat_curve_area_equals_its_height(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 0.5, 0.75], n_cv=2)
        assert df_eval_selective[ut.COL_SCORE].to_list() == [1.0, 1.0, 1.0]
        assert df_eval_selective[ut.COL_SCORE_AURC].iloc[0] == pytest.approx(1.0, abs=1e-12)

    def test_ties_keep_their_input_order(self):
        X, labels = _graded_data()
        # Every sample equally confident: the ranking is the input order, so the first two rows
        # (probabilities 1.0 and 0.9, labels 1 and 1) are both scored correctly.
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, confidence=np.ones(len(labels)), metrics=["accuracy"],
            coverages=[0.25, 1.0], n_cv=2)
        assert df_eval_selective[ut.COL_SCORE].to_list() == [1.0, 0.75]

    def test_precision_on_a_subset_without_positive_predictions_is_zero(self):
        X, labels = _graded_data()
        # Rank the two negative-scored samples first: no positive prediction is retained, so
        # precision has an empty denominator and is reported as 0 rather than warning.
        confidence = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, confidence=confidence, metrics=["precision"],
            coverages=[0.25, 1.0], n_cv=2)
        assert df_eval_selective[ut.COL_SCORE].iloc[0] == 0.0

    def test_label_pos_zero_mirrors_the_curve(self):
        X, labels = _graded_data()
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=["accuracy"], coverages=[0.25, 1.0], n_cv=2, label_pos=0)
        # Scoring class 0 as positive inverts every probability, so the margin ranking is the
        # same and the hard-label decisions are mirrored: the same accuracy curve.
        assert df_eval_selective[ut.COL_SCORE].to_list() == [1.0, 0.75]

    def test_label_pos_zero_controls_asymmetric_metrics(self):
        X, labels = _graded_data()
        metrics = ["precision", "recall", "f1", "roc_auc"]
        df_eval_selective = _aap_graded().eval_selective(
            X, labels, metrics=metrics, coverages=[1.0], n_cv=2, label_pos=0)
        scores = _aap_graded().predict_oof(X, labels, n_cv=2, label_pos=0)[ut.COL_SCORE].to_numpy()
        labels_pred = np.where(scores >= 0.5, 0, 1)
        expected = [precision_score(labels, labels_pred, pos_label=0, zero_division=0),
                    recall_score(labels, labels_pred, pos_label=0, zero_division=0),
                    f1_score(labels, labels_pred, pos_label=0, zero_division=0),
                    roc_auc_score(labels == 0, scores)]
        assert df_eval_selective[ut.COL_SCORE].to_list() == pytest.approx(expected, abs=1e-9)
