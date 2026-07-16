"""Unit tests for the ModelEvaluator class (repeated-CV evaluation + paired comparison)."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.prediction._model_evaluator import _make_unique_names

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False


def _data(n_per_class=20, n_feat=6, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0.6, 1.0, size=(n_per_class, n_feat)),
                   rng.normal(-0.6, 1.0, size=(n_per_class, n_feat))])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


# --------------------------------------------------------------------------- constructor
class TestModelEvaluatorInit:
    def test_default_construction(self):
        me = aa.ModelEvaluator()
        assert me._list_model_names == ["rf"]
        assert me.df_eval_ is None and me.df_scores_ is None

    def test_models_names_and_instances(self):
        me = aa.ModelEvaluator(models=["rf", "svm", SVC(kernel="linear", probability=True)])
        assert len(me._list_estimators) == 3

    def test_models_duplicate_names_made_unique(self):
        me = aa.ModelEvaluator(models=["rf", "rf"])
        assert me._list_model_names == ["rf_1", "rf_2"]

    def test_list_metrics_custom(self):
        me = aa.ModelEvaluator(list_metrics=["accuracy", "mcc"])
        assert me._list_metrics == ["accuracy", "mcc"]

    def test_verbose_and_random_state(self):
        me = aa.ModelEvaluator(verbose=False, random_state=42)
        assert me._random_state == 42

    def test_invalid_empty_models(self):
        with pytest.raises(ValueError):
            aa.ModelEvaluator(models=[])

    def test_invalid_metric_name(self):
        with pytest.raises(ValueError):
            aa.ModelEvaluator(list_metrics=["not_a_metric"])

    def test_invalid_model_name(self):
        with pytest.raises(ValueError):
            aa.ModelEvaluator(models="not_a_model")

    @given(rs=some.integers(min_value=0, max_value=99))
    @settings(max_examples=5)
    def test_random_state_range(self, rs):
        me = aa.ModelEvaluator(random_state=rs)
        assert me._random_state == rs


# --------------------------------------------------------------------------- run
class TestModelEvaluatorRun:
    def test_returns_expected_columns(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=0)
        df_eval = me.run(X=X, labels=labels, n_cv=5, n_rounds=1)
        assert list(df_eval.columns) == ut.COLS_EVAL_MODELEVAL
        assert len(df_eval) == 2 * len(me._list_metrics)

    def test_n_scores_equals_folds_times_rounds(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, n_cv=5, n_rounds=3)
        assert (df_eval[ut.COL_N_SCORES] == 15).all()

    def test_df_scores_one_row_per_fold(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        me.run(X, labels, n_cv=5, n_rounds=2, metrics=["mcc"])
        assert len(me.df_scores_) == 5 * 2  # one score per (round, fold) for one model x one metric
        assert list(me.df_scores_.columns) == ut.COLS_SCORES_MODELEVAL

    def test_metrics_override(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, metrics=["accuracy", "roc_auc"])
        assert set(df_eval[ut.COL_METRIC]) == {"accuracy", "roc_auc"}

    def test_ci_none_gives_nan_bounds(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, ci=None)
        assert df_eval[ut.COL_CI_LOW].isna().all() and df_eval[ut.COL_CI_HIGH].isna().all()

    def test_ci_bounds_bracket_mean(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, n_rounds=3, ci=0.95, random_state=1)
        assert (df_eval[ut.COL_CI_LOW] <= df_eval[ut.COL_SCORE] + 1e-9).all()
        assert (df_eval[ut.COL_CI_HIGH] >= df_eval[ut.COL_SCORE] - 1e-9).all()

    def test_reproducible_same_seed(self):
        X, labels = _data()
        me1 = aa.ModelEvaluator(models=["rf", "svm"], random_state=7)
        me2 = aa.ModelEvaluator(models=["rf", "svm"], random_state=7)
        df1 = me1.run(X, labels, n_rounds=2)
        df2 = me2.run(X, labels, n_rounds=2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_score_in_unit_interval(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, metrics=["accuracy", "balanced_accuracy"])
        assert (df_eval[ut.COL_SCORE] >= 0).all() and (df_eval[ut.COL_SCORE] <= 1).all()

    def test_invalid_n_cv_too_large(self):
        X, labels = _data(n_per_class=4)
        me = aa.ModelEvaluator(models="rf", random_state=0)
        with pytest.raises(ValueError):
            me.run(X, labels, n_cv=10)

    def test_invalid_n_rounds(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        with pytest.raises(ValueError):
            me.run(X, labels, n_rounds=0)

    def test_invalid_ci_out_of_range(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        with pytest.raises(ValueError):
            me.run(X, labels, ci=1.5)

    def test_invalid_non_binary_labels(self):
        X, _ = _data()
        labels = np.array([0, 1, 2] * (len(X) // 3) + [0] * (len(X) % 3))
        me = aa.ModelEvaluator(models="rf", random_state=0)
        with pytest.raises(ValueError):
            me.run(X, labels)

    def test_proba_metric_requires_predict_proba(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=[SVC(kernel="linear")], random_state=0)  # no predict_proba
        with pytest.raises(ValueError):
            me.run(X, labels, metrics=["roc_auc"])

    @given(n_rounds=some.integers(min_value=1, max_value=3))
    @settings(max_examples=3)
    def test_n_rounds_scales_scores(self, n_rounds):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, n_cv=5, n_rounds=n_rounds, metrics=["mcc"])
        assert (df_eval[ut.COL_N_SCORES] == 5 * n_rounds).all()


# --------------------------------------------------------------------------- eval (compare)
class TestModelEvaluatorEval:
    def test_returns_expected_columns(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=0)
        me.run(X, labels)
        df_cmp = me.eval(metric="mcc")
        assert list(df_cmp.columns) == ut.COLS_COMPARE_MODELEVAL
        assert len(df_cmp) == 1  # one pair

    def test_n_pairs(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm", "log_reg"], random_state=0)
        me.run(X, labels)
        df_cmp = me.eval(metric="mcc")
        assert len(df_cmp) == 3  # C(3, 2)

    def test_delta_is_signed_difference(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "log_reg"], random_state=0)
        df_eval = me.run(X, labels, n_rounds=2, metrics=["mcc"])
        df_cmp = me.eval(metric="mcc")
        s = df_eval.set_index(ut.COL_MODEL)[ut.COL_SCORE]
        # delta mean equals difference of per-model means over the same folds
        assert df_cmp[ut.COL_DELTA].iloc[0] == pytest.approx(s["rf"] - s["log_reg"], abs=1e-9)

    def test_pvalue_in_unit_interval(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=0)
        me.run(X, labels, n_rounds=2)
        p = me.eval(metric="mcc")[ut.COL_P_VALUE].iloc[0]
        assert 0.0 <= p <= 1.0

    def test_ci_none(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=0)
        me.run(X, labels)
        df_cmp = me.eval(metric="mcc", ci=None, random_state=0)
        assert df_cmp[ut.COL_CI_LOW].isna().all()

    def test_reproducible_same_seed(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=3)
        me.run(X, labels, n_rounds=2)
        pd.testing.assert_frame_equal(me.eval(metric="mcc", random_state=1),
                                      me.eval(metric="mcc", random_state=1))

    def test_error_before_run(self):
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=0)
        with pytest.raises(ValueError):
            me.eval(metric="mcc")

    def test_error_single_model(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="rf", random_state=0)
        me.run(X, labels)
        with pytest.raises(ValueError):
            me.eval(metric="mcc")

    def test_error_metric_not_in_run(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=0)
        me.run(X, labels, metrics=["accuracy", "mcc"])
        with pytest.raises(ValueError):
            me.eval(metric="roc_auc")  # not computed by run

    def test_identical_models_zero_delta_pvalue_one(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "rf"], random_state=0)
        me.run(X, labels, n_rounds=2, metrics=["mcc"])
        df_cmp = me.eval(metric="mcc")
        assert df_cmp[ut.COL_DELTA].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert df_cmp[ut.COL_P_VALUE].iloc[0] == 1.0


# --------------------------------------------------------------------------- golden values
class TestModelEvaluatorGoldenValues:
    def test_perfect_separation_scores_one(self):
        # Linearly separable -> every metric should be ~1.0 for a capable model.
        X, labels = _data(n_per_class=25, seed=1)
        X = X + labels.reshape(-1, 1) * 6  # push classes far apart
        me = aa.ModelEvaluator(models="rf", random_state=0)
        df_eval = me.run(X, labels, metrics=["accuracy", "mcc"])
        assert (df_eval[ut.COL_SCORE] > 0.95).all()

    def test_real_dataset_smoke(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=15)
        labels = df_seq["label"].to_list()
        df_feat = aa.load_features(name="DOM_GSEC").head(15)
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
        me = aa.ModelEvaluator(models=["rf", "svm"], random_state=42)
        df_eval = me.run(X, labels, n_cv=5, n_rounds=1)
        assert not df_eval[ut.COL_SCORE].isna().any()
        assert len(me.eval(metric="mcc")) == 1


# ------------------------------------------------------------- review-finding regressions (#91)
class TestModelEvaluatorReviewRegressions:
    def test_labels_must_be_zero_one(self):
        # {1, 2} is two-class but not {0,1}: metrics would silently disagree, so it must raise.
        X, _ = _data()
        labels = np.array([1] * (len(X) // 2) + [2] * (len(X) - len(X) // 2))
        me = aa.ModelEvaluator(models="rf", random_state=0)
        with pytest.raises(ValueError):
            me.run(X, labels)

    def test_per_call_seed_reproducible_with_none_constructor(self):
        # Constructor seed None + per-call random_state must still be byte-identical across runs
        # (the estimators are seeded from the effective per-call seed).
        X, labels = _data()
        me1 = aa.ModelEvaluator(models=["rf", "svm"], random_state=None)
        me2 = aa.ModelEvaluator(models=["rf", "svm"], random_state=None)
        df1 = me1.run(X, labels, n_rounds=2, random_state=42)
        df2 = me2.run(X, labels, n_rounds=2, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_eval_default_metric_falls_back_when_mcc_absent(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["rf", "svm"], list_metrics=["accuracy"], random_state=0)
        me.run(X, labels)
        df_cmp = me.eval()  # no metric given, run did not compute mcc -> falls back to 'accuracy'
        assert df_cmp[ut.COL_METRIC].iloc[0] == "accuracy"

    def test_make_unique_names_no_collision(self):
        # A user name colliding with the auto suffix must not produce duplicate labels.
        out = _make_unique_names(["rf", "rf", "rf_1"])
        assert len(out) == len(set(out)) == 3
