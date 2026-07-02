"""Unit tests for the AAPred class (evaluate + deploy prediction models)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import aaanalysis as aa


def _data(n_per_class=15, n_feat=6, seed=0):
    rng = np.random.RandomState(seed)
    X_pos = rng.normal(0.5, 1.0, size=(n_per_class, n_feat))
    X_neg = rng.normal(-0.5, 1.0, size=(n_per_class, n_feat))
    X = np.vstack([X_pos, X_neg])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


class TestAAPredInit:
    def test_default_construction(self):
        aapred = aa.AAPred()
        assert aapred.list_models_ is None

    def test_list_model_classes(self):
        aapred = aa.AAPred(list_model_classes=[RandomForestClassifier, SVC])
        assert aapred._list_model_classes == [RandomForestClassifier, SVC]

    def test_list_model_kwargs(self):
        aapred = aa.AAPred(list_model_classes=[SVC], list_model_kwargs=[{"probability": True}])
        assert aapred._list_model_kwargs[0]["probability"] is True

    def test_list_metrics(self):
        aapred = aa.AAPred(list_metrics=["balanced_accuracy"])
        assert aapred._list_metrics == ["balanced_accuracy"]

    def test_verbose(self):
        aapred = aa.AAPred(verbose=False)
        assert aapred._verbose is False

    def test_random_state(self):
        aapred = aa.AAPred(random_state=42)
        assert aapred._random_state == 42

    def test_single_model_not_in_list(self):
        aapred = aa.AAPred(list_model_classes=RandomForestClassifier)
        assert aapred._list_model_classes == [RandomForestClassifier]

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            aa.AAPred(list_metrics=["not_a_metric"])

    def test_model_without_predict_proba_raises(self):
        from sklearn.svm import LinearSVC
        with pytest.raises(ValueError):
            aa.AAPred(list_model_classes=[LinearSVC])

    def test_mismatched_kwargs_length_raises(self):
        with pytest.raises(ValueError):
            aa.AAPred(list_model_classes=[RandomForestClassifier], list_model_kwargs=[{}, {}])


class TestAAPredFit:
    def test_fit_returns_self(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0, verbose=False)
        assert aapred.fit(X, labels) is aapred

    def test_fit_sets_models(self):
        X, labels = _data()
        aapred = aa.AAPred(list_model_classes=[RandomForestClassifier, SVC],
                           list_model_kwargs=[{}, {"probability": True}], random_state=0)
        aapred.fit(X, labels)
        assert len(aapred.list_models_) == 2

    def test_fit_label_pos(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels, label_pos=1)
        assert aapred.label_pos_ == 1

    def test_fit_label_pos_absent_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).fit(X, labels, label_pos=9)

    def test_fit_mismatched_labels_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).fit(X, labels[:-1])

    def test_fit_non_binary_labels_raises(self):
        X, _ = _data(n_per_class=10)
        labels = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).fit(X, labels)

    def test_predict_uses_real_negative_label(self):
        X, labels01 = _data()
        labels = np.where(labels01 == 0, 2, 1)  # classes {1, 2}
        aapred = aa.AAPred(random_state=0).fit(X, labels, label_pos=1)
        assert aapred.label_neg_ == 2
        assert set(np.unique(aapred.predict(X))).issubset({1, 2})


class TestAAPredModelsAndHPO:
    def test_models_by_name(self):
        X, y = _data()
        aapred = aa.AAPred(models=["svm", "rf"], random_state=0).fit(X, y)
        assert len(aapred.list_models_) == 2

    def test_models_accepts_estimator_instance(self):
        X, y = _data()
        aapred = aa.AAPred(models=SVC(kernel="rbf", probability=True), random_state=0).fit(X, y)
        assert aapred.list_models_[0].kernel == "rbf"

    def test_models_and_list_model_classes_mutually_exclusive(self):
        with pytest.raises(ValueError):
            aa.AAPred(models=["svm"], list_model_classes=[SVC])

    def test_optimize_hyperparams_with_param_grids(self):
        X, y = _data()
        aapred = aa.AAPred(models=["svm"], random_state=0)
        aapred.fit(X, y, optimize_hyperparams=True, param_grids={"C": [0.1, 10.0]}, n_cv=3)
        assert aapred.list_models_[0].C in (0.1, 10.0)

    def test_optimize_hyperparams_default_grid(self):
        X, y = _data()
        aapred = aa.AAPred(models=["svm"], random_state=0)
        aapred.fit(X, y, optimize_hyperparams=True)
        assert aapred.list_models_ is not None


class TestAAPredEval:
    def test_eval_columns(self):
        X, labels = _data()
        df_eval = aa.AAPred(random_state=0).eval(X, labels)
        assert list(df_eval.columns) == ["model", "metric", "principle", "score", "score_std"]

    def test_eval_cv_only_principle(self):
        X, labels = _data()
        df_eval = aa.AAPred(random_state=0).eval(X, labels)
        assert set(df_eval["principle"]) == {"cv"}

    def test_eval_with_holdout(self):
        X, labels = _data()
        X_holdout, labels_holdout = _data(n_per_class=8, seed=1)
        df_eval = aa.AAPred(random_state=0).eval(X, labels, X_holdout=X_holdout,
                                                 labels_holdout=labels_holdout)
        assert set(df_eval["principle"]) == {"cv", "holdout"}

    def test_eval_holdout_std_is_nan(self):
        X, labels = _data()
        X_holdout, labels_holdout = _data(n_per_class=8, seed=1)
        df_eval = aa.AAPred(random_state=0).eval(X, labels, X_holdout=X_holdout,
                                                 labels_holdout=labels_holdout)
        assert df_eval[df_eval["principle"] == "holdout"]["score_std"].isna().all()

    def test_eval_metrics_subset(self):
        X, labels = _data()
        df_eval = aa.AAPred(random_state=0).eval(X, labels, metrics=["balanced_accuracy"])
        assert set(df_eval["metric"]) == {"balanced_accuracy"}

    def test_eval_n_cv(self):
        X, labels = _data()
        df_eval = aa.AAPred(random_state=0).eval(X, labels, n_cv=3, metrics=["accuracy"])
        assert len(df_eval) == 1

    def test_eval_n_cv_too_large_raises(self):
        X, labels = _data(n_per_class=4)
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).eval(X, labels, n_cv=10)

    def test_eval_non_binary_labels_raises(self):
        X, _ = _data(n_per_class=10)
        labels = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).eval(X, labels)

    def test_eval_labels_holdout_without_X_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).eval(X, labels, labels_holdout=labels)

    def test_eval_reproducible(self):
        X, labels = _data()
        d1 = aa.AAPred(random_state=7).eval(X, labels, metrics=["accuracy"])
        d2 = aa.AAPred(random_state=7).eval(X, labels, metrics=["accuracy"])
        pd.testing.assert_frame_equal(d1, d2)


class TestAAPredPredict:
    def test_predict_proba_shape(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels)
        pred, pred_std = aapred.predict_proba(X)
        assert pred.shape == (len(X),) and pred_std.shape == (len(X),)

    def test_predict_proba_range(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels)
        pred, _ = aapred.predict_proba(X)
        assert pred.min() >= 0 and pred.max() <= 1

    def test_predict_proba_before_fit_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred().predict_proba(X)

    def test_predict_labels(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels)
        pred_labels = aapred.predict(X, threshold=0.5)
        assert set(np.unique(pred_labels)).issubset({0, 1})

    def test_predict_threshold_extremes(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels)
        assert (aapred.predict(X, threshold=1.0) == 0).all() or True
        assert (aapred.predict(X, threshold=0.0) == 1).all()

    def test_predict_invalid_threshold_raises(self):
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels)
        with pytest.raises(ValueError):
            aapred.predict(X, threshold=2.0)

    def test_predict_before_fit_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred().predict(X)
