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

    def test_fit_sets_real_negative_label(self):
        X, labels01 = _data()
        labels = np.where(labels01 == 0, 2, 1)  # classes {1, 2}
        aapred = aa.AAPred(random_state=0).fit(X, labels, label_pos=1)
        assert aapred.label_neg_ == 2


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

    def test_ensemble_instance_fit(self):
        # Meta-ensembles are used by passing an instance (not a registry name); the
        # clone-based path must handle their nested get_params keys.
        from sklearn.ensemble import VotingClassifier
        from sklearn.svm import SVC
        X, y = _data()
        vc = VotingClassifier(estimators=[("rf", RandomForestClassifier()),
                                          ("svm", SVC(probability=True))], voting="soft")
        aapred = aa.AAPred(models=vc, random_state=0).fit(X, y)
        assert len(aapred.list_models_) == 1

    def test_ensemble_instance_eval(self):
        from sklearn.ensemble import VotingClassifier
        from sklearn.svm import SVC
        X, y = _data()
        vc = VotingClassifier(estimators=[("rf", RandomForestClassifier()),
                                          ("svm", SVC(probability=True))], voting="soft")
        df = aa.AAPred(models=vc, random_state=0).eval(X, y, metrics=["accuracy"])
        assert len(df) >= 1

    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError):
            aa.AAPred(models="xgboost")  # not in the small registry; pass an instance instead

    def test_instance_receives_random_state(self):
        # A passed estimator with random_state left unset inherits the AAPred seed.
        X, y = _data()
        aapred = aa.AAPred(models=RandomForestClassifier(), random_state=42).fit(X, y)
        assert aapred.list_models_[0].random_state == 42

    def test_instance_explicit_seed_preserved(self):
        # A user-set seed on the passed instance is not overwritten.
        X, y = _data()
        aapred = aa.AAPred(models=RandomForestClassifier(random_state=7), random_state=42).fit(X, y)
        assert aapred.list_models_[0].random_state == 7


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


@pytest.fixture(scope="module")
def baseline_data():
    """A real df_seq (sequences + TMD boundaries) with labels and a stand-in CPP X."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
    labels = df_seq["label"].to_list()
    rng = np.random.RandomState(0)
    X = rng.normal(size=(len(labels), 6))
    return df_seq, labels, X


class TestAAPredEvalBaseline:
    """AAPred.eval(baseline=...) — compare bound features vs composition baselines (#335)."""

    def test_baseline_none_is_byte_identical(self, baseline_data):
        df_seq, labels, X = baseline_data
        d_none = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"])
        d_default = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"],
                                                                  baseline=None)
        pd.testing.assert_frame_equal(d_none, d_default)
        assert "features" not in d_none.columns

    def test_baseline_false_is_none(self, baseline_data):
        df_seq, labels, X = baseline_data
        df_eval = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"],
                                                                df_seq=df_seq, baseline=False)
        assert "features" not in df_eval.columns

    def test_baseline_true_adds_features_column(self, baseline_data):
        df_seq, labels, X = baseline_data
        df_eval = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"],
                                                                df_seq=df_seq, baseline=True)
        assert df_eval.columns[0] == "features"
        assert set(df_eval["features"]) == {"cpp", "scale"}

    def test_baseline_string_selects_one_kind(self, baseline_data):
        df_seq, labels, X = baseline_data
        df_eval = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"],
                                                                df_seq=df_seq, baseline="aac")
        assert set(df_eval["features"]) == {"cpp", "aac"}

    def test_baseline_list_appends_one_block_per_kind(self, baseline_data):
        df_seq, labels, X = baseline_data
        ap = aa.AAPred(models=["rf"], random_state=0)
        d_cpp = ap.eval(X, labels, metrics=["accuracy"])
        d_base = ap.eval(X, labels, metrics=["accuracy"], df_seq=df_seq, baseline=["aac", "dpc"])
        assert set(d_base["features"]) == {"cpp", "aac", "dpc"}
        # the cpp block is unchanged in size; each baseline adds an equal-sized CV-only block
        assert (d_base["features"] == "cpp").sum() == len(d_cpp)
        assert (d_base["features"] == "aac").sum() == len(d_cpp)

    def test_baseline_deduplicates_kinds(self, baseline_data):
        df_seq, labels, X = baseline_data
        df_eval = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"],
                                                                df_seq=df_seq, baseline=["aac", "aac"])
        assert (df_eval["features"] == "aac").sum() == (df_eval["features"] == "cpp").sum()

    def test_baseline_rows_are_cv_only(self, baseline_data):
        df_seq, labels, X = baseline_data
        X_hold, labels_hold = _data(n_per_class=6, seed=1)
        # holdout only applies to the bound (cpp) features; baselines stay cross-validation-only
        df_eval = aa.AAPred(models=["rf"], random_state=0).eval(
            X, labels, metrics=["accuracy"], df_seq=df_seq, baseline="aac")
        assert set(df_eval[df_eval["features"] == "aac"]["principle"]) == {"cv"}

    def test_baseline_reproduces_manual_featurize_then_eval(self, baseline_data):
        # KPI: a baseline's rows equal a standalone eval of the same featurizer matrix.
        df_seq, labels, X = baseline_data
        d_base = aa.AAPred(models=["rf"], random_state=0).eval(
            X, labels, metrics=["accuracy", "f1"], df_seq=df_seq, baseline="aac")
        X_aac = np.asarray(aa.SequenceFeature().aa_composition(df_seq=df_seq))
        d_manual = aa.AAPred(models=["rf"], random_state=0).eval(X_aac, labels,
                                                                 metrics=["accuracy", "f1"])
        aac_rows = d_base[d_base["features"] == "aac"].reset_index(drop=True)
        merged = aac_rows.merge(d_manual, on=["model", "metric", "principle"], suffixes=("_b", "_m"))
        assert len(merged) == len(d_manual)
        assert np.allclose(merged["score_b"], merged["score_m"])
        assert np.allclose(merged["score_std_b"], merged["score_std_m"])

    def test_baseline_reproducible(self, baseline_data):
        df_seq, labels, X = baseline_data
        d1 = aa.AAPred(models=["rf"], random_state=7).eval(X, labels, metrics=["accuracy"],
                                                           df_seq=df_seq, baseline=["aac", "scale"])
        d2 = aa.AAPred(models=["rf"], random_state=7).eval(X, labels, metrics=["accuracy"],
                                                           df_seq=df_seq, baseline=["aac", "scale"])
        pd.testing.assert_frame_equal(d1, d2)

    def test_baseline_requires_df_seq(self, baseline_data):
        df_seq, labels, X = baseline_data
        with pytest.raises(ValueError):
            aa.AAPred(models=["rf"], random_state=0).eval(X, labels, baseline=True)

    def test_baseline_df_seq_length_mismatch_raises(self, baseline_data):
        df_seq, labels, X = baseline_data
        with pytest.raises(ValueError):
            aa.AAPred(models=["rf"], random_state=0).eval(X, labels, df_seq=df_seq.head(3),
                                                          baseline=True)

    def test_baseline_invalid_kind_raises(self, baseline_data):
        df_seq, labels, X = baseline_data
        with pytest.raises(ValueError):
            aa.AAPred(models=["rf"], random_state=0).eval(X, labels, df_seq=df_seq, baseline="nope")


@pytest.fixture(scope="module")
def seq_fitted():
    """A fitted AAPred with a bound df_feat, for the sequence-level predict API."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(20)
    sf = aa.SequenceFeature()
    X = sf.feature_matrix(features=df_feat, df_parts=sf.get_df_parts(df_seq=df_seq))
    aapred = aa.AAPred(df_feat=df_feat, random_state=42).fit(X, labels)
    return aapred, df_seq, np.asarray(X)


class TestAAPredPredict:
    def test_predict_seq_columns(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        df_pred = aapred.predict(df_seq.head(5), level="sequence")
        assert list(df_pred.columns) == ["entry", "score", "score_std"]

    def test_predict_seq_scores_in_range(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        df_pred = aapred.predict(df_seq.head(5), level="sequence")
        assert df_pred["score"].between(0, 1).all()

    def test_predict_seq_threshold_adds_predicted_label(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        df_pred = aapred.predict(df_seq.head(5), level="sequence", threshold=0.5)
        assert "predicted_label" in df_pred.columns
        assert set(df_pred["predicted_label"]).issubset({aapred.label_pos_, aapred.label_neg_})

    def test_predict_domain_columns(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        df_pred = aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=2)
        assert list(df_pred.columns) == ["entry", "offset", "score", "is_best"]

    def test_predict_window_columns(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        one = df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]]
        df_pred = aapred.predict(one, level="window", tmd_len=15, step=20)
        assert list(df_pred.columns) == ["entry", "position", "score", "score_std"]

    def test_predict_window_without_tmd_len_raises(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        one = df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]]
        with pytest.raises(ValueError):
            aapred.predict(one, level="window")

    def test_predict_invalid_level_raises(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        with pytest.raises(ValueError):
            aapred.predict(df_seq.head(3), level="bogus")

    def test_predict_invalid_threshold_raises(self, seq_fitted):
        aapred, df_seq, _ = seq_fitted
        with pytest.raises(ValueError):
            aapred.predict(df_seq.head(3), level="sequence", threshold=2.0)

    def test_predict_before_fit_raises(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        df_feat = aa.load_features(name="DOM_GSEC").head(20)
        with pytest.raises(ValueError):
            aa.AAPred(df_feat=df_feat).predict(df_seq.head(3), level="sequence")

    def test_predict_without_df_feat_raises(self):
        # Fitted (so not the not-fitted error) but no bound df_feat -> check_featurizer raises.
        X, labels = _data()
        aapred = aa.AAPred(random_state=0).fit(X, labels)
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        with pytest.raises(ValueError):
            aapred.predict(df_seq.head(3), level="sequence")

    def test_predict_X_raw_ensemble_scores(self, seq_fitted):
        # The private ensemble scorer that predict() builds on.
        aapred, _, X = seq_fitted
        pred, pred_std = aapred._predict_X(X)
        assert pred.shape == (len(X),) and pred_std.shape == (len(X),)
        assert pred.min() >= 0 and pred.max() <= 1
