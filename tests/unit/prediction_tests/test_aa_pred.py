"""Unit tests for the AAPred class (evaluate + deploy prediction models)."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score)

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _data(n_per_class=15, n_feat=6, seed=0):
    rng = np.random.RandomState(seed)
    X_pos = rng.normal(0.5, 1.0, size=(n_per_class, n_feat))
    X_neg = rng.normal(-0.5, 1.0, size=(n_per_class, n_feat))
    X = np.vstack([X_pos, X_neg])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


class TestAAPredInit:
    def test_default_construction(self):
        aap = aa.AAPred()
        assert aap.list_models_ is None

    def test_list_model_classes(self):
        aap = aa.AAPred(list_model_classes=[RandomForestClassifier, SVC])
        assert aap._list_model_classes == [RandomForestClassifier, SVC]

    def test_list_model_kwargs(self):
        aap = aa.AAPred(list_model_classes=[SVC], list_model_kwargs=[{"probability": True}])
        assert aap._list_model_kwargs[0]["probability"] is True

    def test_list_metrics(self):
        aap = aa.AAPred(list_metrics=["balanced_accuracy"])
        assert aap._list_metrics == ["balanced_accuracy"]

    def test_verbose(self):
        aap = aa.AAPred(verbose=False)
        assert aap._verbose is False

    def test_random_state(self):
        aap = aa.AAPred(random_state=42)
        assert aap._random_state == 42

    def test_single_model_not_in_list(self):
        aap = aa.AAPred(list_model_classes=RandomForestClassifier)
        assert aap._list_model_classes == [RandomForestClassifier]

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
        aap = aa.AAPred(random_state=0, verbose=False)
        assert aap.fit(X, labels) is aap

    def test_fit_sets_models(self):
        X, labels = _data()
        aap = aa.AAPred(list_model_classes=[RandomForestClassifier, SVC],
                           list_model_kwargs=[{}, {"probability": True}], random_state=0)
        aap.fit(X, labels)
        assert len(aap.list_models_) == 2

    def test_fit_label_pos(self):
        X, labels = _data()
        aap = aa.AAPred(random_state=0).fit(X, labels, label_pos=1)
        assert aap.label_pos_ == 1

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
        aap = aa.AAPred(random_state=0).fit(X, labels, label_pos=1)
        assert aap.label_neg_ == 2


class TestAAPredModelsAndHPO:
    def test_models_by_name(self):
        X, y = _data()
        aap = aa.AAPred(models=["svm", "rf"], random_state=0).fit(X, y)
        assert len(aap.list_models_) == 2

    def test_models_accepts_estimator_instance(self):
        X, y = _data()
        aap = aa.AAPred(models=SVC(kernel="rbf", probability=True), random_state=0).fit(X, y)
        assert aap.list_models_[0].kernel == "rbf"

    def test_models_and_list_model_classes_mutually_exclusive(self):
        with pytest.raises(ValueError):
            aa.AAPred(models=["svm"], list_model_classes=[SVC])

    def test_optimize_hyperparams_with_param_grids(self):
        X, y = _data()
        aap = aa.AAPred(models=["svm"], random_state=0)
        aap.fit(X, y, optimize_hyperparams=True, param_grids={"C": [0.1, 10.0]}, n_cv=3)
        assert aap.list_models_[0].C in (0.1, 10.0)

    def test_optimize_hyperparams_default_grid(self):
        X, y = _data()
        aap = aa.AAPred(models=["svm"], random_state=0)
        aap.fit(X, y, optimize_hyperparams=True)
        assert aap.list_models_ is not None

    def test_ensemble_instance_fit(self):
        # Meta-ensembles are used by passing an instance (not a registry name); the
        # clone-based path must handle their nested get_params keys.
        from sklearn.ensemble import VotingClassifier
        from sklearn.svm import SVC
        X, y = _data()
        vc = VotingClassifier(estimators=[("rf", RandomForestClassifier()),
                                          ("svm", SVC(probability=True))], voting="soft")
        aap = aa.AAPred(models=vc, random_state=0).fit(X, y)
        assert len(aap.list_models_) == 1

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
        aap = aa.AAPred(models=RandomForestClassifier(), random_state=42).fit(X, y)
        assert aap.list_models_[0].random_state == 42

    def test_instance_explicit_seed_preserved(self):
        # A user-set seed on the passed instance is not overwritten.
        X, y = _data()
        aap = aa.AAPred(models=RandomForestClassifier(random_state=7), random_state=42).fit(X, y)
        assert aap.list_models_[0].random_state == 7


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


class TestAAPredEvalCV:
    """AAPred.eval(cv=...) — custom CV splitters (e.g. LeaveOneOut) + pooled scoring (#397)."""

    def test_cv_splitter_tags_cv_pooled_principle(self):
        X, labels = _data()
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=LeaveOneOut(), metrics=["balanced_accuracy"])
        assert set(df_eval["principle"]) == {"cv_pooled"}

    def test_cv_pooled_std_is_nan(self):
        # A pooled score is a single estimate over all held-out predictions: no fold distribution.
        X, labels = _data()
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=LeaveOneOut(), metrics=["accuracy"])
        assert df_eval["score_std"].isna().all()

    def test_cv_leaveoneout_bypasses_class_count_cap(self):
        # Imbalanced small set (10 pos vs 3 neg): the int-n_cv cap forbids n_cv>3, but a splitter
        # defines its own folds, so LeaveOneOut evaluates without raising.
        X_pos, _ = _data(n_per_class=10, seed=0)
        X_neg, _ = _data(n_per_class=3, seed=1)
        X = np.vstack([X_pos[:10], X_neg[:3]])
        labels = np.array([1] * 10 + [0] * 3)
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=LeaveOneOut(), metrics=["balanced_accuracy"])
        assert len(df_eval) == 1
        assert set(df_eval["principle"]) == {"cv_pooled"}

    @settings(max_examples=5, deadline=None)
    @given(n_splits=some.integers(min_value=2, max_value=6))
    def test_cv_stratified_kfold_splitter(self, n_splits):
        X, labels = _data(n_per_class=12)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=cv, metrics=["accuracy", "roc_auc"])
        assert set(df_eval["principle"]) == {"cv_pooled"}
        assert ((df_eval["score"] >= 0) & (df_eval["score"] <= 1)).all()

    def test_cv_pooled_roc_auc_uses_probabilities(self):
        # roc_auc needs the positive-class probability; the pooled path must reproduce a manual
        # cross_val_predict(method="predict_proba") reference, not score hard labels.
        X, labels = _data(n_per_class=12)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        est = aa.AAPred(models=["svm"], random_state=0)._list_estimators[0]
        got = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=cv, metrics=["roc_auc"])["score"].iloc[0]
        proba = cross_val_predict(clone(est), X, labels, cv=cv, method="predict_proba")[:, -1]
        ref = roc_auc_score(labels, proba)
        assert abs(got - ref) < 1e-9

    def test_cv_none_is_byte_identical_to_omitting(self):
        # Passing cv=None must be indistinguishable from not passing cv (default per-fold path).
        X, labels = _data()
        d_omit = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"])
        d_none = aa.AAPred(models=["rf"], random_state=0).eval(X, labels, metrics=["accuracy"], cv=None)
        pd.testing.assert_frame_equal(d_omit, d_none)
        assert set(d_none["principle"]) == {"cv"}

    def test_cv_reproducible(self):
        X, labels = _data()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        d1 = aa.AAPred(models=["svm"], random_state=0).eval(X, labels, cv=cv, metrics=["accuracy"])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        d2 = aa.AAPred(models=["svm"], random_state=0).eval(X, labels, cv=cv, metrics=["accuracy"])
        pd.testing.assert_frame_equal(d1, d2)

    def test_cv_invalid_splitter_raises(self):
        X, labels = _data()
        for bad in [5, "loo", object(), [1, 2, 3]]:
            with pytest.raises(ValueError):
                aa.AAPred(models=["svm"], random_state=0).eval(X, labels, cv=bad)


class TestAAPredEvalCVComplex:
    """Interactions of cv= with holdout and baseline (#397)."""

    def test_cv_with_holdout_coexist(self):
        # A pooled cv splitter and a holdout set produce both principle blocks in one table.
        X, labels = _data()
        X_holdout, labels_holdout = _data(n_per_class=8, seed=1)
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=LeaveOneOut(), metrics=["accuracy"],
            X_holdout=X_holdout, labels_holdout=labels_holdout)
        assert set(df_eval["principle"]) == {"cv_pooled", "holdout"}

    def test_cv_threads_through_baseline(self):
        # The splitter must score the baseline matrices too: every row is cv_pooled, and the
        # baseline block is the same size as the cpp block.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        X = np.random.RandomState(0).normal(size=(len(labels), 6))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, metrics=["accuracy"], df_seq=df_seq, baseline="aac", cv=cv)
        assert set(df_eval["principle"]) == {"cv_pooled"}
        assert set(df_eval["features"]) == {"cpp", "aac"}
        assert (df_eval["features"] == "aac").sum() == (df_eval["features"] == "cpp").sum()

    def test_cv_baseline_reproduces_manual_pooled_eval(self):
        # A baseline's pooled rows equal a standalone pooled eval of the same featurizer matrix.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        X = np.random.RandomState(0).normal(size=(len(labels), 6))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        d_base = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, metrics=["accuracy", "balanced_accuracy"], df_seq=df_seq, baseline="aac", cv=cv)
        X_aac = np.asarray(aa.SequenceFeature().aa_composition(df_seq=df_seq))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        d_manual = aa.AAPred(models=["svm"], random_state=0).eval(
            X_aac, labels, metrics=["accuracy", "balanced_accuracy"], cv=cv)
        aac_rows = d_base[d_base["features"] == "aac"].reset_index(drop=True)
        merged = aac_rows.merge(d_manual, on=["model", "metric", "principle"], suffixes=("_b", "_m"))
        assert len(merged) == len(d_manual)
        assert np.allclose(merged["score_b"], merged["score_m"])


class TestAAPredEvalCVGoldenValues:
    """Frozen KPI: pooled LeaveOneOut reproduces the sklearn reference exactly (#397)."""

    def test_leaveoneout_balanced_accuracy_matches_sklearn(self):
        # The exact boilerplate #397 removes:
        #   balanced_accuracy_score(y, cross_val_predict(SVC(kernel="linear"), X, y, cv=LeaveOneOut()))
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal(0.4, 1.0, size=(20, 6)), rng.normal(-0.4, 1.0, size=(9, 6))])
        labels = np.array([1] * 20 + [0] * 9)
        est = SVC(kernel="linear")
        df_eval = aa.AAPred(models=[SVC(kernel="linear")], random_state=0).eval(
            X, labels, cv=LeaveOneOut(), metrics=["balanced_accuracy"])
        got = df_eval["score"].iloc[0]
        ref = balanced_accuracy_score(labels, cross_val_predict(est, X, labels, cv=LeaveOneOut()))
        assert abs(got - ref) < 1e-9

    def test_cv_pooled_reproduces_cross_val_predict_all_label_metrics(self):
        rng = np.random.RandomState(1)
        X = np.vstack([rng.normal(0.5, 1.0, size=(15, 5)), rng.normal(-0.5, 1.0, size=(15, 5))])
        labels = np.array([1] * 15 + [0] * 15)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        est = aa.AAPred(models=["svm"], random_state=0)._list_estimators[0]
        pred = cross_val_predict(clone(est), X, labels, cv=cv)
        refs = {"accuracy": accuracy_score(labels, pred),
                "balanced_accuracy": balanced_accuracy_score(labels, pred),
                "f1": f1_score(labels, pred)}
        df_eval = aa.AAPred(models=["svm"], random_state=0).eval(
            X, labels, cv=cv, metrics=list(refs))
        for metric, ref in refs.items():
            got = df_eval[df_eval["metric"] == metric]["score"].iloc[0]
            assert abs(got - ref) < 1e-9, metric


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

    def test_baseline_all_invalid_row_raises_clear_error(self, baseline_data):
        # A sequence with no canonical residue in the span yields an all-NaN featurizer row;
        # it must raise a clear AAPred-level ValueError, not a cryptic sklearn NaN crash.
        df_seq, labels, X = baseline_data
        df_bad = df_seq.copy().reset_index(drop=True)
        col = df_bad.columns.get_loc("sequence")
        df_bad.iloc[0, col] = "X" * len(df_bad.iloc[0]["sequence"])
        with pytest.raises(ValueError):
            aa.AAPred(models=["rf"], random_state=0).eval(X, labels, df_seq=df_bad, baseline="aac")


@pytest.fixture(scope="module")
def seq_fitted():
    """A fitted AAPred with a bound df_feat, for the sequence-level predict API."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(20)
    sf = aa.SequenceFeature()
    X = sf.feature_matrix(features=df_feat, df_parts=sf.get_df_parts(df_seq=df_seq))
    aap = aa.AAPred(df_feat=df_feat, random_state=42).fit(X, labels)
    return aap, df_seq, np.asarray(X)


class TestAAPredPredict:
    def test_predict_seq_columns(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        df_pred = aap.predict(df_seq.head(5), level="sequence")
        assert list(df_pred.columns) == ["entry", "score", "score_std"]

    def test_predict_seq_scores_in_range(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        df_pred = aap.predict(df_seq.head(5), level="sequence")
        assert df_pred["score"].between(0, 1).all()

    def test_predict_seq_threshold_adds_predicted_label(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        df_pred = aap.predict(df_seq.head(5), level="sequence", threshold=0.5)
        assert "predicted_label" in df_pred.columns
        assert set(df_pred["predicted_label"]).issubset({aap.label_pos_, aap.label_neg_})

    def test_predict_domain_columns(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        df_pred = aap.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=2)
        assert list(df_pred.columns) == ["entry", "offset", "score", "is_best"]

    def test_predict_window_columns(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        one = df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]]
        df_pred = aap.predict(one, level="window", tmd_len=15, step=20)
        assert list(df_pred.columns) == ["entry", "position", "score", "score_std"]

    def test_predict_window_without_tmd_len_raises(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        one = df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]]
        with pytest.raises(ValueError):
            aap.predict(one, level="window")

    def test_predict_invalid_level_raises(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        with pytest.raises(ValueError):
            aap.predict(df_seq.head(3), level="bogus")

    def test_predict_invalid_threshold_raises(self, seq_fitted):
        aap, df_seq, _ = seq_fitted
        with pytest.raises(ValueError):
            aap.predict(df_seq.head(3), level="sequence", threshold=2.0)

    def test_predict_before_fit_raises(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        df_feat = aa.load_features(name="DOM_GSEC").head(20)
        with pytest.raises(ValueError):
            aa.AAPred(df_feat=df_feat).predict(df_seq.head(3), level="sequence")

    def test_predict_without_df_feat_raises(self):
        # Fitted (so not the not-fitted error) but no bound df_feat -> check_featurizer raises.
        X, labels = _data()
        aap = aa.AAPred(random_state=0).fit(X, labels)
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        with pytest.raises(ValueError):
            aap.predict(df_seq.head(3), level="sequence")

    def test_predict_X_raw_ensemble_scores(self, seq_fitted):
        # The private ensemble scorer that predict() builds on.
        aap, _, X = seq_fitted
        pred, pred_std = aap._predict_X(X)
        assert pred.shape == (len(X),) and pred_std.shape == (len(X),)
        assert pred.min() >= 0 and pred.max() <= 1


def _oof_ensemble(random_state=42):
    """A small 3-model ensemble mirroring the study's substratome scoring (sans xgboost)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    return [RandomForestClassifier(n_estimators=50, random_state=random_state),
            CalibratedClassifierCV(SVC(kernel="linear"), cv=3),
            LogisticRegression(max_iter=1000, random_state=random_state)]


class TestAAPredPredictOOF:
    """AAPred.predict_oof — cross-validated out-of-fold per-sample scores for the training set."""

    def test_columns(self):
        X, labels = _data()
        df_pred = aa.AAPred(random_state=0).predict_oof(X, labels)
        assert list(df_pred.columns) == ["score", "score_std"]

    def test_row_aligned_with_X(self):
        X, labels = _data(n_per_class=12)
        df_pred = aa.AAPred(random_state=0).predict_oof(X, labels)
        assert len(df_pred) == len(X)

    def test_scores_in_range(self):
        X, labels = _data()
        df_pred = aa.AAPred(random_state=0).predict_oof(X, labels)
        assert df_pred["score"].between(0, 1).all()

    def test_score_std_non_negative(self):
        X, labels = _data()
        df_pred = aa.AAPred(models=_oof_ensemble(), random_state=42).predict_oof(X, labels)
        assert (df_pred["score_std"] >= 0).all()

    def test_single_model_std_is_zero(self):
        X, labels = _data()
        df_pred = aa.AAPred(models=["rf"], random_state=42).predict_oof(X, labels)
        assert (df_pred["score_std"] == 0).all()

    @pytest.mark.parametrize("n_cv", [2, 3, 5])
    def test_n_cv_values(self, n_cv):
        X, labels = _data(n_per_class=15)
        df_pred = aa.AAPred(random_state=0).predict_oof(X, labels, n_cv=n_cv)
        assert len(df_pred) == len(X)

    def test_label_pos_default_is_one(self):
        X, labels = _data()
        d_default = aa.AAPred(random_state=0).predict_oof(X, labels)
        d_pos1 = aa.AAPred(random_state=0).predict_oof(X, labels, label_pos=1)
        pd.testing.assert_frame_equal(d_default, d_pos1)

    def test_reproducible(self):
        X, labels = _data()
        d1 = aa.AAPred(models=_oof_ensemble(), random_state=7).predict_oof(X, labels)
        d2 = aa.AAPred(models=_oof_ensemble(), random_state=7).predict_oof(X, labels)
        pd.testing.assert_frame_equal(d1, d2)

    def test_does_not_require_fit(self):
        # Unlike predict(), predict_oof cross-validates the constructor models itself.
        X, labels = _data()
        aapred = aa.AAPred(random_state=0)
        df_pred = aapred.predict_oof(X, labels)  # no prior fit
        assert len(df_pred) == len(X)

    def test_does_not_set_deployment_models(self):
        # It must not touch the fitted-deployment state in list_models_.
        X, labels = _data()
        aapred = aa.AAPred(random_state=0)
        aapred.predict_oof(X, labels)
        assert aapred.list_models_ is None

    def test_non_binary_labels_raises(self):
        X, _ = _data(n_per_class=10)
        labels = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).predict_oof(X, labels)

    def test_mismatched_labels_raises(self):
        X, _ = _data(n_per_class=10)
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).predict_oof(X, np.array([1, 0, 1]))

    def test_n_cv_too_large_raises(self):
        X, labels = _data(n_per_class=4)
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).predict_oof(X, labels, n_cv=10)

    def test_n_cv_below_two_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).predict_oof(X, labels, n_cv=1)

    def test_invalid_label_pos_raises(self):
        X, labels = _data()
        with pytest.raises(ValueError):
            aa.AAPred(random_state=0).predict_oof(X, labels, label_pos=5)


class TestAAPredPredictOOFComplex:
    """Cross-parameter and exact-equivalence behavior of AAPred.predict_oof."""

    def test_matches_hand_rolled_ensemble(self):
        # KPI: OOF score/score_std equal the hand-rolled per-model cross_val_predict block
        # (vstack over the ensemble -> mean(0)/std(0)) within 1e-9.
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        X, labels = _data(n_per_class=25, seed=3)
        rs = 42
        # Reference: identical config, scored the notebook way.
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        ref = np.vstack([cross_val_predict(m, X, labels, cv=cv5, method="predict_proba")[:, 1]
                         for m in _oof_ensemble(random_state=rs)])
        ref_score, ref_std = ref.mean(0), ref.std(0)
        df_pred = aa.AAPred(models=_oof_ensemble(random_state=rs), random_state=rs).predict_oof(X, labels)
        assert np.allclose(df_pred["score"].to_numpy(), ref_score, atol=1e-9, rtol=0)
        assert np.allclose(df_pred["score_std"].to_numpy(), ref_std, atol=1e-9, rtol=0)

    def test_label_pos_zero_is_complement_of_one(self):
        # For a single model, the OOF prob of class 0 is 1 - prob of class 1.
        X, labels = _data(n_per_class=20)
        d_pos = aa.AAPred(models=["rf"], random_state=0).predict_oof(X, labels, label_pos=1)
        d_neg = aa.AAPred(models=["rf"], random_state=0).predict_oof(X, labels, label_pos=0)
        assert np.allclose(d_neg["score"].to_numpy(), 1 - d_pos["score"].to_numpy(), atol=1e-9)

    def test_n_cv_changes_scores(self):
        # Different fold counts generally give different out-of-fold scores.
        X, labels = _data(n_per_class=20)
        d3 = aa.AAPred(models=["rf"], random_state=0).predict_oof(X, labels, n_cv=3)
        d5 = aa.AAPred(models=["rf"], random_state=0).predict_oof(X, labels, n_cv=5)
        assert not np.allclose(d3["score"].to_numpy(), d5["score"].to_numpy())

    def test_random_state_changes_folds(self):
        # A different seed reshuffles the stratified folds, changing the OOF scores.
        X, labels = _data(n_per_class=20)
        d1 = aa.AAPred(models=["rf"], random_state=1).predict_oof(X, labels)
        d2 = aa.AAPred(models=["rf"], random_state=2).predict_oof(X, labels)
        assert not np.allclose(d1["score"].to_numpy(), d2["score"].to_numpy())

    def test_matches_eval_cv_roc_auc(self):
        # The per-sample OOF scores must reproduce eval's aggregate CV roc_auc for one model
        # (same folds, same seed) -> a cross-check that predict_oof and eval share the CV contract.
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        X, labels = _data(n_per_class=20, seed=5)
        rs = 11
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        model = RandomForestClassifier(n_estimators=50, random_state=rs)
        oof = cross_val_predict(model, X, labels, cv=cv, method="predict_proba")[:, 1]
        df_pred = aa.AAPred(models=[RandomForestClassifier(n_estimators=50, random_state=rs)],
                            random_state=rs).predict_oof(X, labels)
        assert np.allclose(df_pred["score"].to_numpy(), oof, atol=1e-9)
        assert 0.0 <= roc_auc_score(labels, df_pred["score"].to_numpy()) <= 1.0
