"""This is a script to test the aaanalysis.pipe golden-pipeline contracts.

Two guarantees of the ``ap`` layer are pinned here, independent of any single function's
per-parameter tests:

* **Parity:** ``ap.predict_samples`` at its defaults is byte-identical to the explicit
  ``SequenceFeature.feature_matrix`` -> estimator -> sklearn ``cross_validate`` chain written by
  hand (``find_features(search="fast")`` parity lives in ``test_ap_find_features.py``).
* **Ergonomics:** the ``load_dataset`` -> ``find_features`` -> ``predict_samples`` path reaches a
  cross-validated score in at most 10 statements.
"""
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.pipe as ap


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# Shared seeded fixture data (small DOM_GSEC slice; n=20 -> 40 rows, 20 per class)
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
df_feat = aa.load_features().head(6)

# Written out literally (not imported from the pipeline module) so a silent change of the default
# metric set or the default model set inside predict_samples breaks parity instead of following it.
_METRICS = ["balanced_accuracy", "accuracy", "f1", "precision", "recall", "roc_auc"]
_MAX_STATEMENTS = 10


# I Helper Functions
def _explicit_models(random_state=None):
    """The documented default comparison set, constructed by hand."""
    return {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(random_state=random_state),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
    }


def _explicit_chain(random_state=None):
    """The explicit feature_matrix -> estimator -> cross_validate chain predict_samples mirrors."""
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    rows, fitted = [], {}
    for name, est in _explicit_models(random_state=random_state).items():
        res = cross_validate(est, X, y=labels, cv=5, scoring=_METRICS)
        # cross_validate fits clones, so ``est`` is still unfitted here
        fitted[name] = est.fit(X, labels)
        row = {"feature_set": "features", "model": name, "n_features": int(X.shape[1])}
        for m in _METRICS:
            row[f"{m}_mean"] = float(np.mean(res[f"test_{m}"]))
            row[f"{m}_std"] = float(np.std(res[f"test_{m}"]))
        row["is_shap_ready"] = hasattr(fitted[name], "feature_importances_")
        rows.append(row)
    df_eval = pd.DataFrame(rows)
    best = int(df_eval["balanced_accuracy_mean"].to_numpy().argmax())
    df_eval["is_best"] = [i == best for i in range(len(df_eval))]
    return X, fitted, df_eval


def _count_statements(source):
    """Number of statements in ``source``, counting nested ones (a wrapper cannot hide lines)."""
    return sum(isinstance(node, ast.stmt) for node in ast.walk(ast.parse(source)))


def _called_names(source):
    """Names of every function/method called in ``source``."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None))
    return names


# The golden path as a user would write it: load -> find features -> cross-validated score.
_GOLDEN_PATH = '''
import aaanalysis as aa
import aaanalysis.pipe as ap
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
df_feat, _, _ = ap.find_features(labels=labels, df_seq=df_seq, search="fast", plot=False, random_state=0)
predictors, _, df_eval = ap.predict_samples(list_df_feat=df_feat, df_seq=df_seq, labels=labels,
                                            plot=False, random_state=0)
score = df_eval["balanced_accuracy_mean"].max()
'''


# II Main Functions
class TestPredictSamplesParity:
    """ap.predict_samples defaults are byte-identical to the explicit primitive chain."""

    @pytest.mark.parametrize("random_state", [0, 42])
    def test_df_eval_byte_identical(self, random_state):
        _, _, df_expected = _explicit_chain(random_state=random_state)
        # Pure defaults apart from the seed (plot=True draws the figure but never touches the table)
        _, _, df_eval = ap.predict_samples(list_df_feat=df_feat, df_seq=df_seq, labels=labels,
                                           random_state=random_state)
        pd.testing.assert_frame_equal(df_eval, df_expected, check_exact=True)
        assert df_eval.equals(df_expected)

    def test_fitted_predictors_byte_identical(self):
        X, fitted, _ = _explicit_chain(random_state=0)
        predictors, _, _ = ap.predict_samples(list_df_feat=df_feat, df_seq=df_seq, labels=labels,
                                              plot=False, random_state=0)
        assert list(predictors) == [("features", name) for name in fitted]
        for name, est in fitted.items():
            pred = predictors[("features", name)]
            assert type(pred) is type(est)
            assert pred.get_params() == est.get_params()
            assert np.array_equal(pred.predict(X), est.predict(X))
            assert np.array_equal(pred.predict_proba(X), est.predict_proba(X))

    def test_parity_detects_a_different_chain(self):
        """Guard against a vacuous anchor: a different CV fold count must break equality."""
        _, _, df_expected = _explicit_chain(random_state=0)
        _, _, df_eval = ap.predict_samples(list_df_feat=df_feat, df_seq=df_seq, labels=labels,
                                           n_cv=4, plot=False, random_state=0)
        assert not df_eval.equals(df_expected)


class TestGoldenPathErgonomics:
    """The load -> find_features -> predict_samples path stays within the statement budget."""

    def test_statement_budget(self):
        n = _count_statements(_GOLDEN_PATH)
        assert n <= _MAX_STATEMENTS, (f"golden path needs {n} statements, "
                                      f"exceeding the {_MAX_STATEMENTS}-statement budget")

    def test_path_covers_the_spine(self):
        assert {"load_dataset", "find_features", "predict_samples"} <= _called_names(_GOLDEN_PATH)

    def test_counter_counts_nested_statements(self):
        assert _count_statements("a = 1\nb = 2") == 2
        assert _count_statements("a = 1; b = 2") == 2
        assert _count_statements("if True:\n    a = 1\n    b = 2") == 3

    def test_path_runs_and_reaches_cv_score(self):
        namespace = {}
        exec(compile(_GOLDEN_PATH, "<golden_path>", "exec"), namespace)
        score = namespace["score"]
        assert isinstance(namespace["predictors"], dict) and len(namespace["predictors"]) > 0
        assert np.isfinite(score) and 0.0 <= score <= 1.0
