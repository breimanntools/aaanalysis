"""This is a script to test the aaanalysis.pipe golden-pipeline contracts.

Two guarantees of the ``ap`` layer are pinned here, independent of any single function's
per-parameter tests:

* **Parity:** ``ap.predict_samples`` at its defaults is byte-identical to the explicit
  ``SequenceFeature.feature_matrix`` -> estimator -> sklearn ``cross_validate`` chain written by
  hand (``find_features(search="fast")`` parity lives in ``test_ap_find_features.py``). Parity
  covers the comparison table *and* the learned state of every fitted predictor.
* **Ergonomics:** the ``load_dataset`` -> ``find_features`` -> ``predict_samples`` path reaches a
  cross-validated score in at most 10 statements. The snippet under test is read out of the
  documented code block in ``golden_pipelines.rst``, so the page and the test cannot drift apart.
"""
import ast
import copy
import re
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.pipe as ap


# Shared seeded fixture data (small DOM_GSEC slice; n=20 -> 40 rows, 20 per class)
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
df_feat = aa.load_features().head(6)

# Written out literally (not imported from the pipeline module) so a silent change of the default
# metric set or the default model set inside predict_samples breaks parity instead of following it.
_METRICS = ["balanced_accuracy", "accuracy", "f1", "precision", "recall", "roc_auc"]
_MAX_STATEMENTS = 10

# The documented golden path (docs/source/index/usage_principles/golden_pipelines.rst)
_DOC_PAGE = (Path(__file__).resolve().parents[3]
             / "docs" / "source" / "index" / "usage_principles" / "golden_pipelines.rst")


# I Helper Functions
def _rst_python_blocks(path):
    """Bodies of every ``.. code-block:: python`` directive in an RST file, dedented."""
    lines = Path(path).read_text().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        match = re.match(r"^(\s*)\.\.\s+code-block::\s*python\s*$", lines[i])
        if match is None:
            i += 1
            continue
        indent, body, i = len(match.group(1)), [], i + 1
        while i < len(lines):
            line = lines[i]
            # A blank line may be inside the block; a non-blank line at or left of the directive ends it
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
            i += 1
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return blocks


def _documented_golden_path():
    """The one documented snippet that runs the full find_features -> predict_samples spine."""
    hits = [b for b in _rst_python_blocks(_DOC_PAGE)
            if "ap.find_features(" in b and "ap.predict_samples(" in b]
    if len(hits) != 1:
        raise AssertionError(f"expected exactly one golden-path code block in {_DOC_PAGE}, "
                             f"found {len(hits)}")
    return hits[0] + "\n"


def _explicit_models(random_state=None):
    """The documented default comparison set, constructed by hand."""
    return {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(random_state=random_state),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
    }


@lru_cache(maxsize=1)
def _feature_matrix():
    """The explicit feature matrix ``X`` that predict_samples rebuilds internally."""
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    return sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)


def _explicit_chain(random_state=None):
    """The explicit feature_matrix -> estimator -> cross_validate chain predict_samples mirrors."""
    X = _feature_matrix()
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


def _obj_state(obj):
    """The comparable state of an arbitrary object, or ``None`` if it carries none.

    ``__getstate__`` (defined on every object since Python 3.11) also reaches the state of the
    opaque Cython objects inside a fitted tree ensemble (``DecisionTreeClassifier.tree_``), which
    has no ``__dict__``.
    """
    state = obj.__getstate__() if hasattr(obj, "__getstate__") else None
    if isinstance(state, dict):
        return state
    return dict(vars(obj)) if hasattr(obj, "__dict__") else None


def _state_diff(a, b, path="root"):
    """First path at which two fitted estimators differ, or ``None`` when identical.

    Walks *learned* state recursively (every attribute, including nested ``estimators_`` lists and
    the arrays inside each fitted tree) and compares arrays exactly. Type, shape and dtype must
    match as well, so a numerically equal but differently typed fit is still a difference.
    """
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{path}: array {a.shape}/{a.dtype} != {b.shape}/{b.dtype}"
        equal_nan = a.dtype.kind in "fc"
        if not np.array_equal(a, b, equal_nan=equal_nan):
            return f"{path}: array values differ"
        return None
    if isinstance(a, (str, bytes, bool, int, type(None))):
        return None if a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, float):
        if np.isnan(a) and np.isnan(b):
            return None
        return None if a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _state_diff(x, y, f"{path}.{i}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(map(str, set(a) ^ set(b)))} differ"
        for key in sorted(a, key=repr):
            diff = _state_diff(a[key], b[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    state_a, state_b = _obj_state(a), _obj_state(b)
    if state_a is None or state_b is None:
        return None if a == b else f"{path}: {a!r} != {b!r}"
    return _state_diff(state_a, state_b, path)


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


# The golden path as a user would write it, read from the documentation page it is displayed on.
_GOLDEN_PATH = _documented_golden_path()


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
            # The whole learned state, not only the predictions it happens to produce here
            assert _state_diff(pred, est) is None, f"{name}: {_state_diff(pred, est)}"
            assert np.array_equal(pred.predict(X), est.predict(X))
            assert np.array_equal(pred.predict_proba(X), est.predict_proba(X))

    def test_parity_detects_a_different_chain(self):
        """Guard against a vacuous anchor: a different CV fold count must break equality."""
        _, _, df_expected = _explicit_chain(random_state=0)
        _, _, df_eval = ap.predict_samples(list_df_feat=df_feat, df_seq=df_seq, labels=labels,
                                           n_cv=4, plot=False, random_state=0)
        assert not df_eval.equals(df_expected)

    def test_state_diff_detects_a_different_fit(self):
        """Guard against a vacuous comparator: an equally parametrized but different fit fails."""
        X = _feature_matrix()
        ref = RandomForestClassifier(random_state=0).fit(X, labels)
        other = RandomForestClassifier(random_state=1).fit(X, labels)
        # Same params and same type, so only the learned state can tell the two fits apart
        other.set_params(random_state=0)
        assert type(other) is type(ref) and other.get_params() == ref.get_params()
        assert _state_diff(ref, other) is not None
        assert _state_diff(ref, copy.deepcopy(ref)) is None

    def test_state_diff_walks_nested_estimator_state(self):
        """A difference hidden inside one tree of an ensemble is still caught."""
        X = _feature_matrix()
        ref = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, labels)
        swapped = copy.deepcopy(ref)
        swapped.estimators_[-1] = copy.deepcopy(ref.estimators_[0])
        diff = _state_diff(ref, swapped)
        assert diff is not None and "estimators_" in diff
        truncated = copy.deepcopy(ref)
        truncated.estimators_ = truncated.estimators_[:-1]
        assert "estimators_" in _state_diff(ref, truncated)


class TestGoldenPathErgonomics:
    """The load -> find_features -> predict_samples path stays within the statement budget."""

    def test_statement_budget(self):
        n = _count_statements(_GOLDEN_PATH)
        assert n <= _MAX_STATEMENTS, (f"golden path needs {n} statements, "
                                      f"exceeding the {_MAX_STATEMENTS}-statement budget")

    def test_path_covers_the_spine(self):
        assert {"load_dataset", "find_features", "predict_samples"} <= _called_names(_GOLDEN_PATH)

    def test_snippet_comes_from_the_documentation_page(self):
        """The budget is measured on the snippet the Golden Pipelines page displays."""
        assert _DOC_PAGE.is_file()
        assert _GOLDEN_PATH in [b + "\n" for b in _rst_python_blocks(_DOC_PAGE)]

    def test_counter_counts_nested_statements(self):
        assert _count_statements("a = 1\nb = 2") == 2
        assert _count_statements("a = 1; b = 2") == 2
        assert _count_statements("if True:\n    a = 1\n    b = 2") == 3

    @pytest.mark.slow
    def test_path_runs_and_reaches_cv_score(self):
        namespace = {}
        exec(compile(_GOLDEN_PATH, "<golden_path>", "exec"), namespace)
        score = namespace["score"]
        assert isinstance(namespace["predictors"], dict) and len(namespace["predictors"]) > 0
        assert np.isfinite(score) and 0.0 <= score <= 1.0
