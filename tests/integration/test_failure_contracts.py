"""This is a script to test the failure contract of the golden pipelines + CPP/AAPred path.

Integration tier. A coding agent or a scripted caller mostly meets the golden pipelines
(``ap.find_features``, ``ap.predict_samples``, ``ap.explain_features``) and the core
CPP -> ``AAPred`` path on the *unhappy* path: missing labels, malformed sequences, a
feature matrix that does not match the model, an optional ``pro`` dependency that is not
installed, too few positives to split, an unfitted component. This suite pins that
**failure contract** so it stays stable across releases:

  * each invalid invocation raises a **bare** ``ValueError`` / ``RuntimeError`` (or, for the
    missing-``pro`` case, ``ImportError`` with an install hint) - never a bespoke subclass;
  * the message is **self-explaining** (names the offending input in the package's own voice),
    so a caller can tell "I called it wrong" from "the library broke" without a private
    traceback detail;
  * everything is driven through the **public API only**, so the suite doubles as a smoke of
    the error surface a downstream (e.g. ProtXplain) adapter maps to structured errors.

These are **composition failures** (the higher-tier job): labels-vs-samples misalignment, a
schema/shape mismatch at a seam, an unfitted component. Per-parameter invalid-input sweeps
(``model='xyz'``, out-of-range scalars) stay at the unit tier. The missing-``pro`` stub is
also asserted from an installed wheel by ``tests/_check_public_api_packaged.py``.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

import aaanalysis as aa
from aaanalysis import pipe as ap
from tests import _pipeline

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

pytestmark = pytest.mark.integration

# ``ap.explain_features`` needs SHAP (the ``[pro]`` extra). When absent it degrades to a
# ``missing_feature_stub`` raising ImportError; its real composition failures can only be
# exercised where the extra is installed.
_HAS_PRO = importlib.util.find_spec("shap") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_contract(exc_type, match, call):
    """A failure invocation raises exactly ``exc_type`` (no subclass) with a matching message.

    ``pytest.raises`` matches subclasses too; asserting ``type(...) is exc_type`` pins the
    *bare* ``ValueError`` / ``RuntimeError`` contract (no bespoke exception hierarchy).
    """
    with pytest.raises(exc_type, match=match) as excinfo:
        call()
    assert type(excinfo.value) is exc_type, (
        f"expected a bare {exc_type.__name__}, got {type(excinfo.value).__name__}")
    return excinfo.value


@pytest.fixture(scope="module")
def fitted_aapred(pipeline):
    """A once-fitted ``AAPred`` on the seeded feature matrix (reused by the scoring tests)."""
    return aa.AAPred().fit(X=np.asarray(pipeline["X"]), labels=pipeline["labels"])


# ---------------------------------------------------------------------------
# ap.find_features
# ---------------------------------------------------------------------------
class TestFindFeaturesFailureContract:
    """``ap.find_features`` rejects misaligned/degenerate labels and malformed df_seq early."""

    def test_labels_none(self, pipeline):
        _assert_contract(ValueError, r"'labels' should not be None",
                         lambda: ap.find_features(labels=None, df_seq=pipeline["df_seq"], plot=False))

    def test_labels_single_class(self, pipeline):
        # Composition failure: one class cannot define a test-vs-reference contrast. Without the
        # frontend guard this surfaced as an opaque "produced no valid configurations" RuntimeError.
        n = len(pipeline["df_seq"])
        _assert_contract(ValueError, r"should contain more than one different value",
                         lambda: ap.find_features(labels=[1] * n, df_seq=pipeline["df_seq"], plot=False))

    def test_labels_length_mismatch(self, pipeline):
        _assert_contract(ValueError, r"'labels' \(n=\d+\) should contain \d+ values",
                         lambda: ap.find_features(labels=pipeline["labels"][:-1],
                                                  df_seq=pipeline["df_seq"], plot=False))

    def test_malformed_df_seq(self, pipeline):
        # Malformed input frame: neither the required 'entry' nor a recognized sequence format.
        bad = pd.DataFrame({"foo": [1, 2, 3, 4]})
        _assert_contract(ValueError, r"'df_seq'",
                         lambda: ap.find_features(labels=[0, 1, 0, 1], df_seq=bad, plot=False))


# ---------------------------------------------------------------------------
# ap.predict_samples
# ---------------------------------------------------------------------------
class TestPredictSamplesFailureContract:
    """``ap.predict_samples`` rejects an empty/invalid feature set and misaligned labels."""

    def test_empty_feature_set(self, pipeline):
        _assert_contract(ValueError, r"'list_df_feat' should contain at least one feature DataFrame",
                         lambda: ap.predict_samples(list_df_feat=[], df_seq=pipeline["df_seq"],
                                                    labels=pipeline["labels"], plot=False))

    def test_feature_schema_mismatch(self, pipeline):
        # Schema mismatch at the df_feat -> model seam: a frame missing the canonical CPP columns.
        bad_feat = pd.DataFrame({"foo": [1, 2, 3]})
        _assert_contract(ValueError, r"'df_feat' is missing required columns",
                         lambda: ap.predict_samples(list_df_feat=[bad_feat], df_seq=pipeline["df_seq"],
                                                    labels=pipeline["labels"], plot=False))

    def test_labels_length_mismatch(self, pipeline):
        _assert_contract(ValueError, r"'labels' \(n=\d+\) should contain \d+ values",
                         lambda: ap.predict_samples(list_df_feat=[pipeline["df_feat"]],
                                                    df_seq=pipeline["df_seq"],
                                                    labels=pipeline["labels"][:-1], plot=False))

    def test_labels_single_class(self, pipeline):
        n = len(pipeline["df_seq"])
        _assert_contract(ValueError, r"should contain more than one different value",
                         lambda: ap.predict_samples(list_df_feat=[pipeline["df_feat"]],
                                                    df_seq=pipeline["df_seq"],
                                                    labels=[1] * n, plot=False))


# ---------------------------------------------------------------------------
# ap.explain_features (pro-gated: SHAP)
# ---------------------------------------------------------------------------
class TestExplainFeaturesFailureContract:
    """``ap.explain_features`` degrades to an install hint without ``[pro]``; else fails early."""

    @pytest.mark.skipif(_HAS_PRO, reason="shap installed: the pro stub is not active "
                                         "(the installed-wheel contract is covered by the packaging gate)")
    def test_missing_pro_dependency_install_hint(self, pipeline):
        # Without shap, ap.explain_features is a missing_feature_stub: calling it raises ImportError
        # naming the extra to install. This is the contract a coding agent hits on a base install.
        exc = _assert_contract(ImportError, r"aaanalysis\[pro\]",
                               lambda: ap.explain_features(df_feat=pipeline["df_feat"],
                                                           df_seq=pipeline["df_seq"],
                                                           labels=pipeline["labels"], plot=False))
        assert "explain_features" in str(exc)

    @pytest.mark.skipif(not _HAS_PRO, reason="requires the [pro] extra (shap) for the real function")
    def test_feature_schema_mismatch(self, pipeline):
        bad_feat = pd.DataFrame({"foo": [1, 2, 3]})
        _assert_contract(ValueError, r"'df_feat' is missing required columns",
                         lambda: ap.explain_features(df_feat=bad_feat, df_seq=pipeline["df_seq"],
                                                     labels=pipeline["labels"], plot=False))

    @pytest.mark.skipif(not _HAS_PRO, reason="requires the [pro] extra (shap) for the real function")
    def test_labels_single_class(self, pipeline):
        n = len(pipeline["df_seq"])
        _assert_contract(ValueError, r"should contain more than one different value",
                         lambda: ap.explain_features(df_feat=pipeline["df_feat"], df_seq=pipeline["df_seq"],
                                                     labels=[0] * n, plot=False))


# ---------------------------------------------------------------------------
# CPP construction / run (the interpretable core the pipelines wrap)
# ---------------------------------------------------------------------------
class TestCPPFailureContract:
    """``CPP`` rejects empty/invalid parts, uncovered residues, and degenerate labels."""

    def test_empty_df_parts(self):
        _assert_contract(ValueError, r"'df_parts' should not be empty",
                         lambda: aa.CPP(df_parts=pd.DataFrame()))

    def test_invalid_part_name(self, pipeline):
        # Invalid 'parts' requested from get_df_parts (the parts a caller would feed CPP).
        _assert_contract(ValueError, r"\['not_a_part'\] not valid part",
                         lambda: aa.SequenceFeature().get_df_parts(df_seq=pipeline["df_seq"],
                                                                   list_parts=["not_a_part"]))

    def test_malformed_sequence_uncovered_residue(self, pipeline):
        # Malformed sequence at the parts <-> scales seam: a residue no scale covers.
        df_parts_bad = pipeline["df_parts"].copy()
        first = str(df_parts_bad.iloc[0, 0])
        df_parts_bad.iloc[0, 0] = "Z" + first[1:]
        _assert_contract(ValueError, r"Not all characters in sequences from 'df_parts' are covered",
                         lambda: aa.CPP(df_parts=df_parts_bad, df_scales=pipeline["df_scales"],
                                        verbose=False).run(labels=pipeline["labels"], n_jobs=1))

    def test_labels_single_class(self, pipeline):
        n = len(pipeline["df_parts"])
        _assert_contract(ValueError, r"should contain more than one different value",
                         lambda: aa.CPP(df_parts=pipeline["df_parts"], df_scales=pipeline["df_scales"],
                                        verbose=False).run(labels=[1] * n, n_jobs=1))


# ---------------------------------------------------------------------------
# AAPred (the prediction endpoint the CPP path feeds)
# ---------------------------------------------------------------------------
class TestAAPredFailureContract:
    """``AAPred`` rejects unfitted scoring, too-few-positives splits, and shape mismatch."""

    def test_predict_unfitted(self, pipeline):
        _assert_contract(ValueError, r"'AAPred' is not fitted",
                         lambda: aa.AAPred().predict(df_seq=pipeline["df_seq"]))

    def test_predict_proba_unfitted(self, pipeline):
        _assert_contract(ValueError, r"'AAPred' is not fitted",
                         lambda: aa.AAPred().predict_proba(X=np.asarray(pipeline["X"])))

    def test_too_few_positives_to_split(self, pipeline):
        # More CV folds than the smallest class can populate (the "too few positives to split" case).
        _assert_contract(ValueError, r"should not be greater than the smallest class count",
                         lambda: aa.AAPred().eval(X=np.asarray(pipeline["X"]),
                                                  labels=pipeline["labels"], n_cv=999))

    def test_predict_proba_feature_width_mismatch(self, pipeline, fitted_aapred):
        # Feature matrix / model shape mismatch on the matrix-scoring path: a self-explaining
        # message naming 'X', not scikit-learn's internal "<Estimator> is expecting M features".
        X_narrow = np.asarray(pipeline["X"])[:, :3]
        _assert_contract(ValueError, r"'X' n_features \(\d+\) should match the fitted model's n_features",
                         lambda: fitted_aapred.predict_proba(X=X_narrow))

    def test_eval_holdout_width_mismatch(self, pipeline):
        X = np.asarray(pipeline["X"])
        _assert_contract(ValueError, r"'X_holdout' n_features \(\d+\) should match 'X' n_features",
                         lambda: aa.AAPred().eval(X=X, labels=pipeline["labels"],
                                                  X_holdout=X[:, :3], labels_holdout=pipeline["labels"]))


# ---------------------------------------------------------------------------
# CPPGrid: eval before run (unfitted-component contract, RuntimeError side)
# ---------------------------------------------------------------------------
class TestCPPGridFailureContract:
    """``CPPGrid.eval`` before ``run`` is a runtime-state error (bare RuntimeError)."""

    def test_eval_before_run(self, pipeline):
        _assert_contract(RuntimeError, r"'eval' requires a prior 'run'",
                         lambda: aa.CPPGrid(df_seq=pipeline["df_seq"], labels=pipeline["labels"],
                                            verbose=False).eval())


# ---------------------------------------------------------------------------
# Contract hygiene (the KPIs, asserted, not just eyeballed)
# ---------------------------------------------------------------------------
class TestFailureContractHygiene:
    """The raised errors are bare builtins with self-explaining, input-naming messages."""

    def test_error_types_are_bare_builtins(self, pipeline):
        # No bespoke exception hierarchy: every contract failure is exactly one of these.
        cases = [
            lambda: ap.find_features(labels=[1] * len(pipeline["df_seq"]),
                                     df_seq=pipeline["df_seq"], plot=False),
            lambda: ap.predict_samples(list_df_feat=[], df_seq=pipeline["df_seq"],
                                       labels=pipeline["labels"], plot=False),
            lambda: aa.CPP(df_parts=pd.DataFrame()),
            lambda: aa.AAPred().predict_proba(X=np.asarray(pipeline["X"])),
            lambda: aa.CPPGrid(df_seq=pipeline["df_seq"], labels=pipeline["labels"],
                               verbose=False).eval(),
        ]
        for call in cases:
            with pytest.raises((ValueError, RuntimeError)) as excinfo:
                call()
            assert type(excinfo.value) in (ValueError, RuntimeError)

    def test_messages_name_the_offending_input(self, pipeline):
        # The message alone identifies the user error (no private traceback detail needed): it
        # quotes the offending input name and uses the canonical "should"/"requires" phrasing.
        with pytest.raises(ValueError) as excinfo:
            ap.predict_samples(list_df_feat=[pipeline["df_feat"]], df_seq=pipeline["df_seq"],
                               labels=pipeline["labels"][:-1], plot=False)
        msg = str(excinfo.value)
        assert "'labels'" in msg and "should contain" in msg
