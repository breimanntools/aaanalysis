"""This is a script to test residual dPULearn branch arms via the public API."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _X_labels(n_pos=4, n_unl=6, n_features=5, seed=0):
    """Deterministic PU-encoded feature matrix and labels (1=pos, 2=unl)."""
    rng = np.random.default_rng(seed)
    n = n_pos + n_unl
    X = rng.random((n, n_features))
    labels = np.array([1] * n_pos + [2] * n_unl)
    return X, labels


# I dPULearn.fit branch arms
class TestdPULearnFitBranch:
    """Cover the validation arms reachable through dPULearn.fit()."""

    @settings(max_examples=5, deadline=None)
    @given(metric=some.sampled_from(["bogus", "l1", "minkowski", "", "EUCLIDEAN"]))
    def test_invalid_metric(self, metric):
        # _dpulearn.py:22 — check_metric raises on unknown metric
        X, labels = _X_labels()
        with pytest.raises(ValueError, match="metric"):
            aa.dPULearn(random_state=0).fit(X=X, labels=labels, n_unl_to_neg=1, metric=metric)

    @settings(max_examples=5, deadline=None)
    @given(n_unl_to_neg=some.integers(min_value=7, max_value=20))
    def test_n_unl_to_neg_exceeds_unlabeled(self, n_unl_to_neg):
        # _dpulearn.py:29 — check_n_unl_to_neg: too few unlabeled samples
        X, labels = _X_labels(n_pos=4, n_unl=6)  # only 6 unlabeled
        with pytest.raises(ValueError, match="unlabeled"):
            aa.dPULearn(random_state=0).fit(X=X, labels=labels, n_unl_to_neg=n_unl_to_neg)

    @settings(max_examples=5, deadline=None)
    @given(n_components=some.integers(min_value=2, max_value=3))
    def test_n_components_int_branch(self, n_components):
        # _dpulearn.py:37 — check_n_components: int branch (number of PCs)
        X, labels = _X_labels(n_pos=5, n_unl=8, n_features=6)
        dpul = aa.dPULearn(random_state=0).fit(X=X, labels=labels, n_unl_to_neg=2,
                                               n_components=n_components)
        assert dpul.labels_ is not None
        assert 0 in set(dpul.labels_)

    @settings(max_examples=5, deadline=None)
    @given(n_components=some.sampled_from([0, -1, 1.5, 2.0, "x"]))
    def test_n_components_invalid_value(self, n_components):
        # _dpulearn.py:43-44 — check_n_components except-wrapper re-raises a friendly error
        X, labels = _X_labels(n_pos=4, n_unl=6, n_features=5)
        with pytest.raises(ValueError, match="n_components"):
            aa.dPULearn(random_state=0).fit(X=X, labels=labels, n_unl_to_neg=1,
                                            n_components=n_components)

    def test_n_components_too_large_for_X(self):
        # _dpulearn.py:52 — check_match_X_n_components: n_components >= min(n_feat, n_samp)
        X, labels = _X_labels(n_pos=3, n_unl=4, n_features=3)  # min dim = 3
        with pytest.raises(ValueError, match="n_components"):
            aa.dPULearn(random_state=0).fit(X=X, labels=labels, n_unl_to_neg=1,
                                            n_components=3)


# II dPULearn.compare_sets_negatives branch arm
class TestCompareSetsNegativesBranch:
    """Cover the list_labels/df_seq length-mismatch arm."""

    def test_df_seq_length_mismatch(self):
        # _dpulearn.py:61 — check_match_list_labels_df_seq mismatch
        list_labels = [[0, 1, 2, 0]]
        df_seq = pd.DataFrame({"entry": ["P1", "P2"],
                               "sequence": ["AAAA", "CCCC"]})
        with pytest.raises(ValueError, match="does not match"):
            aa.dPULearn.compare_sets_negatives(list_labels=list_labels, df_seq=df_seq)

    def test_upset_keep_non_neg(self):
        # dpul_compare_sets_neg.py:37->39 — return_upset_data=True, remove_non_neg=False
        list_labels = [[0, 1, 2, 0], [0, 2, 2, 1]]
        upset_data = aa.dPULearn.compare_sets_negatives(
            list_labels=list_labels, return_upset_data=True, remove_non_neg=False)
        assert upset_data is not None

    def test_df_seq_match_and_upset_remove_non_neg(self):
        # _dpulearn.py:60 (match ok) + dpul_compare_sets_neg.py:37
        # (return_upset_data=True with remove_non_neg=True)
        list_labels = [[0, 1, 2, 0], [0, 2, 2, 1]]
        df_seq = pd.DataFrame({"entry": ["P1", "P2", "P3", "P4"],
                               "sequence": ["AAAA", "CCCC", "DDDD", "EEEE"]})
        upset_data = aa.dPULearn.compare_sets_negatives(
            list_labels=list_labels, df_seq=df_seq,
            return_upset_data=True, remove_non_neg=True)
        assert upset_data is not None


# III dPULearn.eval branch arm (KLD with ground-truth negatives)
class TestEvalBranch:
    """Cover the X_neg + comp_kld arm building the KLD_NEG column."""

    def test_eval_with_X_neg_and_kld(self):
        # dpul_eval.py:119 — X_neg is not None and comp_kld -> append COL_AVG_KLD_NEG
        X, labels = _X_labels(n_pos=5, n_unl=8, n_features=5, seed=1)
        dpul = aa.dPULearn(random_state=0).fit(X=X, labels=labels, n_unl_to_neg=3)
        rng = np.random.default_rng(2)
        X_neg = rng.random((4, 5))
        df_eval = aa.dPULearn.eval(X=X, list_labels=[dpul.labels_],
                                   X_neg=X_neg, comp_kld=True)
        assert any("kld" in c.lower() and "neg" in c.lower() for c in df_eval.columns)
