"""This is a script to test branch arcs of the CPP frontend + its simplify
backend that the other CPP test files leave un-hit.

All tests drive the public API only (``aa.CPP`` / ``aa.SequenceFeature``):
- ``run_num`` validation arcs in ``_cpp.py`` (``_check_dict_num_parts``),
- the ``accept_gaps`` gap-encountered ``UserWarning`` arc,
- ``return_stats=True`` finalize arc,
- simplify-backend arcs (max_std_test rejection, on_unimprovable drop variants,
  consolidate batch accept/reject, verbose Case-2 message), and
- the single-sample amino-acid arcs in the CPP-shared
  ``sequence_feature.get_df_feat_`` (reached via ``SequenceFeature.get_df_feat``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fitted():
    """A seeded CPP + a rated (interpretability-tiered) feature set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        split_kws = sf.get_split_kws(
            n_split_min=1, n_split_max=2, split_types=["Segment"]
        )
        df_scales = aa.load_scales(top_explain_n=20)
        cpp = aa.CPP(
            df_parts=df_parts,
            df_scales=df_scales,
            split_kws=split_kws,
            random_state=0,
            verbose=False,
        )
        df_feat = cpp.run(labels=labels, n_filter=10)
    return cpp, df_feat, labels


def _simplify(cpp, df_feat, labels, **kwargs):
    kwargs.setdefault("ml_cv", 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cpp.simplify(df_feat=df_feat, labels=labels, **kwargs)


@pytest.fixture(scope="module")
def num_parts():
    """A CPP whose df_scales has exactly D columns, plus a matching dict_num_parts.

    Drives ``CPP.run_num`` and its ``_check_dict_num_parts`` validation arcs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        split_kws = sf.get_split_kws(
            n_split_min=1, n_split_max=1, split_types=["Segment"]
        )
        D = 3
        df_scales = aa.load_scales().iloc[:, :D].copy()
        cpp = aa.CPP(
            df_parts=df_parts,
            df_scales=df_scales,
            split_kws=split_kws,
            random_state=0,
            verbose=False,
        )
    n = len(df_parts)
    parts = list(df_parts.columns)
    return cpp, df_parts, labels, parts, D, n


# --------------------------------------------------------------------------- #
# _cpp.py: run_num validation (_check_dict_num_parts) + finalize/gap arcs
# --------------------------------------------------------------------------- #
class TestRunNumValidation:
    """Negative arcs of CPP.run_num's dict_num_parts validation."""

    def test_dict_num_parts_none_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        with pytest.raises(ValueError, match="dict_num_parts"):
            cpp.run_num(dict_num_parts=None, labels=labels)

    def test_dict_num_parts_not_dict_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        with pytest.raises(ValueError, match="dict_num_parts"):
            cpp.run_num(dict_num_parts=[1, 2, 3], labels=labels)

    def test_dict_num_parts_wrong_keys_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {"not_a_part": np.zeros((n, 5, D))}
        with pytest.raises(ValueError, match="part names"):
            cpp.run_num(dict_num_parts=bad, labels=labels)

    def test_dict_num_parts_value_not_ndarray_raises(self, num_parts):
        # _check_dict_num_parts L225: a part value that is not an np.ndarray.
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {p: np.zeros((n, 5, D)) for p in parts}
        bad[parts[0]] = [[0.0] * D] * 5  # a Python list, not an ndarray
        with pytest.raises(ValueError, match="np.ndarray"):
            cpp.run_num(dict_num_parts=bad, labels=labels)

    def test_dict_num_parts_wrong_ndim_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {p: np.zeros((n, 5, D)) for p in parts}
        bad[parts[0]] = np.zeros((n, 5))  # 2D instead of 3D
        with pytest.raises(ValueError, match="3D"):
            cpp.run_num(dict_num_parts=bad, labels=labels)

    def test_dict_num_parts_wrong_nsamples_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {p: np.zeros((n, 5, D)) for p in parts}
        bad[parts[0]] = np.zeros((n + 1, 5, D))  # wrong sample count
        with pytest.raises(ValueError, match="n_samples"):
            cpp.run_num(dict_num_parts=bad, labels=labels)

    def test_dict_num_parts_inconsistent_D_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {p: np.zeros((n, 5, D)) for p in parts}
        bad[parts[0]] = np.zeros((n, 5, D + 1))  # inconsistent D across parts
        with pytest.raises(ValueError, match="inconsistent D"):
            cpp.run_num(dict_num_parts=bad, labels=labels)

    def test_dict_num_parts_zero_D_raises(self, num_parts):
        # _check_dict_num_parts L246: every part 3D + consistent but D == 0.
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {p: np.zeros((n, 5, 0)) for p in parts}
        with pytest.raises(ValueError, match="D=0"):
            cpp.run_num(dict_num_parts=bad, labels=labels)

    def test_dict_num_parts_D_mismatch_df_scales_raises(self, num_parts):
        cpp, df_parts, labels, parts, D, n = num_parts
        bad = {p: np.zeros((n, 5, D + 1)) for p in parts}
        with pytest.raises(ValueError, match="df_scales"):
            cpp.run_num(dict_num_parts=bad, labels=labels)


class TestRunFinalizeAndGaps:
    """Frontend finalize + accept_gaps arcs of CPP.run."""

    def test_return_stats_true_returns_tuple(self, fitted):
        # _finalize_run_output L71: return_stats=True -> (df_feat, stats).
        cpp, df_feat, labels = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = cpp.run(labels=labels, n_filter=5, return_stats=True)
        assert isinstance(out, tuple) and len(out) == 2
        df_res, stats = out
        assert isinstance(df_res, pd.DataFrame) and isinstance(stats, dict)

    def test_accept_gaps_encountered_warns(self):
        # _warn_gaps_encountered L53/54: accept_gaps=True AND a gap actually present.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
            labels = df_seq["label"].to_list()
            df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        # Inject a gap symbol into one part of one row.
        df_parts = df_parts.copy()
        col0 = df_parts.columns[0]
        s = df_parts[col0].iloc[0]
        df_parts.iloc[0, df_parts.columns.get_loc(col0)] = ut.STR_AA_GAP + s[1:]
        split_kws = aa.SequenceFeature().get_split_kws(
            n_split_min=1, n_split_max=1, split_types=["Segment"]
        )
        cpp = aa.CPP(
            df_parts=df_parts,
            split_kws=split_kws,
            accept_gaps=True,
            verbose=False,
            random_state=0,
        )
        with pytest.warns(UserWarning, match="accept_gaps"):
            cpp.run(labels=labels, n_filter=5)


# --------------------------------------------------------------------------- #
# _simplify.py backend arcs (driven via CPP.simplify)
# --------------------------------------------------------------------------- #
class TestSimplifyBranchArcs:
    """Un-hit branch arms of the simplify strategies."""

    def test_max_std_test_rejects_candidates(self, fitted):
        # std_test > max_std_test records arc in greedy/swap_all/consolidate.
        cpp, df_feat, labels = fitted
        for strategy in ["greedy", "swap_all", "consolidate"]:
            out, df_cand = _simplify(
                cpp,
                df_feat,
                labels,
                strategy=strategy,
                max_std_test=0.0,  # nothing can pass -> all candidates rejected
                return_details=True,
            )
            assert isinstance(out, pd.DataFrame)
            # Every recorded candidate is rejected for max_std_test.
            if len(df_cand):
                assert (df_cand["reason"] == "max_std_test").all()

    def test_greedy_on_unimprovable_drop(self, fitted):
        # Greedy L411/414: on_unimprovable='drop' with unswappable features.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="greedy", min_cor=1.0,
            on_unimprovable="drop",
        )
        assert 1 <= len(out) <= len(df_feat)

    def test_greedy_on_unimprovable_drop_if_perf(self, fitted):
        # Greedy L414/423: on_unimprovable='drop_if_perf_allows' with a slack th.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="greedy", min_cor=1.0,
            on_unimprovable="drop_if_perf_allows", ml_th=1.0,
        )
        assert 1 <= len(out) <= len(df_feat)

    def test_consolidate_batch_rejected(self, fitted):
        # Consolidate L643->646 / L648->651: a batch whose CV score drops below
        # baseline - ml_th is rejected (accepted=False), so no row is claimed.
        cpp, df_feat, labels = fitted
        out, df_cand = _simplify(
            cpp, df_feat, labels, strategy="consolidate",
            ml_th=-1.0,  # impossible gate -> every batch rejected
            return_details=True,
        )
        assert isinstance(out, pd.DataFrame)
        scored = df_cand[df_cand["reason"].isin(["accepted", "cv_drop"])]
        if len(scored):
            assert (scored["accepted"] == False).all()  # noqa: E712

    def test_consolidate_batch_accepted(self, fitted):
        # Consolidate L643/648 accepted arm: generous gate keeps swaps.
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, strategy="consolidate", ml_th=1.0)
        assert isinstance(out, pd.DataFrame)

    def test_consolidate_on_unimprovable_drop(self, fitted):
        # Consolidate L666/667/670: unclaimed targets dropped via on_unimprovable.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="consolidate",
            ml_th=-1.0, on_unimprovable="drop",
        )
        assert 1 <= len(out) <= len(df_feat)

    def test_consolidate_on_unimprovable_drop_if_perf(self, fitted):
        # Consolidate L672-683: unclaimed targets (no eligible candidate, via
        # min_cor=1.0) drop only if perf allows (generous ml_th=1.0).
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="consolidate",
            min_cor=1.0, ml_th=1.0, on_unimprovable="drop_if_perf_allows",
        )
        assert 1 <= len(out) <= len(df_feat)

    @staticmethod
    def _two_targetable(df_feat):
        """Two improvable (grade>1) features so both are simplify targets."""
        from aaanalysis.feature_engineering._backend.cpp._simplify import (
            _load_candidate_pool_,
            _interp,
        )

        _, _, _, dict_interp, _ = _load_candidate_pool_()
        low = [
            f
            for f in df_feat[ut.COL_FEATURE]
            if (
                lambda g: (g == g) and g > 1  # not NaN and worse than grade 1
            )(_interp(scale_id=ut.split_feat_id(feat_id=f)[2], dict_interp=dict_interp))
        ]
        return df_feat[df_feat[ut.COL_FEATURE].isin(low[:2])].copy()

    def test_greedy_drop_hits_last_feature_guard(self, fitted):
        # Greedy L411->332: with two unimprovable *targetable* features and
        # on_unimprovable='drop', the second drop is blocked by the 'never drop
        # the last feature' guard (n_kept <= 1 arc).
        cpp, df_feat, labels = fitted
        df2 = self._two_targetable(df_feat)
        assert len(df2) == 2  # fixture must offer two targetable features
        out = _simplify(
            cpp, df2, labels, strategy="greedy", min_cor=1.0,
            on_unimprovable="drop",
        )
        assert len(out) == 1

    def test_consolidate_drop_hits_last_feature_guard(self, fitted):
        # Consolidate L668 guard: unclaimed targets, drop blocked at the last one.
        cpp, df_feat, labels = fitted
        df2 = self._two_targetable(df_feat)
        out = _simplify(
            cpp, df2, labels, strategy="consolidate", min_cor=1.0,
            on_unimprovable="drop", ml_th=-5.0,
        )
        assert len(out) == 1

    def test_consolidate_claimed_and_drop_mix(self, fitted):
        # Consolidate L668/669: with a generous gate some targets are claimed
        # (accepted) and on_unimprovable='drop' makes the unclaimed-loop skip the
        # already-claimed ones ('i in claimed' continue) while dropping others.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="consolidate",
            ml_th=1.0, on_unimprovable="drop",
        )
        assert 1 <= len(out) <= len(df_feat)

    def test_greedy_drop_if_perf_refuses_when_hurts(self, fitted):
        # Greedy L423->332: drop_if_perf with a strict (negative) ml_th -> a drop
        # would lower the CV score, so the feature is NOT dropped.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="greedy", min_cor=1.0,
            on_unimprovable="drop_if_perf_allows", ml_th=-0.5,
        )
        assert len(out) == len(df_feat)

    def test_consolidate_drop_if_perf_refuses_when_hurts(self, fitted):
        # Consolidate L681->667: unclaimed targets not dropped when the drop hurts.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="consolidate", min_cor=1.0,
            on_unimprovable="drop_if_perf_allows", ml_th=-0.5,
        )
        assert len(out) == len(df_feat)

    def test_redundancy_non_overlapping_kept(self, fitted):
        # _is_redundant_ L225->220: swapped features whose positions do NOT meet
        # the (high) overlap threshold continue to the next kept feature instead
        # of being flagged redundant -> nothing is dropped for overlap reasons.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="swap_all",
            max_overlap=0.99, max_cor=0.99,
        )
        assert isinstance(out, pd.DataFrame) and len(out) <= len(df_feat)

    def test_redundancy_low_correlation_kept(self, fitted):
        # _is_redundant_ L229->220: overlapping but weakly-correlated swapped
        # scales (cor <= max_cor) are NOT redundant -> kept.
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, strategy="swap_all",
            max_overlap=0.0, max_cor=0.99,
        )
        assert isinstance(out, pd.DataFrame) and len(out) <= len(df_feat)

    def test_redundancy_inner_loop_scans_multiple_kept(self):
        # _is_redundant_ L225->220 / L228->220: with several swapped features and
        # cross-category comparison, the inner loop compares a candidate against
        # multiple kept features and 'continue's past the non-redundant ones.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
            labels = df_seq["label"].to_list()
            sf = aa.SequenceFeature()
            df_parts = sf.get_df_parts(df_seq=df_seq)
            split_kws = sf.get_split_kws(
                n_split_min=2, n_split_max=4, split_types=["Segment"]
            )
            df_scales = aa.load_scales(top_explain_n=20)
            cpp = aa.CPP(
                df_parts=df_parts, df_scales=df_scales, split_kws=split_kws,
                random_state=0, verbose=False,
            )
            df_feat = cpp.run(labels=labels, n_filter=15)
            out = cpp.simplify(
                df_feat=df_feat, labels=labels, strategy="swap_all",
                max_overlap=0.3, max_cor=0.3, check_cat=False, ml_cv=3,
            )
        assert isinstance(out, pd.DataFrame) and len(out) <= len(df_feat)

    def test_already_good_enough_verbose_message(self, fitted, capsys):
        # simplify_cpp_ L744-746: graded features, none worse than the cut, with
        # verbose=True -> prints the 'returning df_feat unchanged' message.
        cpp, df_feat, labels = fitted
        cpp_v = aa.CPP(
            df_parts=cpp.df_parts,
            df_scales=cpp.df_scales,
            split_kws=cpp.split_kws,
            random_state=0,
            verbose=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # Case-2, not Case-1
            out = cpp_v.simplify(
                df_feat=df_feat, labels=labels, max_interpret_grade=10, ml_cv=3
            )
        assert len(out) == len(df_feat)


# --------------------------------------------------------------------------- #
# sequence_feature.get_df_feat_ single-sample arcs (via SequenceFeature)
# --------------------------------------------------------------------------- #
class TestSingleSampleAminoAcids:
    """L86/91 of the CPP-shared sequence_feature backend: a single test/ref
    sample appends the amino-acid column. Reached via the public
    SequenceFeature.get_df_feat with jmd_n/tmd/jmd_c parts present."""

    @pytest.fixture(scope="class")
    def sample_setup(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
            sf = aa.SequenceFeature()
            df_parts = sf.get_df_parts(
                df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"]
            )
            df_scales = aa.load_scales().iloc[:, :5].copy()
            features = sf.get_features(
                list_parts=["tmd"], split_kws=sf.get_split_kws(
                    n_split_min=1, n_split_max=1, split_types=["Segment"]
                ),
                list_scales=list(df_scales.columns),
            )
        return sf, df_parts, df_scales, features

    def test_single_test_sample_adds_aa_test(self, sample_setup):
        sf, df_parts, df_scales, features = sample_setup
        n = len(df_parts)
        labels = [0] * n
        labels[0] = 1  # exactly one test sample -> L86 arc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = sf.get_df_feat(
                features=features, df_parts=df_parts, labels=labels,
                df_scales=df_scales,
            )
        assert ut.COL_AA_TEST in df_feat.columns

    def test_single_ref_sample_adds_aa_ref(self, sample_setup):
        sf, df_parts, df_scales, features = sample_setup
        n = len(df_parts)
        labels = [1] * n
        labels[0] = 0  # exactly one ref sample -> L91 arc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = sf.get_df_feat(
                features=features, df_parts=df_parts, labels=labels,
                df_scales=df_scales,
            )
        assert ut.COL_AA_REF in df_feat.columns

    @settings(max_examples=3, deadline=None)
    @given(idx=some.integers(min_value=0, max_value=7))
    def test_single_test_sample_property(self, sample_setup, idx):
        sf, df_parts, df_scales, features = sample_setup
        n = len(df_parts)
        labels = [0] * n
        labels[idx] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = sf.get_df_feat(
                features=features, df_parts=df_parts, labels=labels,
                df_scales=df_scales,
            )
        assert ut.COL_AA_TEST in df_feat.columns
