"""This is a script to test CPP filter diagnostics (last_filter_stats_ / return_stats) and warnings.

Covers Stage-1 decisions D6 and D7:

* D6 — ``cpp.last_filter_stats_`` is a plain dict set after every ``run`` /
  ``run_num`` with keys ``n_candidates``, ``n_after_prefilter``,
  ``n_after_redundancy``, ``n_final``; ``return_stats=True`` returns
  ``(df_feat, stats)`` without changing the default single-DataFrame return.
* D7 — a ``RuntimeWarning`` fires when the filter removed too many features to
  reach ``n_filter`` (enough candidates existed), and a ``UserWarning`` fires
  when ``accept_gaps=True`` actually encounters a gap in ``df_parts``.
* D5b — a ``UserWarning`` fires when the configuration itself is too sparse to
  generate ``n_filter`` candidate features (``n_candidates < n_filter``); it is
  mutually exclusive with the D7 ``RuntimeWarning``.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

_STATS_KEYS = {"n_candidates", "n_after_prefilter", "n_after_redundancy", "n_final"}


# Helper functions
def _get_df_parts(n=10):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    sf = aa.SequenceFeature()
    return sf.get_df_parts(df_seq=df_seq)


def _labels(n=10):
    # DOM_GSEC is loaded deterministically (same first n per class), so labels
    # read here align row-for-row with _get_df_parts(n).
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return df_seq["label"].to_list()


def _get_run_num_args(n=6, d=4):
    """Build (df_parts, dict_num_parts, df_scales, df_cat, labels) for run_num."""
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3, "MKTAYIAKQRQISFVKSHFSRQ" * 3,
            "GAVLIMFWPSTCYNQDEKRHG" * 3, "LLLLLKKKKKDDDDDEEEEEAA" * 3,
            "WYWYWYWYWYWYWYWYWYWY" * 3, "QQQQQNNNNNSSSSSTTTTTT" * 3][:n]
    df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(n)], "sequence": seqs})
    df_seq["tmd_start"] = 11
    df_seq["tmd_stop"] = df_seq["sequence"].str.len() - 10
    rng = np.random.default_rng(0)
    dict_num = {e: rng.random((len(s), d)) for e, s in zip(df_seq["entry"], df_seq["sequence"])}
    nf = aa.NumericalFeature()
    df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num, jmd_n_len=10, jmd_c_len=10)
    df_scales = pd.DataFrame(rng.random((20, d)), index=list(ut.LIST_CANONICAL_AA),
                                 columns=[f"d{i}" for i in range(d)])
    df_cat = pd.DataFrame({"scale_id": [f"d{i}" for i in range(d)], "category": ["X"] * d,
                              "subcategory": ["x"] * d, "scale_name": [f"d{i}" for i in range(d)],
                              "scale_description": ["d"] * d})
    labels = [1] * (n // 2) + [0] * (n - n // 2)
    return df_parts, dict_num_parts, df_scales, df_cat, labels


def _gapped_df_parts():
    """df_parts whose default composite parts contain a gap (N-terminal pad)."""
    seq20 = "ACDEFGHIKLMNPQRSTVWY"
    df_seq = pd.DataFrame({"entry": ["S1", "S2", "S3", "S4"],
                           "sequence": [seq20, seq20[::-1], seq20, seq20[::-1]],
                           "tmd_start": [2, 2, 2, 2], "tmd_stop": [20, 20, 20, 20]})
    sf = aa.SequenceFeature()
    dp = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10)
    assert dp.apply(lambda c: c.str.contains("-")).to_numpy().any()
    return dp


class TestFilterStats:
    """Normal-case tests for last_filter_stats_ and return_stats (D6)."""

    def test_attr_is_none_before_run(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        assert cpp.last_filter_stats_ is None

    def test_attr_set_after_run(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        cpp.run(labels=_labels(), n_filter=20, n_jobs=1)
        assert isinstance(cpp.last_filter_stats_, dict)

    def test_attr_keys(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        cpp.run(labels=_labels(), n_filter=20, n_jobs=1)
        assert set(cpp.last_filter_stats_) == _STATS_KEYS

    def test_attr_values_are_ints(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        cpp.run(labels=_labels(), n_filter=20, n_jobs=1)
        assert all(isinstance(v, int) for v in cpp.last_filter_stats_.values())

    def test_default_return_is_dataframe(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        out = cpp.run(labels=_labels(), n_filter=20, n_jobs=1)
        assert isinstance(out, pd.DataFrame)

    def test_return_stats_true_returns_tuple(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        out = cpp.run(labels=_labels(), n_filter=20, n_jobs=1, return_stats=True)
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], pd.DataFrame) and isinstance(out[1], dict)

    def test_return_stats_dict_matches_attr(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        df_feat, stats = cpp.run(labels=_labels(), n_filter=20, n_jobs=1, return_stats=True)
        assert stats == cpp.last_filter_stats_

    def test_n_final_matches_df_len(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        df_feat = cpp.run(labels=_labels(), n_filter=15, n_jobs=1)
        assert cpp.last_filter_stats_["n_final"] == len(df_feat)

    def test_funnel_is_monotonic_nonincreasing(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        cpp.run(labels=_labels(), n_filter=20, n_jobs=1)
        s = cpp.last_filter_stats_
        assert s["n_candidates"] >= s["n_after_prefilter"] >= s["n_after_redundancy"] >= s["n_final"]

    def test_return_stats_invalid_type_raises(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        for bad in ["yes", 1, None, [], 0.5]:
            with pytest.raises(ValueError):
                cpp.run(labels=_labels(), n_filter=10, n_jobs=1, return_stats=bad)


class TestFilterStatsRunNum:
    """last_filter_stats_ / return_stats parity on the numerical path (D6)."""

    def test_run_num_attr_set(self):
        dp, dnp, dsc, dca, labels = _get_run_num_args()
        cpp = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca)
        cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1)
        assert set(cpp.last_filter_stats_) == _STATS_KEYS

    def test_run_num_default_return_is_dataframe(self):
        dp, dnp, dsc, dca, labels = _get_run_num_args()
        cpp = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca)
        out = cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1)
        assert isinstance(out, pd.DataFrame)

    def test_run_num_return_stats_tuple(self):
        dp, dnp, dsc, dca, labels = _get_run_num_args()
        cpp = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca)
        out = cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1, return_stats=True)
        assert isinstance(out, tuple) and isinstance(out[1], dict)
        assert set(out[1]) == _STATS_KEYS

    def test_run_num_return_stats_invalid_type_raises(self):
        dp, dnp, dsc, dca, labels = _get_run_num_args()
        cpp = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca)
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1, return_stats=bad)


class TestFilterWarnings:
    """Normal-case tests for the D5b + D7 warnings."""

    def test_shortfall_runtimewarning(self):
        # Enough candidates (45540) exist, but redundancy filtering leaves far
        # fewer than n_filter=5000 -> D7 RuntimeWarning (not the D5b sparse case).
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with pytest.warns(RuntimeWarning, match="n_filter"):
            cpp.run(labels=_labels(), n_filter=5000, n_jobs=1)

    def test_sparse_config_userwarning(self):
        # n_filter beyond what the config can ever generate -> D5b UserWarning.
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with pytest.warns(UserWarning, match="candidate features"):
            cpp.run(labels=_labels(), n_filter=10 ** 7, n_jobs=1)

    def test_sparse_config_not_runtimewarning(self):
        # D5b and D7 are mutually exclusive: the sparse case must NOT raise the
        # D7 RuntimeWarning.
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.warns(UserWarning, match="candidate features"):
                cpp.run(labels=_labels(), n_filter=10 ** 7, n_jobs=1)

    def test_no_shortfall_warning_when_enough(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Asking for few features cannot under-deliver; no RuntimeWarning.
            cpp.run(labels=_labels(), n_filter=5, n_jobs=1)

    def test_gap_userwarning(self):
        cpp = aa.CPP(df_parts=_gapped_df_parts(), accept_gaps=True)
        with pytest.warns(UserWarning, match="accept_gaps"):
            cpp.run(labels=[1, 1, 0, 0], n_filter=3, n_jobs=1)

    def test_no_gap_warning_when_accept_gaps_false_and_no_gap(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cpp.run(labels=_labels(), n_filter=5, n_jobs=1)


class TestFilterStatsComplex:
    """Combinations and edge interactions (D6 + D7 + D5b)."""

    def test_stats_consistent_across_repeat_runs(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        s1 = cpp.run(labels=_labels(), n_filter=15, n_jobs=1, return_stats=True)[1]
        s2 = cpp.run(labels=_labels(), n_filter=15, n_jobs=1, return_stats=True)[1]
        assert s1 == s2

    def test_shortfall_warning_still_attaches_stats(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with pytest.warns(RuntimeWarning):
            df_feat, stats = cpp.run(labels=_labels(), n_filter=5000, n_jobs=1, return_stats=True)
        assert set(stats) == _STATS_KEYS
        assert stats["n_final"] == len(df_feat)

    def test_sparse_config_warning_still_attaches_stats(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        with pytest.warns(UserWarning, match="candidate features"):
            df_feat, stats = cpp.run(labels=_labels(), n_filter=10 ** 7, n_jobs=1, return_stats=True)
        assert set(stats) == _STATS_KEYS
        assert stats["n_final"] == len(df_feat)
        assert stats["n_candidates"] < 10 ** 7  # the config can't reach the requested n_filter

    def test_n_filter_caps_n_final(self):
        cpp = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        df_feat = cpp.run(labels=_labels(), n_filter=12, n_jobs=1)
        assert len(df_feat) <= 12

    def test_gap_warning_and_stats_together(self):
        cpp = aa.CPP(df_parts=_gapped_df_parts(), accept_gaps=True)
        with pytest.warns(UserWarning, match="accept_gaps"):
            df_feat, stats = cpp.run(labels=[1, 1, 0, 0], n_filter=3, n_jobs=1, return_stats=True)
        assert set(stats) == _STATS_KEYS

    def test_run_and_run_num_share_stats_schema(self):
        cpp_seq = aa.CPP(df_parts=_get_df_parts(), df_scales=aa.load_scales(top60_n=38))
        cpp_seq.run(labels=_labels(), n_filter=10, n_jobs=1)
        dp, dnp, dsc, dca, labels = _get_run_num_args()
        cpp_num = aa.CPP(df_parts=dp, df_scales=dsc, df_cat=dca)
        cpp_num.run_num(dict_num_parts=dnp, labels=labels, n_filter=3, n_jobs=1)
        assert set(cpp_seq.last_filter_stats_) == set(cpp_num.last_filter_stats_)
