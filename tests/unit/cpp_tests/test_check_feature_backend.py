"""This is a script to test the CPP feature-validation backend in
``feature_engineering/_backend/check_feature.py`` (check_split_kws / check_parts_len
and related df validators) — the error/warning branches not reached through the
SequenceFeature / CPP frontends with normal inputs.

Backend-direct (a narrow exception to frontend-driven testing) to reach the
internal validation branches deterministically; doubles as error-message coverage.
"""
import warnings
import copy

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.check_feature import (
    check_split_kws,
    check_parts_len,
    check_df_scales,
    check_match_df_scales_features,
    check_match_df_parts_features,
    check_match_df_parts_split_kws,
    check_match_df_scales_df_cat,
    check_match_df_parts_df_scales,
    check_df_cat,
    check_match_df_cat_features,
)

aa.options["verbose"] = False


def _valid_kws():
    return {
        "Segment": {"n_split_min": 1, "n_split_max": 15},
        "Pattern": {"len_max": 15, "n_max": 4, "n_min": 2, "steps": [3, 4]},
        "PeriodicPattern": {"steps": [3, 4]},
    }


class TestCheckSplitKws:
    def test_valid(self):
        assert check_split_kws(split_kws=_valid_kws()) is None

    def test_none_accepted(self):
        assert check_split_kws(split_kws=None) is None

    def test_not_dict(self):
        with pytest.raises(ValueError, match="dict"):
            check_split_kws(split_kws=[1, 2])

    def test_empty_dict(self):
        with pytest.raises(ValueError, match="not empty"):
            check_split_kws(split_kws={})

    def test_invalid_split_type(self):
        with pytest.raises(ValueError, match="invalid"):
            check_split_kws(split_kws={"NoSuchSplit": {}})

    def test_missing_args(self):
        with pytest.raises(ValueError, match="Missing required"):
            check_split_kws(split_kws={"Segment": {"n_split_min": 1}})

    def test_invalid_arg_name(self):
        kws = {"Segment": {"n_split_min": 1, "n_split_max": 2, "extra": 1}}
        with pytest.raises(ValueError, match="invalid"):
            check_split_kws(split_kws=kws)

    def test_wrong_arg_type(self):
        kws = {"Segment": {"n_split_min": "x", "n_split_max": 2}}
        with pytest.raises(ValueError, match="should be"):
            check_split_kws(split_kws=kws)

    def test_list_wrong_element_type(self):
        kws = {"PeriodicPattern": {"steps": [3, "x"]}}
        with pytest.raises(ValueError, match="int"):
            check_split_kws(split_kws=kws)

    def test_segment_min_gt_max(self):
        kws = {"Segment": {"n_split_min": 5, "n_split_max": 2}}
        with pytest.raises(ValueError, match="smaller"):
            check_split_kws(split_kws=kws)

    def test_pattern_n_min_gt_n_max(self):
        kws = {"Pattern": {"len_max": 15, "n_max": 2, "n_min": 5, "steps": [3, 4]}}
        with pytest.raises(ValueError, match="smaller or equal"):
            check_split_kws(split_kws=kws)

    def test_pattern_steps_empty(self):
        kws = {"Pattern": {"len_max": 15, "n_max": 4, "n_min": 2, "steps": []}}
        with pytest.raises(ValueError, match="non-empty"):
            check_split_kws(split_kws=kws)

    def test_pattern_steps_unsorted(self):
        kws = {"Pattern": {"len_max": 15, "n_max": 4, "n_min": 2, "steps": [4, 3]}}
        with pytest.raises(ValueError, match="ascending"):
            check_split_kws(split_kws=kws)

    def test_pattern_step_ge_len_max(self):
        kws = {"Pattern": {"len_max": 3, "n_max": 4, "n_min": 2, "steps": [5, 6]}}
        with pytest.raises(ValueError, match="greater than"):
            check_split_kws(split_kws=kws)

    def test_pattern_zero_yield_warns(self):
        # n_min*steps[0] > len_max -> warns (no valid splits)
        kws = {"Pattern": {"len_max": 4, "n_max": 4, "n_min": 2, "steps": [3]}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_split_kws(split_kws=kws)
        assert any("zero features" in str(x.message) for x in w)

    def test_periodicpattern_wrong_len(self):
        kws = {"PeriodicPattern": {"steps": [3]}}
        with pytest.raises(ValueError, match="exactly 2"):
            check_split_kws(split_kws=kws)

    def test_periodicpattern_unsorted(self):
        kws = {"PeriodicPattern": {"steps": [4, 3]}}
        with pytest.raises(ValueError, match="ascending"):
            check_split_kws(split_kws=kws)


class TestCheckPartsLen:
    def test_valid(self):
        out = check_parts_len(tmd_len=20, jmd_n_len=10, jmd_c_len=10)
        assert out is None or isinstance(out, tuple)

    def test_jmd_n_seq_len_mismatch(self):
        with pytest.raises(ValueError, match="jmd_n_len"):
            check_parts_len(tmd_len=3, jmd_n_len=2, jmd_c_len=3,
                            tmd_seq="AAA", jmd_n_seq="AAA", jmd_c_seq="AAA",
                            check_jmd_seq_len_consistent=True)

    def test_jmd_c_seq_len_mismatch(self):
        with pytest.raises(ValueError, match="jmd_c_len"):
            check_parts_len(tmd_len=3, jmd_n_len=3, jmd_c_len=2,
                            tmd_seq="AAA", jmd_n_seq="AAA", jmd_c_seq="AAA",
                            check_jmd_seq_len_consistent=True)

    def test_ext_len_exceeds_jmd_n(self):
        aa.options["ext_len"] = 10
        try:
            with pytest.raises(ValueError):
                check_parts_len(tmd_len=20, jmd_n_len=2, jmd_c_len=20)
        finally:
            aa.options["ext_len"] = 0

    def test_ext_len_exceeds_jmd_c(self):
        aa.options["ext_len"] = 10
        try:
            with pytest.raises(ValueError):
                check_parts_len(tmd_len=20, jmd_n_len=20, jmd_c_len=2)
        finally:
            aa.options["ext_len"] = 0


class TestCheckFeatureFinishers:
    """Skip / accept-none / multi-sequence branches in the remaining validators."""

    def test_df_scales_none_accept(self):
        assert check_df_scales(df_scales=None, accept_none=True) is None

    def test_match_df_scales_features_none_skip(self):
        from aaanalysis.feature_engineering._backend.check_feature import (
            check_match_df_scales_features as f)
        assert f(df_scales=None, features=["x"]) is None

    def test_df_cat_none_accept(self):
        assert check_df_cat(df_cat=None, accept_none=True) is None

    def test_match_df_cat_features_none_skip(self):
        assert check_match_df_cat_features(df_cat=None, features=["x"]) is None

    def test_match_df_cat_features_missing(self):
        df_cat = pd.DataFrame({ut.COL_SCALE_ID: ["S1"], ut.COL_CAT: ["c"],
                               ut.COL_SUBCAT: ["s"], ut.COL_SCALE_NAME: ["n"]})
        with pytest.raises(ValueError, match="missing in 'df_cat'"):
            check_match_df_cat_features(df_cat=df_cat, features=["TMD-Segment(1,1)-NOPE"])

    def test_match_df_parts_features_multi_too_short(self):
        # two too-short parts -> the multi-sequence error branch
        df_parts = pd.DataFrame({"tmd": ["AC", "AB"]})
        with pytest.raises(ValueError, match="too short"):
            check_match_df_parts_features(
                df_parts=df_parts, features=["TMD-Pattern(N,1,9)-S1"])


class TestCheckDfScalesAndMatches:
    def _df_scales(self):
        order = "ACDEFGHIKLMNPQRSTVWY"
        return pd.DataFrame({"S1": {a: float(i) for i, a in enumerate(order)},
                             "S2": {a: float(i) * 0.1 for i, a in enumerate(order)}})

    def test_df_scales_valid(self):
        assert check_df_scales(df_scales=self._df_scales()) is None

    def test_df_scales_none_not_accepted(self):
        with pytest.raises(ValueError):
            check_df_scales(df_scales=None, accept_none=False)

    def test_match_df_scales_features_missing(self):
        df_scales = self._df_scales()
        # feature references a scale not in df_scales -> mismatch
        with pytest.raises(ValueError):
            check_match_df_scales_features(
                df_scales=df_scales, features=["TMD-Segment(1,1)-NOPE"])

    def test_df_scales_duplicate_columns(self):
        order = "ACDEFGHIKLMNPQRSTVWY"
        df = pd.DataFrame(np.zeros((20, 2)), index=list(order), columns=["S1", "S1"])
        with pytest.raises(ValueError, match="duplicated"):
            check_df_scales(df_scales=df)

    def test_df_scales_duplicate_index(self):
        df = pd.DataFrame({"S1": [1.0, 2.0]}, index=["A", "A"])
        with pytest.raises(ValueError, match="unique"):
            check_df_scales(df_scales=df)

    def test_df_scales_non_numeric(self):
        # force object dtype so np.issubdtype reads it as non-number (a pandas
        # StringDtype would raise TypeError earlier instead of the ValueError branch)
        df = pd.DataFrame({"S1": [1.0, 2.0]}, index=["A", "C"])
        df["S1"] = df["S1"].astype(object)
        with pytest.raises(ValueError, match="numbers"):
            check_df_scales(df_scales=df)

    def test_df_scales_nan(self):
        df = pd.DataFrame({"S1": [1.0, np.nan]}, index=["A", "C"])
        with pytest.raises(ValueError, match="NaN"):
            check_df_scales(df_scales=df)


class TestCheckMatchPartsAndCat:
    def _df_parts(self, seqs):
        return pd.DataFrame({"tmd": list(seqs)})

    def test_match_df_parts_features_too_short(self):
        # a Pattern reaching position 9 needs len>=9; 'AC' (len 2) is too short
        with pytest.raises(ValueError, match="too short"):
            check_match_df_parts_features(
                df_parts=self._df_parts(["AC", "ACDEFGHIKL"]),
                features=["TMD-Pattern(N,1,9)-S1"])

    def test_match_df_parts_features_none_skips(self):
        assert check_match_df_parts_features(df_parts=None, features=["x"]) is None

    def test_match_df_parts_split_kws_too_short(self):
        kws = {"Pattern": {"len_max": 15, "n_max": 4, "n_min": 2, "steps": [3, 4]}}
        with pytest.raises(ValueError, match="too short"):
            check_match_df_parts_split_kws(
                df_parts=self._df_parts(["AC"]), split_kws=kws)

    def test_match_df_parts_df_scales_missing_char(self):
        # 'B' is not a canonical AA in df_scales index -> missing char, no gaps
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: 0.0 for a in order}})
        with pytest.raises(ValueError, match="missing"):
            check_match_df_parts_df_scales(
                df_parts=self._df_parts(["ABC"]), df_scales=df_scales,
                accept_gaps=False)

    def test_match_df_parts_df_scales_accept_gaps(self):
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: 0.0 for a in order}})
        out = check_match_df_parts_df_scales(
            df_parts=self._df_parts(["ABC"]), df_scales=df_scales, accept_gaps=True)
        assert out is not None  # missing char replaced by gap symbol

    def test_match_df_scales_df_cat_missing_in_cat_raises(self):
        df_scales = pd.DataFrame({"S1": [0.0], "S2": [0.0]}, index=["A"])
        df_cat = pd.DataFrame({ut.COL_SCALE_ID: ["S1"], ut.COL_CAT: ["c"],
                               ut.COL_SUBCAT: ["s"], ut.COL_SCALE_NAME: ["n"]})
        # S2 in df_scales but not df_cat -> error
        with pytest.raises(ValueError, match="missing in 'df_cat'"):
            check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales)

    def test_match_df_scales_df_cat_match_ok(self):
        df_scales = pd.DataFrame({"S1": [0.0]}, index=["A"])
        df_cat = pd.DataFrame({ut.COL_SCALE_ID: ["S1"], ut.COL_CAT: ["c"],
                               ut.COL_SUBCAT: ["s"], ut.COL_SCALE_NAME: ["n"]})
        # matching ids -> returns the (filtered) (df_scales, df_cat); no raise.
        # (The missing-in-df_scales warning at 483-484 is unreachable: it needs
        # difference_scales>0, which implies missing_scales_in_df_cat>0 -> raises first.)
        out = check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales)
        assert out is None or isinstance(out, tuple)
