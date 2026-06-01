"""This is a script to test SequenceFeature.get_df_parts() in the anchor (pos) input mode (D5a).

Covers the new (``sequence`` + ``pos`` + ``tmd_len``) format: each 1-based P1 anchor in ``pos``
is exploded into one 3-part row whose TMD is placed right-heavy (for even ``tmd_len``) on the
anchor, ided in the index by ``<entry>_<win_start>-<win_stop>``. Geometry must match
``ut.get_window_offsets`` and stay consistent with the AAWindowSampler P1 convention.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

_SEQ = "ACDEFGHIKLMNPQRSTVWY" * 3  # length 60, no gaps


def _df_anchor(pos, seq=_SEQ, entries=("P1",)):
    return pd.DataFrame({"entry": list(entries),
                         "sequence": [seq] * len(entries),
                         "pos": pos})


class TestGetDfPartsAnchor:
    """Normal cases for the anchor (pos) input mode."""

    def test_scalar_pos_one_row(self):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 1

    def test_list_pos_explodes(self):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([[15, 25, 35]]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 3

    def test_even_tmd_len_is_right_heavy(self):
        # anchor=20, tmd_len=6 -> half_left=2 -> tmd_start=18(1-based) -> seq[17:23]
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4)
        assert dp["tmd"].iloc[0] == _SEQ[17:23]

    def test_odd_tmd_len_symmetric(self):
        # anchor=20, tmd_len=5 -> half_left=2 -> tmd_start=18 -> seq[17:22]
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=5,
                                               jmd_n_len=4, jmd_c_len=4)
        assert dp["tmd"].iloc[0] == _SEQ[17:22]

    def test_entry_win_index_format(self):
        # anchor=20, tmd_len=6 -> tmd 18..23; win_start=18-4=14, win_stop=23+4=27
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4)
        assert list(dp.index) == ["P1_14-27"]

    def test_multi_entry_mixed_cardinality(self):
        df = _df_anchor([[15, 25], 30], entries=("P1", "P2"))
        dp = aa.SequenceFeature().get_df_parts(df_seq=df, tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 3
        assert list(dp.index) == ["P1_9-22", "P1_19-32", "P2_24-37"]

    @pytest.mark.parametrize("tmd_len", [4, 6, 8, 10])
    def test_tmd_length_matches(self, tmd_len):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([30]), tmd_len=tmd_len,
                                               jmd_n_len=5, jmd_c_len=5)
        assert len(dp["tmd"].iloc[0]) == tmd_len

    def test_geometry_matches_window_offsets(self):
        anchor, tmd_len = 25, 8
        half_left, _ = ut.get_window_offsets(tmd_len)
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([anchor]), tmd_len=tmd_len,
                                               jmd_n_len=3, jmd_c_len=3)
        tmd_start0 = anchor - half_left - 1  # 0-based
        assert dp["tmd"].iloc[0] == _SEQ[tmd_start0:tmd_start0 + tmd_len]

    def test_parity_with_explicit_position_based(self):
        # anchor=20, tmd_len=6 -> tmd_start=18, tmd_stop=23 (1-based inclusive)
        df_anchor = _df_anchor([20])
        df_pos = pd.DataFrame({"entry": ["P1"], "sequence": [_SEQ],
                               "tmd_start": [18], "tmd_stop": [23]})
        dp_a = aa.SequenceFeature().get_df_parts(df_seq=df_anchor, tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        dp_p = aa.SequenceFeature().get_df_parts(df_seq=df_pos, jmd_n_len=4, jmd_c_len=4)
        assert dp_a.to_numpy().tolist() == dp_p.to_numpy().tolist()

    def test_jmd_len_changes_entry_win(self):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=6,
                                               jmd_n_len=6, jmd_c_len=2)
        # tmd 18..23; win_start=18-6=12, win_stop=23+2=25
        assert list(dp.index) == ["P1_12-25"]

    def test_duplicate_anchor_duplicate_rows(self):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([[20, 20]]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 2
        assert dp["tmd"].iloc[0] == dp["tmd"].iloc[1]


class TestGetDfPartsAnchorComplex:
    """Combinations and negative cases for the anchor (pos) input mode."""

    def test_tmd_len_none_raises(self):
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=None,
                                              jmd_n_len=4, jmd_c_len=4)

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_tmd_len_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([20]), tmd_len=bad,
                                              jmd_n_len=4, jmd_c_len=4)

    def test_anchor_too_close_to_nterm_raises(self):
        # anchor=2, tmd_len=6, jmd_n_len=4 -> win_start < 1
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([2]), tmd_len=6,
                                              jmd_n_len=4, jmd_c_len=4)

    def test_anchor_too_close_to_cterm_raises(self):
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([59]), tmd_len=6,
                                              jmd_n_len=4, jmd_c_len=4)

    def test_anchor_out_of_sequence_raises(self):
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([1000]), tmd_len=6,
                                              jmd_n_len=4, jmd_c_len=4)

    def test_non_int_pos_raises(self):
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([["x"]]), tmd_len=6,
                                              jmd_n_len=4, jmd_c_len=4)

    def test_all_empty_pos_raises(self):
        with pytest.raises(ValueError):
            aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([None]), tmd_len=6,
                                              jmd_n_len=4, jmd_c_len=4)

    def test_one_bad_anchor_error_names_entry(self):
        df = _df_anchor([[20], [1000]], entries=("GOOD", "BADENTRY"))
        with pytest.raises(ValueError, match="BADENTRY"):
            aa.SequenceFeature().get_df_parts(df_seq=df, tmd_len=6, jmd_n_len=4, jmd_c_len=4)

    def test_anchor_mode_with_list_parts_subset(self):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([[20, 30]]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4, list_parts=["tmd", "jmd_n"])
        assert list(dp.columns) == ["tmd", "jmd_n"]
        assert len(dp) == 2

    def test_anchor_mode_all_parts(self):
        dp = aa.SequenceFeature().get_df_parts(df_seq=_df_anchor([25]), tmd_len=6,
                                               jmd_n_len=4, jmd_c_len=4, all_parts=True)
        assert {"tmd", "jmd_n", "jmd_c"}.issubset(dp.columns)

    def test_empty_list_skips_that_entry_only(self):
        # P1 has no anchors (empty list), P2 has one -> only P2 row survives
        df = pd.DataFrame({"entry": ["P1", "P2"], "sequence": [_SEQ, _SEQ], "pos": [[], [25]]})
        dp = aa.SequenceFeature().get_df_parts(df_seq=df, tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert list(dp.index) == ["P2_19-32"]

    def test_scalar_numpy_int_pos(self):
        df = pd.DataFrame({"entry": ["P1"], "sequence": [_SEQ], "pos": [np.int64(20)]})
        dp = aa.SequenceFeature().get_df_parts(df_seq=df, tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 1
