"""This is a script to test NumericalFeature.get_parts() in the anchor (pos) input mode (D5a).

The (``sequence`` + ``pos`` + ``tmd_len``) format explodes each 1-based P1 anchor into one row and
slices the matching ``dict_num[entry]`` tensor with the SAME boundaries used for the string parts,
so ``df_parts`` rows and ``dict_num_parts`` tensor rows stay aligned (row order = anchor order).
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa

aa.options["verbose"] = False

_SEQ = "ACDEFGHIKLMNPQRSTVWY" * 3  # length 60
_D = 4


def _dict_num(entries, seq=_SEQ):
    rng = np.random.default_rng(0)
    return {e: rng.random((len(seq), _D)) for e in entries}


def _df_anchor(pos, entries=("P1",), seq=_SEQ):
    return pd.DataFrame({"entry": list(entries),
                         "sequence": [seq] * len(entries),
                         "pos": pos})


class TestGetPartsAnchor:
    """Normal cases for the anchor (pos) input mode of get_parts."""

    def test_scalar_pos_one_row(self):
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=_dict_num(["P1"]),
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 1
        assert dnp["tmd"].shape[0] == 1

    def test_list_pos_explodes_both_outputs(self):
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=_df_anchor([[15, 25, 35]]),
                                                  dict_num=_dict_num(["P1"]),
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 3
        assert dnp["tmd"].shape == (3, 6, _D)

    def test_tensor_slice_matches_dict_num(self):
        dn = _dict_num(["P1"])
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=dn,
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        # anchor=20, tmd_len=6 -> tmd_start=18 (1-based) -> 0-based slice 17:23
        assert np.allclose(dnp["tmd"][0], dn["P1"][17:23])

    def test_multi_anchor_tensor_alignment(self):
        dn = _dict_num(["P1"])
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=_df_anchor([[15, 25]]), dict_num=dn,
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        # anchor 15 -> 0-based 12:18 ; anchor 25 -> 0-based 22:28
        assert np.allclose(dnp["tmd"][0], dn["P1"][12:18])
        assert np.allclose(dnp["tmd"][1], dn["P1"][22:28])

    def test_entry_win_index(self):
        dp, _ = aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=_dict_num(["P1"]),
                                                tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert list(dp.index) == ["P1_14-27"]

    def test_dim_preserved(self):
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=_dict_num(["P1"]),
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert dnp["tmd"].shape[-1] == _D

    def test_rows_equal_total_anchors(self):
        df = _df_anchor([[15, 25], 30], entries=("P1", "P2"))
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=df, dict_num=_dict_num(["P1", "P2"]),
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == 3 == dnp["tmd"].shape[0]

    def test_parity_with_explicit_position_based(self):
        dn = _dict_num(["P1"])
        df_anchor = _df_anchor([20])
        df_pos = pd.DataFrame({"entry": ["P1"], "sequence": [_SEQ],
                               "tmd_start": [18], "tmd_stop": [23]})
        _, dnp_a = aa.NumericalFeature().get_parts(df_seq=df_anchor, dict_num=dn,
                                                   tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        _, dnp_p = aa.NumericalFeature().get_parts(df_seq=df_pos, dict_num=dn,
                                                   jmd_n_len=4, jmd_c_len=4)
        assert np.allclose(dnp_a["tmd"], dnp_p["tmd"])


class TestGetPartsAnchorComplex:
    """Combinations and negative cases for the anchor mode of get_parts."""

    def test_tmd_len_none_raises(self):
        with pytest.raises(ValueError):
            aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=_dict_num(["P1"]),
                                            tmd_len=None, jmd_n_len=4, jmd_c_len=4)

    @pytest.mark.parametrize("bad", [0, -2, 1.5])
    def test_tmd_len_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=_dict_num(["P1"]),
                                            tmd_len=bad, jmd_n_len=4, jmd_c_len=4)

    def test_anchor_out_of_range_raises(self):
        with pytest.raises(ValueError):
            aa.NumericalFeature().get_parts(df_seq=_df_anchor([59]), dict_num=_dict_num(["P1"]),
                                            tmd_len=6, jmd_n_len=4, jmd_c_len=4)

    def test_missing_dict_num_entry_raises(self):
        df = _df_anchor([[20], [25]], entries=("P1", "P2"))
        with pytest.raises(ValueError):
            aa.NumericalFeature().get_parts(df_seq=df, dict_num=_dict_num(["P1"]),  # P2 missing
                                            tmd_len=6, jmd_n_len=4, jmd_c_len=4)

    def test_df_parts_and_tensor_rows_stay_aligned(self):
        dn = _dict_num(["P1", "P2"])
        df = _df_anchor([[15, 25], [30, 40]], entries=("P1", "P2"))
        dp, dnp = aa.NumericalFeature().get_parts(df_seq=df, dict_num=dn,
                                                  tmd_len=6, jmd_n_len=4, jmd_c_len=4)
        assert len(dp) == dnp["tmd"].shape[0] == 4
        # last row is P2 @ anchor 40 -> 0-based 37:43
        assert np.allclose(dnp["tmd"][3], dn["P2"][37:43])

    def test_no_dict_num_raises(self):
        with pytest.raises(ValueError):
            aa.NumericalFeature().get_parts(df_seq=_df_anchor([20]), dict_num=None,
                                            tmd_len=6, jmd_n_len=4, jmd_c_len=4)
