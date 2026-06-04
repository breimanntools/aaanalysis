"""This is a script to test branches of CPP's residue-value assignment stage
(``_backend/cpp/_filters/_assign.py``): the seq-mode dict (option A), the dev
option-B 4D path, the n_jobs>1 chunk/merge + empty-chunk guards, and the
dict_num path (per-residue tensor slicing, JMD padding, ext_len, empty parts).

Driven directly at the backend because option-B and the empty-chunk guards are
not reachable from the CPP frontend (dev-bench flag / n_jobs>n_scales), and the
DOM_GSEC fixtures use uniform inputs. Includes a hand-computed golden lookup.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._filters._assign import (
    assign_scale_values_to_seq,
    assign_dict_num_to_parts,
    _slice_dict_num_to_basic_parts,
    _get_dict_part_num,
    _merge_part_vals,
)

aa.options["verbose"] = False

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# I Helper Functions
def _df_scales(n_scales=2):
    cols = {}
    for s in range(n_scales):
        cols[f"S{s + 1}"] = {aa: float(i + 1 + 100 * s) for i, aa in enumerate(AA_ORDER)}
    return pd.DataFrame(cols)


def _df_parts(tmds):
    return pd.DataFrame({"tmd": list(tmds)})


# II Test Classes
class TestAssignScaleValuesGolden:
    """assign_scale_values_to_seq: hand-computed lookup + contract."""

    def test_golden_lookup(self):
        vals, lens = assign_scale_values_to_seq(
            df_parts=_df_parts(["AC"]), df_scales=_df_scales(1), n_jobs=1)
        arr = vals["tmd"]  # (n=1, L_max=2, D=1)
        assert arr.shape == (1, 2, 1)
        assert arr[0, 0, 0] == 1.0   # S1[A]
        assert arr[0, 1, 0] == 2.0   # S1[C]
        assert lens["tmd"].tolist() == [2]

    def test_padding_and_unknown_nan(self):
        # shorter seq padded to L_max with NaN; unknown char '_' also NaN.
        vals, _ = assign_scale_values_to_seq(
            df_parts=_df_parts(["AC", "A"]), df_scales=_df_scales(1), n_jobs=1)
        arr = vals["tmd"]
        assert arr.shape == (2, 2, 1)
        assert np.isnan(arr[1, 1, 0])  # padded position

    def test_two_scales_d_axis(self):
        vals, _ = assign_scale_values_to_seq(
            df_parts=_df_parts(["AC"]), df_scales=_df_scales(2), n_jobs=1)
        assert vals["tmd"].shape == (1, 2, 2)


class TestAssignStagingShapeB:
    """The dev option-B 4D path (_staging_shape='B')."""

    def test_option_b_shapes(self):
        vals, lens = assign_scale_values_to_seq(
            df_parts=_df_parts(["AC", "ACD"]), df_scales=_df_scales(2),
            _staging_shape="B")
        # views into a 4D tensor; per-part (n, L_global, D)
        assert vals["tmd"].shape == (2, 3, 2)
        assert lens["tmd"].tolist() == [2, 3]

    def test_option_b_empty_scales(self):
        empty_scales = pd.DataFrame(index=list(AA_ORDER))  # 0 columns
        vals, lens = assign_scale_values_to_seq(
            df_parts=_df_parts(["AC"]), df_scales=empty_scales, _staging_shape="B")
        assert vals == {} and lens == {}


class TestAssignParallel:
    """n_jobs>1 chunk/merge + empty-chunk guards."""

    def test_parallel_matches_serial(self):
        from joblib import parallel_backend
        parts = _df_parts(["ACDE", "FGHI"])
        scales = _df_scales(2)
        # threading backend runs workers in-process (deterministic + coverable).
        with parallel_backend("threading"):
            v_par, _ = assign_scale_values_to_seq(df_parts=parts, df_scales=scales, n_jobs=2)
        v_ser, _ = assign_scale_values_to_seq(df_parts=parts, df_scales=scales, n_jobs=1)
        assert np.allclose(v_par["tmd"], v_ser["tmd"], equal_nan=True)

    def test_more_jobs_than_scales_empty_chunk(self):
        from joblib import parallel_backend
        # n_jobs(3) > n_scales(2) -> one worker gets an empty scale chunk -> {} skipped.
        parts = _df_parts(["ACDE"])
        with parallel_backend("threading"):
            vals, _ = assign_scale_values_to_seq(df_parts=parts, df_scales=_df_scales(2), n_jobs=3)
        assert vals["tmd"].shape == (1, 4, 2)

    def test_all_empty_chunks_returns_empty(self):
        from joblib import parallel_backend
        # empty scales + n_jobs>1 -> every chunk empty -> _merge_part_vals -> {}, {}.
        empty_scales = pd.DataFrame(index=list(AA_ORDER))
        with parallel_backend("threading"):
            vals, lens = assign_scale_values_to_seq(
                df_parts=_df_parts(["AC"]), df_scales=empty_scales, n_jobs=2)
        assert vals == {} and lens == {}

    def test_merge_helper_all_empty(self):
        assert _merge_part_vals(results=[({}, {}), ({}, {})], list_parts=["tmd"]) == ({}, {})


class TestAssignVerboseAndAuto:
    """verbose progress + auto-n_jobs (None) branches."""

    def test_verbose_n_jobs_1(self, capsys):
        assign_scale_values_to_seq(df_parts=_df_parts(["ACDE", "FGHI"]),
                                   df_scales=_df_scales(2), verbose=True, n_jobs=1)
        # progress output emitted (don't assert exact text — just that it ran)
        capsys.readouterr()

    def test_n_jobs_none_auto(self):
        # n_jobs=None -> auto-compute (small data -> resolves to 1).
        vals, _ = assign_scale_values_to_seq(df_parts=_df_parts(["ACDE"]),
                                             df_scales=_df_scales(2))
        assert vals["tmd"].shape == (1, 4, 2)

    def test_verbose_parallel(self, capsys):
        from joblib import parallel_backend
        with parallel_backend("threading"):
            assign_scale_values_to_seq(df_parts=_df_parts(["ACDE", "FGHI"]),
                                       df_scales=_df_scales(2), verbose=True, n_jobs=2)
        capsys.readouterr()


class TestSliceDictNum:
    """_slice_dict_num_to_basic_parts: JMD padding branches."""

    def test_no_padding(self):
        emb = np.arange(20, dtype=np.float64).reshape(10, 2)
        tmd, jmd_n, jmd_c = _slice_dict_num_to_basic_parts(
            emb=emb, tmd_start=4, tmd_stop=7, jmd_n_len=2, jmd_c_len=2)
        assert jmd_n.shape == (2, 2) and jmd_c.shape == (2, 2)
        assert tmd.shape == (4, 2)  # positions 4..7 inclusive

    def test_n_terminus_padding(self):
        emb = np.arange(20, dtype=np.float64).reshape(10, 2)
        # tmd_start=2 -> n_terminus_len=1 < jmd_n_len=5 -> pad
        _, jmd_n, _ = _slice_dict_num_to_basic_parts(
            emb=emb, tmd_start=2, tmd_stop=9, jmd_n_len=5, jmd_c_len=2)
        assert jmd_n.shape == (5, 2)
        assert np.isnan(jmd_n[0]).all()  # leading NaN pad

    def test_c_terminus_padding(self):
        emb = np.arange(20, dtype=np.float64).reshape(10, 2)
        # tmd_stop=9 -> c_terminus_len=1 < jmd_c_len=5 -> pad
        _, _, jmd_c = _slice_dict_num_to_basic_parts(
            emb=emb, tmd_start=3, tmd_stop=9, jmd_n_len=2, jmd_c_len=5)
        assert jmd_c.shape == (5, 2)
        assert np.isnan(jmd_c[-1]).all()  # trailing NaN pad


class TestGetDictPartNum:
    """_get_dict_part_num: ext_len>0 vs ext_len=0 branches."""

    def test_ext_len_zero_empty_ext(self):
        aa.options["ext_len"] = 0
        tmd = np.arange(8, dtype=np.float64).reshape(4, 2)
        jmd_n = np.zeros((2, 2)); jmd_c = np.zeros((2, 2))
        parts = _get_dict_part_num(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c)
        assert parts["ext_n"].shape == (0, 2)
        assert parts["ext_c"].shape == (0, 2)

    def test_ext_len_positive(self):
        aa.options["ext_len"] = 2
        tmd = np.arange(8, dtype=np.float64).reshape(4, 2)
        jmd_n = np.arange(8, dtype=np.float64).reshape(4, 2)
        jmd_c = np.arange(8, dtype=np.float64).reshape(4, 2)
        parts = _get_dict_part_num(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c)
        assert parts["ext_n"].shape == (2, 2)  # last 2 rows of jmd_n
        assert parts["ext_c"].shape == (2, 2)  # first 2 rows of jmd_c


class TestAssignDictNumToParts:
    """assign_dict_num_to_parts: full path + empty-part (L_max==0) branch."""

    def _df_seq(self):
        return pd.DataFrame({
            ut.COL_ENTRY: ["P1", "P2"],
            ut.COL_TMD_START: [3, 2],
            ut.COL_TMD_STOP: [8, 9],
        })

    def _dict_num(self):
        return {"P1": np.arange(20, dtype=np.float64).reshape(10, 2),
                "P2": np.arange(24, dtype=np.float64).reshape(12, 2)}

    def test_tmd_part_shape(self):
        aa.options["ext_len"] = 0
        vals, lens = assign_dict_num_to_parts(
            df_seq=self._df_seq(), dict_num=self._dict_num(),
            list_parts=["tmd"], jmd_n_len=2, jmd_c_len=2)
        assert vals["tmd"].shape[0] == 2  # n_samples
        assert vals["tmd"].shape[2] == 2  # D
        assert lens["tmd"].tolist() == [6, 8]  # tmd_stop-tmd_start+1

    def test_empty_part_l_max_zero(self):
        # ext_len=0 -> 'ext_n' part is length 0 for every entry -> L_max==0 branch.
        aa.options["ext_len"] = 0
        vals, lens = assign_dict_num_to_parts(
            df_seq=self._df_seq(), dict_num=self._dict_num(),
            list_parts=["ext_n"], jmd_n_len=2, jmd_c_len=2)
        assert vals["ext_n"].shape == (2, 0, 2)
        assert lens["ext_n"].tolist() == [0, 0]

    def test_jmd_padding_via_frontend_path(self):
        # P1 tmd_start=3 -> n_terminus_len=2; request jmd_n_len=5 -> padded.
        aa.options["ext_len"] = 0
        vals, _ = assign_dict_num_to_parts(
            df_seq=self._df_seq(), dict_num=self._dict_num(),
            list_parts=["jmd_n"], jmd_n_len=5, jmd_c_len=2)
        assert vals["jmd_n"].shape == (2, 5, 2)

    def test_multiple_parts_consistent(self):
        aa.options["ext_len"] = 2
        vals, lens = assign_dict_num_to_parts(
            df_seq=self._df_seq(), dict_num=self._dict_num(),
            list_parts=["tmd", "jmd_n", "jmd_c", "tmd_jmd"], jmd_n_len=2, jmd_c_len=2)
        for p in ["tmd", "jmd_n", "jmd_c", "tmd_jmd"]:
            assert vals[p].shape[0] == 2
            assert vals[p].shape[2] == 2
