"""This is a script to test CPPGrid.run() — grid-style CPP configuration sweeps (D4).

Covers the four stage-grouped param dicts (params_parts / params_split / params_scales /
params_cpp), the Cartesian product + list=sweep rule, the lightweight (list_df_feat,
df_params) return (object axes as position index, n_warnings/n_errors counts), the
threads-default backend, soft per-combo errors, fail-fast numeric-arm validation, and
parity with looped CPP.run / CPP.run_num.
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering import CPPGrid

aa.options["verbose"] = False

_C = ut.COL_FEATURE


# Helper functions
def _seq_data(n=8):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return df_seq, df_seq["label"].to_list()


def _scales():
    return aa.load_scales(top60_n=38)


def _num_data(n=4, d=4):
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3] * n
    df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(n)], "sequence": seqs,
                           "tmd_start": 11, "tmd_stop": 50})
    rng = np.random.default_rng(0)
    dict_num = {e: rng.random((60, d)) for e in df_seq["entry"]}
    emb = pd.DataFrame(rng.random((20, d)), index=list(ut.LIST_CANONICAL_AA),
                       columns=[f"d{i}" for i in range(d)])
    labels = [1] * (n // 2) + [0] * (n - n // 2)
    return df_seq, labels, dict_num, emb


def _grid_seq():
    df_seq, labels = _seq_data()
    return CPPGrid(df_seq=df_seq, labels=labels, n_jobs=1), labels


class TestCPPGridInit:
    """Constructor validation (normal + negative)."""

    def test_valid_construction(self):
        df_seq, labels = _seq_data()
        grid = CPPGrid(df_seq=df_seq, labels=labels)
        assert grid.df_seq.shape[0] == len(labels)

    def test_df_seq_none_raises(self):
        with pytest.raises(ValueError):
            CPPGrid(df_seq=None, labels=[1, 0])

    def test_labels_none_raises(self):
        df_seq, _ = _seq_data()
        with pytest.raises(ValueError):
            CPPGrid(df_seq=df_seq, labels=None)

    def test_bad_backend_raises(self):
        df_seq, labels = _seq_data()
        with pytest.raises(ValueError):
            CPPGrid(df_seq=df_seq, labels=labels, backend="dask")

    def test_bad_accept_gaps_raises(self):
        df_seq, labels = _seq_data()
        with pytest.raises(ValueError):
            CPPGrid(df_seq=df_seq, labels=labels, accept_gaps="yes")

    def test_dict_num_wrong_type_raises(self):
        df_seq, labels = _seq_data()
        with pytest.raises(ValueError):
            CPPGrid(df_seq=df_seq, labels=labels, dict_num=[1, 2, 3])

    def test_threads_and_loky_accepted(self):
        df_seq, labels = _seq_data()
        for backend in ("threads", "loky"):
            CPPGrid(df_seq=df_seq, labels=labels, backend=backend)

    def test_accept_gaps_true_accepted(self):
        df_seq, labels = _seq_data()
        CPPGrid(df_seq=df_seq, labels=labels, accept_gaps=True)

    def test_verbose_bool_accepted(self):
        df_seq, labels = _seq_data()
        CPPGrid(df_seq=df_seq, labels=labels, verbose=False)

    def test_bad_verbose_raises(self):
        df_seq, labels = _seq_data()
        with pytest.raises(ValueError):
            CPPGrid(df_seq=df_seq, labels=labels, verbose="x")

    def test_random_state_int_accepted(self):
        df_seq, labels = _seq_data()
        CPPGrid(df_seq=df_seq, labels=labels, random_state=7)

    def test_bad_random_state_raises(self):
        df_seq, labels = _seq_data()
        with pytest.raises(ValueError):
            CPPGrid(df_seq=df_seq, labels=labels, random_state=-5)

    def test_n_jobs_variants_accepted(self):
        df_seq, labels = _seq_data()
        for nj in (1, -1, 2):
            CPPGrid(df_seq=df_seq, labels=labels, n_jobs=nj)


class TestRun:
    """Normal cases for CPPGrid.run (the sweep)."""

    def test_single_combo_one_result(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 20})
        assert len(lst) == 1 and len(dfp) == 1

    def test_sweep_n_filter_count_and_shapes(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": [10, 20, 30]})
        assert len(lst) == 3
        assert [d.shape[0] for d in lst] == [10, 20, 30]

    def test_product_of_two_axes(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_parts={"jmd_n_len": [8, 10]},
                            params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": [10, 15]})
        assert len(lst) == 4 == len(dfp)

    def test_df_params_columns(self):
        grid, _ = _grid_seq()
        _, dfp = grid.run(params_parts={"jmd_n_len": [8, 10]},
                          params_split={"split_types": "Segment"},
                          params_scales=_scales(), params_cpp={"n_filter": 20})
        for col in ("jmd_n_len", "split_types", "n_filter", "df_scales", "n_warnings", "n_errors"):
            assert col in dfp.columns

    def test_df_params_scalar_axis_literal(self):
        grid, _ = _grid_seq()
        _, dfp = grid.run(params_split={"split_types": "Segment"},
                          params_scales=_scales(), params_cpp={"n_filter": [10, 20]})
        assert sorted(dfp["n_filter"].tolist()) == [10, 20]

    def test_df_scales_axis_is_position_index(self):
        grid, _ = _grid_seq()
        s1, s2 = _scales(), aa.load_scales(top60_n=20)
        _, dfp = grid.run(params_split={"split_types": "Segment"},
                          params_scales=[s1, s2], params_cpp={"n_filter": 10})
        assert sorted(dfp["df_scales"].tolist()) == [0, 1]

    def test_empty_params_one_default_combo(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run()
        assert len(lst) == 1 and len(dfp) == 1

    def test_list_df_feat_aligned_to_df_params(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": [10, 20]})
        assert len(lst) == len(dfp)

    def test_parity_with_direct_cpp_run(self):
        df_seq, labels = _seq_data()
        dfs = _scales()
        grid = CPPGrid(df_seq=df_seq, labels=labels, n_jobs=1, random_state=42)
        lst, _ = grid.run(params_split={"split_types": "Segment"},
                          params_scales=dfs, params_cpp={"n_filter": 25})
        sf = aa.SequenceFeature(verbose=False)
        dp = sf.get_df_parts(df_seq=df_seq)
        sk = aa.SequenceFeature.get_split_kws(split_types="Segment")
        cpp = aa.CPP(df_parts=dp, split_kws=sk, df_scales=dfs, verbose=False, random_state=42)
        df_direct = cpp.run(labels=labels, n_filter=25, n_jobs=1)
        assert lst[0][_C].tolist() == df_direct[_C].tolist()

    def test_loky_backend_runs(self):
        df_seq, labels = _seq_data()
        grid = CPPGrid(df_seq=df_seq, labels=labels, n_jobs=1, backend="loky")
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 10})
        assert len(lst) == 1


class TestRunNumeric:
    """Normal cases for the numerical arm."""

    def test_numeric_arm_runs(self):
        df_seq, labels, dict_num, emb = _num_data()
        grid = CPPGrid(df_seq=df_seq, labels=labels, dict_num=dict_num, n_jobs=1)
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=emb, params_cpp={"n_filter": [5, 8]})
        assert len(lst) == 2
        assert [d.shape[0] for d in lst] == [5, 8]

    def test_numeric_parity_with_run_num(self):
        df_seq, labels, dict_num, emb = _num_data()
        grid = CPPGrid(df_seq=df_seq, labels=labels, dict_num=dict_num, n_jobs=1, random_state=1)
        lst, _ = grid.run(params_split={"split_types": "Segment"},
                          params_scales=emb, params_cpp={"n_filter": 6})
        nf = aa.NumericalFeature()
        dp, dnp = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        sk = aa.SequenceFeature.get_split_kws(split_types="Segment")
        from aaanalysis.feature_engineering._cpp_grid import _resolve_df_cat
        cpp = aa.CPP(df_parts=dp, split_kws=sk, df_scales=emb, df_cat=_resolve_df_cat(emb),
                     verbose=False, random_state=1)
        df_direct = cpp.run_num(dict_num_parts=dnp, labels=labels, n_filter=6, n_jobs=1)
        assert lst[0][_C].tolist() == df_direct[_C].tolist()


class TestRunComplex:
    """Combinations + negative cases."""

    def test_product_three_axes(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_parts={"jmd_n_len": [8, 10]},
                            params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": [10, 15, 20]})
        assert len(lst) == 6 == len(dfp)

    def test_scales_sweep_times_n_filter(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=[_scales(), aa.load_scales(top60_n=20)],
                            params_cpp={"n_filter": [10, 20]})
        assert len(lst) == 4
        assert sorted(dfp["df_scales"].unique().tolist()) == [0, 1]

    def test_object_axis_recorded_as_index(self):
        # steps_pattern wrapped in an outer list to sweep two list-valued candidates.
        grid, _ = _grid_seq()
        _, dfp = grid.run(params_split={"split_types": "Pattern",
                                        "steps_pattern": [[3, 4], [2, 3]]},
                          params_scales=_scales(), params_cpp={"n_filter": 10})
        assert sorted(dfp["steps_pattern"].tolist()) == [0, 1]

    def test_sparse_config_counts_warning(self):
        grid, _ = _grid_seq()
        _, dfp = grid.run(params_split={"split_types": "Segment"},
                          params_scales=_scales(), params_cpp={"n_filter": 10 ** 7})
        assert dfp["n_warnings"].iloc[0] == 1

    def test_per_combo_error_is_soft(self):
        # n_filter=-1 is invalid -> that combo raises inside CPP.run and is captured.
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": [20, -1]})
        assert dfp["n_errors"].sum() == 1
        bad = dfp.index[dfp["n_errors"] == 1][0]
        assert lst[bad] is None
        good = dfp.index[dfp["n_errors"] == 0][0]
        assert lst[good] is not None

    def test_params_parts_not_dict_raises(self):
        grid, _ = _grid_seq()
        with pytest.raises(ValueError):
            grid.run(params_parts=[1, 2], params_scales=_scales())

    def test_params_scales_bad_entry_raises(self):
        grid, _ = _grid_seq()
        with pytest.raises(ValueError):
            grid.run(params_scales=[_scales(), "not_a_df"])

    def test_numeric_d_mismatch_fail_fast(self):
        df_seq, labels, dict_num, _ = _num_data(d=4)
        bad_emb = pd.DataFrame(np.random.default_rng(0).random((20, 3)),
                               index=list(ut.LIST_CANONICAL_AA), columns=[f"d{i}" for i in range(3)])
        grid = CPPGrid(df_seq=df_seq, labels=labels, dict_num=dict_num, n_jobs=1)
        with pytest.raises(ValueError, match="columns"):
            grid.run(params_split={"split_types": "Segment"}, params_scales=bad_emb,
                     params_cpp={"n_filter": 5})

    def test_list_parts_swept_when_wrapped(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_parts={"list_parts": [["tmd"], ["jmd_n", "jmd_c"]]},
                            params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 10})
        assert len(lst) == 2
        assert sorted(dfp["list_parts"].tolist()) == [0, 1]

    def test_all_good_combos_no_errors(self):
        grid, _ = _grid_seq()
        _, dfp = grid.run(params_split={"split_types": "Segment"},
                          params_scales=_scales(), params_cpp={"n_filter": [10, 20]})
        assert dfp["n_errors"].sum() == 0

    def test_params_split_not_dict_raises(self):
        grid, _ = _grid_seq()
        with pytest.raises(ValueError):
            grid.run(params_split=[1, 2], params_scales=_scales())

    def test_params_cpp_not_dict_raises(self):
        grid, _ = _grid_seq()
        with pytest.raises(ValueError):
            grid.run(params_cpp=[1, 2], params_scales=_scales())

    def test_bad_n_jobs_raises_at_run(self):
        df_seq, labels = _seq_data()
        grid = CPPGrid(df_seq=df_seq, labels=labels, n_jobs=0)
        with pytest.raises(ValueError):
            grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                     params_cpp={"n_filter": 10})

    def test_n_jobs_two_parallel_run(self):
        df_seq, labels = _seq_data()
        grid = CPPGrid(df_seq=df_seq, labels=labels, n_jobs=2)
        lst, dfp = grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                            params_cpp={"n_filter": [10, 15]})
        assert len(lst) == 2 and dfp["n_errors"].sum() == 0


class TestRunShortcuts:
    """The n_filter collapse + parts/split caching shortcuts (run CPP once, slice the rest)."""

    @staticmethod
    def _count_calls(monkeypatch, cls, method):
        calls = {"n": 0}
        orig = getattr(cls, method)

        def counting(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        monkeypatch.setattr(cls, method, counting)
        return calls

    def test_n_filter_sweep_runs_cpp_once(self, monkeypatch):
        calls = self._count_calls(monkeypatch, aa.CPP, "run")
        grid, _ = _grid_seq()
        grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                 params_cpp={"n_filter": [10, 25, 50]})
        assert calls["n"] == 1  # one run at max(50); 10 and 25 are head() slices

    def test_n_filter_sweep_exact_vs_independent(self):
        df_seq, labels = _seq_data()
        dfs = _scales()
        grid = CPPGrid(df_seq=df_seq, labels=labels, n_jobs=1, random_state=0)
        lst, _ = grid.run(params_split={"split_types": "Segment"}, params_scales=dfs,
                          params_cpp={"n_filter": [10, 25, 50]})
        sf = aa.SequenceFeature(verbose=False)
        dp = sf.get_df_parts(df_seq=df_seq)
        sk = aa.SequenceFeature.get_split_kws(split_types="Segment")
        for j, n in enumerate([10, 25, 50]):
            indep = aa.CPP(df_parts=dp, split_kws=sk, df_scales=dfs, verbose=False,
                           random_state=0).run(labels=labels, n_filter=n, n_jobs=1)
            assert lst[j][_C].tolist() == indep[_C].tolist()

    def test_sliced_member_preserves_shortfall_warning(self):
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                            params_cpp={"n_filter": [10, 10 ** 7]})
        assert dfp.loc[0, "n_warnings"] == 0    # 10 delivered fully
        assert dfp.loc[1, "n_warnings"] == 1    # 1e7 can't be reached -> warns

    def test_n_filter_collapses_per_other_cpp_group(self, monkeypatch):
        calls = self._count_calls(monkeypatch, aa.CPP, "run")
        grid, _ = _grid_seq()
        # 2 max_std_test x 3 n_filter = 6 combos, but only 2 runs (one per max_std_test group)
        grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                 params_cpp={"max_std_test": [0.2, 0.3], "n_filter": [10, 25, 50]})
        assert calls["n"] == 2

    def test_invalid_n_filter_in_sweep_soft_errors_runs_once(self, monkeypatch):
        calls = self._count_calls(monkeypatch, aa.CPP, "run")
        grid, _ = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                            params_cpp={"n_filter": [20, -1]})
        assert calls["n"] == 1 and dfp["n_errors"].sum() == 1

    def test_df_parts_built_once_across_n_filter(self, monkeypatch):
        calls = self._count_calls(monkeypatch, aa.SequenceFeature, "get_df_parts")
        grid, _ = _grid_seq()
        grid.run(params_split={"split_types": "Segment"}, params_scales=_scales(),
                 params_cpp={"n_filter": [10, 25, 50]})
        assert calls["n"] == 1  # parts cached: built once for the shared parts-config
