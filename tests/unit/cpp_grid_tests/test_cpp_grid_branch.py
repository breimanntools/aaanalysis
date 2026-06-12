"""Branch-coverage tests for CPPGrid, exercised ONLY through the public aa.CPPGrid API.

Targets the under-covered guard arms in feature_engineering/_cpp_grid.py:
* the sweep-footgun UserWarning on a flat list-valued knob,
* the soft per-combo error paths (parts-build failure, all-invalid n_filter),
* the n_warnings filter-shortfall derivation,
* the eval() None/empty df_feat arm.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering import CPPGrid

aa.options["verbose"] = False


def _seq_data(n=10):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return df_seq, df_seq["label"].to_list()


def _scales():
    return aa.load_scales(top60_n=38)


def _grid_seq(n=10):
    df_seq, labels = _seq_data(n=n)
    return CPPGrid(df_seq=df_seq, labels=labels, n_jobs=1)


class TestRunWarnFootgun:
    """_warn_sweep_footgun true arm (line 139): a flat list-valued knob warns."""

    def test_flat_steps_pattern_warns(self):
        grid = _grid_seq()
        with pytest.warns(UserWarning, match="flat list"):
            grid.run(params_split={"split_types": "Segment", "steps_pattern": [3, 4]},
                     params_scales=_scales(), params_cpp={"n_filter": 10})

    def test_flat_list_parts_warns(self):
        grid = _grid_seq()
        with pytest.warns(UserWarning, match="flat list"):
            grid.run(params_parts={"list_parts": ["tmd", "jmd_n"]},
                     params_scales=_scales(), params_cpp={"n_filter": 10})


class TestRunSoftErrors:
    """Soft per-combo error arms: parts-build failure (394) and all-invalid n_filter (405)."""

    def test_parts_kwarg_invalid_at_build_soft_errors(self):
        # An invalid list_parts raises *inside* get_df_parts (parts-cache build),
        # so the member is recorded as a soft error rather than crashing run().
        grid = _grid_seq()
        lst, dfp = grid.run(params_parts={"list_parts": "bogus_part"},
                            params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 10})
        assert lst[0] is None
        assert dfp["n_errors"].iloc[0] == 1

    def test_parts_build_failure_soft_errors(self):
        # An impossible jmd length makes _build_parts raise; the member is recorded
        # as a soft error (df_feat is None, n_errors == 1) rather than crashing run().
        grid = _grid_seq()
        lst, dfp = grid.run(params_parts={"jmd_n_len": 10_000},
                            params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 10})
        assert lst[0] is None
        assert dfp["n_errors"].iloc[0] == 1

    def test_all_invalid_n_filter_soft_errors(self):
        # n_filter=0 is < 1 -> invalid; the whole group has no valid member (line 405).
        grid = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 0})
        assert lst[0] is None
        assert dfp["n_errors"].iloc[0] == 1

    def test_mixed_valid_invalid_n_filter(self):
        # One invalid (0) and one valid (10) value in the same n_filter group.
        grid = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": [0, 10]})
        n_err = dfp["n_errors"].tolist()
        assert sum(n_err) == 1
        # The valid member produced a table; the invalid one is None.
        assert sum(x is not None for x in lst) == 1


class TestRunCacheDedup:
    """Cache-dedup branch arms (359->357, 367->365): repeated sub-config keys are reused."""

    def test_duplicate_parts_axis_reuses_cache(self):
        # Two identical jmd_n_len candidates collapse to one parts-cache key, so the
        # second product entry hits the "key already cached" branch.
        grid = _grid_seq()
        lst, dfp = grid.run(params_parts={"jmd_n_len": [10, 10]},
                            params_split={"split_types": "Segment"},
                            params_scales=_scales(), params_cpp={"n_filter": 10})
        assert len(lst) == 2

    def test_duplicate_split_axis_reuses_cache(self):
        # Two identical split candidates collapse to one split-cache key.
        grid = _grid_seq()
        lst, dfp = grid.run(params_split={"split_types": ["Segment", "Segment"]},
                            params_scales=_scales(), params_cpp={"n_filter": 10})
        assert len(lst) == 2


class TestRunNWarnings:
    """n_warnings derivation arms (102/106): sparse-config / filter-shortfall."""

    def test_filter_shortfall_sets_n_warnings(self):
        # A large n_filter on a small sweep forces the filter funnel to fall short,
        # so the derived per-combo n_warnings is >= 1 for at least one config.
        grid = _grid_seq()
        _, dfp = grid.run(params_split={"split_types": "Segment"},
                          params_scales=_scales(), params_cpp={"n_filter": 5000})
        assert dfp["n_warnings"].iloc[0] >= 1


class TestEvalNoneFeat:
    """eval() None/empty df_feat arm (line 499): errored configs get NaN quality, sort last."""

    def test_eval_with_errored_config(self):
        grid = _grid_seq()
        grid.run(params_split={"split_types": "Segment"},
                 params_scales=_scales(), params_cpp={"n_filter": [0, 10]})
        df_eval = grid.eval()
        # The errored (n_filter=0) row carries NaN quality and sorts last.
        assert df_eval[ut.COL_AVG_ABS_AUC].isna().any()
        assert np.isnan(df_eval[ut.COL_AVG_ABS_AUC].iloc[-1])
