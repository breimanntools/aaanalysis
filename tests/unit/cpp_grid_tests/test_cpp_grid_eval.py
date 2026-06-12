"""This is a script to test CPPGrid.eval() — scoring/ranking swept configurations."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering import CPPGrid
from aaanalysis.template_classes import Tool

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helpers --------------------------------------------------------------
def _grid_after_run(n=10, n_filter=(10, 25)):
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].to_list()
    grid = CPPGrid(df_seq=df_seq, labels=labels, n_jobs=1, random_state=0)
    grid.run(params_cpp={"n_filter": list(n_filter)})
    return grid


@pytest.fixture(scope="module")
def grid_default():
    """One default CPPGrid (n=10, n_filter=(10, 25)), run once and shared across the
    read-only eval tests. ``eval`` is non-mutating (it only reads ``df_params_`` /
    ``list_df_feat_`` and returns a fresh frame), so a single post-run grid is safe to
    reuse — this collapses ~13 redundant ``grid.run()`` calls into one per worker."""
    return _grid_after_run(n=10, n_filter=(10, 25))


# Normal cases ---------------------------------------------------------
class TestCPPGridEval:
    """Positive and parameter-level negative tests for eval."""

    def test_is_tool_subclass(self):
        assert issubclass(CPPGrid, Tool)

    def test_returns_dataframe(self, grid_default):
        assert isinstance(grid_default.eval(), pd.DataFrame)

    def test_one_row_per_config(self, grid_default):
        assert len(grid_default.eval()) == len(grid_default.df_params_)

    def test_has_quality_columns(self, grid_default):
        df_eval = grid_default.eval()
        for col in (ut.COL_AVG_ABS_AUC, "avg_abs_mean_dif", "n_features"):
            assert col in df_eval.columns

    def test_keeps_param_columns(self, grid_default):
        df_eval = grid_default.eval()
        assert "n_filter" in df_eval.columns

    def test_sorted_best_first_by_auc(self, grid_default):
        df_eval = grid_default.eval(sort_by=ut.COL_AVG_ABS_AUC)
        vals = df_eval[ut.COL_AVG_ABS_AUC].dropna().to_list()
        assert vals == sorted(vals, reverse=True)

    def test_index_maps_to_list_df_feat(self, grid_default):
        df_eval = grid_default.eval()
        i = df_eval.index[0]
        assert grid_default.list_df_feat_[i] is not None

    def test_avg_abs_auc_matches_manual_mean(self, grid_default):
        df_eval = grid_default.eval()
        i = df_eval.index[0]
        expected = float(grid_default.list_df_feat_[i][ut.COL_ABS_AUC].mean())
        assert np.isclose(df_eval.loc[i, ut.COL_AVG_ABS_AUC], expected)

    def test_n_features_matches_len(self, grid_default):
        df_eval = grid_default.eval()
        i = df_eval.index[0]
        assert df_eval.loc[i, "n_features"] == len(grid_default.list_df_feat_[i])

    def test_ascending_override(self, grid_default):
        df_eval = grid_default.eval(sort_by=ut.COL_AVG_ABS_AUC, ascending=True)
        vals = df_eval[ut.COL_AVG_ABS_AUC].dropna().to_list()
        assert vals == sorted(vals)

    # Negative tests
    def test_eval_before_run_raises(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=6)
        grid = CPPGrid(df_seq=df_seq, labels=df_seq["label"].to_list(), n_jobs=1)
        with pytest.raises(RuntimeError):
            grid.eval()

    def test_invalid_sort_by_column(self, grid_default):
        with pytest.raises(ValueError):
            grid_default.eval(sort_by="does_not_exist")

    def test_invalid_sort_by_type(self, grid_default):
        with pytest.raises(ValueError):
            grid_default.eval(sort_by=123)


# Combinations ---------------------------------------------------------
class TestCPPGridEvalComplex:
    """Combinations and edge interactions for eval."""

    def test_run_sets_attributes(self, grid_default):
        assert grid_default.df_params_ is not None and grid_default.list_df_feat_ is not None

    def test_best_feature_table_accessible(self, grid_default):
        df_eval = grid_default.eval()
        best = grid_default.list_df_feat_[df_eval.index[0]]
        assert isinstance(best, pd.DataFrame) and ut.COL_ABS_AUC in best.columns

    def test_single_config_eval(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        grid = CPPGrid(df_seq=df_seq, labels=df_seq["label"].to_list(), n_jobs=1, random_state=0)
        grid.run(params_cpp={"n_filter": 10})
        df_eval = grid.eval()
        assert len(df_eval) == 1

    def test_sort_by_n_features_ascending_default(self, grid_default):
        df_eval = grid_default.eval(sort_by="n_features")
        vals = df_eval["n_features"].dropna().to_list()
        assert vals == sorted(vals)

    def test_rerun_refreshes_eval(self):
        grid = _grid_after_run(n_filter=(10, 25))
        first = len(grid.eval())
        grid.run(params_cpp={"n_filter": 10})
        assert len(grid.eval()) == 1 and first == 2
