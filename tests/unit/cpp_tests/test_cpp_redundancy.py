"""Tests for the CPP.run redundancy criterion (``redundancy='legacy'|'exact'``)."""
import inspect
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


def _setup():
    df_seq = aa.load_dataset(name="DOM_GSEC")
    labels = df_seq[ut.COL_LABEL].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    return df_parts, labels


class TestCPPRedundancy:
    def test_legacy_is_the_default(self):
        # default and explicit 'legacy' must be byte-identical (reproducibility contract)
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        d_default = cpp.run(labels=labels, n_jobs=1).reset_index(drop=True)
        d_legacy = cpp.run(labels=labels, redundancy="legacy", n_jobs=1).reset_index(drop=True)
        pd.testing.assert_frame_equal(d_default, d_legacy)

    def test_exact_differs_from_legacy(self):
        # DOM_GSEC has multi-digit positions -> the two criteria select different features
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        legacy = set(cpp.run(labels=labels, redundancy="legacy", n_jobs=1)[ut.COL_FEATURE])
        exact = set(cpp.run(labels=labels, redundancy="exact", n_jobs=1)[ut.COL_FEATURE])
        assert legacy != exact
        assert 0 < len(exact) <= 100

    def test_exact_is_deterministic(self):
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False, random_state=42)
        a = set(cpp.run(labels=labels, redundancy="exact", n_jobs=1)[ut.COL_FEATURE])
        b = set(cpp.run(labels=labels, redundancy="exact", n_jobs=1)[ut.COL_FEATURE])
        assert a == b

    def test_invalid_redundancy_raises(self):
        df_parts, labels = _setup()
        cpp = aa.CPP(df_parts=df_parts, verbose=False)
        with pytest.raises(ValueError):
            cpp.run(labels=labels, redundancy="not-a-mode", n_jobs=1)

    def test_run_num_exposes_redundancy_default_legacy(self):
        params = inspect.signature(aa.CPP.run_num).parameters
        assert "redundancy" in params
        assert params["redundancy"].default == "legacy"
