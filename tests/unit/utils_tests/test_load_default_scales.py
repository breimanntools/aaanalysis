"""This is a script to test ut.load_default_scales() and its private memoization helper.

Focus: the override-only contract. The default frame is memoized in a private
``@lru_cache`` loader, and the user-facing ``options['df_scales'|'df_cat']`` keys are an
override only — the library must never write them, so they stay ``None`` until the user
sets them (regression guard for the old options-as-cache leak).
"""
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut


class TestLoadDefaultScales:
    """Default path: bundled frame, no global-state side effects."""

    def test_returns_dataframe(self):
        assert isinstance(ut.load_default_scales(), pd.DataFrame)
        assert isinstance(ut.load_default_scales(scale_cat=True), pd.DataFrame)

    def test_default_path_does_not_write_df_scales_option(self):
        # The leak: load_default_scales must not populate the user-visible option key.
        assert aa.options["df_scales"] is None
        ut.load_default_scales()
        assert aa.options["df_scales"] is None

    def test_default_path_does_not_write_df_cat_option(self):
        assert aa.options["df_cat"] is None
        ut.load_default_scales(scale_cat=True)
        assert aa.options["df_cat"] is None

    def test_returns_independent_copies(self):
        a = ut.load_default_scales()
        b = ut.load_default_scales()
        assert a is not b  # fresh copy each call
        assert a.equals(b)

    def test_mutating_result_does_not_corrupt_cache(self):
        a = ut.load_default_scales()
        a.iloc[:, :] = 0.0
        b = ut.load_default_scales()
        assert not (b == 0.0).all().all()  # cached default untouched


class TestLoadDefaultScalesOverride:
    """Override path: options[...] set by the user wins and is returned as a copy."""

    def test_df_scales_override_returned(self):
        custom = pd.DataFrame({"A": [0.1], "B": [0.2]})
        aa.options["df_scales"] = custom
        out = ut.load_default_scales()
        assert out.equals(custom)
        assert out is not custom  # copy, not the user's object

    def test_df_cat_override_returned(self):
        custom = pd.DataFrame({"scale_id": ["X"], "category": ["c"]})
        aa.options["df_cat"] = custom
        out = ut.load_default_scales(scale_cat=True)
        assert out.equals(custom)
        assert out is not custom

    def test_override_does_not_leak_into_other_axis(self):
        custom = pd.DataFrame({"A": [0.1]})
        aa.options["df_scales"] = custom
        # scale_cat=True must still load the default categories, not the scales override.
        out_cat = ut.load_default_scales(scale_cat=True)
        assert not out_cat.equals(custom)
