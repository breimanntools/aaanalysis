"""This is a script to test the aaanalysis.options Settings surface: the
verbose / random_state / jmd-length option overrides, _check_option validation
branches, and the dict-like Settings dunder methods.

Complements test_n_jobs.py (which covers the n_jobs / resolve_n_jobs contract).
"""
import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.config import (
    check_verbose,
    check_random_state,
    check_auto_font,
    check_legend_title_bold,
    check_jmd_n_len,
    check_jmd_c_len,
    options,
)


class TestOptionOverrides:
    """Per-call value is used when the option is 'off'/None; the option wins otherwise."""

    # verbose
    def test_verbose_option_overrides_per_call(self):
        aa.options["verbose"] = True
        assert check_verbose(verbose=False) is True

    def test_verbose_off_uses_per_call(self):
        assert aa.options["verbose"] == "off"
        assert check_verbose(verbose=True) is True

    # random_state
    def test_random_state_option_overrides_per_call(self):
        aa.options["random_state"] = 42
        assert check_random_state(random_state=7) == 42

    def test_random_state_off_uses_per_call(self):
        assert aa.options["random_state"] == "off"
        assert check_random_state(random_state=7) == 7

    def test_random_state_off_accepts_none(self):
        assert check_random_state(random_state=None) is None

    # jmd_n_len / jmd_c_len
    def test_jmd_n_len_option_overrides_per_call(self):
        aa.options["jmd_n_len"] = 12
        assert check_jmd_n_len(jmd_n_len=5) == 12

    def test_jmd_n_len_none_uses_per_call(self):
        assert aa.options["jmd_n_len"] is None
        assert check_jmd_n_len(jmd_n_len=5) == 5

    def test_jmd_c_len_option_overrides_per_call(self):
        aa.options["jmd_c_len"] = 8
        assert check_jmd_c_len(jmd_c_len=3) == 8

    def test_jmd_c_len_none_uses_per_call(self):
        assert aa.options["jmd_c_len"] is None
        assert check_jmd_c_len(jmd_c_len=3) == 3


class TestCheckOptionBranches:
    """Setting each option routes through _check_option's validation branch."""

    # Positive: valid values accepted and stored
    def test_set_random_state(self):
        aa.options["random_state"] = 99
        assert aa.options["random_state"] == 99

    def test_set_jmd_n_len(self):
        aa.options["jmd_n_len"] = 10
        assert aa.options["jmd_n_len"] == 10

    def test_set_jmd_c_len(self):
        aa.options["jmd_c_len"] = 11
        assert aa.options["jmd_c_len"] == 11

    def test_set_name_jmd_n(self):
        aa.options["name_jmd_n"] = "Nterm"
        assert aa.options["name_jmd_n"] == "Nterm"

    def test_set_ext_len(self):
        aa.options["ext_len"] = 3
        assert aa.options["ext_len"] == 3

    def test_set_df_scales(self):
        df = pd.DataFrame({"A": [0.1, 0.2]})
        aa.options["df_scales"] = df
        assert aa.options["df_scales"] is df

    def test_auto_font_default_on(self):
        assert aa.options["auto_font"] is True
        assert check_auto_font() is True

    def test_set_auto_font(self):
        try:
            aa.options["auto_font"] = True
            assert aa.options["auto_font"] is True
            assert check_auto_font() is True
        finally:
            aa.options["auto_font"] = False

    def test_legend_title_bold_default_off(self):
        assert aa.options["legend_title_bold"] is False
        assert check_legend_title_bold() is False

    def test_set_legend_title_bold(self):
        try:
            aa.options["legend_title_bold"] = True
            assert aa.options["legend_title_bold"] is True
            assert check_legend_title_bold() is True
        finally:
            aa.options["legend_title_bold"] = False

    def test_invalid_legend_title_bold_type(self):
        with pytest.raises(ValueError):
            aa.options["legend_title_bold"] = "yes"

    # Negative: invalid values rejected at assignment time
    def test_invalid_random_state(self):
        with pytest.raises(ValueError):
            aa.options["random_state"] = -5

    def test_invalid_jmd_n_len(self):
        with pytest.raises(ValueError):
            aa.options["jmd_n_len"] = -1

    def test_invalid_name_jmd_n_none(self):
        with pytest.raises(ValueError):
            aa.options["name_jmd_n"] = None

    def test_invalid_ext_len_negative(self):
        with pytest.raises(ValueError):
            aa.options["ext_len"] = -2

    def test_invalid_ext_len_float(self):
        with pytest.raises(ValueError):
            aa.options["ext_len"] = 1.5

    def test_invalid_df_scales_type(self):
        with pytest.raises(ValueError):
            aa.options["df_scales"] = "not_a_df"

    def test_invalid_auto_font_type(self):
        with pytest.raises(ValueError):
            aa.options["auto_font"] = "yes"


class TestSettingsDunders:
    """Dict-like interface of the Settings instance."""

    def test_contains_known_key(self):
        assert "verbose" in aa.options

    def test_contains_unknown_key(self):
        assert "not_an_option" not in aa.options

    def test_str_returns_dict_repr(self):
        s = str(aa.options)
        assert s.startswith("{") and "verbose" in s

    def test_getitem_unknown_key_returns_none(self):
        assert aa.options["not_an_option"] is None

    def test_setitem_unknown_key_raises(self):
        with pytest.raises(KeyError):
            aa.options["not_an_option"] = 1

    def test_singleton_is_module_options(self):
        assert aa.options is options
