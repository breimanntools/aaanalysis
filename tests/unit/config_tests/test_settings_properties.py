"""This is a script to test property/golden invariants of the aaanalysis.options Settings object.

Complements test_options.py (validation branches + dunders) and test_n_jobs.py (n_jobs contract)
with the layer they lacked: a golden snapshot of the shipped defaults, a set->get roundtrip
property across every option key, the __str__ completeness invariant, and the "off"-override
contract expressed directly.
"""
import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.config import options, check_verbose


# Golden snapshot of the shipped defaults (freezing the public options surface).
_EXPECTED_DEFAULTS = {
    "verbose": "off",
    "random_state": "off",
    "n_jobs": "off",
    "allow_multiprocessing": True,
    "name_tmd": "TMD",
    "name_jmd_n": "JMD-N",
    "name_jmd_c": "JMD-C",
    "ext_len": 0,
    "jmd_n_len": None,
    "jmd_c_len": None,
    "df_scales": None,
    "df_cat": None,
    "auto_font": True,
    "legend_title_bold": False,
}

# One representative valid value per scalar key for the roundtrip property.
_VALID_VALUES = {
    "verbose": True,
    "random_state": 42,
    "n_jobs": 2,
    "allow_multiprocessing": False,
    "name_tmd": "MyTMD",
    "name_jmd_n": "N",
    "name_jmd_c": "C",
    "ext_len": 3,
    "jmd_n_len": 5,
    "jmd_c_len": 7,
    "auto_font": True,
    "legend_title_bold": True,
}


class TestSettingsGolden:
    """Frozen-value checks on the defaults and key set."""

    def test_default_values_match_golden(self):
        for key, expected in _EXPECTED_DEFAULTS.items():
            assert options[key] == expected, key

    def test_no_unexpected_keys(self):
        assert set(options._settings.keys()) == set(_EXPECTED_DEFAULTS.keys())


class TestSettingsProperties:
    """Invariants that must hold for any option key/value."""

    @pytest.mark.parametrize("key,value", list(_VALID_VALUES.items()))
    def test_set_then_get_roundtrip(self, key, value):
        options[key] = value
        assert options[key] == value

    @pytest.mark.parametrize("key", ["df_scales", "df_cat"])
    def test_dataframe_roundtrip(self, key):
        df = pd.DataFrame({"a": [0.1, 0.2]})
        options[key] = df
        assert options[key].equals(df)

    def test_str_contains_every_key(self):
        text = str(options)
        for key in _EXPECTED_DEFAULTS:
            assert key in text

    def test_contains_matches_keys(self):
        for key in _EXPECTED_DEFAULTS:
            assert key in options
        assert "not_a_real_option" not in options


class TestOffOverrideContract:
    """The "off" contract: per-call arg wins when the option is "off"; the option wins otherwise."""

    def test_off_defers_to_call_arg(self):
        options["verbose"] = "off"
        assert check_verbose(True) is True
        assert check_verbose(False) is False

    def test_set_option_overrides_call_arg(self):
        options["verbose"] = True
        assert check_verbose(False) is True  # option wins over the per-call argument
