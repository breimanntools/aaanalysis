"""Tests for the session-persistent ``options['plot_settings']`` surface (issue #131).

Covers: the default (unset) path is a byte-identical no-op; a set dict reproduces an
explicit :func:`aaanalysis.plot_settings` call (both via the helper and through the real
shared plot entry point); an explicit call is never overridden; and invalid values raise
the house ``ValueError``.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa
from aaanalysis.plotting import _plot_settings as ps


@pytest.fixture(autouse=True)
def _restore_plot_state():
    """Isolate global rcParams / options / session state mutated by these tests."""
    yield
    aa.options["plot_settings"] = None
    ps._LAST_APPLIED_PLOT_SETTINGS = None
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close("all")


class TestPlotSettingsOption:
    """The opt-in dict, its lazy application, and validation."""

    def test_default_is_none(self):
        assert aa.options["plot_settings"] is None

    def test_none_accepted(self):
        aa.options["plot_settings"] = None
        assert aa.options["plot_settings"] is None

    # (b) Unset -> byte-identical: the helper must not touch rcParams
    def test_unset_is_byte_identical_noop(self):
        ps._LAST_APPLIED_PLOT_SETTINGS = None
        aa.options["plot_settings"] = None
        mpl.rcParams.update(mpl.rcParamsDefault)
        before = dict(mpl.rcParams)
        ps.apply_session_plot_settings()
        after = dict(mpl.rcParams)
        assert before == after
        assert ps._LAST_APPLIED_PLOT_SETTINGS is None

    def test_empty_dict_is_noop(self):
        ps._LAST_APPLIED_PLOT_SETTINGS = None
        aa.options["plot_settings"] = {}
        mpl.rcParams.update(mpl.rcParamsDefault)
        before = dict(mpl.rcParams)
        ps.apply_session_plot_settings()
        assert dict(mpl.rcParams) == before
        assert ps._LAST_APPLIED_PLOT_SETTINGS is None

    # (a) Set -> matches an explicit plot_settings(**kws) call
    def test_set_matches_explicit_plot_settings(self):
        kws = dict(font_scale=1.7, weight_bold=True, no_ticks=True)
        ps._LAST_APPLIED_PLOT_SETTINGS = None
        aa.options["plot_settings"] = kws
        ps.apply_session_plot_settings()
        rc_session = dict(mpl.rcParams)
        aa.plot_settings(**kws)
        rc_explicit = dict(mpl.rcParams)
        assert rc_session == rc_explicit

    # (a) Regression through the real shared plot entry point (plot_gco), not the helper
    def test_set_applies_via_real_plot_entry_point(self):
        kws = dict(font_scale=1.5)
        ps._LAST_APPLIED_PLOT_SETTINGS = None
        mpl.rcParams.update(mpl.rcParamsDefault)
        aa.options["plot_settings"] = kws
        df_feat = aa.load_features()
        aa.CPPPlot().feature_map(df_feat=df_feat)
        rc_via_plot = dict(mpl.rcParams)
        plt.close("all")
        assert ps._LAST_APPLIED_PLOT_SETTINGS == kws
        aa.plot_settings(**kws)
        assert rc_via_plot["font.size"] == mpl.rcParams["font.size"]

    # An explicit call made after the style is in place is preserved (takes precedence)
    def test_explicit_call_preserved_while_option_unchanged(self):
        ps._LAST_APPLIED_PLOT_SETTINGS = None
        aa.options["plot_settings"] = dict(font_scale=1.2)
        ps.apply_session_plot_settings()          # session style now applied
        aa.plot_settings(font_scale=2.0)          # explicit override for subsequent figures
        styled = dict(mpl.rcParams)
        ps.apply_session_plot_settings()          # option unchanged -> not re-applied
        assert dict(mpl.rcParams) == styled

    # Idempotent for an unchanged value, but re-applies when the option value changes
    def test_reapplies_only_on_value_change(self):
        ps._LAST_APPLIED_PLOT_SETTINGS = None
        aa.options["plot_settings"] = dict(font_scale=1.3)
        ps.apply_session_plot_settings()
        first = dict(mpl.rcParams)
        ps.apply_session_plot_settings()          # same value -> no-op
        assert dict(mpl.rcParams) == first
        aa.options["plot_settings"] = dict(font_scale=3.0)
        ps.apply_session_plot_settings()          # changed value -> re-applied
        assert mpl.rcParams["font.size"] != first["font.size"]

    # (c) Invalid values raise the house ValueError
    def test_invalid_int_raises(self):
        with pytest.raises(ValueError):
            aa.options["plot_settings"] = 5

    def test_invalid_str_raises(self):
        with pytest.raises(ValueError):
            aa.options["plot_settings"] = "talk"

    def test_invalid_list_raises(self):
        with pytest.raises(ValueError):
            aa.options["plot_settings"] = ["font_scale"]
