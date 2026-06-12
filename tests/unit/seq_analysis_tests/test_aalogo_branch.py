"""Branch-coverage tests for AAlogo / AAlogoPlot, exercised ONLY through the public API.

Targets the under-covered guard / branch arms in:
* seq_analysis/_aalogo.py (no-required-part raise),
* seq_analysis/_aalogo_plot.py (tmd_len<1 raise, verbose prints),
* seq_analysis/_backend/_aalogo/aalogo.py (no-TMD early return),
* seq_analysis/_backend/_aalogo/aalogo_plot.py (P-site x-axis, probability scaling,
  left name annotation, bit-score bar).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import aaanalysis as aa


# Helpers
def _df_seq(n=12):
    return aa.load_dataset(name="DOM_GSEC", n=n)


def _df_parts(n=12):
    sf = aa.SequenceFeature()
    return sf.get_df_parts(df_seq=_df_seq(n=n))


def _df_logo(n=12, logo_type="probability"):
    return aa.AAlogo(logo_type=logo_type).get_df_logo(df_parts=_df_parts(n=n))


def _df_logo_info(n=12):
    return aa.AAlogo().get_df_logo_info(df_parts=_df_parts(n=n))


class TestGetDfLogoNoRequiredPart:
    """_aalogo.py line 24: df_parts has a valid part column but none of jmd_n/tmd/jmd_c."""

    def test_no_seq_part_column_raises(self):
        # 'tmd_n' is a valid part (passes check_df_parts) but is NOT one of
        # COLS_SEQ_PARTS, so check_match_df_parts_logo_parts must raise.
        df_parts = pd.DataFrame({"tmd_n": ["ACDEF", "GHIKL", "MNPQR"]})
        with pytest.raises(ValueError, match="at least one"):
            aa.AAlogo().get_df_logo(df_parts=df_parts)


class TestGetDfLogoNoTmdEarlyReturn:
    """backend aalogo.py line 13: _retrieve_tmd_aligned early-returns when no TMD column."""

    def test_jmd_only_parts_logo(self):
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=_df_seq(), list_parts=["jmd_n", "jmd_c"])
        assert "tmd" not in df_parts.columns
        df_logo = aa.AAlogo().get_df_logo(df_parts=df_parts)
        assert isinstance(df_logo, pd.DataFrame)
        assert len(df_logo) > 0


class TestSingleLogoPartsLenRaise:
    """_aalogo_plot.py line 49: jmd_n_len + jmd_c_len leave tmd_len < 1 -> raise."""

    def test_jmd_lens_exceed_logo_length_raises(self):
        df_logo = _df_logo()
        # JMD lengths sum to >= logo length, so derived tmd_len < 1.
        big = len(df_logo)
        aal_plot = aa.AAlogoPlot(jmd_n_len=big, jmd_c_len=big, verbose=False)
        with pytest.raises(ValueError, match="logo length"):
            aal_plot.single_logo(df_logo=df_logo)


class TestSingleLogoVerbose:
    """_aalogo_plot.py line 315: the verbose print arm in single_logo."""

    def test_verbose_single_logo_prints(self, capfd):
        df_logo = _df_logo()
        aal_plot = aa.AAlogoPlot(verbose=True)
        fig, _ = aal_plot.single_logo(df_logo=df_logo)
        out = capfd.readouterr().out
        assert "single logo" in out.lower()
        plt.close("all")


class TestSingleLogoPSites:
    """backend aalogo_plot.py line 105: target_p1_site replaces the JMD/TMD x-axis."""

    def test_target_p1_site_single(self):
        df_logo = _df_logo()
        aal_plot = aa.AAlogoPlot(verbose=False)
        fig, _ = aal_plot.single_logo(df_logo=df_logo, target_p1_site=5)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestSingleLogoNameDataLeft:
    """backend aalogo_plot.py line 62: name_data_pos='left' annotation arm."""

    def test_name_data_left(self):
        df_logo = _df_logo()
        aal_plot = aa.AAlogoPlot(verbose=False)
        fig, _ = aal_plot.single_logo(df_logo=df_logo, name_data="GSEC",
                                      name_data_pos="left")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestSingleLogoBitScoreBar:
    """backend aalogo_plot.py line 17: bit-score bar (show_right=True default) arm."""

    def test_with_df_logo_info(self):
        df_logo = _df_logo()
        df_logo_info = _df_logo_info()
        aal_plot = aa.AAlogoPlot(verbose=False)
        fig, axes = aal_plot.single_logo(df_logo=df_logo, df_logo_info=df_logo_info)
        # df_logo_info present -> a (ax_logo, ax_info) tuple is returned.
        assert isinstance(axes, tuple) and len(axes) == 2
        plt.close("all")


class TestMultiLogoVerbose:
    """_aalogo_plot.py line 478: the verbose print arm in multi_logo."""

    def test_verbose_multi_logo_prints(self, capfd):
        df_logo = _df_logo()
        aal_plot = aa.AAlogoPlot(verbose=True)
        fig, _ = aal_plot.multi_logo(list_df_logo=[df_logo, df_logo])
        out = capfd.readouterr().out
        assert "logos" in out.lower()
        plt.close("all")


class TestMultiLogoProbabilityScaling:
    """backend aalogo_plot.py lines 167/179/187: probability y_max scaling + yticks in multi."""

    def test_probability_multi_logo(self):
        # Probability logos sum to 1 per position -> y_max <= 1 -> the *100 scaling
        # arm (167/179) and the percentage yticks arm (187) all fire.
        df_logo = _df_logo(logo_type="probability")
        aal_plot = aa.AAlogoPlot(logo_type="probability", verbose=False)
        fig, axes = aal_plot.multi_logo(list_df_logo=[df_logo, df_logo])
        assert len(axes) == 2
        plt.close("all")


class TestMultiLogoPSites:
    """backend aalogo_plot.py line 192: target_p1_site in multi_logo."""

    def test_target_p1_site_multi(self):
        df_logo = _df_logo()
        aal_plot = aa.AAlogoPlot(verbose=False)
        fig, axes = aal_plot.multi_logo(list_df_logo=[df_logo, df_logo],
                                        target_p1_site=5)
        assert len(axes) == 2
        plt.close("all")
