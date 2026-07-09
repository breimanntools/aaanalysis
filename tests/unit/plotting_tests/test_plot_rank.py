"""This is a script to test the plot_rank() per-protein rank scatter (D12)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.plotting import plot_rank

aa.options["verbose"] = False


# Helper functions
def _df(n_sub=10, n_hold=5, n_non=15, seed=0):
    rng = np.random.default_rng(seed)
    n = n_sub + n_hold + n_non
    return pd.DataFrame({"score": rng.random(n),
                         "group": ["substrate"] * n_sub + ["hold-out"] * n_hold
                                  + ["non-substrate"] * n_non})


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestPlotRank:
    """Normal cases for plot_rank."""

    def test_returns_fig_ax(self):
        fig, ax = plot_rank(df_rank=_df())
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    def test_auto_font_width_grows_with_n_else_fixed(self):
        # Omitted figsize participates in auto_font: on -> width grows with the number of
        # ranked proteins; explicit figsize or auto_font off -> the fixed (7, 5) default.
        def width(n):
            plot_rank(df_rank=_df(n_sub=n, n_hold=0, n_non=0))
            w = float(plt.gcf().get_size_inches()[0]); plt.close("all"); return w
        aa.options["auto_font"] = True
        assert width(600) > width(20)                                  # grows with N
        plot_rank(df_rank=_df(n_sub=300, n_hold=0, n_non=0), figsize=(7, 5))
        assert round(float(plt.gcf().get_size_inches()[0]), 1) == 7.0   # explicit honored
        plt.close("all")
        aa.options["auto_font"] = False
        plot_rank(df_rank=_df(n_sub=300, n_hold=0, n_non=0))
        assert round(float(plt.gcf().get_size_inches()[0]), 1) == 7.0   # off -> fixed
        aa.options["auto_font"] = True

    def test_one_collection_per_group(self):
        fig, ax = plot_rank(df_rank=_df())
        assert len(ax.collections) == 3

    def test_legend_labels_match_groups(self):
        fig, ax = plot_rank(df_rank=_df())
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert set(labels) == {"substrate", "hold-out", "non-substrate"}

    def test_threshold_list_draws_lines(self):
        fig, ax = plot_rank(df_rank=_df(), threshold=[0.5, 0.8])
        assert len(ax.get_lines()) == 2

    def test_threshold_scalar_draws_one_line(self):
        fig, ax = plot_rank(df_rank=_df(), threshold=0.5)
        assert len(ax.get_lines()) == 1

    def test_no_threshold_no_lines(self):
        fig, ax = plot_rank(df_rank=_df())
        assert len(ax.get_lines()) == 0

    def test_canonical_substrate_color_is_green(self):
        fig, ax = plot_rank(df_rank=_df(), group_order=["substrate", "hold-out", "non-substrate"])
        # first scatter == substrate -> COLOR_POS
        face = ax.collections[0].get_facecolor()[0]
        assert np.allclose(face[:3], matplotlib.colors.to_rgb(ut.COLOR_POS))

    def test_custom_dict_color_applied(self):
        fig, ax = plot_rank(df_rank=_df(n_sub=5, n_hold=0, n_non=5),
                            group_order=["substrate", "non-substrate"],
                            dict_color={"substrate": "#000000", "non-substrate": "#ffffff"})
        assert np.allclose(ax.collections[0].get_facecolor()[0][:3], (0, 0, 0))

    def test_draws_on_passed_ax(self):
        fig0, ax0 = plt.subplots()
        fig, ax = plot_rank(df_rank=_df(), ax=ax0)
        assert ax is ax0 and fig is fig0

    def test_ranking_is_descending(self):
        fig, ax = plot_rank(df_rank=_df())
        xs, ys = [], []
        for coll in ax.collections:
            offs = coll.get_offsets()
            xs.extend(offs[:, 0]); ys.extend(offs[:, 1])
        order = np.argsort(xs)
        ys_by_rank = np.array(ys)[order]
        assert np.all(np.diff(ys_by_rank) <= 1e-9)  # score non-increasing with rank

    def test_custom_columns(self):
        df = _df().rename(columns={"score": "s", "group": "g"})
        fig, ax = plot_rank(df_rank=df, col_score="s", col_group="g")
        assert len(ax.collections) == 3

    def test_group_order_controls_draw_order(self):
        df = _df(n_sub=4, n_hold=0, n_non=4)
        order = ["non-substrate", "substrate"]
        fig, ax = plot_rank(df_rank=df, group_order=order)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == order

    def test_figsize_applied(self):
        fig, ax = plot_rank(df_rank=_df(), figsize=(10, 4))
        assert tuple(fig.get_size_inches()) == (10.0, 4.0)

    def test_marker_size_custom_applied(self):
        fig, ax = plot_rank(df_rank=_df(), marker_size=99)
        assert np.allclose(ax.collections[0].get_sizes(), 99)

    def test_xlabel_custom(self):
        fig, ax = plot_rank(df_rank=_df(), xlabel="Rank X")
        assert ax.get_xlabel() == "Rank X"

    def test_ylabel_custom(self):
        fig, ax = plot_rank(df_rank=_df(), ylabel="Score Y")
        assert ax.get_ylabel() == "Score Y"

    def test_fontsize_labels_applied(self):
        fig, ax = plot_rank(df_rank=_df(), fontsize_labels=15)
        assert ax.xaxis.label.get_size() == 15

    def test_dict_color_partial_with_fallback(self):
        df = _df(n_sub=5, n_hold=5, n_non=5)
        fig, ax = plot_rank(df_rank=df, group_order=["substrate", "hold-out", "non-substrate"],
                            dict_color={"substrate": "#123456"})
        # explicit color wins for substrate; the others fall back without error
        assert np.allclose(ax.collections[0].get_facecolor()[0][:3],
                           matplotlib.colors.to_rgb("#123456"))


class TestPlotRankComplex:
    """Negative cases and combinations."""

    def test_missing_score_col_raises(self):
        df = _df().drop(columns=["score"])
        with pytest.raises(ValueError):
            plot_rank(df_rank=df)

    def test_missing_group_col_raises(self):
        df = _df().drop(columns=["group"])
        with pytest.raises(ValueError):
            plot_rank(df_rank=df)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(n_sub=0, n_hold=0, n_non=0))

    def test_group_order_missing_group_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), group_order=["substrate"])  # omits two present groups

    @pytest.mark.parametrize("bad", ["x", None, [0.5, "y"]])
    def test_threshold_non_numeric_raises(self, bad):
        if bad is None:
            return  # None means "no threshold" -> valid; skip
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), threshold=bad)

    def test_negative_marker_size_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), marker_size=-5)

    def test_single_group(self):
        df = pd.DataFrame({"score": [0.1, 0.9, 0.5], "group": ["a", "a", "a"]})
        fig, ax = plot_rank(df_rank=df)
        assert len(ax.collections) == 1

    def test_unknown_groups_use_fallback_palette(self):
        df = pd.DataFrame({"score": np.linspace(0, 1, 6),
                           "group": ["g1", "g2", "g3", "g1", "g2", "g3"]})
        fig, ax = plot_rank(df_rank=df)
        assert len(ax.collections) == 3

    def test_negative_threshold_allowed(self):
        df = pd.DataFrame({"score": [-0.5, 0.2, 0.8], "group": ["a", "a", "a"]})
        fig, ax = plot_rank(df_rank=df, threshold=-0.1)
        assert len(ax.get_lines()) == 1

    def test_bad_col_score_name_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), col_score="nope")

    def test_bad_col_group_name_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), col_group="nope")

    # Wrong-TYPE negatives for the cosmetic params (clean ValueError, not a deep mpl crash)
    def test_figsize_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), figsize="big")

    def test_figsize_wrong_length_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), figsize=(1, 2, 3))

    def test_xlabel_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), xlabel=123)

    def test_ylabel_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), ylabel=123)

    def test_fontsize_labels_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), fontsize_labels="big")

    def test_ax_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), ax="not_an_axes")

    def test_dict_color_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), dict_color="red")

    def test_group_order_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df(), group_order="substrate")
