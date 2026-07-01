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

    def test_non_numeric_score_raises(self):
        # A non-numeric score column must fail loudly (scatter mode), not draw garbage.
        df = pd.DataFrame({"score": ["hi", "lo", "mid"], "group": ["a", "a", "b"]})
        with pytest.raises(ValueError):
            plot_rank(df_rank=df)

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


# Helper for the additive ranked-candidates (bar) mode
def _df_cand(n_sub=3, n_non=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_sub + n_non
    names = [f"GENE{i}" for i in range(n)]
    return pd.DataFrame(
        {"score": rng.uniform(0, 100, n),
         "std": rng.uniform(1, 10, n),
         "class": ["substrate"] * n_sub + ["non-substrate"] * n_non},
        index=names,
    )


class TestPlotRankCandidatesMode:
    """The additive col_class / col_std ranked-candidates (horizontal-bar) variant."""

    def test_col_class_switches_to_bars(self):
        fig, ax = plot_rank(df_rank=_df_cand(), col_score="score", col_class="class")
        # barh produces a BarContainer (no scatter collections in this mode)
        assert len(ax.containers) >= 1 and len(ax.collections) == 0

    def test_yticklabels_are_candidate_names(self):
        df = _df_cand()
        fig, ax = plot_rank(df_rank=df, col_score="score", col_class="class")
        labels = {t.get_text() for t in ax.get_yticklabels()}
        assert labels == set(df.index.astype(str))

    def test_col_std_draws_error_bars(self):
        df = _df_cand()
        fig, ax = plot_rank(df_rank=df, col_score="score", col_class="class", col_std="std")
        # errorbar adds an extra container beyond the BarContainer
        assert len(ax.containers) >= 2

    def test_threshold_draws_vertical_line(self):
        fig, ax = plot_rank(df_rank=_df_cand(), col_score="score", col_class="class",
                            threshold=50)
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
        assert len(dashed) == 1

    def test_bar_colors_follow_class(self):
        df = _df_cand(n_sub=2, n_non=2)
        fig, ax = plot_rank(df_rank=df, col_score="score", col_class="class",
                            group_order=["substrate", "non-substrate"])
        green = matplotlib.colors.to_rgb(ut.COLOR_POS)
        facecolors = {tuple(np.round(p.get_facecolor()[:3], 4)) for p in ax.patches}
        assert tuple(np.round(green, 4)) in facecolors

    def test_default_labels_substituted_in_bar_mode(self):
        # scatter defaults ("Protein rank") must not leak onto the score axis
        fig, ax = plot_rank(df_rank=_df_cand(), col_score="score", col_class="class")
        assert ax.get_xlabel() == "Prediction score" and ax.get_ylabel() == ""

    def test_explicit_labels_respected_in_bar_mode(self):
        fig, ax = plot_rank(df_rank=_df_cand(), col_score="score", col_class="class",
                            xlabel="My score", ylabel="My genes")
        assert ax.get_xlabel() == "My score" and ax.get_ylabel() == "My genes"

    def test_col_std_without_col_class_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df_cand(), col_score="score", col_std="std")

    def test_missing_col_class_col_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df_cand(), col_score="score", col_class="nope")

    def test_missing_col_std_col_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df_cand(), col_score="score", col_class="class", col_std="nope")

    def test_col_class_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df_cand(), col_score="score", col_class=123)

    def test_col_std_wrong_type_raises(self):
        with pytest.raises(ValueError):
            plot_rank(df_rank=_df_cand(), col_score="score", col_class="class", col_std=123)

    def test_non_numeric_score_bar_mode_raises(self):
        df = pd.DataFrame({"score": ["hi", "lo"], "class": ["substrate", "non-substrate"]},
                          index=["A", "B"])
        with pytest.raises(ValueError):
            plot_rank(df_rank=df, col_score="score", col_class="class")

    def test_non_numeric_std_raises(self):
        df = _df_cand()
        df["std"] = df["std"].astype(str)
        with pytest.raises(ValueError):
            plot_rank(df_rank=df, col_score="score", col_class="class", col_std="std")

    def test_draws_on_passed_ax_bar_mode(self):
        fig0, ax0 = plt.subplots()
        fig, ax = plot_rank(df_rank=_df_cand(), col_score="score", col_class="class", ax=ax0)
        assert ax is ax0 and fig is fig0


class TestPlotRankDefaultRegression:
    """Guard: the default scatter path stays byte-identical to the pre-``col_class`` output.

    The expected values below are FROZEN from the scatter implementation before the additive
    ranked-candidates mode was added (``_df()`` seed=0, threshold=[0.5, 0.8]). They are NOT
    recomputed from the current code, so any change to the scatter branch (sort order, ranking,
    group->color mapping, or threshold drawing) makes these assertions fail.
    """

    # Golden snapshot of the first drawn group ("substrate") as (rank, score) pairs.
    GOLDEN_SUBSTRATE_OFFSETS = [
        (3.0, 0.935072), (4.0, 0.912756), (8.0, 0.81327), (10.0, 0.729497),
        (15.0, 0.636962), (17.0, 0.606636), (18.0, 0.543625), (23.0, 0.269787),
        (26.0, 0.040974), (29.0, 0.016528),
    ]
    GOLDEN_LABELS = ["substrate", "hold-out", "non-substrate"]
    GOLDEN_SIZES = [10, 5, 15]

    def test_scatter_offsets_frozen(self):
        # Exact ranked (rank, score) coordinates of the substrate group must not drift.
        fig, ax = plot_rank(df_rank=_df(), threshold=[0.5, 0.8])
        offs = [tuple(np.round(xy, 6)) for xy in ax.collections[0].get_offsets()]
        assert offs == [tuple(np.round(g, 6)) for g in self.GOLDEN_SUBSTRATE_OFFSETS]

    def test_scatter_groups_labels_and_sizes_frozen(self):
        fig, ax = plot_rank(df_rank=_df(), threshold=[0.5, 0.8])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        sizes = [len(c.get_offsets()) for c in ax.collections]
        assert labels == self.GOLDEN_LABELS and sizes == self.GOLDEN_SIZES

    def test_scatter_group_colors_frozen(self):
        fig, ax = plot_rank(df_rank=_df(),
                            group_order=["substrate", "hold-out", "non-substrate"])
        expected = [ut.COLOR_POS, ut.COLOR_REL_NEG, ut.COLOR_NEG]  # green / brown / magenta
        for coll, col in zip(ax.collections, expected):
            assert np.allclose(coll.get_facecolor()[0][:3], matplotlib.colors.to_rgb(col))

    def test_scatter_thresholds_are_horizontal_frozen(self):
        # Scatter mode draws thresholds as HORIZONTAL dashed lines at the given y-values.
        fig, ax = plot_rank(df_rank=_df(), threshold=[0.5, 0.8])
        lines = sorted((round(float(ln.get_ydata()[0]), 6), ln.get_linestyle())
                       for ln in ax.get_lines())
        assert lines == [(0.5, "--"), (0.8, "--")]
        # each threshold line is flat (constant y) -> truly horizontal
        assert all(len(set(np.round(ln.get_ydata(), 6))) == 1 for ln in ax.get_lines())

    def test_new_args_default_dont_touch_scatter(self):
        # Passing the new args at their explicit defaults yields exactly the golden scatter.
        fig, ax = plot_rank(df_rank=_df(), threshold=[0.5, 0.8], col_std=None, col_class=None)
        offs = [tuple(np.round(xy, 6)) for xy in ax.collections[0].get_offsets()]
        assert offs == [tuple(np.round(g, 6)) for g in self.GOLDEN_SUBSTRATE_OFFSETS]
        assert len(ax.patches) == 0  # scatter path, not the bar path

    def test_default_axis_labels_unchanged(self):
        fig, ax = plot_rank(df_rank=_df())
        assert ax.get_xlabel() == "Protein rank" and ax.get_ylabel() == "Max score per protein"
