"""
Tests the dense-plot row-label overlap gate for ``CPPPlot.feature_map``.

The scale subcategory row labels collide once the grid has more than a handful
of rows; at the full AAontology breadth (74 subcategories) the label column
becomes an unreadable blur. These tests force-render the figure and assert that
no two *row labels* overlap. They are the first layout/visual assertions in the
suite (plain matplotlib bbox introspection, no image snapshots).
"""
import io
import re

import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pytest

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.cpp.cpp_plot_feature_map import (
    derive_feature_map_figsize,
)

from ._text_overlap import get_label_overlaps, make_dense_df_feat


N_SUBCATS = [20, 36, 55, 74]
_NUMERIC = re.compile(r"^\[?-?\d+(\.\d+)?%?\]?$")


def _numeric_tick_overlaps(fig):
    """Mutual overlaps among the numeric tick labels *of a single axis* (targets
    the importance-bar ticks, e.g. the historical '40' over '0'). Scoped per axis
    so harmless cross-axis coincidences (colorbar '%' vs heatmap position ticks)
    are not flagged; the text row-label axis has no numeric ticks."""
    fig.savefig(io.BytesIO(), format="png", dpi=fig.dpi)
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        for ticklabels in (ax.get_xticklabels(), ax.get_yticklabels()):
            items = [(t.get_text().strip(), t.get_window_extent(renderer=r))
                     for t in ticklabels
                     if t.get_visible() and _NUMERIC.match(t.get_text().strip())]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    it = Bbox.intersection(items[i][1], items[j][1])
                    if it and it.width > 0 and it.height > 0:
                        bad.append((items[i][0], items[j][0]))
    return bad


class TestOverlapDetector:
    """Self-validation of the detector itself (independent of any plot fix)."""

    def test_detects_stacked_labels(self):
        # Three labels drawn at (almost) the same point must be flagged.
        fig, ax = plt.subplots()
        names = ["Alpha", "Beta", "Gamma"]
        for i, name in enumerate(names):
            ax.text(0.5, 0.5 + i * 0.0001, name, ha="center", va="center")
        bad = get_label_overlaps(fig, names)
        assert len(bad) > 0
        plt.close(fig)

    def test_no_false_positive_when_spaced(self):
        # Well-separated labels must report clean.
        fig, ax = plt.subplots(figsize=(6, 6))
        names = ["Alpha", "Beta", "Gamma"]
        for i, name in enumerate(names):
            ax.text(0.5, 0.1 + i * 0.4, name, ha="center", va="center")
        bad = get_label_overlaps(fig, names)
        assert bad == []
        plt.close(fig)

    def test_scoped_to_named_labels_only(self):
        # Text artists not in `names` are ignored even when overlapping.
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "IGNORED", ha="center", va="center")
        ax.text(0.5, 0.5, "ALSO_IGNORED", ha="center", va="center")
        bad = get_label_overlaps(fig, names=["Something", "Else"])
        assert bad == []
        plt.close(fig)


class TestFeatureMapLabelOverlap:
    """The row-label overlap gate for feature_map at increasing grid density."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        aa.options["verbose"] = "off"
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", N_SUBCATS)
    def test_no_row_label_overlap(self, n_subcat):
        df_feat = make_dense_df_feat(n_subcat)
        subcats = list(dict.fromkeys(df_feat["subcategory"]))
        fig, ax = aa.CPPPlot().feature_map(df_feat)
        bad = get_label_overlaps(fig, subcats)
        assert bad == [], (
            f"{len(bad)} overlapping subcategory row labels at "
            f"n_subcat={n_subcat} (first few: {bad[:3]})"
        )

    @pytest.mark.parametrize("n_subcat", [20, 74])
    def test_no_importance_bar_tick_overlap(self, n_subcat):
        # The top/right cumulative-importance bar numeric ticks must not collide
        # (historically '40' over '0'); independent of grid density.
        df_feat = make_dense_df_feat(n_subcat)
        fig, ax = aa.CPPPlot().feature_map(df_feat)
        bad = _numeric_tick_overlaps(fig)
        assert bad == [], f"overlapping numeric ticks: {bad}"

    @pytest.mark.parametrize("n_subcat", N_SUBCATS)
    def test_row_label_font_above_floor(self, n_subcat):
        # Shrinking to fit must never take the row-label font below ~5pt.
        df_feat = make_dense_df_feat(n_subcat)
        subcats = set(df_feat["subcategory"])
        fig, ax = aa.CPPPlot().feature_map(df_feat)
        fig.canvas.draw()
        sizes = [t.get_fontsize() for a in fig.axes for t in a.texts
                 if t.get_text().strip() in subcats and t.get_visible()]
        assert sizes, "no row-label text artists found"
        assert min(sizes) >= 5.0, f"row-label font {min(sizes)}pt < 5pt floor"


class TestHeatmapLabelOverlap:
    """The same row-label gate applies to standalone heatmap (shared plot path)."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        aa.options["verbose"] = "off"
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", [20, 74])
    def test_no_row_label_overlap(self, n_subcat):
        df_feat = make_dense_df_feat(n_subcat)
        subcats = list(dict.fromkeys(df_feat["subcategory"]))
        fig, ax = aa.CPPPlot().heatmap(df_feat)
        assert get_label_overlaps(fig, subcats) == []


class TestOptionIsolation:
    """The autouse reset fixture must clear auto_font between tests (no leakage)."""

    def test_auto_font_defaults_false_each_test(self):
        # If a prior test left auto_font=True and the reset fixture missed it,
        # this would fail. Guards that 'auto_font' is in the reset defaults.
        assert aa.options["auto_font"] is False


class TestAutoFontFeatureMap:
    """The global auto_font option: off = unchanged size; on = grid-derived size."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        aa.options["auto_font"] = False
        aa.options["verbose"] = "off"
        plt.close("all")

    def test_derive_figsize_monotonic_and_bounded(self):
        f = derive_feature_map_figsize
        # grows with rows and columns
        assert f(n_subcat=74, n_positions=40)[1] > f(n_subcat=20, n_positions=40)[1]
        assert f(n_subcat=40, n_positions=80)[0] > f(n_subcat=40, n_positions=20)[0]
        # clamped to sane bounds
        w, h = f(n_subcat=5000, n_positions=5000)
        assert 6.0 <= w <= 20.0 and 4.0 <= h <= 20.0
        w, h = f(n_subcat=1, n_positions=1)
        assert 6.0 <= w <= 20.0 and 4.0 <= h <= 20.0

    def test_off_keeps_default_figsize(self):
        aa.options["auto_font"] = False
        df_feat = make_dense_df_feat(74)
        fig, ax = aa.CPPPlot().feature_map(df_feat)
        assert tuple(round(float(v), 2) for v in fig.get_size_inches()) == (8.0, 8.0)

    def test_on_derives_larger_figure_for_dense_grid(self):
        df_feat = make_dense_df_feat(74)
        aa.options["auto_font"] = True
        fig, ax = aa.CPPPlot().feature_map(df_feat)
        _, h = (float(v) for v in fig.get_size_inches())
        assert h > 8.0  # dense grid grew beyond the (8, 8) default

    def test_on_improves_legibility_no_overlap(self):
        df_feat = make_dense_df_feat(74)
        subcats = list(dict.fromkeys(df_feat["subcategory"]))

        aa.options["auto_font"] = False
        fig_off, _ = aa.CPPPlot().feature_map(df_feat)
        fig_off.canvas.draw()
        fs_off = min(t.get_fontsize() for a in fig_off.axes for t in a.texts
                     if t.get_text().strip() in set(subcats))

        aa.options["auto_font"] = True
        fig_on, _ = aa.CPPPlot().feature_map(df_feat)
        fig_on.canvas.draw()
        fs_on = min(t.get_fontsize() for a in fig_on.axes for t in a.texts
                    if t.get_text().strip() in set(subcats))

        assert fs_on >= fs_off >= 5.0  # auto_font is at least as legible, never below floor
        assert get_label_overlaps(fig_on, subcats) == []

    def test_explicit_figsize_wins_over_auto_font(self):
        df_feat = make_dense_df_feat(74)
        aa.options["auto_font"] = True
        fig, ax = aa.CPPPlot().feature_map(df_feat, figsize=(10, 6))
        assert tuple(round(float(v), 2) for v in fig.get_size_inches()) == (10.0, 6.0)

    def test_figsize_none_is_auto_derived_when_on(self):
        # figsize=None with auto_font on must not crash and should derive a size.
        df_feat = make_dense_df_feat(74)
        aa.options["auto_font"] = True
        fig, ax = aa.CPPPlot().feature_map(df_feat, figsize=None)
        _, h = (float(v) for v in fig.get_size_inches())
        assert h > 8.0
        assert get_label_overlaps(fig, list(dict.fromkeys(df_feat["subcategory"]))) == []
