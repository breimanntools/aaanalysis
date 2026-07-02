"""
Layout invariants for the ``auto_font`` constant-cell-size sizing of the CPP
composite plots (``CPPPlot.feature_map`` / ``heatmap`` / ``profile``).

The guarantee under ``auto_font=True`` (the default) is: every grid *cell* keeps a
constant physical size regardless of how many subcategory rows or residue-position
columns the grid has — the *figure* grows instead. That keeps fonts fixed (they are
measured in points, so they do not scale with the figure) and therefore legible and
non-overlapping at any data size, without the caller hand-tuning
``plot_settings(font_scale=...)``.

These assert the invariants directly (constant cell size; font neither shrunk nor
enlarged; no overlap; figure grows) rather than fragile absolute-pixel snapshots,
so they are robust to the freetype/matplotlib build. The pixel-exact check lives in
the pytest-mpl ``visual_regression`` suite.
"""
import io
import re

import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pytest

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.cpp._utils_cpp_plot_sizing import (
    CELL_W_IN,
    CELL_H_IN,
)

from ._text_overlap import get_label_overlaps, make_dense_df_feat


N_SUBCATS = [20, 36, 55, 74]
# Grids dense enough that the auto figure exceeds the default (8, 8) floor, so the
# constant-cell-size invariant applies (below the floor, auto_font only grows the
# figure to the default and cells are larger — see TestSmallGridFloor).
DENSE_SUBCATS = [40, 55, 74]
N_POSITIONS = 40  # default tmd_len 20 + jmd_n 10 + jmd_c 10
CELL_TOL = 0.06   # relative tolerance on the per-cell size (freetype/layout jitter)


def _cell_size(fig, ax, n_rows, n_cols):
    """Per-cell (width, height) in inches from the axes' figure-fraction * figsize."""
    fig.canvas.draw()
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    pos = ax.get_position()
    return pos.width * w_in / n_cols, pos.height * h_in / n_rows


def _row_label_fonts(fig, subcats):
    fig.canvas.draw()
    subcats = set(subcats)
    return [t.get_fontsize() for a in fig.axes for t in a.texts
            if t.get_text().strip() in subcats and t.get_visible()]


_NUMERIC = re.compile(r"^\[?-?\d+(\.\d+)?%?\]?$")


def _numeric_tick_overlaps(fig):
    """Mutual overlaps among numeric tick labels of a single axis (targets the
    importance-bar ticks, historically '40' over '0')."""
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
    """Self-validation of the overlap detector itself (independent of any fix)."""

    def test_detects_stacked_labels(self):
        fig, ax = plt.subplots()
        names = ["Alpha", "Beta", "Gamma"]
        for i, name in enumerate(names):
            ax.text(0.5, 0.5 + i * 0.0001, name, ha="center", va="center")
        assert len(get_label_overlaps(fig, names)) > 0
        plt.close(fig)

    def test_no_false_positive_when_spaced(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        names = ["Alpha", "Beta", "Gamma"]
        for i, name in enumerate(names):
            ax.text(0.5, 0.1 + i * 0.4, name, ha="center", va="center")
        assert get_label_overlaps(fig, names) == []
        plt.close(fig)

    def test_scoped_to_named_labels_only(self):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "IGNORED", ha="center", va="center")
        ax.text(0.5, 0.5, "ALSO_IGNORED", ha="center", va="center")
        assert get_label_overlaps(fig, names=["Something", "Else"]) == []
        plt.close(fig)


class TestConstantCellSize:
    """auto_font=True holds each cell at a constant physical size across densities."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", DENSE_SUBCATS)
    def test_feature_map_cell_size_is_target(self, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat))
        cw, ch = _cell_size(fig, ax, n_subcat, N_POSITIONS)
        assert abs(cw - CELL_W_IN) <= CELL_W_IN * CELL_TOL, cw
        assert abs(ch - CELL_H_IN) <= CELL_H_IN * CELL_TOL, ch

    @pytest.mark.parametrize("n_subcat", [55, 74])
    def test_heatmap_cell_size_is_target(self, n_subcat):
        fig, ax = aa.CPPPlot().heatmap(make_dense_df_feat(n_subcat))
        cw, ch = _cell_size(fig, ax, n_subcat, N_POSITIONS)
        assert abs(cw - CELL_W_IN) <= CELL_W_IN * CELL_TOL, cw
        assert abs(ch - CELL_H_IN) <= CELL_H_IN * CELL_TOL, ch

    def test_cell_size_constant_across_densities(self):
        # The whole point: the cell does NOT shrink as the grid gets denser (above the floor).
        heights = []
        for n in (40, 74):
            fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n))
            heights.append(_cell_size(fig, ax, n, N_POSITIONS)[1])
        assert abs(heights[0] - heights[1]) <= CELL_H_IN * CELL_TOL, heights


class TestSmallGridFloor:
    """auto_font only GROWS the figure: sparse grids never collapse below the default."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", [1, 5, 15])
    def test_feature_map_not_smaller_than_default(self, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat))
        w, h = (float(v) for v in fig.get_size_inches())
        # Never shrink below the (8, 8) default footprint (so decorations/fonts fit).
        assert w >= 8.0 - 1e-6 and h >= 8.0 - 1e-6, (n_subcat, w, h)

    @pytest.mark.parametrize("n_subcat", [1, 5, 15])
    def test_heatmap_not_smaller_than_default(self, n_subcat):
        fig, ax = aa.CPPPlot().heatmap(make_dense_df_feat(n_subcat))
        w, h = (float(v) for v in fig.get_size_inches())
        assert w >= 8.0 - 1e-6 and h >= 8.0 - 1e-6, (n_subcat, w, h)


class TestFontStableAndNoOverlap:
    """Fonts stay fixed (never shrunk, never gigantic) and labels never overlap."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", N_SUBCATS)
    def test_feature_map_no_row_label_overlap(self, n_subcat):
        df_feat = make_dense_df_feat(n_subcat)
        subcats = list(dict.fromkeys(df_feat["subcategory"]))
        fig, ax = aa.CPPPlot().feature_map(df_feat)
        bad = get_label_overlaps(fig, subcats)
        assert bad == [], f"{len(bad)} overlaps at n={n_subcat} (first: {bad[:3]})"

    @pytest.mark.parametrize("n_subcat", [20, 74])
    def test_heatmap_no_row_label_overlap(self, n_subcat):
        df_feat = make_dense_df_feat(n_subcat)
        subcats = list(dict.fromkeys(df_feat["subcategory"]))
        fig, ax = aa.CPPPlot().heatmap(df_feat)
        assert get_label_overlaps(fig, subcats) == []

    def test_font_not_shrunk_and_constant_across_densities(self):
        # The shrink gate must NOT fire in the auto path: the row-label font is the
        # same at n=20 and n=74 (the figure grows instead of the font dropping).
        f20 = min(_row_label_fonts(_fm(20)[0], make_dense_df_feat(20)["subcategory"]))
        f74 = min(_row_label_fonts(_fm(74)[0], make_dense_df_feat(74)["subcategory"]))
        assert f20 == f74, (f20, f74)

    def test_font_not_gigantic(self):
        # Guards the last-round failure: sizing must never inflate the font.
        for n in (20, 74):
            fig, ax = _fm(n)
            assert max(_row_label_fonts(fig, make_dense_df_feat(n)["subcategory"])) <= 20.0

    @pytest.mark.parametrize("n_subcat", [20, 74])
    def test_no_importance_bar_tick_overlap(self, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat))
        assert _numeric_tick_overlaps(fig) == []

    def test_col_cat_scale_name_many_rows_no_overlap(self):
        df_feat = make_dense_df_feat(74)
        names = list(dict.fromkeys(df_feat["scale_name"]))
        fig, ax = aa.CPPPlot().feature_map(df_feat, col_cat="scale_name")
        assert get_label_overlaps(fig, names) == []


def _fm(n_subcat, **kwargs):
    return aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat), **kwargs)


class TestFigureGrows:
    """The figure grows with the grid instead of cramming."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    def test_dense_grid_grows_figure_height(self):
        h20 = float(_fm(20)[0].get_size_inches()[1])
        h74 = float(_fm(74)[0].get_size_inches()[1])
        assert h74 > h20 > 0

    def test_profile_widens_with_sequence_length(self):
        w_short = float(aa.CPPPlot().profile(make_dense_df_feat(12), tmd_len=10)[0].get_size_inches()[0])
        w_long = float(aa.CPPPlot().profile(make_dense_df_feat(12), tmd_len=60)[0].get_size_inches()[0])
        assert w_long > w_short


class TestAutoFontOffAndExplicit:
    """auto_font=False reproduces the fixed default; explicit figsize always wins."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    def test_off_keeps_default_figsize(self):
        aa.options["auto_font"] = False
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(74))
        assert tuple(round(float(v), 2) for v in fig.get_size_inches()) == (8.0, 8.0)

    def test_explicit_figsize_wins_over_auto_font(self):
        aa.options["auto_font"] = True
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(74), figsize=(10, 6))
        assert tuple(round(float(v), 2) for v in fig.get_size_inches()) == (10.0, 6.0)

    def test_single_subcategory_does_not_crash(self):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(1))
        assert fig is not None

    @pytest.mark.parametrize("auto", [True, False])
    @pytest.mark.parametrize("method", ["feature_map", "heatmap", "profile"])
    def test_figsize_none_works_in_both_states(self, method, auto):
        aa.options["auto_font"] = auto
        fig, ax = getattr(aa.CPPPlot(), method)(make_dense_df_feat(20), figsize=None)
        assert fig is not None

    def test_seq_char_fill_off_when_auto_font_off(self):
        # seq_char_fill=None follows auto_font: off (unchanged spacing) when auto_font off.
        aa.options["auto_font"] = False
        df = make_dense_df_feat(20)
        f_none, _ = aa.CPPPlot().feature_map(df, tmd_seq="A" * 20, jmd_n_seq="A" * 10,
                                             jmd_c_seq="A" * 10, figsize=None)
        f_false, _ = aa.CPPPlot().feature_map(df, tmd_seq="A" * 20, jmd_n_seq="A" * 10,
                                              jmd_c_seq="A" * 10, seq_char_fill=False, figsize=None)
        f_none.canvas.draw(); f_false.canvas.draw()
        # None resolves to False under auto_font=False -> identical seq-letter sizing.
        s_none = [t.get_fontsize() for t in f_none.axes[0].texts]
        s_false = [t.get_fontsize() for t in f_false.axes[0].texts]
        assert s_none == s_false


class TestForcedFigsizeFallback:
    """auto_font on + a forced too-small figsize falls back to the shrink gate."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    def test_fallback_shrinks_but_holds_5pt_floor(self):
        # No room to grow (fixed small figure) -> labels shrink to fit, never < 5pt.
        aa.options["auto_font"] = True
        df_feat = make_dense_df_feat(74)
        fig, ax = aa.CPPPlot().feature_map(df_feat, figsize=(6, 6))
        sizes = _row_label_fonts(fig, df_feat["subcategory"])
        assert sizes and min(sizes) >= 5.0, min(sizes) if sizes else None


class TestOptionIsolation:
    """The reset fixture restores the (now True) auto_font default between tests."""

    def test_auto_font_defaults_true_each_test(self):
        assert aa.options["auto_font"] is True
