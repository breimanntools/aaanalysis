"""
Layout invariants for the constant-cell-size sizing of the CPP composite plots
(``CPPPlot.feature_map`` / ``heatmap`` / ``profile``).

The guarantee (on by default via ``auto_font``, and driven explicitly by ``cell_size``)
is: every grid *cell* keeps a constant physical size regardless of how many subcategory
rows or residue-position columns the grid has — the *figure* shrinks for a small grid and
grows for a large one. That keeps fonts fixed (they are measured in points, so they do not
scale with the figure) and the residue letters, which are fitted to the cell, consistent —
legible and non-overlapping at any data size, and NOTHING clips the figure edge.

These assert the invariants directly (constant cell size; no edge clip; font neither shrunk
nor enlarged; no overlap; capped/constant part labels) rather than fragile absolute-pixel
snapshots, so they are robust to the freetype/matplotlib build. The pixel-exact check lives
in the pytest-mpl ``visual_regression`` suite.
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
    HEATMAP_CELL_H_IN,
)

from ._text_overlap import get_label_overlaps, make_dense_df_feat


N_SUBCATS = [20, 36, 55, 74]
# The cell stays constant at ANY density now (no lower floor): sparse grids shrink the
# figure below the (8, 8) default, dense grids grow it, and the cell hits the target either way.
DENSE_SUBCATS = [40, 55, 74]
SPARSE_SUBCATS = [1, 5, 15]
N_POSITIONS = 40  # default tmd_len 20 + jmd_n 10 + jmd_c 10
CELL_TOL = 0.03   # relative tolerance on the per-cell size (freetype/layout jitter)
CLIP_TOL_IN = 0.02  # inches of tolerated content overflow past the figure edge
PART_NAMES = {"JMD-N", "TMD", "JMD-C"}


def _cell_size(fig, ax, n_rows, n_cols):
    """Per-cell (width, height) in inches from the axes' figure-fraction * figsize."""
    fig.canvas.draw()
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    pos = ax.get_position()
    return pos.width * w_in / n_cols, pos.height * h_in / n_rows


def _edge_clip_in(fig):
    """Max inches by which any drawn content spills past the figure box [0,w]x[0,h]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    tb = fig.get_tightbbox(r)  # inches, figure lower-left origin
    return max(max(0.0, -tb.x0), max(0.0, -tb.y0), max(0.0, tb.x1 - w_in), max(0.0, tb.y1 - h_in))


def _part_label_size_and_gap(fig, ax):
    """(max part-label pt, points from sequence-band bottom to part-label top), or (None, None)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    part_ax = next((a for a in fig.axes
                    if any(t.get_text() in PART_NAMES for t in a.get_xticklabels())), None)
    if part_ax is None:
        return None, None
    pl = [t for t in part_ax.get_xticklabels() if t.get_text() in PART_NAMES]
    sl = [t for t in ax.get_xticklabels(which="both") if t.get_text().strip()]
    if not pl or not sl:
        return None, None
    size = max(t.get_fontsize() for t in pl)
    gap = (min(t.get_window_extent(r).y0 for t in sl)
           - max(t.get_window_extent(r).y1 for t in pl)) / fig.dpi * 72.0
    return size, gap


def _df_feat_len(n_subcat, tmd_len):
    """A valid df_feat for ANY tmd_len (feature spans the whole TMD), varying the row count.

    ``make_dense_df_feat`` hardcodes positions valid only at the default tmd_len; the
    TMD-spanning ``Segment(1,1)`` feature stays valid for any tmd_len, so the part-label tests
    can vary the sequence length.
    """
    df = make_dense_df_feat(n_subcat).copy()
    df["feature"] = ["TMD-Segment(1,1)-" + f.split("-")[-1] for f in df["feature"]]
    return df


def _row_label_fonts(fig, subcats):
    fig.canvas.draw()
    subcats = set(subcats)
    return [t.get_fontsize() for a in fig.axes for t in a.texts
            if t.get_text().strip() in subcats and t.get_visible()]


def _seq_letter_overlaps(fig, ax):
    """(#residue letters, worst px overlap between adjacent letters). >0 px means glyphs collide."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    labs = [t for t in ax.xaxis.get_ticklabels(which="both") if len(t.get_text().strip()) == 1]
    labs = sorted(labs, key=lambda t: t.get_window_extent(r).x0)
    boxes = [t.get_window_extent(r) for t in labs]
    worst = max((a.x1 - b.x0 for a, b in zip(boxes, boxes[1:])), default=0.0)
    return len(labs), worst


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
    """The cell holds a constant physical size across every density (sparse to dense)."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", SPARSE_SUBCATS + DENSE_SUBCATS)
    def test_feature_map_cell_size_is_target(self, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat))
        cw, ch = _cell_size(fig, ax, n_subcat, N_POSITIONS)
        assert abs(cw - CELL_W_IN) <= CELL_W_IN * CELL_TOL, cw
        assert abs(ch - CELL_H_IN) <= CELL_H_IN * CELL_TOL, ch

    @pytest.mark.parametrize("n_subcat", [5, 55, 74])
    def test_heatmap_cell_size_is_target(self, n_subcat):
        # The standalone heatmap uses a taller target cell height (HEATMAP_CELL_H_IN) than
        # the feature map, so its subcategory row labels do not overlap; width matches.
        fig, ax = aa.CPPPlot().heatmap(make_dense_df_feat(n_subcat))
        cw, ch = _cell_size(fig, ax, n_subcat, N_POSITIONS)
        assert abs(cw - CELL_W_IN) <= CELL_W_IN * CELL_TOL, cw
        assert abs(ch - HEATMAP_CELL_H_IN) <= HEATMAP_CELL_H_IN * CELL_TOL, ch

    def test_cell_size_constant_across_densities(self):
        # The whole point: the cell is the SAME at a sparse grid and a dense grid (the figure
        # resizes, not the cell). Spanning sparse (5) to dense (74) exercises shrink AND grow.
        heights = []
        for n in (5, 74):
            fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n))
            heights.append(_cell_size(fig, ax, n, N_POSITIONS)[1])
        assert abs(heights[0] - heights[1]) <= CELL_H_IN * CELL_TOL, heights


class TestSparseGridShrinks:
    """A sparse grid shrinks the figure below the default and still hits the cell, no clip."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", SPARSE_SUBCATS)
    def test_feature_map_shrinks_and_hits_cell(self, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat))
        w, h = (float(v) for v in fig.get_size_inches())
        cw, ch = _cell_size(fig, ax, n_subcat, N_POSITIONS)
        # Few rows -> the figure is shorter than the old (8, 8) floor (the cell no longer balloons).
        assert h < 8.0, (n_subcat, h)
        assert abs(ch - CELL_H_IN) <= CELL_H_IN * CELL_TOL, ch
        assert _edge_clip_in(fig) <= CLIP_TOL_IN, _edge_clip_in(fig)

    @pytest.mark.parametrize("n_subcat", SPARSE_SUBCATS)
    def test_heatmap_shrinks_and_hits_cell(self, n_subcat):
        fig, ax = aa.CPPPlot().heatmap(make_dense_df_feat(n_subcat))
        h = float(fig.get_size_inches()[1])
        cw, ch = _cell_size(fig, ax, n_subcat, N_POSITIONS)
        assert h < 8.0, (n_subcat, h)
        assert abs(ch - HEATMAP_CELL_H_IN) <= HEATMAP_CELL_H_IN * CELL_TOL, ch
        assert _edge_clip_in(fig) <= CLIP_TOL_IN, _edge_clip_in(fig)


class TestNoEdgeClip:
    """Nothing clips the figure edge, at any density (the objective 'perfect in any setting' gate)."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("n_subcat", SPARSE_SUBCATS + N_SUBCATS)
    def test_feature_map_no_clip(self, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(n_subcat))
        assert _edge_clip_in(fig) <= CLIP_TOL_IN, (n_subcat, _edge_clip_in(fig))

    @pytest.mark.parametrize("n_subcat", SPARSE_SUBCATS + N_SUBCATS)
    def test_heatmap_no_clip(self, n_subcat):
        fig, ax = aa.CPPPlot().heatmap(make_dense_df_feat(n_subcat))
        assert _edge_clip_in(fig) <= CLIP_TOL_IN, (n_subcat, _edge_clip_in(fig))


class TestPartLabelsCappedAndConstant:
    """The TMD/JMD part labels stay capped in size and a constant gap below the sequence band."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("tmd_len", [6, 20, 100])
    def test_part_label_size_capped(self, tmd_len):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 8
        fig, ax = aa.CPPPlot().feature_map(
            _df_feat_len(20, tmd_len), tmd_len=tmd_len, add_imp_bar_top=False,
            tmd_seq=seq[:tmd_len], jmd_n_seq=seq[:10], jmd_c_seq=seq[:10])
        size, _ = _part_label_size_and_gap(fig, ax)
        assert size is not None and size <= 12.01, size

    def test_part_label_gap_constant_across_lengths(self):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 8
        gaps = []
        for tmd_len in (6, 100):
            fig, ax = aa.CPPPlot().feature_map(
                _df_feat_len(20, tmd_len), tmd_len=tmd_len, add_imp_bar_top=False,
                tmd_seq=seq[:tmd_len], jmd_n_seq=seq[:10], jmd_c_seq=seq[:10])
            gaps.append(_part_label_size_and_gap(fig, ax)[1])
            plt.close("all")
        assert gaps[0] is not None and abs(gaps[0] - gaps[1]) <= 1.0, gaps


class TestCellSizeParam:
    """The explicit cell_size param sets the exact cell, composes with figsize, and never clips."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("cell_size", [(0.25, 0.25), (0.10, 0.12)])
    def test_feature_map_cell_size_is_exact(self, cell_size):
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(25), cell_size=cell_size)
        cw, ch = _cell_size(fig, ax, 25, N_POSITIONS)
        assert abs(cw - cell_size[0]) <= cell_size[0] * CELL_TOL, cw
        assert abs(ch - cell_size[1]) <= cell_size[1] * CELL_TOL, ch
        assert _edge_clip_in(fig) <= CLIP_TOL_IN, _edge_clip_in(fig)

    def test_cell_size_drives_even_with_auto_font_off(self):
        aa.options["auto_font"] = False
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(25), cell_size=(0.2, 0.2))
        cw, ch = _cell_size(fig, ax, 25, N_POSITIONS)
        assert abs(cw - 0.2) <= 0.2 * CELL_TOL and abs(ch - 0.2) <= 0.2 * CELL_TOL, (cw, ch)

    @pytest.mark.parametrize("cell_size", [(0.0, 0.2), (0.2, -0.1), (0.2,), 0.2])
    def test_invalid_cell_size_raises(self, cell_size):
        with pytest.raises(ValueError):
            aa.CPPPlot().feature_map(make_dense_df_feat(20), cell_size=cell_size)


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

    def test_explicit_default_figsize_also_wins(self):
        # An explicit (8, 8) -- the old sentinel -- must be honored as a fixed size, not
        # auto-sized. This is what lets embedded consumers (e.g. the structure explorer)
        # pin a predictable figure independent of the auto_font default.
        aa.options["auto_font"] = True
        fig, ax = aa.CPPPlot().feature_map(make_dense_df_feat(74), figsize=(8, 8))
        assert tuple(round(float(v), 2) for v in fig.get_size_inches()) == (8.0, 8.0)

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


class TestFigsizeSentinel:
    """'Explicit figsize wins' holds for every composite CPP plot: passing a figsize -- even the
    old per-method default -- is honored as a fixed size; omitting it (None) triggers auto-sizing."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    def test_explicit_figsize_is_honored(self):
        cpp = aa.CPPPlot()
        df = aa.load_features(name="DOM_GSEC").head(40)
        for name, fs in [("feature_map", (8, 8)), ("heatmap", (8, 8)),
                         ("profile", (7, 5)), ("ranking", (7, 5))]:
            getattr(cpp, name)(df, figsize=fs)
            w, h = plt.gcf().get_size_inches()
            assert (round(float(w), 1), round(float(h), 1)) == fs, (name, w, h)
            plt.close("all")

    def test_omitted_figsize_autosizes_heatmap(self):
        # Omitted figsize -> auto: a dense grid grows taller than the fixed (8, 8) default.
        aa.CPPPlot().heatmap(aa.load_features(name="DOM_GSEC").head(40))
        assert plt.gcf().get_size_inches()[1] > 8.0


class TestSeqCharFill:
    """seq_char_fill is accepted by heatmap and profile (not only feature_map)."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    def test_heatmap_profile_accept_seq_char_fill(self):
        df = aa.load_features(name="DOM_GSEC")
        df_seq = aa.load_dataset(name="DOM_GSEC", n=1)
        sf = aa.SequenceFeature()
        seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=sf.get_df_parts(df_seq=df_seq),
                                 sample=df_seq["entry"].iloc[0])
        cpp = aa.CPPPlot()
        for name in ("heatmap", "profile"):
            for fill in (True, False):
                getattr(cpp, name)(df, seq_char_fill=fill, **seq_kws)
                plt.close("all")


class TestSeqCharNeverOverlap:
    """The sequence residue letters must NEVER overlap, at any grid width; the colored band
    stays gap-free (one full-width cell per residue in fill mode)."""

    _SEQ = "ACDEFGHIKLMNPQRSTVWY" * 10

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")
        aa.options["auto_font"] = True

    @pytest.mark.parametrize("n_subcat", [5, 25, 40])
    @pytest.mark.parametrize("tmd_len", [6, 20, 95, 150])
    def test_feature_map_no_seq_overlap(self, tmd_len, n_subcat):
        fig, ax = aa.CPPPlot().feature_map(
            _df_feat_len(n_subcat, tmd_len), tmd_len=tmd_len, add_imp_bar_top=False,
            tmd_seq=self._SEQ[:tmd_len], jmd_n_seq=self._SEQ[:10], jmd_c_seq=self._SEQ[:10])
        n, worst = _seq_letter_overlaps(fig, ax)
        assert n == tmd_len + 20, n
        assert worst <= 1.0, f"seq letters overlap by {worst:.1f}px (tmd={tmd_len}, nsub={n_subcat})"

    @pytest.mark.parametrize("figsize", [None, (10, 7), (20, 9)])
    def test_no_seq_overlap_across_figsize(self, figsize):
        # All-W/M: the widest glyphs, the worst case for adjacent-letter collision.
        seq = "WM" * 90
        fig, ax = aa.CPPPlot().feature_map(
            _df_feat_len(25, 95), tmd_len=95, add_imp_bar_top=False, figsize=figsize,
            tmd_seq=seq[:95], jmd_n_seq=seq[:10], jmd_c_seq=seq[:10])
        _, worst = _seq_letter_overlaps(fig, ax)
        assert worst <= 1.0, f"seq letters overlap by {worst:.1f}px (figsize={figsize})"

    def test_seq_band_one_cell_per_residue(self):
        # fill mode (auto_font on): a full-width colored cell behind every residue = gap-free band.
        aa.options["auto_font"] = True
        fig, ax = aa.CPPPlot().feature_map(
            _df_feat_len(25, 95), tmd_len=95, add_imp_bar_top=False,
            tmd_seq=self._SEQ[:95], jmd_n_seq=self._SEQ[:10], jmd_c_seq=self._SEQ[:10])
        n_cells = sum(1 for p in ax.patches if p.get_gid() == "_seq_cell")
        assert n_cells == 95 + 20, n_cells


class TestSeqSizeModes:
    """seq_size: 'auto' (length-adaptive) | fraction of the cell height (<=1) | points (>1).
    Every mode keeps the residue letters non-overlapping."""

    _SEQ = "ACDEFGHIKLMNPQRSTVWY" * 10

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")
        aa.options["auto_font"] = True

    def _seq_fs(self, tmd_len, seq_size):
        fig, ax = aa.CPPPlot().feature_map(
            _df_feat_len(20, tmd_len), tmd_len=tmd_len, add_imp_bar_top=False, seq_size=seq_size,
            tmd_seq=self._SEQ[:tmd_len], jmd_n_seq=self._SEQ[:10], jmd_c_seq=self._SEQ[:10])
        _, worst = _seq_letter_overlaps(fig, ax)
        labs = [t for t in ax.xaxis.get_ticklabels(which="both") if len(t.get_text().strip()) == 1]
        fs = max(t.get_fontsize() for t in labs)
        plt.close("all")
        assert worst <= 1.0, f"overlap {worst:.1f}px at seq_size={seq_size}"
        return fs

    def test_auto_steps_down_for_short_tmd(self):
        assert self._seq_fs(20, "auto") > self._seq_fs(4, "auto")

    def test_fraction_smaller_than_full(self):
        assert self._seq_fs(20, 0.6) < self._seq_fs(20, 0.9)

    @pytest.mark.parametrize("pt", [8.0, 12.0])
    def test_points_sets_absolute_size(self, pt):
        assert abs(self._seq_fs(20, pt) - pt) <= 0.6

    def test_invalid_seq_size_raises(self):
        with pytest.raises(ValueError):
            aa.CPPPlot().feature_map(
                _df_feat_len(20, 20), tmd_len=20, seq_size=-1,
                tmd_seq=self._SEQ[:20], jmd_n_seq=self._SEQ[:10], jmd_c_seq=self._SEQ[:10])


class TestTightLayoutFreeze:
    """feature_map / heatmap compose their own furniture layout (colorbar + legends below the
    grid). A later plt.tight_layout() -- the standard notebook idiom -- must NOT re-pack the axes
    and pull that furniture back onto the heatmap: the returned figure neutralizes tight_layout."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @pytest.mark.parametrize("method", ["feature_map", "heatmap"])
    @pytest.mark.parametrize("figsize", [None, (9, 6)])
    def test_tight_layout_is_neutralized(self, method, figsize):
        fig, ax = getattr(aa.CPPPlot(), method)(make_dense_df_feat(30), figsize=figsize)
        fig.canvas.draw()
        before = ax.get_position().bounds
        plt.tight_layout()   # pyplot idiom -> gcf().tight_layout()
        fig.tight_layout()   # and a direct call on the figure
        after = ax.get_position().bounds
        assert all(abs(b - a) < 1e-6 for b, a in zip(before, after)), (method, figsize, before, after)


class TestNoSequenceBarHeight:
    """With no sequence, the TMD/JMD region is a solid colored bar; it must be a substantial,
    sequence-band-like track, not the razor-thin default strip."""

    def setup_method(self):
        aa.options["verbose"] = False

    def teardown_method(self):
        plt.close("all")

    @staticmethod
    def _tmd_jmd_bar_heights(ax):
        # The TMD/JMD bar rectangles span whole regions (width > 1 position) and are drawn
        # unclipped just below the grid; the narrow category sidebar rectangles are < 1 wide.
        import matplotlib.patches as mpatches
        return [p.get_height() for p in ax.patches
                if isinstance(p, mpatches.Rectangle) and p.get_width() > 1 and not p.get_clip_on()]

    @pytest.mark.parametrize("method", ["feature_map", "heatmap"])
    def test_no_seq_bar_is_substantial(self, method):
        fig, ax = getattr(aa.CPPPlot(), method)(make_dense_df_feat(20))
        fig.canvas.draw()
        heights = self._tmd_jmd_bar_heights(ax)
        assert heights, "no TMD/JMD bar rectangle found in the no-sequence layout"
        # One grid row spans 1.0 data unit; the thin default bar is ~0.4 rows, the thickened one
        # is well above two-thirds of a row. Guards against the height factor regressing to ~1.
        assert max(heights) > 0.7, max(heights)
