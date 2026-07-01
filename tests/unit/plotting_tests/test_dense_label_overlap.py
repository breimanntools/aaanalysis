"""
Tests the dense-plot row-label overlap gate for ``CPPPlot.feature_map``.

The scale subcategory row labels collide once the grid has more than a handful
of rows; at the full AAontology breadth (74 subcategories) the label column
becomes an unreadable blur. These tests force-render the figure and assert that
no two *row labels* overlap. They are the first layout/visual assertions in the
suite (plain matplotlib bbox introspection, no image snapshots).
"""
import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa

from ._text_overlap import get_label_overlaps, make_dense_df_feat


N_SUBCATS = [20, 36, 55, 74]


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
