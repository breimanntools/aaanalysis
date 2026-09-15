"""This is a script to test AAPredPlot.group_cluster(kind='dendrogram') and its layouts."""
import io
import importlib.util
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import matplotlib.colors as mcolors

from aaanalysis.prediction._backend.aa_pred import aa_pred_plot_clustermap as cm_backend
from aaanalysis.prediction._backend.aa_pred import aa_pred_plot_dendrogram as dn_backend
from aaanalysis.prediction._backend.aa_pred.aa_pred_plot_linkage import (sample_correlation_,
                                                                         sample_linkage_)

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

LAYOUTS = ["rectangular", "circular"]
HAS_FASTCLUSTER = importlib.util.find_spec("fastcluster") is not None


# Helper functions
def _imp_data(n=12, n_feat=20, seed=0):
    rng = np.random.RandomState(seed)
    return np.vstack([rng.normal(0.6, 0.4, (n // 2, n_feat)),
                      rng.normal(-0.6, 0.4, (n - n // 2, n_feat))])


def _names(n):
    return [f"P{i:02d}" for i in range(n)]


def _tree_ax(r):
    return r.ax


def _leaf_names(r):
    """Leaf names in drawing order (top-to-bottom / counterclockwise from east)."""
    fig, ax = r
    if ax.name == "polar":
        return [t.get_text() for t in ax.texts]
    # Rectangular: names sit on the right-most strip (or the tree axes without tracks).
    ax_names = max((a for a in fig.axes if a.get_yticklabels() and a.get_visible()
                    and any(t.get_text() for t in a.get_yticklabels())),
                   key=lambda a: a.get_position().x0)
    return [t.get_text() for t in ax_names.get_yticklabels()]


def _heatmap_row_names(r):
    return [t.get_text() for t in r.ax.get_yticklabels()]


def _rgba(color):
    return tuple(np.round(mcolors.to_rgba(color), 3))


def _patch_colors(ax):
    return {tuple(np.round(p.get_facecolor(), 3)) for p in ax.patches}


def _png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    return buf.getvalue()


def _assert_linkage_equal(a, b):
    if HAS_FASTCLUSTER:
        np.testing.assert_allclose(a, b)
    else:
        assert np.array_equal(a, b)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestGroupCluster:
    """Normal cases: one parameter per test."""

    # Positive tests
    @settings(max_examples=4, deadline=None)
    @given(layout=some.sampled_from(LAYOUTS))
    def test_kind_dendrogram_returns_fig_ax(self, layout):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout=layout)
        fig, ax = r
        assert fig is not None and ax is not None
        assert ax.get_figure() is fig
        plt.close("all")

    def test_kind_clustermap_default_layout(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="clustermap", layout="rectangular")
        assert r.ax is not None

    def test_layout_rectangular_is_cartesian(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="rectangular")
        assert r.ax.name == "rectilinear"

    def test_layout_circular_is_polar(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular")
        assert r.ax.name == "polar"

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=2, max_value=30))
    def test_X_n_samples(self, n):
        X = np.random.RandomState(n).rand(n, 6)
        r = aa.AAPredPlot().group_cluster(X, kind="dendrogram", names=_names(n))
        assert sorted(_leaf_names(r)) == _names(n)
        plt.close("all")

    def test_X_dataframe(self):
        df = pd.DataFrame(_imp_data(), columns=[f"f{i}" for i in range(20)])
        r = aa.AAPredPlot().group_cluster(df, kind="dendrogram", layout="circular")
        assert r.ax.name == "polar"

    @settings(max_examples=4, deadline=None)
    @given(layout=some.sampled_from(LAYOUTS))
    def test_labels(self, layout):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout=layout,
                                          labels=["sub", "non"] * 6)
        legends = [a.get_legend() for a in r.fig.axes if a.get_legend() is not None]
        assert len(legends) == 1
        assert sorted(t.get_text() for t in legends[0].get_texts()) == ["non", "sub"]
        plt.close("all")

    def test_labels_row(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram",
                                          labels_row=["hi", "lo"] * 6, legend_title_row="Conf")
        legends = [a.get_legend() for a in r.fig.axes if a.get_legend() is not None]
        assert [lg.get_title().get_text() for lg in legends] == ["Conf"]

    def test_dict_color_rectangular_strip(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram",
                                          labels=["sub", "non"] * 6,
                                          dict_color={"sub": "red", "non": "blue"})
        strip = [a for a in r.fig.axes if a is not r.ax and a.patches][0]
        colors = {tuple(np.round(p.get_facecolor(), 3)) for p in strip.patches}
        assert colors == {(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)}

    def test_dict_color_circular_ring(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular",
                                          labels=["sub", "non"] * 6,
                                          dict_color={"sub": "red", "non": "blue"})
        colors = {tuple(np.round(p.get_facecolor(), 3)) for p in r.ax.patches
                  if type(p).__name__ == "Rectangle"}
        assert {(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)} <= colors

    def test_dict_color_row(self):
        # The lone `labels_row` track is drawn as one strip whose bars carry exactly the
        # requested colors (6 'hi' bars and 6 'lo' bars), and no heatmap/colorbar is drawn.
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram",
                                          labels_row=["hi", "lo"] * 6,
                                          dict_color_row={"hi": "tab:red", "lo": "tab:blue"})
        strip = [a for a in r.fig.axes if a is not r.ax and a.patches][0]
        drawn = [tuple(np.round(p.get_facecolor(), 3)) for p in strip.patches]
        assert _patch_colors(strip) == {_rgba("tab:red"), _rgba("tab:blue")}
        assert drawn.count(_rgba("tab:red")) == drawn.count(_rgba("tab:blue")) == 6
        assert not r.ax.images and not r.ax.patches  # tree axes only: no heatmap, no strip

    def test_dict_color_row_circular_ring(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular",
                                          labels_row=["hi", "lo"] * 6,
                                          dict_color_row={"hi": "tab:red", "lo": "tab:blue"})
        drawn = [tuple(np.round(p.get_facecolor(), 3)) for p in r.ax.patches]
        assert set(drawn) == {_rgba("tab:red"), _rgba("tab:blue")}
        assert drawn.count(_rgba("tab:red")) == drawn.count(_rgba("tab:blue")) == 6

    @settings(max_examples=5, deadline=None)
    @given(title=some.text(alphabet="abcdefgh ", min_size=1, max_size=15))
    def test_legend_title(self, title):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", labels=[1, 0] * 6,
                                          legend_title=title)
        legends = [a.get_legend() for a in r.fig.axes if a.get_legend() is not None]
        assert legends[0].get_title().get_text() == title
        plt.close("all")

    @settings(max_examples=4, deadline=None)
    @given(layout=some.sampled_from(LAYOUTS))
    def test_names(self, layout):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout=layout,
                                          names=_names(12))
        assert sorted(_leaf_names(r)) == _names(12)
        plt.close("all")

    def test_names_thinned_when_dense(self):
        r = aa.AAPredPlot().group_cluster(np.random.RandomState(0).rand(130, 5),
                                          kind="dendrogram", layout="circular")
        assert 0 < len(r.ax.texts) <= 60

    @settings(max_examples=4, deadline=None)
    @given(w=some.floats(min_value=4, max_value=10), h=some.floats(min_value=4, max_value=10))
    def test_figsize(self, w, h):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", figsize=(w, h))
        assert np.allclose(r.fig.get_size_inches(), (w, h))
        plt.close("all")

    def test_figsize_default_per_layout(self):
        r_rect = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram")
        r_circ = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular")
        assert tuple(r_rect.fig.get_size_inches()) == (7, 9)
        assert tuple(r_circ.fig.get_size_inches()) == (9, 10)

    def test_title(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", title="Tree")
        assert r.fig.get_suptitle() == "Tree"

    def test_cmap_cbar_label_ignored(self):
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", cmap="viridis",
                                          cbar_label="r")
        assert r.ax is not None

    def test_constant_row_does_not_crash(self):
        data = _imp_data()
        data[0] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r = aa.AAPredPlot().group_cluster(data, kind="dendrogram",
                                               layout="circular")
        assert r.ax is not None

    # Negative tests
    def test_invalid_layout(self):
        for layout in ["radial", "", "Circular", None, 1, ["circular"]]:
            with pytest.raises(ValueError, match="'layout'"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout=layout)

    def test_layout_with_clustermap_raises(self):
        with pytest.raises(ValueError, match="'layout'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="clustermap", layout="circular")

    def test_invalid_kind(self):
        for kind in ["dendogram", "tree", "", None, 3]:
            with pytest.raises(ValueError, match="'kind'"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind=kind)

    def test_invalid_X(self):
        for X in [np.random.RandomState(0).rand(1, 5), None, "abc",
                  np.array([[1.0, np.nan], [0.5, 0.2], [0.1, 0.9]])]:
            with pytest.raises(ValueError, match=r"'X'|n_samples"):
                aa.AAPredPlot().group_cluster(X, kind="dendrogram")

    def test_invalid_labels_length(self):
        for labels in [[1, 0, 1], [1] * 13]:
            with pytest.raises(ValueError, match="'labels'"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", labels=labels)

    def test_invalid_labels_row_length(self):
        with pytest.raises(ValueError, match="'labels_row'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", labels_row=[1, 0])

    def test_unhashable_labels(self):
        with pytest.raises(ValueError, match="'labels'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram",
                                          labels=[[1]] * 12)

    def test_invalid_names(self):
        for names in [["a", "b"], _names(13), [["a"]] * 12]:
            with pytest.raises(ValueError, match="'names'"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", names=names)

    def test_invalid_dict_color(self):
        for dict_color in [{"sub": "red"}, {"sub": "red", "non": "not_a_color"}]:
            with pytest.raises(ValueError, match=r"colors|'dict_color'"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram",
                                              labels=["sub", "non"] * 6, dict_color=dict_color)

    def test_invalid_dict_color_row(self):
        with pytest.raises(ValueError, match=r"colors|'dict_color_row'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram",
                                          labels_row=["hi", "lo"] * 6, dict_color_row={"hi": "red"})

    def test_invalid_legend_title(self):
        for kws in [dict(legend_title=1), dict(legend_title_row=["x"])]:
            with pytest.raises(ValueError, match="'legend_title"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", **kws)

    def test_invalid_figsize(self):
        for figsize in [(0, 5), "big", (5,), (-1, -1)]:
            with pytest.raises(ValueError, match="'figsize"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", figsize=figsize)

    def test_invalid_title(self):
        for title in [1, ["t"]]:
            with pytest.raises(ValueError, match="'title'"):
                aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", title=title)

    def test_invalid_cmap(self):
        # `cmap` is validated for both kinds, even though the dendrogram ignores it.
        for kind in ["clustermap", "dendrogram"]:
            for cmap in ["not_a_cmap", "", None, 5, ["GnBu"]]:
                with pytest.raises(ValueError, match="'cmap'"):
                    aa.AAPredPlot().group_cluster(_imp_data(), kind=kind, cmap=cmap)

    def test_invalid_cbar_label(self):
        for kind in ["clustermap", "dendrogram"]:
            for cbar_label in [1, ["r"], 0.5]:
                with pytest.raises(ValueError, match="'cbar_label'"):
                    aa.AAPredPlot().group_cluster(_imp_data(), kind=kind, cbar_label=cbar_label)


class TestGroupClusterComplex:
    """Combinations of parameters."""

    # Positive tests
    @settings(max_examples=4, deadline=None)
    @given(layout=some.sampled_from(LAYOUTS))
    def test_two_annotations_two_legends(self, layout):
        r = aa.AAPredPlot().group_cluster(
            _imp_data(), kind="dendrogram", layout=layout,
            labels=["sub", "non"] * 6, dict_color={"sub": "tab:green", "non": "tab:gray"},
            legend_title="Prediction group",
            labels_row=["hi", "lo", "mid"] * 4, legend_title_row="Confidence",
            names=_names(12), title="Tree", figsize=(8, 8))
        legends = [a.get_legend() for a in r.fig.axes if a.get_legend() is not None]
        assert [lg.get_title().get_text() for lg in legends] == ["Prediction group", "Confidence"]
        plt.close("all")

    def test_rectangular_axes_count(self):
        # tree + one strip per annotation + one legend axes per annotation
        r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", labels=[1, 0] * 6,
                                          labels_row=[0, 1] * 6)
        assert len(r.fig.axes) == 5

    def test_circular_rings_per_annotation(self):
        r1 = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular",
                                           labels=[1, 0] * 6)
        r2 = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular",
                                           labels=[1, 0] * 6, labels_row=[0, 1] * 6)
        assert len(r1.ax.patches) == 12
        assert len(r2.ax.patches) == 24

    def test_no_annotation_no_legend(self):
        for layout in LAYOUTS:
            r = aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout=layout)
            assert all(a.get_legend() is None for a in r.fig.axes)

    def test_tight_layout_compatible(self):
        import warnings
        for layout in LAYOUTS:
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout=layout,
                                          labels=[1, 0] * 6, labels_row=[0, 1] * 6,
                                          title="t")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                plt.tight_layout()
            plt.close("all")

    def test_end_to_end_render_both_layouts(self):
        X = _imp_data(n=30, n_feat=15, seed=3)
        group = ["Substrate"] * 12 + ["Known"] * 8 + ["dPULearn"] * 10
        for layout in LAYOUTS:
            fig, ax = aa.AAPredPlot().group_cluster(X, kind="dendrogram", layout=layout,
                                                    labels=group, labels_row=[1, 0, 0] * 10,
                                                    names=_names(30), title="Cohort tree")
            plt.tight_layout()
            png = _png_bytes(fig)
            assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10000
            plt.close("all")

    # Negative tests
    def test_circular_labels_mismatch(self):
        with pytest.raises(ValueError, match="'labels_row'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular",
                                          labels=[1, 0] * 6, labels_row=[1, 0])

    def test_clustermap_circular_with_labels(self):
        with pytest.raises(ValueError, match="'layout'"):
            aa.AAPredPlot().group_cluster(_imp_data(), labels=[1, 0] * 6, layout="circular")

    def test_bad_layout_and_bad_kind(self):
        with pytest.raises(ValueError, match="'kind'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="tree", layout="radial")

    def test_dict_color_row_without_matching_labels_row(self):
        with pytest.raises(ValueError, match="'dict_color_row'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", layout="circular",
                                          labels_row=["a", "b"] * 6, dict_color_row={"c": "red"})

    def test_names_mismatch_with_annotations(self):
        with pytest.raises(ValueError, match="'names'"):
            aa.AAPredPlot().group_cluster(_imp_data(), kind="dendrogram", labels=[1, 0] * 6,
                                          names=_names(5))


class TestGroupClusterGoldenValues:
    """Topology equivalence between the dendrogram and the clustermap kinds."""

    def test_clustermap_receives_seaborn_equivalent_linkage(self, monkeypatch):
        # The linkage group_cluster now hands to seaborn equals what seaborn computed on its own.
        captured = {}
        original = sns.clustermap

        def spy(*args, **kwargs):
            captured.update(kwargs)
            grid = original(*args, **kwargs)
            captured["grid"] = grid
            return grid
        monkeypatch.setattr(cm_backend.sns, "clustermap", spy)
        X = _imp_data(n=14, seed=5)
        aa.AAPredPlot().group_cluster(X, labels=[1, 0] * 7)
        assert captured["row_linkage"] is not None and captured["col_linkage"] is not None
        assert captured["row_linkage"] is captured["col_linkage"]
        corr_df = sample_correlation_(data=X)
        own = original(corr_df)
        _assert_linkage_equal(captured["grid"].dendrogram_row.linkage, own.dendrogram_row.linkage)
        _assert_linkage_equal(captured["grid"].dendrogram_col.linkage, own.dendrogram_col.linkage)
        assert captured["grid"].dendrogram_row.reordered_ind == own.dendrogram_row.reordered_ind
        assert captured["grid"].dendrogram_col.reordered_ind == own.dendrogram_col.reordered_ind

    @pytest.mark.skipif(HAS_FASTCLUSTER, reason="seaborn's own linkage uses fastcluster here")
    def test_clustermap_figure_byte_identical_to_seaborn_linkage(self, monkeypatch):
        # Rendering with seaborn computing the linkage itself (the previous behavior) gives the
        # exact same PNG as rendering with the explicitly passed linkage.
        X = _imp_data(n=16, seed=2)
        kws = dict(labels=["a", "b"] * 8, labels_row=[1, 1, 0, 0] * 4, names=_names(16),
                   title="t")
        png_new = _png_bytes(aa.AAPredPlot().group_cluster(X, **kws).fig)
        plt.close("all")
        original = sns.clustermap

        def without_linkage(*args, **kwargs):
            kwargs.pop("row_linkage", None)
            kwargs.pop("col_linkage", None)
            return original(*args, **kwargs)
        monkeypatch.setattr(cm_backend.sns, "clustermap", without_linkage)
        png_old = _png_bytes(aa.AAPredPlot().group_cluster(X, **kws).fig)
        assert png_new == png_old

    def test_same_linkage_matrix_for_both_kinds(self, monkeypatch):
        # Topology KPI: the merge pairs AND merge distances of the linkage seaborn clusters the
        # clustermap with equal those of the linkage the dendrogram kind draws (same input).
        X = _imp_data(n=16, seed=11)
        names = _names(16)
        captured = {}
        original = sns.clustermap

        def spy_clustermap(*args, **kwargs):
            grid = original(*args, **kwargs)
            captured["grid"] = grid
            return grid
        monkeypatch.setattr(cm_backend.sns, "clustermap", spy_clustermap)
        aa.AAPredPlot().group_cluster(X, names=names)
        plt.close("all")
        original_linkage = dn_backend.sample_linkage_

        def spy_linkage(*args, **kwargs):
            out = original_linkage(*args, **kwargs)
            captured["dendrogram_linkage"] = out
            return out
        monkeypatch.setattr(dn_backend, "sample_linkage_", spy_linkage)
        aa.AAPredPlot().group_cluster(X, kind="dendrogram", names=names)
        link_cm = np.asarray(captured["grid"].dendrogram_row.linkage)
        link_dn = np.asarray(captured["dendrogram_linkage"])
        assert link_cm.shape == link_dn.shape == (15, 4)
        assert np.array_equal(link_cm[:, :2], link_dn[:, :2])   # merge pairs
        assert np.array_equal(link_cm[:, 2], link_dn[:, 2])     # merge distances
        assert np.array_equal(link_cm, link_dn)                 # + cluster sizes

    def test_drawn_tree_heights_equal_clustermap_linkage(self, monkeypatch):
        # The linkage is not only shared but actually drawn: every rectangular link is drawn at
        # the merge distance of the clustermap's row linkage.
        X = _imp_data(n=14, seed=12)
        captured = {}
        original = sns.clustermap

        def spy_clustermap(*args, **kwargs):
            grid = original(*args, **kwargs)
            captured["grid"] = grid
            return grid
        monkeypatch.setattr(cm_backend.sns, "clustermap", spy_clustermap)
        aa.AAPredPlot().group_cluster(X)
        plt.close("all")
        r = aa.AAPredPlot().group_cluster(X, kind="dendrogram")
        # Rectangular segments are (distance, leaf position): the merge height is the max x.
        drawn = sorted(float(np.max(seg[:, 0])) for seg in r.ax.collections[0].get_segments())
        expected = sorted(np.asarray(captured["grid"].dendrogram_row.linkage)[:, 2])
        assert np.allclose(drawn, expected)

    @settings(max_examples=4, deadline=None)
    @given(seed=some.integers(min_value=0, max_value=1000))
    def test_rectangular_leaf_order_equals_clustermap_rows(self, seed):
        X = np.random.RandomState(seed).rand(12, 8)
        r_cm = aa.AAPredPlot().group_cluster(X, names=_names(12))
        r_dn = aa.AAPredPlot().group_cluster(X, kind="dendrogram", names=_names(12))
        assert _leaf_names(r_dn) == _heatmap_row_names(r_cm)
        plt.close("all")

    def test_circular_leaf_order_equals_clustermap_rows(self):
        X = _imp_data(n=20, seed=7)
        r_cm = aa.AAPredPlot().group_cluster(X, names=_names(20))
        r_dn = aa.AAPredPlot().group_cluster(X, kind="dendrogram", layout="circular",
                                             names=_names(20))
        assert _leaf_names(r_dn) == _heatmap_row_names(r_cm)

    def test_tree_segments_match_linkage(self):
        X = _imp_data(n=10, seed=4)
        n_links = len(sample_linkage_(corr_df=sample_correlation_(data=X)))
        assert n_links == 9
        r_rect = aa.AAPredPlot().group_cluster(X, kind="dendrogram")
        r_circ = aa.AAPredPlot().group_cluster(X, kind="dendrogram", layout="circular")
        assert len(r_rect.ax.collections[0].get_segments()) == n_links
        # circular: two radial legs + one arc per link
        assert len(r_circ.ax.collections[0].get_segments()) == 3 * n_links

    def test_hand_computed_two_pair_topology(self):
        # Samples 0/2 and 1/3 share a profile: the first two merges join exactly those pairs,
        # and each pair is adjacent in the leaf order.
        base_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base_b = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        X = np.vstack([base_a, base_b, base_a + [0.01, 0, 0, 0, 0], base_b + [0, 0.02, 0, 0, 0]])
        linkage = sample_linkage_(corr_df=sample_correlation_(data=X))
        first_merges = {frozenset(map(int, row[:2])) for row in linkage[:2]}
        assert first_merges == {frozenset({0, 2}), frozenset({1, 3})}
        names = ["a0", "b1", "a2", "b3"]
        order = _leaf_names(aa.AAPredPlot().group_cluster(X, kind="dendrogram", names=names))
        idx = {name: i for i, name in enumerate(order)}
        assert abs(idx["a0"] - idx["a2"]) == 1 and abs(idx["b1"] - idx["b3"]) == 1

    def test_hand_computed_correlation_with_constant_row(self):
        X = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 5.0, 5.0]])
        expected = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(sample_correlation_(data=X).to_numpy(), expected)
