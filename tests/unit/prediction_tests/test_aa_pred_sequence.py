"""Unit tests for AAPred sequence-level prediction (seq / domain / window) + their plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import matplotlib.colors as mcolors

import aaanalysis as aa

aa.options["verbose"] = False

_CYAN = mcolors.to_rgb("#00E5FF")  # ut.COLOR_LINK_HIGHLIGHT, the highlight-span color


def _cyan_spans(ax):
    """Highlight-span patches (bright-cyan axvspans) on ``ax``."""
    return [p for p in ax.patches
            if tuple(round(c, 3) for c in p.get_facecolor()[:3]) == tuple(round(c, 3) for c in _CYAN)]


@pytest.fixture(scope="module")
def fitted():
    df_seq = aa.load_dataset(name="DOM_GSEC")
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(10)
    sf = aa.SequenceFeature()
    X = sf.feature_matrix(features=df_feat, df_parts=sf.get_df_parts(df_seq=df_seq))
    aapred = aa.AAPred(df_feat=df_feat, random_state=42).fit(X, labels)
    return aapred, df_seq, df_feat


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class TestFeaturizerBinding:
    def test_df_feat_stored(self, fitted):
        _, _, df_feat = fitted
        aapred = aa.AAPred(df_feat=df_feat)
        assert aapred._df_feat is not None

    def test_df_scales_stored(self):
        df_scales = aa.load_scales()
        aapred = aa.AAPred(df_scales=df_scales)
        assert aapred._df_scales is not None

    def test_predict_seq_without_df_feat_raises(self, fitted):
        _, df_seq, _ = fitted
        aapred = aa.AAPred(random_state=0)
        # fit on a dummy X so it is "fitted" but has no df_feat
        aapred.fit(np.random.RandomState(0).rand(20, 10), np.array([0, 1] * 10))
        with pytest.raises(ValueError):
            aapred.predict(df_seq.head(3), level="seq")


class TestPredictSeq:
    def test_columns(self, fitted):
        aapred, df_seq, _ = fitted
        df_pred = aapred.predict(df_seq.head(5), level="seq")
        assert list(df_pred.columns) == ["entry", "score", "score_std"]

    def test_one_row_per_protein(self, fitted):
        aapred, df_seq, _ = fitted
        df_pred = aapred.predict(df_seq.head(5), level="seq")
        assert len(df_pred) == 5

    def test_scores_in_range(self, fitted):
        aapred, df_seq, _ = fitted
        df_pred = aapred.predict(df_seq.head(5), level="seq")
        assert df_pred["score"].between(0, 1).all()

    def test_threshold_adds_predicted_label(self, fitted):
        aapred, df_seq, _ = fitted
        df_pred = aapred.predict(df_seq.head(5), level="seq", threshold=0.5)
        assert "predicted_label" in df_pred.columns
        assert set(df_pred["predicted_label"]).issubset({aapred.label_pos_, aapred.label_neg_})

    def test_list_parts(self, fitted):
        aapred, df_seq, _ = fitted
        df_pred = aapred.predict(df_seq.head(3), level="seq",
                                 list_parts=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"])
        assert len(df_pred) == 3

    def test_before_fit_raises(self, fitted):
        _, df_seq, df_feat = fitted
        with pytest.raises(ValueError):
            aa.AAPred(df_feat=df_feat).predict(df_seq.head(3), level="seq")

    def test_invalid_level_raises(self, fitted):
        aapred, df_seq, _ = fitted
        with pytest.raises(ValueError):
            aapred.predict(df_seq.head(3), level="not_a_level")


class TestPredictDomain:
    def test_columns(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=2)
        assert list(df.columns) == ["entry", "offset", "score", "is_best"]

    def test_offsets_scanned(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=2)
        assert set(df["offset"]) == {-2, -1, 0, 1, 2}

    def test_exactly_one_best(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=3)
        assert df["is_best"].sum() == 1

    def test_best_is_argmax(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=3)
        assert df.loc[df["is_best"], "score"].iloc[0] == df["score"].max()

    def test_list_parts(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=1,
                            list_parts=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"])
        assert len(df) >= 1


class TestPredictWindow:
    def _one(self, df_seq):
        return df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]]

    def test_columns(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(self._one(df_seq), level="window", tmd_len=15, step=20)
        assert list(df.columns) == ["entry", "position", "score", "score_std"]

    def test_positions_sorted_and_spaced(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(self._one(df_seq), level="window", tmd_len=15, step=10)
        pos = df["position"].to_numpy()
        assert (np.diff(pos) == 10).all()

    def test_step(self, fitted):
        aapred, df_seq, _ = fitted
        d1 = aapred.predict(self._one(df_seq), level="window", tmd_len=15, step=5)
        d2 = aapred.predict(self._one(df_seq), level="window", tmd_len=15, step=10)
        assert len(d1) > len(d2)

    def test_jmd_lengths(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(self._one(df_seq), level="window", tmd_len=15, step=25,
                            jmd_n_len=8, jmd_c_len=8)
        assert len(df) > 0

    def test_missing_tmd_len_raises(self, fitted):
        aapred, df_seq, _ = fitted
        with pytest.raises(ValueError):
            aapred.predict(self._one(df_seq), level="window")

    def test_list_parts(self, fitted):
        aapred, df_seq, _ = fitted
        df = aapred.predict(self._one(df_seq), level="window", tmd_len=15, step=25,
                            list_parts=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"])
        assert len(df) > 0


class TestPlotWindow:
    def _dw(self, fitted, step=15):
        aapred, df_seq, _ = fitted
        return aapred.predict(df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]],
                              level="window", tmd_len=15, step=step)

    def test_returns_fig_ax(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dw(fitted), kind="window")
        assert fig is not None and ax is not None

    def test_threshold(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dw(fitted), kind="window", threshold=0.75)
        assert any(l.get_linestyle() == "--" for l in ax.get_lines())

    def test_entry_multi_requires_entry(self, fitted):
        aapred, df_seq, _ = fitted
        two = df_seq[df_seq["entry"].isin(["P05067", "P14925"])][["entry", "sequence"]]
        dw = aapred.predict(two, level="window", tmd_len=15, step=30)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(dw, kind="window")
        fig, ax = aa.AAPredPlot().predict_sample(dw, kind="window", entry="P05067")
        assert ax is not None

    def test_list_annotations(self, fitted):
        dw = self._dw(fitted)
        tracks = [{"values": np.linspace(0, 1, len(dw)), "label": "pLDDT", "cmap": "RdYlBu"}]
        fig, ax = aa.AAPredPlot().predict_sample(dw, kind="window", list_annotations=tracks)
        assert ax is not None

    def test_color_labels(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dw(fitted), kind="window", color="#123456",
                                          xlabel="Pos", ylabel="P")
        assert ax.get_ylabel() == "P"

    def test_ax_figsize(self, fitted):
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_sample(self._dw(fitted), kind="window", ax=ax0, figsize=(10, 4))
        assert ax is ax0


class TestPlotDomain:
    def _dd(self, fitted):
        aapred, df_seq, _ = fitted
        return aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=3)

    def test_returns_fig_ax(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain")
        assert fig is not None and ax is not None

    def test_best_marked(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain")
        assert ax.get_legend() is not None

    def test_entry_color_labels(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain", entry="P05067",
                                          color="#123456", xlabel="Offset", ylabel="Score")
        assert ax.get_xlabel() == "Offset"

    def test_ax_figsize(self, fitted):
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain", ax=ax0, figsize=(6, 4))
        assert ax is ax0


@pytest.fixture(scope="module")
def scales():
    return aa.load_scales(), aa.load_scales(name="scales_cat")


class TestPlotWindowTracks:
    """Multi-track sequence viewer for kind='window' (importance, subcats, sequence, annotations)."""

    def _dw(self, fitted, step=45):
        aapred, df_seq, _ = fitted
        return aapred.predict(df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]],
                              level="window", tmd_len=15, step=step)

    def _one(self, fitted):
        _, df_seq, _ = fitted
        return df_seq[df_seq["entry"] == "P05067"]

    def test_numeric_annotation_is_line_track(self, fitted):
        dw = self._dw(fitted)
        tracks = [{"values": np.linspace(0, 1, len(dw)), "label": "pLDDT", "color": "tab:orange"}]
        fig, ax = aa.AAPredPlot().predict_sample(dw, kind="window", list_annotations=tracks)
        # base + one line track; the track carries a Line2D, not an image
        assert len(fig.axes) == 2
        assert len(fig.axes[1].images) == 0 and len(fig.axes[1].get_lines()) >= 1

    def test_categorical_annotation_is_imshow_fallback(self, fitted):
        dw = self._dw(fitted)
        n = len(dw)
        cats = ["H"] * (n // 2) + ["E"] * (n - n // 2)
        tracks = [{"values": cats, "label": "SS", "cmap": "coolwarm"}]
        fig, ax = aa.AAPredPlot().predict_sample(dw, kind="window", list_annotations=tracks)
        assert len(fig.axes[1].images) == 1  # categorical -> imshow strip

    def test_subcats_add_line_tracks(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        subcats = list(df_feat["subcategory"].unique()[:2])
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dw(fitted), kind="window", entry="P05067", subcats=subcats,
            df_seq=self._one(fitted), df_scales=df_scales, df_cat=df_cat)
        # base + one track per subcat + sequence row
        assert len(fig.axes) == len(subcats) + 2

    def test_subcats_default_load_scales_when_omitted(self, fitted):
        # df_scales/df_cat omitted -> bundled scales loaded internally
        aapred, df_seq, df_feat = fitted
        subcats = list(df_feat["subcategory"].unique()[:1])
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dw(fitted), kind="window", entry="P05067", subcats=subcats,
            df_seq=self._one(fitted))
        assert len(fig.axes) == 3  # base + 1 subcat + sequence

    def test_df_feat_adds_importance_track(self, fitted):
        aapred, df_seq, df_feat = fitted
        fig, ax = aa.AAPredPlot().predict_sample(self._dw(fitted), kind="window", df_feat=df_feat)
        assert len(fig.axes) == 2  # base + importance
        assert len(fig.axes[1].get_lines()) >= 1

    def test_sequence_row_renders_letters_when_short(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dw(fitted, step=45), kind="window", entry="P05067", df_seq=self._one(fitted))
        seq_ax = fig.axes[-1]
        assert len(seq_ax.texts) > 0  # short region -> per-residue letters drawn

    def test_sequence_row_suppresses_letters_when_long(self, fitted):
        # step=5 -> many positions (> 80) -> only the position axis, no letters
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dw(fitted, step=5), kind="window", entry="P05067", df_seq=self._one(fitted))
        assert len(fig.axes[-1].texts) == 0

    def test_missing_inputs_omit_tracks(self, fitted):
        # subcats requested but no df_seq -> no subcat/sequence tracks (base only)
        aapred, df_seq, df_feat = fitted
        subcats = list(df_feat["subcategory"].unique()[:1])
        fig, ax = aa.AAPredPlot().predict_sample(self._dw(fitted), kind="window", subcats=subcats)
        assert len(fig.axes) == 1

    def test_full_stack_returns_base_axes_on_top(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        n = len(self._dw(fitted))
        ann = [{"values": np.linspace(0, 1, n), "label": "pLDDT"}]
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dw(fitted), kind="window", entry="P05067", threshold=0.5,
            list_annotations=ann, subcats=list(df_feat["subcategory"].unique()[:1]),
            df_feat=df_feat, df_seq=self._one(fitted), df_scales=df_scales, df_cat=df_cat)
        assert ax is fig.axes[0]  # returned axes is the base profile (top track)
        assert len(fig.axes) == 5  # base + importance + subcat + annotation + sequence


class TestPlotDomainTracks:
    """Multi-track viewer for kind='domain' (offsets mapped to residues via tmd_start)."""

    def _dd(self, fitted, window=5):
        aapred, df_seq, _ = fitted
        return aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=window)

    def _one(self, fitted):
        _, df_seq, _ = fitted
        return df_seq[df_seq["entry"] == "P05067"]

    def test_df_feat_importance_track(self, fitted):
        aapred, df_seq, df_feat = fitted
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain", df_feat=df_feat)
        assert len(fig.axes) == 2 and ax is fig.axes[0]

    def test_subcats_and_sequence_via_tmd_start(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        subcats = list(df_feat["subcategory"].unique()[:1])
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dd(fitted), kind="domain", entry="P05067", subcats=subcats,
            df_seq=self._one(fitted), df_scales=df_scales, df_cat=df_cat)
        # base + subcat + sequence (offsets -> residues via tmd_start)
        assert len(fig.axes) == 3
        assert len(fig.axes[-1].texts) > 0  # short offset span -> residue letters

    def test_user_annotation_line_track(self, fitted):
        dd = self._dd(fitted)
        ann = [{"values": np.linspace(0, 1, len(dd)), "label": "cons"}]
        fig, ax = aa.AAPredPlot().predict_sample(dd, kind="domain", list_annotations=ann)
        assert len(fig.axes) == 2

    def test_missing_df_seq_omits_residue_tracks(self, fitted):
        # subcats requested but no df_seq -> subcat/sequence tracks omitted (base only)
        aapred, df_seq, df_feat = fitted
        subcats = list(df_feat["subcategory"].unique()[:1])
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain", subcats=subcats)
        assert len(fig.axes) == 1

    def test_returns_fig_ax(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain")
        assert fig is not None and ax is not None


class TestPredictSampleInvalidKind:
    def test_bad_kind_raises(self, fitted):
        # A group kind (or any non-positional kind) must be rejected by predict_sample,
        # naming only the two valid sample kinds.
        aapred, df_seq, _ = fitted
        dw = aapred.predict(df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]],
                            level="window", tmd_len=15, step=15)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(dw, kind="hist")
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(dw, kind="not_a_kind")


class TestPredictSampleHighlightWindow:
    """Region highlighting + zoom for kind='window' (residue-position spans)."""

    def _dw(self, fitted, step=45):
        aapred, df_seq, _ = fitted
        return aapred.predict(df_seq[df_seq["entry"] == "P05067"][["entry", "sequence"]],
                              level="window", tmd_len=15, step=step)

    def _one(self, fitted):
        _, df_seq, _ = fitted
        return df_seq[df_seq["entry"] == "P05067"]

    def _mid(self, dw):
        pos = dw["position"].to_numpy()
        return int(pos[len(pos) // 2])

    def test_single_tuple_spans_every_track_axes(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        dw = self._dw(fitted)
        mid = self._mid(dw)
        subcats = list(df_feat["subcategory"].unique()[:1])
        fig, ax = aa.AAPredPlot().predict_sample(
            dw, kind="window", entry="P05067", df_seq=self._one(fitted), df_scales=df_scales,
            df_cat=df_cat, subcats=subcats, df_feat=df_feat, highlight=(mid, mid + 20))
        assert len(fig.axes) > 1  # base + several tracks
        # one cyan span on the base axes and on every track axes
        assert all(len(_cyan_spans(a)) == 1 for a in fig.axes)

    def test_list_of_tuples_spans_every_track_axes(self, fitted):
        dw = self._dw(fitted)
        mid = self._mid(dw)
        fig, ax = aa.AAPredPlot().predict_sample(
            dw, kind="window", entry="P05067", df_seq=self._one(fitted),
            highlight=[(mid, mid + 15), (mid + 30, mid + 45)])
        assert len(fig.axes) == 2  # base + sequence row
        assert all(len(_cyan_spans(a)) == 2 for a in fig.axes)  # two regions on every axes

    def test_zoom_restricts_xlim_to_region_plus_pad(self, fitted):
        dw = self._dw(fitted)
        mid = self._mid(dw)
        full = aa.AAPredPlot().predict_sample(dw, kind="window", entry="P05067")[1].get_xlim()
        fig, ax = aa.AAPredPlot().predict_sample(
            dw, kind="window", entry="P05067", highlight=(mid, mid + 20), zoom=True)
        lo, hi = ax.get_xlim()
        assert lo <= mid and hi >= mid + 20            # region stays inside the view
        assert lo >= mid - 10 and hi <= mid + 20 + 10  # padded by only a few residues
        assert (hi - lo) < (full[1] - full[0])         # genuinely zoomed in

    def test_zoom_reveals_sequence_letters_keyed_off_visible_span(self, fitted):
        # step=5 -> >80 positions: letters suppressed at full view, but the short zoomed
        # window must render per-residue letters (visible-length keying, not data-length).
        dw = self._dw(fitted, step=5)
        mid = self._mid(dw)
        full = aa.AAPredPlot().predict_sample(
            dw, kind="window", entry="P05067", df_seq=self._one(fitted))
        assert len(full[0].axes[-1].texts) == 0
        zoomed = aa.AAPredPlot().predict_sample(
            dw, kind="window", entry="P05067", df_seq=self._one(fitted),
            highlight=(mid, mid + 30), zoom=True)
        assert len(zoomed[0].axes[-1].texts) > 0

    def test_returns_fig_ax(self, fitted):
        dw = self._dw(fitted)
        mid = self._mid(dw)
        fig, ax = aa.AAPredPlot().predict_sample(
            dw, kind="window", entry="P05067", highlight=(mid, mid + 20), zoom=True)
        assert fig is not None and ax is fig.axes[0]

    def test_start_gt_stop_raises(self, fitted):
        dw = self._dw(fitted)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(dw, kind="window", highlight=(60, 40))

    def test_non_int_bounds_raise(self, fitted):
        dw = self._dw(fitted)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(dw, kind="window", highlight=(50.5, 60.0))
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(dw, kind="window", highlight=[(10, 20), (30, "x")])


class TestPredictSampleHighlightDomain:
    """Region highlighting + zoom for kind='domain' (boundary-offset spans)."""

    def _dd(self, fitted, window=8):
        aapred, df_seq, _ = fitted
        return aapred.predict(df_seq[df_seq["entry"] == "P05067"], level="domain", window=window)

    def _one(self, fitted):
        _, df_seq, _ = fitted
        return df_seq[df_seq["entry"] == "P05067"]

    def test_single_tuple_spans_every_track_axes(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        subcats = list(df_feat["subcategory"].unique()[:1])
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dd(fitted), kind="domain", entry="P05067", df_seq=self._one(fitted),
            df_scales=df_scales, df_cat=df_cat, subcats=subcats, highlight=(-2, 2))
        assert len(fig.axes) == 3  # base + subcat + sequence
        assert all(len(_cyan_spans(a)) == 1 for a in fig.axes)

    def test_list_of_tuples_spans_every_track_axes(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dd(fitted), kind="domain", entry="P05067",
            highlight=[(-5, -3), (2, 4)])
        assert all(len(_cyan_spans(a)) == 2 for a in fig.axes)

    def test_zoom_restricts_xlim_and_returns_fig_ax(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(
            self._dd(fitted), kind="domain", entry="P05067", highlight=(-2, 2), zoom=True)
        lo, hi = ax.get_xlim()
        assert lo <= -2 and hi >= 2
        assert fig is not None and ax is fig.axes[0]

    def test_invalid_highlight_raises(self, fitted):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain", highlight=(3, -3))
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(self._dd(fitted), kind="domain", highlight=(1.5, 2.5))


def _track_axes(fig, ax):
    """Axes that share the residue x-axis with the heatmap ``ax`` (excludes the colorbar)."""
    return [a for a in fig.axes if a is ax or ax in a.get_shared_x_axes().get_siblings(a)]


class TestPredictSampleSequence:
    """Full-sequence subcategory x residue heatmap (kind='sequence')."""

    def _one(self, fitted):
        _, df_seq, _ = fitted
        return df_seq[df_seq["entry"] == "P05067"]

    def _subcats(self, df_feat, n=3):
        return list(pd.unique(df_feat["subcategory"]))[:n]

    def _seq_len(self, one):
        return len(one["sequence"].iloc[0])

    def test_registered_kind(self):
        from aaanalysis.prediction._aa_pred_plot import LIST_SAMPLE_KINDS
        assert "sequence" in LIST_SAMPLE_KINDS

    def test_returns_fig_ax(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            df_scales=df_scales, df_cat=df_cat, subcats=self._subcats(df_feat))
        assert fig is not None and ax is not None and ax is fig.axes[0]

    def test_rows_equal_n_subcats(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        subcats = self._subcats(df_feat, n=4)
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            df_scales=df_scales, df_cat=df_cat, subcats=subcats)
        matrix = ax.images[0].get_array()
        assert matrix.shape[0] == len(subcats)  # one heatmap row per subcategory

    def test_columns_span_full_sequence(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        one = self._one(fitted)
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=one,
            df_scales=df_scales, df_cat=df_cat, subcats=self._subcats(df_feat))
        matrix = ax.images[0].get_array()
        assert matrix.shape[1] == self._seq_len(one)  # one column per residue (full sequence)

    def test_subcats_none_uses_all_capped(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        from aaanalysis.prediction._aa_pred_plot import _SUBCAT_ROW_CAP
        n_all = df_cat["subcategory"].nunique()
        assert n_all > _SUBCAT_ROW_CAP  # data really has more subcategories than the cap
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            df_scales=df_scales, df_cat=df_cat, subcats=None)
        assert ax.images[0].get_array().shape[0] == _SUBCAT_ROW_CAP

    def test_subcats_none_below_cap_uses_all(self, fitted, scales):
        # A small df_cat (< cap subcategories) shows all of them, uncapped.
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        keep = list(pd.unique(df_cat["subcategory"]))[:5]
        df_cat_small = df_cat[df_cat["subcategory"].isin(keep)]
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            df_scales=df_scales, df_cat=df_cat_small, subcats=None)
        assert ax.images[0].get_array().shape[0] == 5

    def test_default_loads_scales_when_omitted(self, fitted):
        # df_scales/df_cat omitted -> bundled scales loaded internally (heatmap still drawn)
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            subcats=self._subcats(fitted[2]))
        assert len(ax.images) == 1

    def test_sequence_row_below_heatmap(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            df_scales=df_scales, df_cat=df_cat, subcats=self._subcats(df_feat))
        tracks = _track_axes(fig, ax)
        assert len(tracks) == 2  # heatmap + sequence row (colorbar excluded)
        # the sequence row carries the x-label; the heatmap does not
        assert tracks[-1].get_xlabel() == "Residue position"

    def test_data_adds_prediction_track_above(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        one = self._one(fitted)
        dw = aapred.predict(one[["entry", "sequence"]], level="window", tmd_len=15, step=40)
        fig, ax = aa.AAPredPlot().predict_sample(
            data=dw, kind="sequence", entry="P05067", df_seq=one,
            df_scales=df_scales, df_cat=df_cat, subcats=self._subcats(df_feat), threshold=0.5)
        tracks = _track_axes(fig, ax)
        assert len(tracks) == 3  # prediction + heatmap + sequence row
        assert tracks[0] is not ax and len(tracks[0].get_lines()) >= 1  # top track is the profile

    def test_missing_df_seq_raises(self, fitted):
        _, _, df_feat = fitted
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(kind="sequence", subcats=self._subcats(df_feat))

    def test_data_optional_no_df_seq_still_raises(self, fitted):
        # data is optional for sequence, but df_seq is not -> still raises without it
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(kind="sequence")

    def test_multi_entry_requires_entry(self, fitted, scales):
        _, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        two = df_seq[df_seq["entry"].isin(["P05067", "P14925"])]
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(kind="sequence", df_seq=two,
                                           subcats=self._subcats(df_feat))
        fig, ax = aa.AAPredPlot().predict_sample(kind="sequence", df_seq=two, entry="P05067",
                                                 subcats=self._subcats(df_feat))
        assert ax is not None

    def test_unknown_entry_raises(self, fitted):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(kind="sequence", df_seq=self._one(fitted),
                                           entry="NOT_A_PROTEIN")

    def test_ax_draws_heatmap_only(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        fig0, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted), ax=ax0,
            df_scales=df_scales, df_cat=df_cat, subcats=self._subcats(df_feat))
        assert ax is ax0 and len(ax0.images) == 1


class TestPredictSampleSequenceHighlight:
    """Region highlighting + zoom for kind='sequence' (residue-position columns)."""

    def _one(self, fitted):
        _, df_seq, _ = fitted
        return df_seq[df_seq["entry"] == "P05067"]

    def _subcats(self, df_feat, n=3):
        return list(pd.unique(df_feat["subcategory"]))[:n]

    def test_single_tuple_spans_every_track_axes(self, fitted, scales):
        aapred, df_seq, df_feat = fitted
        df_scales, df_cat = scales
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted), df_scales=df_scales,
            df_cat=df_cat, subcats=self._subcats(df_feat), highlight=(100, 150))
        tracks = _track_axes(fig, ax)
        assert all(len(_cyan_spans(a)) == 1 for a in tracks)  # one span on heatmap + sequence row

    def test_list_of_tuples_spans_every_track_axes(self, fitted):
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=self._one(fitted),
            subcats=self._subcats(fitted[2]), highlight=[(100, 150), (300, 340)])
        tracks = _track_axes(fig, ax)
        assert all(len(_cyan_spans(a)) == 2 for a in tracks)

    def test_zoom_restricts_xlim_and_reveals_letters(self, fitted):
        one = self._one(fitted)
        subcats = self._subcats(fitted[2])
        # full view: >80 residues -> sequence letters suppressed
        full = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=one, subcats=subcats)
        seq_ax_full = _track_axes(full[0], full[1])[-1]
        assert len(seq_ax_full.texts) == 0
        # zoom into a short region: xlim restricted + per-residue letters drawn
        fig, ax = aa.AAPredPlot().predict_sample(
            kind="sequence", entry="P05067", df_seq=one, subcats=subcats,
            highlight=(690, 720), zoom=True)
        lo, hi = ax.get_xlim()
        assert lo <= 690 and hi >= 720
        assert (hi - lo) < len(one["sequence"].iloc[0])  # genuinely zoomed in
        seq_ax = _track_axes(fig, ax)[-1]
        assert len(seq_ax.texts) > 0

    def test_invalid_highlight_raises(self, fitted):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(kind="sequence", df_seq=self._one(fitted),
                                           subcats=self._subcats(fitted[2]), highlight=(150, 100))
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_sample(kind="sequence", df_seq=self._one(fitted),
                                           subcats=self._subcats(fitted[2]), highlight=(10.5, 20.0))
