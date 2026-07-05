"""Unit tests for AAPred sequence-level prediction (seq / domain / window) + their plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa

aa.options["verbose"] = False


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
