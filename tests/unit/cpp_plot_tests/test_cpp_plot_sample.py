"""Tests for the sample-level shortcut of CPPPlot.ranking/profile/feature_map/heatmap.

``ranking`` keeps a flat ``sample=`` (entry name only; it has no ``df_parts`` to map a position).
``profile`` / ``heatmap`` / ``feature_map`` take the bundled ``sample_kws=dict(sample, df_seq,
df_parts)`` — the alternative to providing the TMD-JMD sequences directly. Each resolved call must be
byte-identical to the equivalent explicit ``col_imp=...`` (+ ``**seq_kws``) call, ``sample_kws=None``
must not change the default output, and ``sample_kws`` OVERRIDES any explicitly passed sequences.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import aaanalysis as aa
from aaanalysis.feature_engineering._cpp_plot import unpack_sample_kws

aa.options["verbose"] = False

# Fixtures shared across the module (small, deterministic)
DF_SEQ = aa.load_dataset(name="DOM_GSEC", n=5)
ACC = DF_SEQ["entry"].to_list()[0]
SF = aa.SequenceFeature(verbose=False)
DF_PARTS = SF.get_df_parts(df_seq=DF_SEQ)
SEQ_KWS = SF.get_seq_kws(df_seq=DF_SEQ, df_parts=DF_PARTS, sample=ACC)
SKW = dict(sample=ACC, df_seq=DF_SEQ, df_parts=DF_PARTS)


def _df_feat():
    df_feat = aa.load_features().head(40).copy()
    df_feat[f"feat_impact_{ACC}"] = np.linspace(-2, 3, len(df_feat))
    return df_feat


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    return buf.getvalue()


class TestUnpackSampleKws:
    """The bundle validator accepts the fixed key set and rejects the rest."""

    def test_none_passthrough(self):
        assert unpack_sample_kws(None) == (None, None, None)

    def test_returns_triplet(self):
        assert unpack_sample_kws(dict(sample=ACC, df_seq=DF_SEQ, df_parts=DF_PARTS))[0] == ACC

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError):
            unpack_sample_kws(dict(sample=ACC, bogus=1))

    def test_missing_sample_raises(self):
        with pytest.raises(ValueError):
            unpack_sample_kws(dict(df_seq=DF_SEQ, df_parts=DF_PARTS))

    def test_not_a_dict_raises(self):
        with pytest.raises(ValueError):
            unpack_sample_kws([ACC])


class TestSampleGoldenRanking:
    """ranking keeps the flat sample= (entry name only, resolve_seq=False)."""

    def test_sample_equals_explicit(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.ranking(df_feat=df_feat, shap_plot=True, col_imp=f"feat_impact_{ACC}")
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.ranking(df_feat=df_feat, sample=ACC)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_int_position_rejected(self):
        # ranking has no df_parts to map a position to the entry-name-keyed impact column
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.ranking(df_feat=_df_feat(), sample=0)

    def test_sample_none_is_default(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.ranking(df_feat=df_feat)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.ranking(df_feat=df_feat, sample=None)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b


class TestSampleGoldenProfile:
    """profile(sample_kws=...) equals the explicit col_imp + seq_kws call."""

    def test_sample_kws_equals_explicit(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.profile(df_feat=df_feat, shap_plot=True, col_imp=f"feat_impact_{ACC}", **SEQ_KWS)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.profile(df_feat=df_feat, sample_kws=SKW)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_sample_kws_none_is_default(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.profile(df_feat=df_feat)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.profile(df_feat=df_feat, sample_kws=None)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b


class TestSampleGoldenFeatureMap:
    """feature_map(sample_kws=...) equals the explicit col_imp + seq_kws call."""

    def test_sample_kws_equals_explicit(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.feature_map(df_feat=df_feat, shap_plot=True, col_imp=f"feat_impact_{ACC}", **SEQ_KWS)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.feature_map(df_feat=df_feat, sample_kws=SKW)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_int_position_equals_name(self):
        # sample=0 (row position) maps to the entry name, so it equals sample=ACC
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.feature_map(df_feat=df_feat, sample_kws=SKW)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.feature_map(df_feat=df_feat, sample_kws=dict(sample=0, df_seq=DF_SEQ, df_parts=DF_PARTS))
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_sample_kws_overrides_explicit_parts(self):
        # sample_kws WINS over explicitly passed sequences: adding a stray tmd_seq changes nothing,
        # since the sample's df_parts geometry is used (faithful to the data the features map to).
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.feature_map(df_feat=df_feat, sample_kws=SKW)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.feature_map(df_feat=df_feat, sample_kws=SKW, tmd_seq="A" * 5)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b


class TestSampleGoldenHeatmap:
    """heatmap(sample_kws=...) draws the sample's sequence band, equal to the explicit seq_kws call."""

    def test_sample_kws_equals_explicit(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = aa.load_features().head(40).copy()
        fig_a, _ = cpp.heatmap(df_feat=df_feat, **SEQ_KWS)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.heatmap(df_feat=df_feat, sample_kws=SKW)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b


class TestSampleErrors:
    """Negative cases for the sample shortcut."""

    def test_ranking_bad_sample_type_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.ranking(df_feat=_df_feat(), sample=1.5)

    def test_profile_sample_kws_without_df_seq_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.profile(df_feat=_df_feat(), sample_kws=dict(sample=ACC, df_parts=DF_PARTS))

    def test_profile_sample_kws_without_df_parts_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.profile(df_feat=_df_feat(), sample_kws=dict(sample=ACC, df_seq=DF_SEQ))

    def test_feature_map_sample_kws_without_parts_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.feature_map(df_feat=_df_feat(), sample_kws=dict(sample=ACC))
