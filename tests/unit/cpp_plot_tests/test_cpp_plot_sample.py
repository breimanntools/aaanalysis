"""This script tests the ``sample=`` shortcut of CPPPlot.ranking/profile/feature_map.

The shortcut resolves ``col_imp='feat_impact_<sample>'`` (all three methods) and the TMD-JMD
sequence parts from ``df_parts`` (profile / feature_map), so each ``sample=...`` call must be
byte-identical to the equivalent explicit ``col_imp=...`` (+ ``**seq_kws``) call, and passing
``sample=None`` must not change the default output at all.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import aaanalysis as aa

aa.options["verbose"] = False

# Fixtures shared across the module (small, deterministic)
DF_SEQ = aa.load_dataset(name="DOM_GSEC", n=5)
ACC = DF_SEQ["entry"].to_list()[0]
SF = aa.SequenceFeature(verbose=False)
DF_PARTS = SF.get_df_parts(df_seq=DF_SEQ)
SEQ_KWS = SF.get_seq_kws(df_seq=DF_SEQ, df_parts=DF_PARTS, sample=ACC)


def _df_feat():
    df_feat = aa.load_features().head(40).copy()
    df_feat[f"feat_impact_{ACC}"] = np.linspace(-2, 3, len(df_feat))
    return df_feat


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    return buf.getvalue()


class TestSampleGoldenRanking:
    """ranking(sample=...) equals the explicit col_imp call (resolve_seq=False)."""

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
    """profile(sample=...) equals the explicit col_imp + seq_kws call (resolve_seq=True)."""

    def test_sample_equals_explicit(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.profile(df_feat=df_feat, shap_plot=True, col_imp=f"feat_impact_{ACC}", **SEQ_KWS)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.profile(df_feat=df_feat, sample=ACC, df_seq=DF_SEQ, df_parts=DF_PARTS)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_sample_none_is_default(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.profile(df_feat=df_feat)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.profile(df_feat=df_feat, sample=None)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b


class TestSampleGoldenFeatureMap:
    """feature_map(sample=...) equals the explicit col_imp + seq_kws call (resolve_seq=True)."""

    def test_sample_equals_explicit(self):
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.feature_map(df_feat=df_feat, shap_plot=True, col_imp=f"feat_impact_{ACC}", **SEQ_KWS)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.feature_map(df_feat=df_feat, sample=ACC, df_seq=DF_SEQ, df_parts=DF_PARTS)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_int_position_equals_name(self):
        # sample=0 (row position) maps to the entry name, so it equals sample=ACC
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.feature_map(df_feat=df_feat, sample=ACC, df_seq=DF_SEQ, df_parts=DF_PARTS)
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.feature_map(df_feat=df_feat, sample=0, df_seq=DF_SEQ, df_parts=DF_PARTS)
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b

    def test_explicit_parts_not_overridden(self):
        # An explicitly passed tmd_seq must survive the sample= resolution.
        cpp = aa.CPPPlot(verbose=False)
        df_feat = _df_feat()
        fig_a, _ = cpp.feature_map(df_feat=df_feat, shap_plot=True, col_imp=f"feat_impact_{ACC}",
                                   jmd_n_seq=SEQ_KWS["jmd_n_seq"], tmd_seq=SEQ_KWS["tmd_seq"],
                                   jmd_c_seq=SEQ_KWS["jmd_c_seq"])
        png_a = _png(fig_a)
        plt.close("all")
        fig_b, _ = cpp.feature_map(df_feat=df_feat, sample=ACC, df_seq=DF_SEQ, df_parts=DF_PARTS,
                                   tmd_seq=SEQ_KWS["tmd_seq"])
        png_b = _png(fig_b)
        plt.close("all")
        assert png_a == png_b


class TestSampleErrors:
    """Negative cases for the sample= shortcut."""

    def test_ranking_bad_sample_type_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.ranking(df_feat=_df_feat(), sample=1.5)

    def test_profile_sample_without_df_seq_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.profile(df_feat=_df_feat(), sample=ACC, df_parts=DF_PARTS)

    def test_profile_sample_without_df_parts_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.profile(df_feat=_df_feat(), sample=ACC, df_seq=DF_SEQ)

    def test_feature_map_sample_without_parts_raises(self):
        cpp = aa.CPPPlot(verbose=False)
        with pytest.raises(ValueError):
            cpp.feature_map(df_feat=_df_feat(), sample=ACC)
