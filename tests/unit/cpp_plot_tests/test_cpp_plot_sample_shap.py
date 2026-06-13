"""
This script tests the sample= / df_seq= SHAP convenience of CPPPlot.profile() and
CPPPlot.feature_map() (issue #214): with shap_plot=True, the feat_impact_<name>
column and the per-sample TMD-JMD parts are resolved internally, and name_test is
auto-resolved (feature_map). The explicit col_imp=/**args_seq path stays working.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hypothesis import given, settings, strategies as st
import pytest
import aaanalysis as aa

aa.options["verbose"] = False
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Constants and Helper functions
N_SEQ = 10
N_FEAT = 10


def get_df_seq():
    aa.options["verbose"] = False
    return aa.load_dataset(name="DOM_GSEC", n=N_SEQ)


def get_args_seq(df_seq, sample):
    sf = aa.SequenceFeature(verbose=False)
    return sf.get_args_seq(df_seq=df_seq, sample=sample)


def get_df_feat_impact(name):
    """Feature table carrying a single signed feat_impact_<name> column."""
    aa.options["verbose"] = False
    df_feat = aa.load_features().head(N_FEAT).reset_index(drop=True)
    signs = [1 if i % 2 == 0 else -1 for i in range(len(df_feat))]
    df_feat[f"feat_impact_{name}"] = [s * (1 + i) for i, s in enumerate(signs)]
    return df_feat


def xtick_text(ax):
    return [t.get_text() for t in ax.get_xticklabels(which="both")]


class TestProfileSampleShap:
    """profile(sample=, df_seq=, shap_plot=True) resolves col_imp + seq parts."""

    def test_sample_matches_explicit_col_imp_and_args_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        args_seq = get_args_seq(df_seq, entry)
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.profile(df_feat=df_feat, shap_plot=True,
                                     col_imp=f"feat_impact_{entry}", **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    @settings(max_examples=4, deadline=None)
    @given(n=st.integers(min_value=0, max_value=N_SEQ - 1))
    def test_int_sample_matches_entry(self, n):
        df_seq = get_df_seq()
        entry = df_seq.loc[n, "entry"]
        df_feat = get_df_feat_impact(entry)
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=n)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    def test_custom_name_single_impact_col_resolves(self):
        """name != accession: the single feat_impact_* column is still resolved."""
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact("APP")  # custom name, accession differs
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_backcompat_explicit_col_imp_args_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        args_seq = get_args_seq(df_seq, entry)
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat, shap_plot=True,
                                   col_imp=f"feat_impact_{entry}", **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_explicit_col_imp_wins_over_sample(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry,
                                   col_imp=f"feat_impact_{entry}")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_error_entry_and_sample(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should not be provided together"):
            cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, entry=entry, sample=0)

    def test_error_ambiguous_impact_columns(self):
        """Multiple feat_impact_* columns and no exact match -> clear error."""
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact("APP")
        df_feat["feat_impact_OTHER"] = 1.0
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="col_imp"):
            cpp_plot.profile(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry)


class TestFeatureMapSampleShap:
    """feature_map(sample=, df_seq=, shap_plot=True) resolves col_imp + seq + name_test."""

    def test_sample_matches_explicit_path(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        args_seq = get_args_seq(df_seq, entry)
        col = f"feat_impact_{entry}"
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, col_imp=col, col_val=col,
                                         name_test=entry, **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry,
                                         col_val=col)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    def test_name_test_auto_resolves_to_sample(self):
        """name_test defaults to the sample name (shown in the colorbar label)."""
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        # mean_dif_<name> col_val -> colorbar label reads "<name_test> - <name_ref>"
        df_feat[f"mean_dif_{entry}"] = df_feat["mean_dif"]
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=entry,
                                       col_val=f"mean_dif_{entry}")
        labels = [axx.get_ylabel() for axx in fig.axes]
        labels += [axx.get_xlabel() for axx in fig.axes]
        labels += [t.get_text() for axx in fig.axes for t in axx.texts]
        assert any(entry in (s or "") for s in labels)
        plt.close("all")

    def test_int_sample_runs(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[2, "entry"]
        df_feat = get_df_feat_impact(entry)
        col = f"feat_impact_{entry}"
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, df_seq=df_seq, sample=2,
                                       col_val=col)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_backcompat_explicit_path(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        args_seq = get_args_seq(df_seq, entry)
        col = f"feat_impact_{entry}"
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, col_imp=col, col_val=col,
                                       name_test=entry, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_error_entry_and_sample(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat_impact(entry)
        col = f"feat_impact_{entry}"
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should not be provided together"):
            cpp_plot.feature_map(df_feat=df_feat, shap_plot=True, df_seq=df_seq, entry=entry, sample=0,
                                 col_val=col)
