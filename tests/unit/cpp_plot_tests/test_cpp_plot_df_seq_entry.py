"""
This script tests the df_seq=/entry= convenience input of CPPPlot.profile() and
CPPPlot.feature_map() (issue #127): deriving the per-sample TMD-JMD sequence parts
internally must match the explicit **args_seq path, and stay back-compatible.
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
COL_FEAT_IMPACT_TEST = "feat_impact_test"


def get_df_seq():
    aa.options["verbose"] = False
    return aa.load_dataset(name="DOM_GSEC", n=N_SEQ)


def get_args_seq(df_seq, n=0):
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[n, ["jmd_n", "tmd", "jmd_c"]]
    return dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)


def get_df_feat(n=10):
    aa.options["verbose"] = False
    df_feat = aa.load_features().head(n)
    df_feat.insert(0, COL_FEAT_IMPACT_TEST, [2] * len(df_feat))
    return df_feat


def xtick_text(ax):
    """The rendered TMD-JMD sequence (where the per-sample seq parts manifest)."""
    return [t.get_text() for t in ax.get_xticklabels(which="both")]


class TestProfileDfSeqEntry:
    """Positive + negative cases for profile(df_seq=, entry=)."""

    def test_entry_path_matches_explicit_args_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        args_seq = get_args_seq(df_seq, n=0)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.profile(df_feat=df_feat, **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.profile(df_feat=df_feat, df_seq=df_seq, entry=entry)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    @settings(max_examples=4, deadline=None)
    @given(n=st.integers(min_value=0, max_value=N_SEQ - 1))
    def test_entry_path_matches_per_sample(self, n):
        df_seq = get_df_seq()
        entry = df_seq.loc[n, "entry"]
        args_seq = get_args_seq(df_seq, n=n)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.profile(df_feat=df_feat, **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.profile(df_feat=df_feat, df_seq=df_seq, entry=entry)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    def test_backcompat_explicit_args_seq_still_works(self):
        df_seq = get_df_seq()
        args_seq = get_args_seq(df_seq, n=0)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_no_entry_no_df_seq_unchanged(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_error_entry_and_explicit_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        args_seq = get_args_seq(df_seq, n=0)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should not be provided together"):
            cpp_plot.profile(df_feat=df_feat, df_seq=df_seq, entry=entry, **args_seq)

    def test_error_entry_without_df_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="df_seq"):
            cpp_plot.profile(df_feat=df_feat, entry=entry)

    def test_error_df_seq_without_entry(self):
        df_seq = get_df_seq()
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should be given"):
            cpp_plot.profile(df_feat=df_feat, df_seq=df_seq)

    def test_error_unknown_entry(self):
        df_seq = get_df_seq()
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.profile(df_feat=df_feat, df_seq=df_seq, entry="NOT_AN_ENTRY")


class TestFeatureMapDfSeqEntry:
    """Positive + negative cases for feature_map(df_seq=, entry=)."""

    def test_entry_path_matches_explicit_args_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        args_seq = get_args_seq(df_seq, n=0)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.feature_map(df_feat=df_feat, **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.feature_map(df_feat=df_feat, df_seq=df_seq, entry=entry)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    @settings(max_examples=4, deadline=None)
    @given(n=st.integers(min_value=0, max_value=N_SEQ - 1))
    def test_entry_path_matches_per_sample(self, n):
        df_seq = get_df_seq()
        entry = df_seq.loc[n, "entry"]
        args_seq = get_args_seq(df_seq, n=n)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig1, ax1 = cpp_plot.feature_map(df_feat=df_feat, **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.feature_map(df_feat=df_feat, df_seq=df_seq, entry=entry)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2

    def test_backcompat_explicit_args_seq_still_works(self):
        df_seq = get_df_seq()
        args_seq = get_args_seq(df_seq, n=0)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_no_entry_no_df_seq_unchanged(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.feature_map(df_feat=df_feat)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_error_entry_and_explicit_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        args_seq = get_args_seq(df_seq, n=0)
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should not be provided together"):
            cpp_plot.feature_map(df_feat=df_feat, df_seq=df_seq, entry=entry, **args_seq)

    def test_error_entry_without_df_seq(self):
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="df_seq"):
            cpp_plot.feature_map(df_feat=df_feat, entry=entry)

    def test_error_unknown_entry(self):
        df_seq = get_df_seq()
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError):
            cpp_plot.feature_map(df_feat=df_feat, df_seq=df_seq, entry="NOT_AN_ENTRY")

    def test_entry_honors_jmd_len(self):
        """Derived parts must honor the instance jmd_n_len/jmd_c_len, matching get_args_seq."""
        df_seq = get_df_seq()
        entry = df_seq.loc[0, "entry"]
        df_feat = get_df_feat()
        sf = aa.SequenceFeature(verbose=False)
        # Explicit parts computed with the SAME jmd lengths as the instance
        args_seq = sf.get_args_seq(df_seq=df_seq, sample=entry, jmd_n_len=5, jmd_c_len=5)
        cpp_plot = aa.CPPPlot(jmd_n_len=5, jmd_c_len=5)
        fig1, ax1 = cpp_plot.feature_map(df_feat=df_feat, **args_seq)
        t1 = xtick_text(ax1)
        plt.close("all")
        fig2, ax2 = cpp_plot.feature_map(df_feat=df_feat, df_seq=df_seq, entry=entry)
        t2 = xtick_text(ax2)
        plt.close("all")
        assert t1 == t2
