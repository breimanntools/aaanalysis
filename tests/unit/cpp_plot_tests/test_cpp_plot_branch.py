"""
This script targets specific BRANCH-coverage arcs of CPPPlot and its backend
that the per-method test files leave un-hit. Every test drives a public
``aa.CPPPlot`` method (with ``aa.CPP``/``aa.load_features`` producing inputs);
no private backend function is called directly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False

N_SEQ = 10
COL_FEAT_IMPACT_TEST = "feat_impact_test"
COL_MEAN_DIF_TEST = "mean_dif_test"


# Helper functions
def get_df_feat(n=10):
    aa.options["verbose"] = False
    df_feat = aa.load_features().head(n).copy()
    df_feat.insert(0, COL_FEAT_IMPACT_TEST, np.linspace(-2, 2, len(df_feat)))
    df_feat.insert(0, COL_MEAN_DIF_TEST, [1.0] * len(df_feat))
    return df_feat


def get_args_seq(n=0):
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_SEQ)
    jmd_n_seq, tmd_seq, jmd_c_seq = df_seq.loc[n, ["jmd_n", "tmd", "jmd_c"]]
    return dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)


def get_df_feat_no_jmd_n(n=10):
    aa.options["verbose"] = False
    df = aa.load_features()
    return df[~df["feature"].str.contains("JMD_N")].head(n).copy()


def get_df_feat_no_jmd_c(n=10):
    aa.options["verbose"] = False
    df = aa.load_features()
    return df[~df["feature"].str.contains("JMD_C")].head(n).copy()


def get_df_feat_tmd_only(n=8):
    aa.options["verbose"] = False
    df = aa.load_features()
    return df[df["feature"].str.startswith("TMD-")].head(n).copy()


def make_df_eval(n_rows=3, n_cat=None):
    if n_cat is None:
        n_cat = len(ut.LIST_CAT)
    return pd.DataFrame({
        "name": [f"S{i}" for i in range(n_rows)],
        "n_features": [(50, np.random.randint(0, 40, size=n_cat).tolist()) for _ in range(n_rows)],
        "avg_ABS_AUC": np.random.rand(n_rows),
        "range_ABS_AUC": [np.sort(np.random.rand(5)).tolist() for _ in range(n_rows)],
        "avg_MEAN_DIF": [(np.random.rand(), -np.random.rand()) for _ in range(n_rows)],
        "n_clusters": np.random.randint(1, 10, size=n_rows),
        "avg_n_feat_per_clust": np.random.uniform(1, 5, size=n_rows),
        "std_n_feat_per_clust": np.random.uniform(0, 3, size=n_rows),
    })


class TestCPPPlotBranch:
    """Targeted positive/negative arcs through the public CPPPlot methods."""

    # --- heatmap: JMD-length-0 axvline + adjust-xticks arcs -----------------
    def test_heatmap_jmd_n_len_0(self):
        # _utils_cpp_plot_map.py L178->180 (skip jmd_n axvline);
        # _utils_cpp_plot_positions.py _get_ends / _adjust_xticks_labels JMD-N arm
        cpp_plot = aa.CPPPlot(jmd_n_len=0)
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat_no_jmd_n())
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_heatmap_jmd_c_len_0(self):
        # _utils_cpp_plot_map.py L180->182 (skip jmd_c axvline)
        cpp_plot = aa.CPPPlot(jmd_c_len=0)
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat_no_jmd_c())
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_heatmap_jmd_n_len_0_with_seq(self):
        # _utils_cpp_plot_positions.py L88-89 (_adjust_xticks_labels: drop JMD-N)
        args_seq = get_args_seq()
        args_seq["jmd_n_seq"] = ""
        cpp_plot = aa.CPPPlot(jmd_n_len=0)
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat_no_jmd_n(), **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_heatmap_jmd_c_len_0_with_seq(self):
        # _utils_cpp_plot_positions.py L92-93 (_adjust_xticks_labels: drop JMD-C)
        args_seq = get_args_seq()
        args_seq["jmd_c_seq"] = ""
        cpp_plot = aa.CPPPlot(jmd_c_len=0)
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat_no_jmd_c(), **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- heatmap: cmap == 'CPP' / 'SHAP' returns None (frontend L100) -------
    def test_heatmap_cmap_cpp_token(self):
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat(), cmap=ut.STR_CMAP_CPP)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_heatmap_cmap_shap_token(self):
        cpp_plot = aa.CPPPlot()
        df_feat = get_df_feat()
        fig, ax = cpp_plot.heatmap(df_feat=df_feat, shap_plot=True,
                                   col_val=COL_FEAT_IMPACT_TEST, cmap=ut.STR_CMAP_SHAP)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- heatmap: cbar_kws label + orientation branches ---------------------
    def test_heatmap_cbar_kws_label_orientation(self):
        # _utils_cpp_plot_map.py L102-108, L116-118, L125-138 (cbar label/spine)
        cpp_plot = aa.CPPPlot()
        cbar_kws = dict(label="value", orientation="horizontal", ticksize=8)
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat(), cbar_kws=cbar_kws)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- heatmap: fontsize_tmd_jmd=0 skips second axis (positions L271->278) -
    def test_heatmap_fontsize_tmd_jmd_zero(self):
        args_seq = get_args_seq()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.heatmap(df_feat=get_df_feat(), seq_size=8,
                                   fontsize_tmd_jmd=0, **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- profile: _scale_ylim warning arms (cpp_plot_profile.py L22-32) ------
    def _shap_profile_with_ylim(self, ylim):
        df_feat = get_df_feat()
        args_seq = get_args_seq()
        cpp_plot = aa.CPPPlot(verbose=True)
        fig, ax = cpp_plot.profile(df_feat=df_feat, shap_plot=True,
                                   col_imp=COL_FEAT_IMPACT_TEST, ylim=ylim,
                                   normalize=False, **args_seq)
        return fig, ax

    def test_profile_ylim_max_too_small(self):
        # L28-30: max_val > max_y
        with pytest.warns(UserWarning):
            fig, ax = self._shap_profile_with_ylim((-1000, 1))
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_profile_ylim_min_too_large(self):
        # L25-27: min_val < min_y
        with pytest.warns(UserWarning):
            fig, ax = self._shap_profile_with_ylim((-1, 1000))
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    def test_profile_ylim_both_off(self):
        # L22-24: min_val < min_y and max_val > max_y
        with pytest.warns(UserWarning):
            fig, ax = self._shap_profile_with_ylim((-1, 1))
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- profile: xtick_size=0 skips set_xticklabels (cpp_plot_profile L100->102)
    def test_profile_xtick_size_zero(self):
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=get_df_feat(), xtick_size=0)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- profile: wide figure so seq chars never overlap, exercising the
    #     step-wise font-size break (positions _get_optimal_fontsize L55) -----
    def test_profile_seq_wide_figure(self):
        args_seq = get_args_seq()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=get_df_feat(n=8), figsize=(40, 5), **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")

    # --- ranking: col_dif range > 2 skips *100 (cpp_plot_ranking L17->19) ----
    def test_ranking_col_dif_large_range(self):
        df_feat = aa.load_features().head(15).copy()
        df_feat["mean_dif"] = np.linspace(-5, 5, len(df_feat))
        cpp_plot = aa.CPPPlot()
        fig, axes = cpp_plot.ranking(df_feat=df_feat, n_top=10, xlim_dif=(-10, 10))
        assert isinstance(fig, plt.Figure) and isinstance(axes, np.ndarray)
        plt.close("all")

    # --- ranking: TMD-only label (cpp_plot_ranking _get_tmd_jmd_label L64) ---
    def test_ranking_tmd_only_label(self):
        cpp_plot = aa.CPPPlot(jmd_n_len=0, jmd_c_len=0)
        fig, axes = cpp_plot.ranking(df_feat=get_df_feat_tmd_only(), n_top=6)
        assert isinstance(fig, plt.Figure) and isinstance(axes, np.ndarray)
        plt.close("all")

    # --- feature: ax=None with an already-open figure (feature L161-162,
    #     elements set_figsize L38) ------------------------------------------
    def test_feature_ax_none_open_figure(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        feature = aa.load_features()["feature"].iloc[0]
        cpp_plot = aa.CPPPlot()
        plt.figure()  # ensure plt.get_fignums() is non-empty
        _, ax = cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    # --- feature_map: verbose seq-size print with tmd_seq (frontend L1487) ---
    def test_feature_map_verbose_tmd_seq(self):
        args_seq = get_args_seq()
        cpp_plot = aa.CPPPlot(verbose=True)
        fig, ax = cpp_plot.feature_map(df_feat=get_df_feat(), **args_seq)
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        plt.close("all")


class TestCPPPlotBranchNegative:
    """Negative arcs: frontend validation guards that other tests miss."""

    # --- eval: list_cat duplicate / missing color (frontend L44, L48) -------
    def test_eval_list_cat_duplicate(self):
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should not contain duplicates"):
            cpp_plot.eval(df_eval=make_df_eval(n_cat=2), list_cat=["Energy", "Energy"])
        plt.close("all")

    def test_eval_list_cat_missing_color(self):
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="miss a color"):
            cpp_plot.eval(df_eval=make_df_eval(n_cat=2),
                          dict_color={"Energy": "b"}, list_cat=["Energy", "Polarity"])
        plt.close("all")

    # --- eval: < 2 feature sets (frontend check_df_eval L55) ----------------
    def test_eval_single_feature_set(self):
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="at least two"):
            cpp_plot.eval(df_eval=make_df_eval(n_rows=1))
        plt.close("all")

    # --- feature: names_to_show not in df_seq (frontend L92) ----------------
    def test_feature_names_to_show_missing(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_seq = df_seq.copy()
        df_seq["name"] = [f"p{i}" for i in range(len(df_seq))]
        feature = aa.load_features()["feature"].iloc[0]
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="names_to_show"):
            cpp_plot.feature(feature=feature, df_seq=df_seq, labels=labels,
                             names_to_show=["does_not_exist"])
        plt.close("all")

    # --- ranking: col_dif mismatch under shap_plot (frontend L114) ----------
    def test_ranking_col_dif_shap_mismatch(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="col_dif"):
            cpp_plot.ranking(df_feat=df_feat, shap_plot=True,
                             col_dif="bad_name", col_imp=COL_FEAT_IMPACT_TEST)
        plt.close("all")

    # --- profile: shap_plot and add_legend_cat both True (frontend L154) ----
    def test_profile_shap_and_legend_cat(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="can not be both True"):
            cpp_plot.profile(df_feat=df_feat, shap_plot=True,
                             col_imp=COL_FEAT_IMPACT_TEST, add_legend_cat=True)
        plt.close("all")

    # --- feature_map: imp_ths minimum <= 0 (frontend check_imp_tuples L166) -
    def test_feature_map_imp_ths_non_positive(self):
        df_feat = get_df_feat()
        cpp_plot = aa.CPPPlot()
        with pytest.raises(ValueError, match="should be > 0"):
            cpp_plot.feature_map(df_feat=df_feat, imp_ths=(0, 0.5, 1))
        plt.close("all")

    # --- update_seq_size: ax sequence shorter than minimum (frontend L179) --
    def test_update_seq_size_ax_too_short(self):
        df_feat = get_df_feat(n=8)
        args_seq = get_args_seq()
        cpp_plot = aa.CPPPlot()
        fig, ax = cpp_plot.profile(df_feat=df_feat, **args_seq)
        cpp_big = aa.CPPPlot(jmd_n_len=40, jmd_c_len=40)
        with pytest.raises(ValueError, match="shorter than minimum"):
            cpp_big.update_seq_size(ax=ax, fig=fig)
        plt.close("all")
