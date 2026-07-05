"""Unit tests for the AAPredPlot class (visualize AAPred results via predict/eval kinds)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa


def _data(n_per_class=15, n_feat=6, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0.5, 1, (n_per_class, n_feat)),
                   rng.normal(-0.5, 1, (n_per_class, n_feat))])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


def _df_eval(holdout=False):
    X, labels = _data()
    kwargs = {}
    if holdout:
        Xh, lh = _data(n_per_class=8, seed=1)
        kwargs = dict(X_holdout=Xh, labels_holdout=lh)
    return aa.AAPred(random_state=0).eval(X, labels, metrics=["accuracy", "balanced_accuracy"], **kwargs)


def _df_eval_grid(models, metrics):
    """Synthetic df_eval with a chosen number of models x metrics (controls 1D vs 2D)."""
    import aaanalysis.utils as ut
    rows = [{ut.COL_MODEL: m, ut.COL_METRIC: me, ut.COL_PRINCIPLE: "cv",
             ut.COL_SCORE: 0.6 + 0.05 * (i + j), ut.COL_SCORE_STD: 0.02}
            for i, m in enumerate(models) for j, me in enumerate(metrics)]
    return pd.DataFrame(rows)


def _is_bar(ax):
    return any(type(p).__name__ == "Rectangle" for p in ax.patches)


def _scores(seed=0):
    X, labels = _data(seed=seed)
    aapred = aa.AAPred(random_state=seed).fit(X, labels)
    pred, _ = aapred._predict_X(X)
    return pred, labels


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class TestAAPredPlotEval:
    def test_returns_fig_ax(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval(), kind="eval")
        assert fig is not None and ax is not None

    def test_df_eval_required_cols(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(pd.DataFrame({"model": ["a"]}), kind="eval")

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(_df_eval(), kind="not_a_kind")

    def test_baseline(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval(), baseline=0.5)
        assert any(l.get_linestyle() == "--" for l in ax.get_lines())

    def test_holdout_hatched(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval(holdout=True))
        assert ax is not None

    def test_ax_passthrough(self):
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().eval(_df_eval(), ax=ax0)
        assert ax is ax0

    def test_figsize(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval(), figsize=(8, 4))
        assert ax is not None

    def test_dict_color(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval(), dict_color={"RandomForestClassifier": "#123456"})
        assert ax is not None

    def test_ylabel(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval(), ylabel="Balanced accuracy")
        assert ax.get_ylabel() == "Balanced accuracy"

    def test_multi_model_is_bars_with_hue(self):
        # Comparing methods (multiple models) -> grouped bars, one color (hue) per model.
        fig, ax = aa.AAPredPlot().eval(_df_eval_grid(["svm", "rf", "log"], ["acc", "auc"]))
        assert _is_bar(ax)
        colors = {tuple(round(c, 4) for c in p.get_facecolor())
                  for p in ax.patches if type(p).__name__ == "Rectangle"}
        assert len(colors) >= 3  # a distinct hue per model

    def test_single_model_is_bars(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval_grid(["svm"], ["acc", "auc", "f1"]))
        assert _is_bar(ax)


def _df_comp():
    return pd.DataFrame({"group": ["A", "A", "B", "B"],
                         "condition": ["c1", "c2", "c1", "c2"],
                         "value": [61.0, 60.0, 71.0, 74.0]})


class TestAAPredPlotEvalComparison:
    def test_returns_fig_ax(self):
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison")
        assert r.fig is not None and r.ax is not None

    def test_group_condition_value_cols(self):
        df = _df_comp().rename(columns={"group": "method", "condition": "cond", "value": "acc"})
        r = aa.AAPredPlot().eval(df, kind="comparison", group="method", condition="cond", value="acc")
        assert r.ax is not None

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(_df_comp().drop(columns=["value"]), kind="comparison")

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(_df_comp().iloc[0:0], kind="comparison")

    def test_baseline_and_baseline_label(self):
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison", baseline=50, baseline_label="chance")
        assert any(line.get_linestyle() == "--" for line in r.ax.get_lines())

    def test_annotate_and_fmt(self):
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison", annotate=True, annotation_fmt="{:.0f}")
        assert any(t.get_text() for t in r.ax.texts)

    def test_orders_and_colors(self):
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison", group_order=["B", "A"],
                                 condition_order=["c2", "c1"], colors=["red", "blue"])
        assert r.ax is not None

    def test_bar_width_and_rotation(self):
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison", bar_width=0.6, xtick_rotation=30)
        assert r.ax is not None

    def test_labels_title_ylim_fontsize(self):
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison", xlabel="x", ylabel="y", title="t",
                                 ylim=(0, 100), fontsize_annotations=8)
        assert r.ax.get_title() == "t"

    def test_ax_figsize_passthrough(self):
        _, ax = plt.subplots()
        r = aa.AAPredPlot().eval(_df_comp(), kind="comparison", ax=ax, figsize=(6, 4))
        assert r.ax is ax


def _df_rank(n=8):
    rng = np.random.RandomState(0)
    return pd.DataFrame({"name": [f"G{i}" for i in range(n)],
                         "score": np.linspace(30, 95, n),
                         "group": (["sub", "non"] * n)[:n],
                         "std": np.linspace(1, 4, n)})


class TestAAPredPlotPredictRanking:
    def test_returns_fig_ax(self):
        r = aa.AAPredPlot().predict_cohort(_df_rank(), kind="ranking")
        assert r.fig is not None and r.ax is not None

    def test_col_name_score(self):
        df = _df_rank().rename(columns={"name": "gene", "score": "p"})
        r = aa.AAPredPlot().predict_cohort(df, kind="ranking", col_name="gene", col_score="p")
        assert r.ax is not None

    def test_col_group_and_std(self):
        r = aa.AAPredPlot().predict_cohort(_df_rank(), kind="ranking", col_group="group", col_std="std")
        assert len(r.ax.patches) == 8

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(_df_rank().drop(columns=["score"]), kind="ranking")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(_df_rank().iloc[0:0], kind="ranking")

    def test_top_n_and_ascending(self):
        r = aa.AAPredPlot().predict_cohort(_df_rank(), kind="ranking", top_n=3, ascending=True)
        assert len(r.ax.patches) == 3

    def test_cutoffs_and_colors(self):
        r = aa.AAPredPlot().predict_cohort(_df_rank(), kind="ranking", col_group="group",
                                    colors={"sub": "red", "non": "blue"}, cutoffs=(60,))
        assert any(line.get_linestyle() == "--" for line in r.ax.get_lines())

    def test_figsize_height_scales_with_items(self):
        h8 = aa.AAPredPlot().predict_cohort(_df_rank(8), kind="ranking").fig.get_size_inches()[1]
        h20 = aa.AAPredPlot().predict_cohort(_df_rank(20), kind="ranking").fig.get_size_inches()[1]
        assert h20 > h8

    def test_ax_xlabel_title(self):
        _, ax = plt.subplots()
        r = aa.AAPredPlot().predict_cohort(_df_rank(), kind="ranking", ax=ax, xlabel="score", title="t")
        assert r.ax is ax


def _imp_data(n=12, n_feat=20, seed=0):
    rng = np.random.RandomState(seed)
    return np.vstack([rng.normal(0.6, 0.2, (n // 2, n_feat)),
                      rng.normal(-0.6, 0.2, (n - n // 2, n_feat))])


class TestAAPredPlotPredictClustermap:
    def test_returns_fig_ax(self):
        r = aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap")
        assert r.fig is not None and r.ax is not None

    def test_labels_and_names(self):
        r = aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap", labels=np.array([1, 0] * 6),
                                    names=[f"P{i}" for i in range(12)])
        assert r.ax is not None

    def test_colors_and_cmap(self):
        r = aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap", labels=np.array([1, 0] * 6),
                                    colors={1: "red", 0: "blue"}, cmap="viridis")
        assert r.ax is not None

    def test_min_samples_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(np.random.RandomState(0).rand(1, 5), kind="clustermap")

    def test_labels_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap", labels=np.array([1, 0, 1]))

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap", names=["a", "b"])

    def test_figsize_cbar_label_title(self):
        r = aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap", figsize=(6, 6),
                                    cbar_label="r", title="t")
        assert r.ax is not None

    def test_constant_row_does_not_crash(self):
        # A zero-variance importance row yields NaN correlations; must be sanitized.
        data = _imp_data()
        data[0] = 0.0
        r = aa.AAPredPlot().predict_cohort(data, kind="clustermap", labels=np.array([1, 0] * 6))
        assert r.ax is not None

    def test_string_labels_allowed(self):
        r = aa.AAPredPlot().predict_cohort(_imp_data(), kind="clustermap", labels=["sub", "non"] * 6)
        assert r.ax is not None


class TestAAPredPlotPredictHist:
    def test_returns_fig_ax(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist")
        assert fig is not None and ax is not None

    def test_labels_separated(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist", labels=labels)
        assert len(ax.get_legend().get_texts()) == 2

    def test_bins(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist", bins=5)
        assert ax is not None

    def test_thresholds(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist", thresholds=[0.4, 0.6])
        assert sum(l.get_linestyle() == "--" for l in ax.get_lines()) == 2

    def test_dict_color(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist", labels=labels,
                                          dict_color={1: "#111111", 0: "#222222"})
        assert ax is not None

    def test_labels_length_mismatch_raises(self):
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(scores, kind="hist", labels=labels[:-1])

    def test_xlabel_ylabel(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist", xlabel="Score", ylabel="Count")
        assert ax.get_xlabel() == "Score" and ax.get_ylabel() == "Count"

    def test_ax_figsize(self):
        scores, labels = _scores()
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="hist", ax=ax0, figsize=(6, 4))
        assert ax is ax0


class TestAAPredPlotPredictScatter:
    def test_returns_fig_ax(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2)
        assert fig is not None and ax is not None

    def test_missing_scores_y_raises(self):
        s1, labels = _scores(0)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(s1, kind="scatter")

    def test_scores_x_scores_y_length_mismatch(self):
        s1, labels = _scores(0)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s1[:-1])

    def test_labels_color(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2, labels=labels)
        assert ax.get_legend() is not None

    def test_diagonal_off(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2, diagonal=False)
        assert ax is not None

    def test_marker_size(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2, marker_size=50)
        assert ax is not None

    def test_dict_color(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2, labels=labels,
                                          dict_color={1: "#111111", 0: "#222222"})
        assert ax is not None

    def test_xlabel_ylabel(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2, xlabel="P1", ylabel="P2")
        assert ax.get_xlabel() == "P1" and ax.get_ylabel() == "P2"

    def test_ax_figsize(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_cohort(s1, kind="scatter", scores_y=s2, ax=ax0, figsize=(5, 5))
        assert ax is ax0


class TestAAPredPlotPredictCutoff:
    def test_returns_fig_ax(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="cutoff")
        assert fig is not None and ax is not None

    def test_monotone_non_increasing(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="cutoff")
        y = ax.get_lines()[0].get_ydata()
        assert np.all(np.diff(y) <= 1e-9)

    def test_n_steps(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="cutoff", n_steps=11)
        assert len(ax.get_lines()[0].get_ydata()) == 11

    def test_thresholds(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="cutoff", thresholds=[0.5])
        assert any(l.get_linestyle() == "--" for l in ax.get_lines())

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(np.array([]), kind="cutoff")

    def test_xlabel_ylabel(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="cutoff", xlabel="Cutoff", ylabel="Percent")
        assert ax.get_xlabel() == "Cutoff" and ax.get_ylabel() == "Percent"

    def test_ax_figsize(self):
        scores, labels = _scores()
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_cohort(scores, kind="cutoff", ax=ax0, figsize=(6, 4))
        assert ax is ax0


class TestAAPredPlotPredictCohortInvalidKind:
    def test_bad_kind_raises(self):
        # A sample (positional) kind or any unknown kind must be rejected by predict_cohort.
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(scores, kind="window")
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_cohort(scores, kind="not_a_kind")
