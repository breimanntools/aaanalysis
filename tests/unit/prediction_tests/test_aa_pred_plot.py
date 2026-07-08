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


def _grid(n_rows=4, n_cols=5, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.uniform(50, 100, (n_rows, n_cols)),
                        index=[f"r{i}" for i in range(n_rows)],
                        columns=[f"c{j}" for j in range(n_cols)])


def _box_cells(ax):
    """Return the set of (row, col) full-cell boxes (unfilled Rectangles placed at (col, row))."""
    boxes = [p for p in ax.patches if type(p).__name__ == "Rectangle" and p.get_fill() is False]
    return {(round(p.get_xy()[1]), round(p.get_xy()[0])) for p in boxes}


def _box_cell(ax):
    """Return the single boxed (row, col), or None if none."""
    cells = _box_cells(ax)
    if not cells:
        return None
    assert len(cells) == 1
    return next(iter(cells))


class TestAAPredPlotEvalHeatmap:
    def test_returns_fig_ax(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap")
        assert r.fig is not None and r.ax is not None

    def test_boxes_best_cell_by_default(self):
        # highlight defaults to 1: the box lands on the argmax cell, flush to the cell (no inset).
        g = _grid()
        r = aa.AAPredPlot().eval(g, kind="heatmap")
        i, j = np.unravel_index(np.argmax(g.to_numpy()), g.shape)
        assert _box_cell(r.ax) == (int(i), int(j))
        box = [p for p in r.ax.patches if type(p).__name__ == "Rectangle" and p.get_fill() is False][0]
        assert box.get_width() == 1 and box.get_height() == 1  # full-cell frame

    def test_highlight_top_n(self):
        # highlight=3 boxes the three highest-value cells.
        g = _grid()
        r = aa.AAPredPlot().eval(g, kind="heatmap", highlight=3)
        flat = g.to_numpy().ravel()
        top3 = {tuple(int(v) for v in np.unravel_index(k, g.shape))
                for k in np.argsort(flat)[::-1][:3]}
        assert _box_cells(r.ax) == top3

    def test_highlight_min(self):
        g = _grid()
        r = aa.AAPredPlot().eval(g, kind="heatmap", highlight="min")
        i, j = np.unravel_index(np.argmin(g.to_numpy()), g.shape)
        assert _box_cell(r.ax) == (int(i), int(j))

    def test_highlight_explicit_cell(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", highlight=(1, 2))
        assert _box_cell(r.ax) == (1, 2)

    def test_highlight_explicit_cell_list(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", highlight=[(0, 0), (3, 4)])
        assert _box_cells(r.ax) == {(0, 0), (3, 4)}

    def test_highlight_max_ignores_nan(self):
        g = _grid()
        g.iloc[0, 0] = np.nan  # NaN must not be picked as the max
        r = aa.AAPredPlot().eval(g, kind="heatmap", highlight="max")
        i, j = np.unravel_index(np.nanargmax(g.to_numpy()), g.shape)
        assert _box_cell(r.ax) == (int(i), int(j))

    def test_highlight_none_draws_no_box(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", highlight=None)
        assert _box_cell(r.ax) is None

    def test_highlight_zero_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(_grid(), kind="heatmap", highlight=0)

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(pd.DataFrame(columns=["a", "b"]), kind="heatmap", highlight=None)

    def test_vmin_vmax_cmap_cbar_label(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", vmin=50, vmax=100, cmap="magma",
                                 cbar_label="Balanced accuracy [%]")
        assert r.ax is not None

    def test_annotate_and_fmt(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", annotate=True, annotation_fmt=".1f")
        assert r.ax is not None

    def test_annotate_false(self):
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", annotate=False)
        assert r.ax is not None

    def test_title_and_ax(self):
        fig, ax = plt.subplots()
        r = aa.AAPredPlot().eval(_grid(), kind="heatmap", ax=ax, title="t")
        assert r.ax is ax and ax.get_title() == "t"

    def test_bad_highlight_string_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(_grid(), kind="heatmap", highlight="best")

    def test_highlight_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(_grid(), kind="heatmap", highlight=(9, 9))

    def test_non_numeric_grid_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(pd.DataFrame({"a": ["x", "y"]}), kind="heatmap")


def _df_rank(n=8):
    rng = np.random.RandomState(0)
    return pd.DataFrame({"name": [f"G{i}" for i in range(n)],
                         "score": np.linspace(30, 95, n),
                         "group": (["sub", "non"] * n)[:n],
                         "std": np.linspace(1, 4, n)})


class TestAAPredPlotPredictRanking:
    def test_returns_fig_ax(self):
        r = aa.AAPredPlot().predict_group(_df_rank(), kind="ranking")
        assert r.fig is not None and r.ax is not None

    def test_col_name_score(self):
        df = _df_rank().rename(columns={"name": "gene", "score": "p"})
        r = aa.AAPredPlot().predict_group(df, kind="ranking", col_name="gene", col_score="p")
        assert r.ax is not None

    def test_col_group_and_std(self):
        r = aa.AAPredPlot().predict_group(_df_rank(), kind="ranking", col_group="group", col_std="std")
        assert len(r.ax.patches) == 8

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(_df_rank().drop(columns=["score"]), kind="ranking")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(_df_rank().iloc[0:0], kind="ranking")

    def test_top_n_and_ascending(self):
        r = aa.AAPredPlot().predict_group(_df_rank(), kind="ranking", top_n=3, ascending=True)
        assert len(r.ax.patches) == 3

    def test_cutoffs_and_colors(self):
        r = aa.AAPredPlot().predict_group(_df_rank(), kind="ranking", col_group="group",
                                    colors={"sub": "red", "non": "blue"}, cutoffs=(60,))
        assert any(line.get_linestyle() == "--" for line in r.ax.get_lines())

    def test_figsize_height_scales_with_items(self):
        h8 = aa.AAPredPlot().predict_group(_df_rank(8), kind="ranking").fig.get_size_inches()[1]
        h20 = aa.AAPredPlot().predict_group(_df_rank(20), kind="ranking").fig.get_size_inches()[1]
        assert h20 > h8

    def test_ax_xlabel_title(self):
        _, ax = plt.subplots()
        r = aa.AAPredPlot().predict_group(_df_rank(), kind="ranking", ax=ax, xlabel="score", title="t")
        assert r.ax is ax


def _imp_data(n=12, n_feat=20, seed=0):
    rng = np.random.RandomState(seed)
    return np.vstack([rng.normal(0.6, 0.2, (n // 2, n_feat)),
                      rng.normal(-0.6, 0.2, (n - n // 2, n_feat))])


class TestAAPredPlotPredictClustermap:
    def test_returns_fig_ax(self):
        r = aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap")
        assert r.fig is not None and r.ax is not None

    def test_labels_and_names(self):
        r = aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap", labels=np.array([1, 0] * 6),
                                    names=[f"P{i}" for i in range(12)])
        assert r.ax is not None

    def test_colors_and_cmap(self):
        r = aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap", labels=np.array([1, 0] * 6),
                                    colors={1: "red", 0: "blue"}, cmap="viridis")
        assert r.ax is not None

    def test_min_samples_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(np.random.RandomState(0).rand(1, 5), kind="clustermap")

    def test_labels_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap", labels=np.array([1, 0, 1]))

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap", names=["a", "b"])

    def test_figsize_cbar_label_title(self):
        r = aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap", figsize=(6, 6),
                                    cbar_label="r", title="t")
        assert r.ax is not None

    def test_constant_row_does_not_crash(self):
        # A zero-variance importance row yields NaN correlations; must be sanitized.
        data = _imp_data()
        data[0] = 0.0
        r = aa.AAPredPlot().predict_group(data, kind="clustermap", labels=np.array([1, 0] * 6))
        assert r.ax is not None

    def test_string_labels_allowed(self):
        r = aa.AAPredPlot().predict_group(_imp_data(), kind="clustermap", labels=["sub", "non"] * 6)
        assert r.ax is not None


class TestAAPredPlotPredictHist:
    def test_returns_fig_ax(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist")
        assert fig is not None and ax is not None

    def test_labels_separated(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", labels=labels)
        assert len(ax.get_legend().get_texts()) == 2

    def test_bins(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", bins=5)
        assert ax is not None

    def test_thresholds(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", thresholds=[0.4, 0.6])
        assert sum(l.get_linestyle() == "--" for l in ax.get_lines()) == 2

    def test_dict_color(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", labels=labels,
                                          dict_color={1: "#111111", 0: "#222222"})
        assert ax is not None

    def test_labels_length_mismatch_raises(self):
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(scores, kind="hist", labels=labels[:-1])

    def test_xlabel_ylabel(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", xlabel="Score", ylabel="Count")
        assert ax.get_xlabel() == "Score" and ax.get_ylabel() == "Count"

    def test_ax_figsize(self):
        scores, labels = _scores()
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", ax=ax0, figsize=(6, 4))
        assert ax is ax0

    def test_band_colors_bars_by_confidence(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", band=True,
                                                thresholds=[0.3, 0.7])
        # Three bands (2 thresholds) -> bars fall into at most three distinct face colors,
        # and at least two bands are populated for a spread score distribution.
        face_colors = {tuple(np.round(p.get_facecolor(), 4)) for p in ax.patches
                       if type(p).__name__ == "Rectangle"}
        assert 2 <= len(face_colors) <= 3

    def test_band_custom_colors(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", band=True,
                                                thresholds=[0.5], colors=["#cccccc", "#4477aa"])
        assert ax is not None

    def test_band_cmap(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="hist", band=True,
                                                thresholds=[0.3, 0.7], cmap="magma")
        assert ax is not None

    def test_band_requires_thresholds(self):
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(scores, kind="hist", band=True)

    def test_band_mutually_exclusive_with_labels(self):
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(scores, kind="hist", band=True,
                                          thresholds=[0.5], labels=labels)

    def test_band_bad_colors_length_raises(self):
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(scores, kind="hist", band=True,
                                          thresholds=[0.3, 0.7], colors=["#000000"])


class TestAAPredPlotPredictScatter:
    def test_returns_fig_ax(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2)
        assert fig is not None and ax is not None

    def test_missing_scores_y_raises(self):
        s1, labels = _scores(0)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(s1, kind="scatter")

    def test_scores_x_scores_y_length_mismatch(self):
        s1, labels = _scores(0)
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s1[:-1])

    def test_labels_color(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2, labels=labels)
        assert ax.get_legend() is not None

    def test_diagonal_off(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2, diagonal=False)
        assert ax is not None

    def test_marker_size(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2, marker_size=50)
        assert ax is not None

    def test_dict_color(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2, labels=labels,
                                          dict_color={1: "#111111", 0: "#222222"})
        assert ax is not None

    def test_xlabel_ylabel(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2, xlabel="P1", ylabel="P2")
        assert ax.get_xlabel() == "P1" and ax.get_ylabel() == "P2"

    def test_ax_figsize(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_group(s1, kind="scatter", scores_y=s2, ax=ax0, figsize=(5, 5))
        assert ax is ax0


class TestAAPredPlotPredictCutoff:
    def test_returns_fig_ax(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="cutoff")
        assert fig is not None and ax is not None

    def test_monotone_non_increasing(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="cutoff")
        y = ax.get_lines()[0].get_ydata()
        assert np.all(np.diff(y) <= 1e-9)

    def test_n_steps(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="cutoff", n_steps=11)
        assert len(ax.get_lines()[0].get_ydata()) == 11

    def test_thresholds(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="cutoff", thresholds=[0.5])
        assert any(l.get_linestyle() == "--" for l in ax.get_lines())

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(np.array([]), kind="cutoff")

    def test_xlabel_ylabel(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="cutoff", xlabel="Cutoff", ylabel="Percent")
        assert ax.get_xlabel() == "Cutoff" and ax.get_ylabel() == "Percent"

    def test_ax_figsize(self):
        scores, labels = _scores()
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().predict_group(scores, kind="cutoff", ax=ax0, figsize=(6, 4))
        assert ax is ax0


class TestAAPredPlotPredictGroupInvalidKind:
    def test_bad_kind_raises(self):
        # A sample (positional) kind or any unknown kind must be rejected by predict_group.
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(scores, kind="window")
        with pytest.raises(ValueError):
            aa.AAPredPlot().predict_group(scores, kind="not_a_kind")
