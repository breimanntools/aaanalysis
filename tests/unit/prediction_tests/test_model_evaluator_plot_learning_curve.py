"""This is a script to test ModelEvaluatorPlot.learning_curve()."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

METRICS = ["accuracy", "mcc", "f1"]
COLORS = ["tab:red", "tab:green", "black", "magenta"]


# I Helper Functions
def _data(n_per_class=20, n_feat=4, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal(0.8, 1.0, size=(n_per_class, n_feat)),
                   rng.normal(-0.8, 1.0, size=(n_per_class, n_feat))])
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return X, labels


def _df_manual(n_models=2, ci=True):
    """Hand-made learning-curve table (two sizes per model) for exact, fast assertions."""
    rows = []
    for i in range(n_models):
        base = 0.5 - 0.2 * i
        rows.append([f"m{i + 1}", 4, "mcc", base, 0.1, base - 0.1 if ci else np.nan,
                     base + 0.1 if ci else np.nan, 5])
        rows.append([f"m{i + 1}", 8, "mcc", base + 0.2, 0.2, base + 0.05 if ci else np.nan,
                     base + 0.35 if ci else np.nan, 5])
    return pd.DataFrame(rows, columns=ut.COLS_CURVE_MODELEVAL)


@pytest.fixture(scope="module")
def df_curve():
    X, labels = _data()
    me = aa.ModelEvaluator(models=["log_reg", "svm"], random_state=0, verbose=False)
    return me.learning_curve(X, labels, train_sizes=[0.25, 0.5, 1.0], metrics=METRICS)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# II Main Functions
class TestPlotLearningCurve:
    """Normal cases: one parameter per test (positive and negative)."""

    # Positive tests
    def test_df_curve(self, df_curve):
        out = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        assert isinstance(out, ut.FigAxResult)
        fig, ax = out
        assert ax.get_figure() is fig
        assert len(ax.get_lines()) == 2

    @settings(max_examples=3, deadline=None)
    @given(n_models=some.integers(min_value=1, max_value=4))
    def test_df_curve_hand_made(self, n_models):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=_df_manual(n_models=n_models))
        assert len(ax.get_lines()) == n_models
        plt.close("all")

    def test_df_curve_forwards_attributes_to_ax(self, df_curve):
        out = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        assert out.get_ylabel() == "mcc"

    @settings(max_examples=3, deadline=None)
    @given(metric=some.sampled_from(METRICS))
    def test_metric(self, df_curve, metric):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric=metric)
        assert ax.get_ylabel() == metric
        sub = df_curve[(df_curve[ut.COL_METRIC] == metric) & (df_curve[ut.COL_MODEL] == "log_reg")]
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), sub[ut.COL_SCORE].to_numpy())
        plt.close("all")

    def test_metric_default(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        assert ax.get_ylabel() == "mcc"
        df_no_mcc = df_curve[df_curve[ut.COL_METRIC] != "mcc"]
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_no_mcc)
        assert ax.get_ylabel() == "accuracy"

    @settings(max_examples=3, deadline=None)
    @given(width=some.floats(min_value=2, max_value=10), height=some.floats(min_value=2, max_value=8))
    def test_figsize(self, df_curve, width, height):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, figsize=(width, height))
        np.testing.assert_allclose(fig.get_size_inches(), [width, height])
        plt.close("all")

    @settings(max_examples=4, deadline=None)
    @given(colors=some.lists(some.sampled_from(COLORS), min_size=2, max_size=2, unique=True))
    def test_colors(self, df_curve, colors):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=colors)
        drawn = [matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()]
        assert drawn == [matplotlib.colors.to_hex(c) for c in colors]
        plt.close("all")

    def test_colors_default(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        drawn = [matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()]
        expected = [matplotlib.colors.to_hex(c) for c in ut.plot_get_clist_(n_colors=2)]
        assert drawn == expected

    def test_colors_single_string_for_single_model(self):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=_df_manual(n_models=1), colors="tab:red")
        assert matplotlib.colors.to_hex(ax.get_lines()[0].get_color()) == matplotlib.colors.to_hex("tab:red")

    @settings(max_examples=2, deadline=None)
    @given(show_ci=some.booleans())
    def test_show_ci(self, df_curve, show_ci):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, show_ci=show_ci)
        assert len(ax.collections) == (2 if show_ci else 0)
        plt.close("all")

    def test_x_is_train_size(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve)
        np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), [8, 16, 32])
        assert "Training size" in ax.get_xlabel()

    # Negative tests
    def test_invalid_df_curve_none(self):
        with pytest.raises(ValueError, match="should not be None"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=None)

    def test_invalid_df_curve_type(self):
        for df in ["df", 5, [1, 2, 3]]:
            with pytest.raises(ValueError, match="should be DataFrame"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df)

    def test_invalid_df_curve_missing_columns(self, df_curve):
        for col in [ut.COL_TRAIN_SIZE, ut.COL_CI_LOW, ut.COL_SCORE_STD, ut.COL_MODEL]:
            with pytest.raises(ValueError, match="missing required columns"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve.drop(columns=col))

    def test_invalid_df_curve_empty(self, df_curve):
        with pytest.raises(ValueError, match="at least one row"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve.iloc[0:0])

    def test_invalid_df_curve_values(self, df_curve):
        df_bad_model = df_curve.copy()
        df_bad_model.loc[df_bad_model.index[0], ut.COL_MODEL] = ""
        with pytest.raises(ValueError, match="non-empty string"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_model)
        df_bad_numeric = df_curve.copy()
        df_bad_numeric[ut.COL_SCORE] = df_bad_numeric[ut.COL_SCORE].astype(str)
        with pytest.raises(ValueError, match="numeric curve values"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_numeric)
        df_bad_train_size = df_curve.copy()
        df_bad_train_size[ut.COL_TRAIN_SIZE] = df_bad_train_size[ut.COL_TRAIN_SIZE].astype(float)
        df_bad_train_size.loc[df_bad_train_size.index[0], ut.COL_TRAIN_SIZE] = 1.5
        with pytest.raises(ValueError, match="integer training sizes"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_train_size)
        df_bad_score = df_curve.copy()
        df_bad_score.loc[df_bad_score.index[0], ut.COL_SCORE] = np.nan
        with pytest.raises(ValueError, match="finite training-size"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_score)
        df_bad_std = df_curve.copy()
        df_bad_std.loc[df_bad_std.index[0], ut.COL_SCORE_STD] = -0.1
        with pytest.raises(ValueError, match="non-negative score standard deviations"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_std)
        df_bad_ci = df_curve.copy()
        df_bad_ci.loc[df_bad_ci.index[0], ut.COL_CI_LOW] = np.nan
        with pytest.raises(ValueError, match="two finite CI bounds or two NaN"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_ci)
        df_bad_count = df_curve.copy()
        df_bad_count.loc[df_bad_count.index[0], ut.COL_N_SCORES] = 0
        with pytest.raises(ValueError, match="positive integer score counts"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_bad_count)

    def test_invalid_df_curve_structure(self, df_curve):
        df_duplicate = pd.concat([df_curve, df_curve.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="one row per model"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_duplicate)
        first_size = df_curve[ut.COL_TRAIN_SIZE].iloc[0]
        df_one_size = df_curve[df_curve[ut.COL_TRAIN_SIZE] == first_size]
        with pytest.raises(ValueError, match="at least two training sizes"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_one_size)

    def test_invalid_metric_unknown(self, df_curve):
        for metric in ["roc_auc", "not_a_metric", "recall"]:
            with pytest.raises(ValueError, match="should be one of"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric=metric)

    def test_invalid_metric_type(self, df_curve):
        for metric in [1, 1.5, ["mcc"]]:
            with pytest.raises(ValueError, match="should be string"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric=metric)

    def test_invalid_figsize_values(self, df_curve):
        for figsize in [(0, 0), (-1, 4), (4, 0)]:
            with pytest.raises(ValueError, match="figsize"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, figsize=figsize)

    def test_invalid_figsize_type(self, df_curve):
        for figsize in ["big", (4,), (1, 2, 3)]:
            with pytest.raises(ValueError, match="should be a tuple"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, figsize=figsize)

    def test_invalid_figsize_none(self, df_curve):
        with pytest.raises(ValueError, match="should not be None"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, figsize=None)

    def test_invalid_colors_too_few(self, df_curve):
        for colors in [["tab:red"], "tab:red", []]:
            with pytest.raises(ValueError, match="at least one color per model"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=colors)

    def test_invalid_colors_values(self, df_curve):
        for colors in [["not_a_color", "tab:red"], [1, 2], ["tab:red", None]]:
            with pytest.raises(ValueError, match="valid color|should not contain 'None'"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=colors)

    def test_invalid_colors_type(self, df_curve):
        for colors in [("tab:red", "tab:blue"), np.array(["tab:red", "tab:blue"]),
                       pd.Series(["tab:red", "tab:blue"])]:
            with pytest.raises(ValueError, match="string or a list"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=colors)

    def test_invalid_show_ci_none(self, df_curve):
        with pytest.raises(ValueError, match="should not be None"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, show_ci=None)

    def test_invalid_show_ci_type(self, df_curve):
        for show_ci in [1, 0, "yes", 1.0]:
            with pytest.raises(ValueError, match="should be bool"):
                aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, show_ci=show_ci)


class TestPlotLearningCurveComplex:
    """Combinations and edge interactions."""

    # Positive tests
    def test_single_model_without_legend(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models="log_reg", random_state=0, verbose=False)
        df = me.learning_curve(X, labels, train_sizes=[0.5, 1.0], metrics=["mcc"])
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df)
        assert ax.get_legend() is None
        assert len(ax.get_lines()) == 1

    def test_multi_model_legend(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric="f1")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["log_reg", "svm"]

    def test_ci_none_falls_back_to_std_band(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["log_reg", "svm"], random_state=0, verbose=False)
        df = me.learning_curve(X, labels, train_sizes=[0.25, 1.0], metrics=["mcc"], ci=None)
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df, show_ci=True)
        for coll in ax.collections:
            assert np.isfinite(coll.get_paths()[0].vertices).all()

    def test_all_params_combined(self, df_curve):
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric="accuracy", figsize=(5, 3),
                                                       colors=["black", "gray", "red"], show_ci=False)
        assert len(ax.collections) == 0 and ax.get_ylabel() == "accuracy"

    def test_unsorted_rows_are_plotted_by_size(self, df_curve):
        df_shuffled = df_curve.sample(frac=1.0, random_state=0)
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df_shuffled, metric="mcc")
        for line in ax.get_lines():
            assert np.all(np.diff(line.get_xdata()) > 0)

    def test_absolute_sizes_curve(self):
        X, labels = _data()
        me = aa.ModelEvaluator(models=["log_reg", "svm"], random_state=0, verbose=False)
        df = me.learning_curve(X, labels, train_sizes=[4, 12, 32], metrics=["mcc"])
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df, show_ci=True)
        np.testing.assert_array_equal(ax.get_lines()[1].get_xdata(), [4, 12, 32])
        assert len(ax.collections) == 2

    # Negative tests
    def test_invalid_metric_after_filtering(self, df_curve):
        with pytest.raises(ValueError, match="should be one of"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve[df_curve[ut.COL_METRIC] == "mcc"],
                                                 metric="f1")

    def test_invalid_colors_with_show_ci(self, df_curve):
        with pytest.raises(ValueError, match="at least one color per model"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=["red"], show_ci=True)

    def test_invalid_df_from_run(self):
        X, labels = _data()
        df_eval = aa.ModelEvaluator(models="log_reg", random_state=0, verbose=False).run(X, labels)
        with pytest.raises(ValueError, match="missing required columns"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_eval)

    def test_invalid_figsize_with_metric(self, df_curve):
        with pytest.raises(ValueError, match="should not be None"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, metric="mcc", figsize=None)

    def test_invalid_show_ci_with_colors(self, df_curve):
        with pytest.raises(ValueError, match="should be bool"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_curve, colors=["red", "blue"], show_ci="false")

    def test_invalid_empty_metric_selection(self, df_curve):
        df_empty_metric = df_curve[df_curve[ut.COL_METRIC] == "not_a_metric"]
        with pytest.raises(ValueError, match="at least one row"):
            aa.ModelEvaluatorPlot.learning_curve(df_curve=df_empty_metric)


class TestPlotLearningCurveGoldenValues:
    """Hand-computed geometry: one line per model, band iff show_ci, exact band and line values."""

    def test_n_lines_equals_n_models(self):
        for n_models in [1, 2, 3, 4]:
            fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=_df_manual(n_models=n_models))
            assert len(ax.get_lines()) == n_models
            assert len(ax.collections) == n_models
            plt.close("all")

    def test_ci_band_present_iff_show_ci(self):
        df = _df_manual(n_models=3)
        for show_ci, n_bands in [(True, 3), (False, 0)]:
            fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df, show_ci=show_ci)
            assert len(ax.collections) == n_bands
            plt.close("all")

    def test_band_equals_ci_columns(self):
        df = _df_manual(n_models=2)  # m1: ci 0.4-0.6 at size 4, 0.55-0.85 at size 8
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df, show_ci=True)
        verts = ax.collections[0].get_paths()[0].vertices
        np.testing.assert_allclose(verts[:, 1].min(), 0.4)
        np.testing.assert_allclose(verts[:, 1].max(), 0.85)
        assert sorted(set(verts[:, 0])) == [4.0, 8.0]

    def test_band_falls_back_to_std(self):
        df = _df_manual(n_models=1, ci=False)  # m1: 0.5 +/- 0.1 at size 4, 0.7 +/- 0.2 at size 8
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df, show_ci=True)
        verts = ax.collections[0].get_paths()[0].vertices
        np.testing.assert_allclose(verts[:, 1].min(), 0.4)
        np.testing.assert_allclose(verts[:, 1].max(), 0.9)

    def test_line_values_are_the_score_column(self):
        df = _df_manual(n_models=2)
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df)
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [0.5, 0.7])
        np.testing.assert_allclose(ax.get_lines()[1].get_ydata(), [0.3, 0.5])
        np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), [4, 8])

    def test_default_colors_follow_first_appearance_order(self):
        df = _df_manual(n_models=3)
        fig, ax = aa.ModelEvaluatorPlot.learning_curve(df_curve=df)
        drawn = [matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()]
        expected = [matplotlib.colors.to_hex(c) for c in ut.plot_get_clist_(n_colors=3)]
        assert drawn == expected
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["m1", "m2", "m3"]
