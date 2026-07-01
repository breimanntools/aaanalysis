"""Unit tests for the AAPredPlot class (visualize AAPred results)."""
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


def _scores(seed=0):
    X, labels = _data(seed=seed)
    aapred = aa.AAPred(random_state=seed).fit(X, labels)
    pred, _ = aapred.predict_proba(X)
    return pred, labels


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class TestAAPredPlotEval:
    def test_returns_fig_ax(self):
        fig, ax = aa.AAPredPlot().eval(_df_eval())
        assert fig is not None and ax is not None

    def test_df_eval_required_cols(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().eval(pd.DataFrame({"model": ["a"]}))

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


class TestAAPredPlotHist:
    def test_returns_fig_ax(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().hist(scores)
        assert fig is not None and ax is not None

    def test_labels_separated(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().hist(scores, labels=labels)
        assert len(ax.get_legend().get_texts()) == 2

    def test_bins(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().hist(scores, bins=5)
        assert ax is not None

    def test_thresholds(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().hist(scores, thresholds=[0.4, 0.6])
        assert sum(l.get_linestyle() == "--" for l in ax.get_lines()) == 2

    def test_dict_color(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().hist(scores, labels=labels, dict_color={1: "#111111", 0: "#222222"})
        assert ax is not None

    def test_labels_length_mismatch_raises(self):
        scores, labels = _scores()
        with pytest.raises(ValueError):
            aa.AAPredPlot().hist(scores, labels=labels[:-1])

    def test_xlabel_ylabel(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().hist(scores, xlabel="Score", ylabel="Count")
        assert ax.get_xlabel() == "Score" and ax.get_ylabel() == "Count"

    def test_ax_figsize(self):
        scores, labels = _scores()
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().hist(scores, ax=ax0, figsize=(6, 4))
        assert ax is ax0


class TestAAPredPlotScatter:
    def test_returns_fig_ax(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().scatter(s1, s2)
        assert fig is not None and ax is not None

    def test_scores_x_scores_y_length_mismatch(self):
        s1, labels = _scores(0)
        with pytest.raises(ValueError):
            aa.AAPredPlot().scatter(scores_x=s1, scores_y=s1[:-1])

    def test_labels_color(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().scatter(s1, s2, labels=labels)
        assert ax.get_legend() is not None

    def test_diagonal_off(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().scatter(s1, s2, diagonal=False)
        assert ax is not None

    def test_marker_size(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().scatter(s1, s2, marker_size=50)
        assert ax is not None

    def test_dict_color(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().scatter(s1, s2, labels=labels, dict_color={1: "#111111", 0: "#222222"})
        assert ax is not None

    def test_xlabel_ylabel(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax = aa.AAPredPlot().scatter(s1, s2, xlabel="P1", ylabel="P2")
        assert ax.get_xlabel() == "P1" and ax.get_ylabel() == "P2"

    def test_ax_figsize(self):
        s1, labels = _scores(0)
        s2, _ = _scores(1)
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().scatter(s1, s2, ax=ax0, figsize=(5, 5))
        assert ax is ax0


class TestAAPredPlotCutoff:
    def test_returns_fig_ax(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().cutoff(scores)
        assert fig is not None and ax is not None

    def test_monotone_non_increasing(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().cutoff(scores)
        y = ax.get_lines()[0].get_ydata()
        assert np.all(np.diff(y) <= 1e-9)

    def test_n_steps(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().cutoff(scores, n_steps=11)
        assert len(ax.get_lines()[0].get_ydata()) == 11

    def test_color(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().cutoff(scores, color="#123456")
        assert ax is not None

    def test_thresholds(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().cutoff(scores, thresholds=[0.5])
        assert any(l.get_linestyle() == "--" for l in ax.get_lines())

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            aa.AAPredPlot().cutoff(np.array([]))

    def test_xlabel_ylabel(self):
        scores, labels = _scores()
        fig, ax = aa.AAPredPlot().cutoff(scores, xlabel="Cutoff", ylabel="Percent")
        assert ax.get_xlabel() == "Cutoff" and ax.get_ylabel() == "Percent"

    def test_ax_figsize(self):
        scores, labels = _scores()
        fig, ax0 = plt.subplots()
        fig, ax = aa.AAPredPlot().cutoff(scores, ax=ax0, figsize=(6, 4))
        assert ax is ax0
