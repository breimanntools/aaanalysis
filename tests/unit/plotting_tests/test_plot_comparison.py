"""This is a script to test the plot_comparison() grouped comparison barplot (#311)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.plotting import plot_comparison

aa.options["verbose"] = False


# Helper functions
def _df(groups=("Scale-based", "CPP"), conditions=("No expansion", "Random", "dPULearn")):
    rows = []
    base = {"Scale-based": 60, "CPP": 80}
    for g in groups:
        for k, c in enumerate(conditions):
            rows.append({"group": g, "condition": c, "value": base.get(g, 70) + k})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestPlotComparison:
    """Normal cases for plot_comparison."""

    def test_returns_axes(self):
        ax = plot_comparison(df_eval=_df())
        assert isinstance(ax, plt.Axes)

    def test_bar_count_groups_times_conditions(self):
        # 2 groups x 3 conditions = 6 bars
        ax = plot_comparison(df_eval=_df())
        assert len(ax.patches) == 6

    def test_bar_count_three_groups(self):
        ax = plot_comparison(df_eval=_df(groups=("A", "B", "C")))
        assert len(ax.patches) == 9

    def test_baseline_drawn_when_set(self):
        ax = plot_comparison(df_eval=_df(), baseline=50)
        assert len(ax.lines) == 1

    def test_baseline_none_no_line(self):
        ax = plot_comparison(df_eval=_df(), baseline=None)
        assert len(ax.lines) == 0

    def test_baseline_label_default_text(self):
        ax = plot_comparison(df_eval=_df(), baseline=50)
        assert any("chance" in t.get_text() for t in ax.texts)

    def test_baseline_label_custom(self):
        ax = plot_comparison(df_eval=_df(), baseline=50, baseline_label="random (50%)")
        assert any("random (50%)" == t.get_text() for t in ax.texts)

    def test_baseline_label_empty_suppresses_text(self):
        ax = plot_comparison(df_eval=_df(), baseline=50, baseline_label="")
        # 6 value labels, no baseline text
        assert len(ax.texts) == 6

    def test_annotate_true_writes_value_labels(self):
        ax = plot_comparison(df_eval=_df(), annotate=True, baseline=None)
        assert len(ax.texts) == 6

    def test_annotate_false_no_value_labels(self):
        ax = plot_comparison(df_eval=_df(), annotate=False, baseline=None)
        assert len(ax.texts) == 0

    def test_legend_labels_are_groups(self):
        ax = plot_comparison(df_eval=_df())
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert set(labels) == {"Scale-based", "CPP"}

    def test_group_order_controls_bar_order(self):
        ax = plot_comparison(df_eval=_df(), group_order=["CPP", "Scale-based"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CPP", "Scale-based"]

    def test_condition_order_controls_ticks(self):
        order = ["dPULearn", "Random", "No expansion"]
        ax = plot_comparison(df_eval=_df(), condition_order=order)
        assert [t.get_text() for t in ax.get_xticklabels()] == order

    def test_colors_list(self):
        ax = plot_comparison(df_eval=_df(), colors=["tab:gray", "tab:red"])
        assert len(ax.patches) == 6

    def test_colors_dict(self):
        ax = plot_comparison(df_eval=_df(), colors={"Scale-based": "tab:gray", "CPP": "tab:red"})
        assert len(ax.patches) == 6

    def test_custom_columns(self):
        df = _df().rename(columns={"group": "method", "condition": "setting", "value": "acc"})
        ax = plot_comparison(df_eval=df, group="method", condition="setting", value="acc")
        assert len(ax.patches) == 6

    def test_aggregates_repeated_cells(self):
        df = pd.concat([_df(), _df()], ignore_index=True)  # duplicate (group, condition) rows
        ax = plot_comparison(df_eval=df)
        assert len(ax.patches) == 6

    def test_passing_existing_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = plot_comparison(df_eval=_df(), ax=ax_in)
        assert ax_out is ax_in

    def test_labels_and_title(self):
        ax = plot_comparison(df_eval=_df(), xlabel="X", ylabel="Balanced accuracy [%]",
                             title="Bench")
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Balanced accuracy [%]"
        assert ax.get_title() == "Bench"

    def test_ylim_respected(self):
        ax = plot_comparison(df_eval=_df(), ylim=(0, 108))
        assert ax.get_ylim() == (0, 108)

    def test_bar_width_and_figsize_and_fontsize(self):
        ax = plot_comparison(df_eval=_df(), bar_width=0.6, figsize=(8, 5),
                             fontsize_annotations=8)
        assert len(ax.patches) == 6


class TestPlotComparisonErrors:
    """Negative cases: bad input must raise ValueError."""

    def test_missing_column_raises(self):
        df = _df().drop(columns=["value"])
        with pytest.raises(ValueError):
            plot_comparison(df_eval=df)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df().iloc[0:0])

    def test_non_df_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=[1, 2, 3])

    def test_bad_group_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), group=123)

    def test_bad_baseline_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), baseline="50")

    def test_bad_annotate_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), annotate="yes")

    def test_bar_width_out_of_range_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), bar_width=1.5)

    def test_bar_width_zero_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), bar_width=0)

    def test_incomplete_group_order_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), group_order=["CPP"])

    def test_colors_list_too_short_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), colors=["tab:gray"])

    def test_colors_dict_missing_group_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), colors={"CPP": "tab:red"})

    def test_bad_ylim_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), ylim=(10,))
