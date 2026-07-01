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

    def test_returns_fig_ax(self):
        fig, ax = plot_comparison(df_eval=_df())
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    def test_bar_count_groups_times_conditions(self):
        # 2 groups x 3 conditions = 6 bars
        fig, ax = plot_comparison(df_eval=_df())
        assert len(ax.patches) == 6

    def test_bar_count_three_groups(self):
        fig, ax = plot_comparison(df_eval=_df(groups=("A", "B", "C")))
        assert len(ax.patches) == 9

    def test_baseline_drawn_when_set(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=50)
        assert len(ax.lines) == 1

    def test_baseline_none_no_line(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=None)
        assert len(ax.lines) == 0

    def test_baseline_label_default_text(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=50)
        assert any("chance" in t.get_text() for t in ax.get_legend().get_texts())

    def test_baseline_label_custom(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=50, baseline_label="random (50%)")
        assert any("random (50%)" == t.get_text() for t in ax.get_legend().get_texts())

    def test_baseline_label_empty_suppresses_legend_entry(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=50, baseline_label="")
        # Only the groups appear in the legend, no baseline entry.
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Scale-based", "CPP"]

    def test_annotate_true_writes_value_labels(self):
        fig, ax = plot_comparison(df_eval=_df(), annotate=True, baseline=None)
        assert len(ax.texts) == 6

    def test_annotate_false_no_value_labels(self):
        fig, ax = plot_comparison(df_eval=_df(), annotate=False, baseline=None)
        assert len(ax.texts) == 0

    def test_integer_values_labelled_without_decimals(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=None)
        assert all("." not in t.get_text() for t in ax.texts)

    def test_fractional_values_keep_precision(self):
        # AUC-like values in [0, 1] must NOT collapse to "0"/"1" integer labels.
        df = pd.DataFrame([{"group": g, "condition": c, "value": v}
                           for g, c, v in [("A", "x", 0.62), ("A", "y", 0.71),
                                           ("B", "x", 0.83), ("B", "y", 0.90)]])
        fig, ax = plot_comparison(df_eval=df, baseline=None)
        labels = sorted(t.get_text() for t in ax.texts)
        assert labels == ["0.62", "0.71", "0.83", "0.90"]

    def test_annotation_fmt_override(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=None, annotation_fmt="{:.1f}")
        assert all("." in t.get_text() for t in ax.texts)
        assert any(t.get_text() == "60.0" for t in ax.texts)

    def test_legend_labels_are_groups(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=None)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert set(labels) == {"Scale-based", "CPP"}

    def test_group_order_controls_bar_order(self):
        fig, ax = plot_comparison(df_eval=_df(), baseline=None, group_order=["CPP", "Scale-based"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["CPP", "Scale-based"]

    def test_condition_order_controls_ticks(self):
        order = ["dPULearn", "Random", "No expansion"]
        fig, ax = plot_comparison(df_eval=_df(), condition_order=order)
        assert [t.get_text() for t in ax.get_xticklabels()] == order

    def test_colors_list(self):
        fig, ax = plot_comparison(df_eval=_df(), colors=["tab:gray", "tab:red"])
        assert len(ax.patches) == 6

    def test_colors_dict(self):
        fig, ax = plot_comparison(df_eval=_df(), colors={"Scale-based": "tab:gray", "CPP": "tab:red"})
        assert len(ax.patches) == 6

    def test_custom_columns(self):
        df = _df().rename(columns={"group": "method", "condition": "setting", "value": "acc"})
        fig, ax = plot_comparison(df_eval=df, group="method", condition="setting", value="acc")
        assert len(ax.patches) == 6

    def test_aggregates_repeated_cells(self):
        df = pd.concat([_df(), _df()], ignore_index=True)  # duplicate (group, condition) rows
        fig, ax = plot_comparison(df_eval=df)
        assert len(ax.patches) == 6

    def test_repeated_cells_use_mean(self):
        # Two rows for the same (group, condition) with different values -> the mean.
        df = pd.DataFrame([{"group": "A", "condition": "x", "value": 60},
                           {"group": "A", "condition": "x", "value": 80}])
        fig, ax = plot_comparison(df_eval=df, baseline=None)
        assert ax.patches[0].get_height() == pytest.approx(70)

    def test_passing_existing_ax(self):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = plot_comparison(df_eval=_df(), ax=ax_in)
        assert ax_out is ax_in and fig_out is fig_in

    def test_labels_and_title(self):
        fig, ax = plot_comparison(df_eval=_df(), xlabel="X", ylabel="Balanced accuracy [%]",
                                  title="Bench")
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Balanced accuracy [%]"
        assert ax.get_title() == "Bench"

    def test_duplicate_group_order_deduped(self):
        # A repeated entry in the explicit order must not create a duplicate bar row.
        fig, ax = plot_comparison(df_eval=_df(), baseline=None,
                                  group_order=["Scale-based", "Scale-based", "CPP"])
        assert len(ax.patches) == 6
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Scale-based", "CPP"]

    def test_duplicate_condition_order_deduped(self):
        fig, ax = plot_comparison(df_eval=_df(),
                                  condition_order=["No expansion", "No expansion", "Random", "dPULearn"])
        assert len(ax.patches) == 6
        assert [t.get_text() for t in ax.get_xticklabels()] == ["No expansion", "Random", "dPULearn"]

    def test_missing_cell_partial_grid(self):
        # A (group, condition) combination absent from df_eval leaves a gap, no crash / label.
        df = pd.DataFrame([{"group": "A", "condition": "x", "value": 60},
                           {"group": "B", "condition": "y", "value": 70}])
        fig, ax = plot_comparison(df_eval=df, baseline=None)
        assert len(ax.patches) == 4  # 2 groups x 2 conditions
        assert len(ax.texts) == 2    # only the 2 present cells are labelled

    def test_ylim_respected(self):
        fig, ax = plot_comparison(df_eval=_df(), ylim=(0, 108))
        assert ax.get_ylim() == (0, 108)

    def test_bar_width_and_figsize_and_fontsize(self):
        fig, ax = plot_comparison(df_eval=_df(), bar_width=0.6, figsize=(8, 5),
                                  fontsize_annotations=8)
        assert len(ax.patches) == 6

    def test_xtick_rotation_applied(self):
        fig, ax = plot_comparison(df_eval=_df(), xtick_rotation=30)
        assert all(t.get_rotation() == 30 for t in ax.get_xticklabels())

    def test_xtick_rotation_default_horizontal(self):
        fig, ax = plot_comparison(df_eval=_df())
        assert all(t.get_rotation() == 0 for t in ax.get_xticklabels())


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

    def test_duplicate_columns_raise(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), group="group", condition="group")

    def test_bad_baseline_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), baseline="50")

    def test_bad_annotate_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), annotate="yes")

    def test_bad_annotation_fmt_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), annotation_fmt=2)

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

    def test_non_numeric_value_column_raises(self):
        df = _df()
        df["value"] = df["value"].astype(str)
        with pytest.raises(ValueError):
            plot_comparison(df_eval=df)

    def test_bad_xtick_rotation_type_raises(self):
        with pytest.raises(ValueError):
            plot_comparison(df_eval=_df(), xtick_rotation="45")
