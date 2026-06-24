"""This script tests the aaanalysis.pipe.plot_eval() evaluation-grid plot."""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

import aaanalysis as aa
import aaanalysis.pipe as aap
from aaanalysis.pipe._eval_plot import (_sweep_axes, _display_order, _best_row, _axis_levels,
                                        _level_label, _refine_str, plot_eval)

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")
aa.options["verbose"] = False


def _make_eval(parts=("tmd",), pats=("none",), nsmax=(15,), nexp=(50,), nfilt=(100,)):
    """Build a synthetic find_features-style sweep table over the given per-axis level lists."""
    rows = []
    for lp, pm, ns, ne, nf in itertools.product(parts, pats, nsmax, nexp, nfilt):
        rows.append({"list_parts": lp, "split_types": pm, "pattern_mode": pm, "n_split_max": ns,
                     "n_explain": ne, "n_filter": nf, "n_features": nf,
                     "cv_bacc_mean": 0.5 + 0.1 * np.cos(nf / 40.0) + 0.02 * len(pm),
                     "cv_bacc_std": 0.03})
    df = pd.DataFrame(rows)
    df["rank"] = df["cv_bacc_mean"].rank(ascending=False, method="first").astype(int)
    df["is_selected"] = False
    df.loc[df["cv_bacc_mean"].idxmax(), "is_selected"] = True
    return df


# Per-layout fixtures (0/1/2/3/4 varying axes).
_DF_0D = _make_eval()
_DF_1D = _make_eval(nfilt=(25, 50, 75, 100, 125, 150))
_DF_1D_CAT = _make_eval(parts=("tmd", "tmd_jmd", "jmd_n_tmd"))
_DF_2D = _make_eval(pats=("none", "p1", "p2", "p1+p2"), nfilt=(25, 50, 75, 100))
_DF_3D = _make_eval(pats=("none", "p1", "p2", "p1+p2"), nsmax=(10, 15), nfilt=(25, 50, 75, 100))
_DF_4D = _make_eval(parts=("A", "B", "C", "D"), pats=("none", "p1"), nexp=(50, None),
                    nfilt=(25, 50, 75, 100))


@pytest.fixture(autouse=True)
def _close_figs():
    """Close every figure after each test so the Agg figure registry stays clean."""
    yield
    plt.close("all")


class TestPlotEval:
    """Positive tests for aap.plot_eval(): adaptive layout, color, best-marking, annotations."""

    # Layout dispatch by number of varying axes
    def test_zero_axes_returns_none(self):
        assert plot_eval(_DF_0D) is None

    def test_one_axis_returns_figure(self):
        fig = plot_eval(_DF_1D)
        assert isinstance(fig, Figure)

    def test_one_axis_numeric_draws_line(self):
        fig = plot_eval(_DF_1D)
        assert any(len(ax.lines) > 0 for ax in fig.axes)

    def test_one_axis_categorical_draws_bars(self):
        fig = plot_eval(_DF_1D_CAT)
        assert any(len(ax.patches) > 0 for ax in fig.axes)

    def test_two_axes_single_heatmap(self):
        fig = plot_eval(_DF_2D)
        n_heatmaps = sum(1 for ax in fig.axes if ax.images)
        assert n_heatmaps == 1

    def test_three_axes_facets(self):
        fig = plot_eval(_DF_3D)
        n_heatmaps = sum(1 for ax in fig.axes if ax.images)
        assert n_heatmaps >= 2

    def test_four_axes_facets(self):
        fig = plot_eval(_DF_4D)
        n_heatmaps = sum(1 for ax in fig.axes if ax.images)
        assert n_heatmaps >= 2

    # Color encoding
    def test_heatmap_uses_viridis(self):
        fig = plot_eval(_DF_2D)
        cmaps = {ax.images[0].get_cmap().name for ax in fig.axes if ax.images}
        assert cmaps == {"viridis"}

    def test_facets_share_viridis(self):
        fig = plot_eval(_DF_4D)
        cmaps = {ax.images[0].get_cmap().name for ax in fig.axes if ax.images}
        assert cmaps == {"viridis"}

    def test_has_colorbar(self):
        fig = plot_eval(_DF_2D)
        # The colorbar adds at least one extra axes beyond the single heatmap.
        assert len(fig.axes) >= 2

    # Best-config marking
    def test_best_marked_in_one_axis(self):
        fig = plot_eval(_DF_1D)
        assert any(len(ax.collections) > 0 for ax in fig.axes)

    def test_best_marked_in_heatmap(self):
        fig = plot_eval(_DF_2D)
        assert any(len(ax.collections) > 0 for ax in fig.axes)

    def test_best_falls_back_to_argmax_without_is_selected(self):
        df = _DF_2D.drop(columns=["is_selected"])
        assert _best_row(df, "cv_bacc_mean") == df["cv_bacc_mean"].idxmax()

    # Suptitle / refinement annotation
    def test_default_suptitle_names_axes(self):
        fig = plot_eval(_DF_2D)
        assert "n_filter" in fig._suptitle.get_text()

    def test_dict_refine_annotation_in_suptitle(self):
        dr = {"base": 0.70, "simplify": 0.73, "simplify_kept": True, "rfe": 0.73, "rfe_kept": False}
        fig = plot_eval(_DF_2D, dict_refine=dr)
        text = fig._suptitle.get_text()
        assert "refine" in text and "simplify" in text and "rfe" in text

    def test_no_refine_annotation_when_absent(self):
        fig = plot_eval(_DF_2D)
        assert "refine" not in fig._suptitle.get_text()

    # Parameters
    def test_custom_title(self):
        fig = plot_eval(_DF_2D, title="my sweep")
        assert fig._suptitle.get_text().startswith("my sweep")

    def test_custom_score_col(self):
        df = _DF_2D.rename(columns={"cv_bacc_mean": "my_score"})
        fig = plot_eval(df, score_col="my_score")
        assert isinstance(fig, Figure)

    def test_std_col_none_ok(self):
        fig = plot_eval(_DF_1D, std_col=None)
        assert isinstance(fig, Figure)

    def test_missing_std_col_ignored(self):
        df = _DF_1D.drop(columns=["cv_bacc_std"])
        fig = plot_eval(df)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self):
        fig = plot_eval(_DF_2D, figsize=(5, 4))
        assert isinstance(fig, Figure)

    def test_none_level_labeled_all(self):
        fig = plot_eval(_DF_4D)
        assert isinstance(fig, Figure)


class TestPlotEvalErrors:
    """Negative tests: one rejected input per test."""

    def test_df_eval_not_dataframe(self):
        with pytest.raises(ValueError):
            plot_eval([1, 2, 3])

    def test_df_eval_empty(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2D.iloc[0:0])

    def test_score_col_missing(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2D, score_col="not_a_column")

    def test_score_col_non_numeric(self):
        df = _DF_2D.copy()
        df["cv_bacc_mean"] = "x"
        with pytest.raises(ValueError):
            plot_eval(df)

    def test_std_col_wrong_type(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_1D, std_col=5)

    def test_title_wrong_type(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2D, title=123)

    def test_dict_refine_wrong_type(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2D, dict_refine=["base", 0.7])

    def test_dict_refine_non_numeric_value(self):
        with pytest.raises(ValueError):
            plot_eval(_DF_2D, dict_refine={"base": "x", "simplify": 0.7})

    def test_all_nan_score_rejected(self):
        df = _DF_2D.copy()
        df["cv_bacc_mean"] = np.nan
        with pytest.raises(ValueError):
            plot_eval(df)


class TestPlotEvalRobustness:
    """Regression tests for the hardening fixes (non-unique index, refinement Δ, color scale)."""

    def test_non_unique_index_renders(self):
        df = _DF_2D.copy()
        df.index = [0] * len(df)  # concatenated-sweep style duplicate index
        assert isinstance(plot_eval(df), Figure)

    def test_refine_running_base_rfe_noop(self):
        # simplify kept (0.70 -> 0.73); RFE a no-op returns the post-simplify score -> Δ must be 0.
        text = _refine_str({"base": 0.70, "simplify": 0.73, "simplify_kept": True,
                            "rfe": 0.73, "rfe_kept": False})
        assert "simplify: 0.730 (Δ+0.030, kept)" in text
        assert "rfe: 0.730 (Δ+0.000, no-op)" in text

    def test_refine_simplify_not_kept_running_base_unchanged(self):
        # simplify NOT kept (0.68 < 0.70); RFE measured against the original base, not the drop.
        text = _refine_str({"base": 0.70, "simplify": 0.68, "simplify_kept": False,
                            "rfe": 0.72, "rfe_kept": True})
        assert "rfe: 0.720 (Δ+0.020, kept)" in text

    def test_global_color_scale_matches_raw_range(self):
        fig = plot_eval(_DF_2D)
        im = next(ax.images[0] for ax in fig.axes if ax.images)
        lo, hi = im.get_clim()
        assert lo == pytest.approx(_DF_2D["cv_bacc_mean"].min())
        assert hi == pytest.approx(_DF_2D["cv_bacc_mean"].max())

    def test_facets_share_global_color_scale(self):
        fig = plot_eval(_DF_4D)
        clims = {ax.images[0].get_clim() for ax in fig.axes if ax.images}
        assert len(clims) == 1  # every facet panel on one shared scale


class TestPlotEvalHelpers:
    """Unit tests for the axis-detection / ordering helpers."""

    def test_sweep_axes_only_varying(self):
        assert _sweep_axes(_DF_2D) == ["n_filter", "pattern_mode"]

    def test_sweep_axes_none_when_single_config(self):
        assert _sweep_axes(_DF_0D) == []

    def test_sweep_axes_excludes_split_types_duplicate(self):
        # pattern_mode represents the split axis; split_types must not be counted separately.
        assert "split_types" not in _sweep_axes(_DF_2D)

    def test_display_order_puts_n_filter_first(self):
        axes = _sweep_axes(_DF_4D)
        ordered = _display_order(_DF_4D, axes)
        assert ordered[0] == "n_filter"

    def test_display_order_by_cardinality(self):
        # After n_filter, higher-cardinality axes lead (so they become the inner heatmap).
        axes = _sweep_axes(_DF_4D)
        ordered = _display_order(_DF_4D, axes)
        cards = [_DF_4D[a].nunique(dropna=False) for a in ordered[1:]]
        assert cards == sorted(cards, reverse=True)

    def test_axis_levels_numeric_sorted(self):
        assert _axis_levels(_DF_1D, "n_filter") == [25, 50, 75, 100, 125, 150]

    def test_axis_levels_none_last(self):
        levels = _axis_levels(_DF_4D, "n_explain")
        assert levels[-1] is None

    def test_level_label_none_is_all(self):
        assert _level_label("n_explain", None) == "all"

    def test_level_label_numeric_int(self):
        assert _level_label("n_filter", 50) == "50"


class TestPlotEvalComplex:
    """Integration with a real find_features sweep table and a property test."""

    def test_real_find_features_df_eval(self):
        # A genuine (small) balanced sweep -> 2 swept axes (split levers); plot must render.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
        labels = df_seq["label"].to_list()
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="balanced",
                                          kws={"n_filter": 25, "n_explain": 20}, plot=False,
                                          random_state=0, n_jobs=1)
        assert len(df_eval) > 1 and bool(df_eval["is_selected"].any())
        fig = plot_eval(df_eval)
        assert isinstance(fig, Figure)

    @settings(max_examples=25)
    @given(n_pat=st.integers(min_value=1, max_value=4),
           n_nf=st.integers(min_value=1, max_value=6),
           n_ns=st.integers(min_value=1, max_value=2))
    def test_property_returns_none_or_figure(self, n_pat, n_nf, n_ns):
        df = _make_eval(pats=("none", "p1", "p2", "p1+p2")[:n_pat],
                        nsmax=(10, 15)[:n_ns],
                        nfilt=(25, 50, 75, 100, 125, 150)[:n_nf])
        result = plot_eval(df)
        n_axes = sum(x > 1 for x in (n_pat, n_nf, n_ns))
        if n_axes == 0:
            assert result is None
        else:
            assert isinstance(result, Figure)
        plt.close("all")
