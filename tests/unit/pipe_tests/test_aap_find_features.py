"""This script tests the aaanalysis.pipe.find_features() staged CPP AutoML golden pipeline."""
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.pipe as aap
from aaanalysis.pipe._find_features import (_resolve_config, _resolve_models, _load_scale_spec,
                                            _cv_scores, _pareto_mask, _axis_impact, _MODES,
                                            _PART_SETS, _SPLIT_TYPE_SETS)

aa.options["verbose"] = False


# Shared seeded fixture data (small DOM_GSEC slice; n=20 -> 40 rows, 20 per class).
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
sf = aa.SequenceFeature(verbose=False)
# kws that shrink a search to a tiny Stage-1 grid (one scale, one n_split) so tests stay fast.
SMALL = {"n_explain": 30, "n_split_max": 15}


def _explicit_fast(random_state=0):
    """The explicit single-CPP chain that find_features(search='fast') mirrors byte-for-byte."""
    cfg = _resolve_config(search="fast")
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=cfg["part_sets"][0])
    df_scales = _load_scale_spec(cfg["scale_specs"][0])
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, random_state=random_state, verbose=False)
    df_feat = cpp.run(labels=labels, label_test=1, label_ref=0, n_filter=cfg["n_filter_vals"][0],
                      max_cor=cfg["max_cor"], max_overlap=cfg["max_overlap"], n_jobs=1)
    df_feat = cpp.simplify(df_feat=df_feat, labels=labels, strategy=cfg["simplify_strategy"],
                           label_test=1, label_ref=0)
    df_scales_all = aa.load_scales(name="scales")
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts,
                          df_scales=df_scales_all, n_jobs=1)
    tm = aa.TreeModel(random_state=random_state, verbose=False)
    tm.fit(X, labels=labels)
    return tm.add_feat_importance(df_feat=df_feat, sort=True)


class TestFindFeatures:
    """Positive and negative tests for find_features(), one parameter per test.

    Selection parameters (``model`` / ``cv`` / ``metric`` / ``simplify``) are exercised in the cheap
    single-configuration ``"fast"`` mode (same scorer, one-row ``df_eval``); the staged search and
    multi-objective behaviour get their own ``"balanced"`` tests below.
    """

    # Positive tests
    def test_returns_triple(self):
        df_feat, ax, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                                 plot=True, random_state=0, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        assert isinstance(ax, Axes)
        assert isinstance(df_eval, pd.DataFrame)

    def test_plot_false_returns_none_axes(self):
        _, ax, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                     plot=False, random_state=0, n_jobs=1)
        assert ax is None

    def test_df_feat_importance_sorted(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          plot=False, random_state=0, n_jobs=1)
        assert "feat_importance" in df_feat.columns
        imp = df_feat["feat_importance"].to_numpy()
        assert np.all(imp[:-1] >= imp[1:])

    def test_search_fast_single_row_eval(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                                plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) > 0 and len(df_eval) == 1
        assert df_eval["stage"].iloc[0] == "single"

    def test_simplify_parameter(self):
        for simplify in [True, False]:
            df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                              simplify=simplify, plot=False, random_state=0, n_jobs=1)
            assert len(df_feat) > 0

    @pytest.mark.parametrize("model", ["svm", "rf", "log_reg"])
    def test_model_parameter(self, model):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          model=model, plot=False, random_state=0, n_jobs=1)
        assert df_eval["balanced_accuracy_mean"].notna().all()

    def test_model_list_averages(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          model=["svm", "rf"], plot=False, random_state=0, n_jobs=1)
        assert df_eval["balanced_accuracy_mean"].notna().all()

    def test_cv_parameter(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          cv=3, plot=False, random_state=0, n_jobs=1)
        assert len(df_eval) == 1

    def test_metric_parameter(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          metric="accuracy", plot=False, random_state=0, n_jobs=1)
        assert "accuracy_mean" in df_eval.columns

    def test_metric_list_pareto_columns(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          metric=["balanced_accuracy", "f1"], plot=False,
                                          random_state=0, n_jobs=1)
        assert {"balanced_accuracy_mean", "f1_mean"}.issubset(df_eval.columns)

    def test_kws_parameter(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          kws={"n_filter": 75}, plot=False, random_state=0, n_jobs=1)
        assert sorted(df_eval["n_filter"].unique()) == [75]

    def test_subcategories_parameter(self):
        subs = sorted(aa.load_scales(name="scales_cat")["subcategory"].unique())[:5]
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, subcategories=subs, search="fast",
                                          simplify=False, plot=False, random_state=0, n_jobs=1)
        assert set(df_feat["subcategory"]).issubset(set(subs))

    def test_top_n_parameter(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          top_n=10, plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) == 10

    def test_label_test_ref_parameters(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          label_test=1, label_ref=0, plot=False,
                                          random_state=0, n_jobs=1)
        assert len(df_feat) > 0

    def test_name_test_ref_parameters(self):
        _, ax, _ = aap.find_features(labels, df_seq=df_seq, search="fast", plot=True,
                                     name_test="Target", name_ref="Control", random_state=0, n_jobs=1)
        assert isinstance(ax, Axes)

    def test_n_jobs_parameter(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) > 0

    def test_verbose_parameter(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          plot=False, random_state=0, n_jobs=1, verbose=True)
        assert len(df_feat) > 0

    def test_df_seq_parameter(self):
        df_feat, _, _ = aap.find_features(labels=labels, df_seq=df_seq, search="fast",
                                          plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) > 0

    # Negative tests
    def test_invalid_search(self):
        for bad in ["nope", "optimization", "", 1]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search=bad, plot=False)

    def test_invalid_df_seq(self):
        for bad in [None, pd.DataFrame({"x": [1]}), "not_a_df"]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=bad, search="fast", plot=False)

    def test_invalid_model(self):
        for bad in ["xgboost", "", 1]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", model=bad, plot=False)

    def test_invalid_cv(self):
        for bad in [1, 0, -1, "x"]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", cv=bad, plot=False)

    def test_invalid_simplify(self):
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", simplify=bad, plot=False)

    def test_invalid_top_n(self):
        for bad in [0, -1, "x"]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", top_n=bad, plot=False)

    def test_invalid_random_state(self):
        for bad in [-1, "x", 1.5]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", random_state=bad, plot=False)

    def test_invalid_plot(self):
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", plot=bad)

    def test_invalid_verbose(self):
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, search="fast", plot=False, verbose=bad)

    def test_invalid_kws_unknown_key(self):
        with pytest.raises(ValueError):
            aap.find_features(labels, df_seq=df_seq, search="balanced",
                              kws={"not_a_lever": 1}, plot=False)

    def test_invalid_kws_type(self):
        with pytest.raises(ValueError):
            aap.find_features(labels, df_seq=df_seq, search="balanced", kws="not_a_dict", plot=False)

    def test_invalid_subcategories_no_match(self):
        with pytest.raises(ValueError):
            aap.find_features(labels, df_seq=df_seq, subcategories=["__nonexistent__"],
                              search="fast", plot=False, random_state=0)


class TestFindFeaturesComplex:
    """Fast parity, the staged search, multi-objective selection, and reproducibility."""

    def test_fast_byte_identical_to_explicit_chain(self):
        df_m = _explicit_fast(random_state=0)
        df_a, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                       plot=False, random_state=0, n_jobs=1)
        assert df_m.equals(df_a)

    def test_reproducible_same_seed_fast(self):
        df_1, _, e_1 = aap.find_features(labels, df_seq=df_seq, search="fast",
                                         plot=False, random_state=7, n_jobs=1)
        df_2, _, e_2 = aap.find_features(labels, df_seq=df_seq, search="fast",
                                         plot=False, random_state=7, n_jobs=1)
        assert df_1.equals(df_2) and e_1.equals(e_2)

    def test_top_n_after_importance_sort(self):
        df_full, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                          plot=False, random_state=0, n_jobs=1)
        df_top, _, _ = aap.find_features(labels, df_seq=df_seq, search="fast",
                                         top_n=5, plot=False, random_state=0, n_jobs=1)
        assert df_top.equals(df_full.head(5))

    def test_balanced_staged_search(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="balanced",
                                                kws=SMALL, plot=False, random_state=0, n_jobs=1)
        assert set(df_eval["stage"]).issubset({"sensitivity", "n_filter", "refine"})
        assert "sensitivity" in set(df_eval["stage"]) and "n_filter" in set(df_eval["stage"])
        assert int(df_eval["is_selected"].sum()) == 1
        sel = df_eval[df_eval["is_selected"]].iloc[0]
        assert int(sel["n_features"]) == len(df_feat)
        for col in ["stage", "pattern_mode", "scale", "n_filter", "n_features",
                    "balanced_accuracy_mean", "balanced_accuracy_std", "is_pareto", "rank"]:
            assert col in df_eval.columns

    def test_balanced_multi_metric_pareto(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="balanced", kws=SMALL,
                                                metric=["balanced_accuracy", "f1"], plot=False,
                                                random_state=0, n_jobs=1)
        assert {"balanced_accuracy_mean", "f1_mean"}.issubset(df_eval.columns)
        assert int(df_eval["is_pareto"].sum()) >= 1
        assert int(df_eval["is_selected"].sum()) == 1

    def test_reproducible_same_seed_balanced(self):
        d1, _, e1 = aap.find_features(labels, df_seq=df_seq, search="balanced", kws=SMALL,
                                      plot=False, random_state=7, n_jobs=1)
        d2, _, e2 = aap.find_features(labels, df_seq=df_seq, search="balanced", kws=SMALL,
                                      plot=False, random_state=7, n_jobs=1)
        assert d1.equals(d2) and e1.equals(e2)

    @pytest.mark.slow
    def test_exhaustive_sweeps_parts(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, search="exhaustive",
                                                kws={"n_filter": 100, "n_split_max": 15}, plot=False,
                                                random_state=0, n_jobs=1)
        assert len(df_feat) > 0
        sens = df_eval[df_eval["stage"] == "sensitivity"]
        assert sens["list_parts"].nunique() > 1   # Part axis is swept in exhaustive

    def test_combined_subcategories_top_n_plot(self):
        subs = sorted(aa.load_scales(name="scales_cat")["subcategory"].unique())[:8]
        df_feat, ax, _ = aap.find_features(labels, df_seq=df_seq, subcategories=subs,
                                           search="balanced", kws=SMALL, top_n=8, plot=True,
                                           random_state=1, n_jobs=1)
        assert len(df_feat) <= 8 and isinstance(ax, Axes)

    def test_fast_ax_eval_empty(self):
        from matplotlib.figure import Figure
        _, ax, _ = aap.find_features(labels, df_seq=df_seq, search="fast", plot=True,
                                     random_state=0, n_jobs=1)
        assert isinstance(ax, Axes) and ax.eval == []

    def test_balanced_ax_eval_publication_figures(self):
        from matplotlib.figure import Figure
        _, ax, _ = aap.find_features(labels, df_seq=df_seq, search="balanced",
                                     kws={"n_split_max": 15}, plot=True, random_state=0, n_jobs=1)
        assert len(ax.eval) >= 1 and all(isinstance(f, Figure) for f in ax.eval)


class TestFindFeaturesHelpers:
    """Unit tests for the internal selection / sensitivity helpers."""

    def test_resolve_config_modes(self):
        for mode in _MODES:
            cfg = _resolve_config(search=mode)
            assert cfg["scale_specs"] and cfg["n_filter_vals"]

    def test_resolve_config_kws_pins_levers(self):
        cfg = _resolve_config(search="balanced", kws={"n_filter": 42, "n_explain": 20})
        assert cfg["n_filter_vals"] == [42]
        assert cfg["scale_specs"] == [("explain", 20)] and cfg["sweep_scales"] is False

    def test_resolve_config_unknown_kws_raises(self):
        with pytest.raises(ValueError):
            _resolve_config(search="balanced", kws={"bogus": 1})

    def test_resolve_models_list(self):
        models = _resolve_models(["svm", "rf"], random_state=0)
        assert len(models) == 2

    def test_pareto_mask_dominance(self):
        means = [[0.8, 0.7], [0.7, 0.8], [0.9, 0.9]]
        mask = _pareto_mask(means)
        assert mask[2] and not mask[0] and not mask[1]

    def test_pareto_mask_tradeoff(self):
        mask = _pareto_mask([[0.9, 0.5], [0.5, 0.9]])
        assert mask.all()

    def test_axis_impact_zero_when_flat(self):
        df = pd.DataFrame({"axis": ["a", "a", "b", "b"], "m_mean": [0.9, 0.9, 0.9, 0.9]})
        assert _axis_impact(df, "axis", ["m"]) == 0.0

    def test_axis_impact_positive(self):
        df = pd.DataFrame({"axis": ["a", "a", "b", "b"], "m_mean": [0.6, 0.7, 0.9, 1.0]})
        assert _axis_impact(df, "axis", ["m"]) > 0

    def test_cv_scores_per_metric(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([rng.normal(0, 1, 40) for _ in range(5)])
        y = [0] * 20 + [1] * 20
        out = _cv_scores(X, y, models=_resolve_models("svm", 0), cv=3,
                         metrics=["balanced_accuracy", "f1"], random_state=0)
        assert set(out) == {"balanced_accuracy", "f1"} and len(out["f1"]) == 2

    def test_split_type_sets_start_with_segment(self):
        for sts in _SPLIT_TYPE_SETS:
            assert sts[0] == "Segment"
