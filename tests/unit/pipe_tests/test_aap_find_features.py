"""This script tests the aaanalysis.pipe.find_features() CPP AutoML golden pipeline."""
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.pipe as aap
from aaanalysis.pipe._find_features import (_resolve_config, _load_scales_breadth, _MODES,
                                            _PART_SETS, _SPLIT_TYPE_SETS)

aa.options["verbose"] = False


# Shared seeded fixture data (small DOM_GSEC slice; n=20 -> 40 rows, 20 per class)
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=_PART_SETS[0])


def _explicit_fast(random_state=0):
    """The explicit single-CPP chain that find_features(optimization='fast') mirrors byte-for-byte."""
    cfg = _resolve_config(optimization="fast")
    df_scales = _load_scales_breadth(top_explain_n=cfg["scale_breadths"][0])
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

    Parameters that only feed the cross-validated selection score (``model`` / ``cv`` / ``metric``
    / ``simplify``) are exercised in the cheap single-configuration ``"fast"`` mode, which builds the
    same one-row ``df_eval`` via the same scorer; the sweep/selection behaviour gets its own
    ``"balanced"`` tests below.
    """

    # Positive tests
    def test_returns_triple(self):
        df_feat, ax, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                                 plot=True, random_state=0, n_jobs=1)
        assert isinstance(df_feat, pd.DataFrame)
        assert isinstance(ax, Axes)
        assert isinstance(df_eval, pd.DataFrame)

    def test_plot_false_returns_none_axes(self):
        df_feat, ax, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                                 plot=False, random_state=0, n_jobs=1)
        assert ax is None

    def test_df_feat_has_importance_column_sorted(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          plot=False, random_state=0, n_jobs=1)
        assert "feat_importance" in df_feat.columns
        imp = df_feat["feat_importance"].to_numpy()
        assert np.all(imp[:-1] >= imp[1:])

    def test_optimization_fast(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                                plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) > 0 and len(df_eval) == 1

    def test_simplify_parameter(self):
        for simplify in [True, False]:
            df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                              simplify=simplify, plot=False, random_state=0, n_jobs=1)
            assert len(df_feat) > 0

    @pytest.mark.parametrize("model", ["svm", "rf", "log_reg"])
    def test_model_parameter(self, model):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          model=model, plot=False, random_state=0, n_jobs=1)
        assert df_eval["cv_bacc_mean"].notna().all()

    def test_cv_parameter(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          cv=3, plot=False, random_state=0, n_jobs=1)
        assert len(df_eval) == 1

    def test_metric_parameter(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          metric="accuracy", plot=False, random_state=0, n_jobs=1)
        assert df_eval["cv_bacc_mean"].notna().all()

    def test_kws_parameter(self):
        _, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          kws={"n_filter": 75}, plot=False, random_state=0, n_jobs=1)
        assert sorted(df_eval["n_filter"].unique()) == [75]

    def test_subcategories_parameter(self):
        # simplify=False: simplify may legitimately swap to correlated scales outside the subset.
        subs = sorted(aa.load_scales(name="scales_cat")["subcategory"].unique())[:5]
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, subcategories=subs,
                                          optimization="fast", simplify=False, plot=False,
                                          random_state=0, n_jobs=1)
        assert set(df_feat["subcategory"]).issubset(set(subs))

    def test_top_n_parameter(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          top_n=10, plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) == 10

    def test_label_test_ref_parameters(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          label_test=1, label_ref=0, plot=False,
                                          random_state=0, n_jobs=1)
        assert len(df_feat) > 0

    def test_name_test_ref_parameters(self):
        df_feat, ax, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast", plot=True,
                                           name_test="Target", name_ref="Control",
                                           random_state=0, n_jobs=1)
        assert isinstance(ax, Axes)

    def test_n_jobs_parameter(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) > 0

    def test_verbose_parameter(self):
        df_feat, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          plot=False, random_state=0, n_jobs=1, verbose=True)
        assert len(df_feat) > 0

    def test_df_seq_parameter(self):
        df_feat, _, _ = aap.find_features(labels=labels, df_seq=df_seq, optimization="fast",
                                          plot=False, random_state=0, n_jobs=1)
        assert len(df_feat) > 0

    # Negative tests
    def test_invalid_optimization(self):
        for bad in ["nope", "quick", "", 1]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization=bad, plot=False)

    def test_invalid_df_seq(self):
        for bad in [None, pd.DataFrame({"x": [1]}), "not_a_df"]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=bad, optimization="fast", plot=False)

    def test_invalid_model(self):
        for bad in ["xgboost", "", 1]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", model=bad, plot=False)

    def test_invalid_cv(self):
        for bad in [1, 0, -1, "x"]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", cv=bad, plot=False)

    def test_invalid_simplify(self):
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", simplify=bad, plot=False)

    def test_invalid_top_n(self):
        for bad in [0, -1, "x"]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", top_n=bad, plot=False)

    def test_invalid_random_state(self):
        for bad in [-1, "x", 1.5]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", random_state=bad, plot=False)

    def test_invalid_plot(self):
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", plot=bad)

    def test_invalid_verbose(self):
        for bad in ["yes", 1, None]:
            with pytest.raises(ValueError):
                aap.find_features(labels, df_seq=df_seq, optimization="fast", plot=False, verbose=bad)

    def test_invalid_kws_unknown_key(self):
        with pytest.raises(ValueError):
            aap.find_features(labels, df_seq=df_seq, optimization="balanced",
                              kws={"not_a_lever": 1}, plot=False)

    def test_invalid_kws_type(self):
        with pytest.raises(ValueError):
            aap.find_features(labels, df_seq=df_seq, optimization="balanced",
                              kws="not_a_dict", plot=False)

    def test_invalid_subcategories_no_match(self):
        with pytest.raises(ValueError):
            aap.find_features(labels, df_seq=df_seq, subcategories=["__nonexistent__"],
                              optimization="fast", plot=False, random_state=0)


class TestFindFeaturesComplex:
    """The byte-identical fast parity contract, the sweep/selection behaviour, and reproducibility."""

    def test_fast_byte_identical_to_explicit_chain(self):
        df_m = _explicit_fast(random_state=0)
        df_a, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                       plot=False, random_state=0, n_jobs=1)
        assert df_m.equals(df_a)

    def test_reproducible_same_seed_fast(self):
        df_1, _, e_1 = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                         plot=False, random_state=7, n_jobs=1)
        df_2, _, e_2 = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                         plot=False, random_state=7, n_jobs=1)
        assert df_1.equals(df_2) and e_1.equals(e_2)

    def test_top_n_after_importance_sort(self):
        df_full, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                          plot=False, random_state=0, n_jobs=1)
        df_top, _, _ = aap.find_features(labels, df_seq=df_seq, optimization="fast",
                                         top_n=5, plot=False, random_state=0, n_jobs=1)
        assert df_top.equals(df_full.head(5))

    def test_balanced_sweep_and_selection(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="balanced",
                                                plot=False, random_state=0, n_jobs=1)
        # df_eval is a multi-row sweep with exactly one selected, best-first ranked, configuration.
        for col in ["pattern_mode", "n_filter", "n_features", "cv_bacc_mean", "cv_bacc_std",
                    "rank", "is_selected"]:
            assert col in df_eval.columns
        assert len(df_eval) > 1 and int(df_eval["is_selected"].sum()) == 1
        means = df_eval.sort_values("rank")["cv_bacc_mean"].to_numpy()
        assert np.all(means[:-1] >= means[1:])
        assert len(df_feat) > 0

    def test_reproducible_same_seed_balanced(self):
        df_1, _, e_1 = aap.find_features(labels, df_seq=df_seq, optimization="balanced",
                                         plot=False, random_state=7, n_jobs=1)
        df_2, _, e_2 = aap.find_features(labels, df_seq=df_seq, optimization="balanced",
                                         plot=False, random_state=7, n_jobs=1)
        assert df_1.equals(df_2) and e_1.equals(e_2)

    @pytest.mark.slow
    def test_exhaustive_sweeps_parts_and_scales(self):
        df_feat, _, df_eval = aap.find_features(labels, df_seq=df_seq, optimization="exhaustive",
                                                kws={"n_filter": 100}, plot=False,
                                                random_state=0, n_jobs=1)
        assert len(df_feat) > 0
        # exhaustive varies the Part region set and the Scale breadth across configurations
        # (n_explain is None for the full-scale breadth, so count it with dropna=False).
        assert df_eval["list_parts"].nunique() > 1
        assert df_eval["n_explain"].nunique(dropna=False) > 1

    def test_combined_subcategories_top_n_plot(self):
        subs = sorted(aa.load_scales(name="scales_cat")["subcategory"].unique())[:8]
        df_feat, ax, df_eval = aap.find_features(labels, df_seq=df_seq, subcategories=subs,
                                                 optimization="balanced", top_n=8, plot=True,
                                                 random_state=1, n_jobs=1)
        assert len(df_feat) <= 8
        assert isinstance(ax, Axes)


class TestFindFeaturesHelpers:
    """Unit tests for the internal config/scale helpers."""

    def test_resolve_config_modes(self):
        for mode in _MODES:
            cfg = _resolve_config(optimization=mode)
            assert "n_filter_vals" in cfg and len(cfg["part_sets"]) >= 1

    def test_resolve_config_kws_pins_levers(self):
        cfg = _resolve_config(optimization="balanced", kws={"n_filter": 42, "n_explain": 20})
        assert cfg["n_filter_vals"] == [42]
        assert cfg["scale_breadths"] == [20]
        assert cfg["sweep_scales"] is False

    def test_resolve_config_unknown_kws_raises(self):
        with pytest.raises(ValueError):
            _resolve_config(optimization="balanced", kws={"bogus": 1})

    def test_split_type_sets_start_with_segment(self):
        for sts in _SPLIT_TYPE_SETS:
            assert sts[0] == "Segment"
