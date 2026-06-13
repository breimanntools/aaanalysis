"""This is a script to test the end-to-end protocol workflows.

E2e tier (ADR-0031): one assertion-bearing test per ``protocols/*.ipynb`` —
the protocol notebooks run end to end but assert nothing and are not in CI, so
these are their checked analogues. Each runs the real pipeline at a tiny, seeded
size and asserts a *final artifact* (range/shape/finiteness), never a frozen
exact value. Where a protocol cell uses a ``pro`` feature (protocol9's
ShapModel) the core path is substituted. A couple of degenerate-dataset
negatives confirm a clear error instead of a cryptic crash.
"""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
from tests import _pipeline

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

pytestmark = pytest.mark.e2e

DF_FEAT_COLS = {"feature", "abs_auc", "mean_dif", "category"}


# ---------------------------------------------------------------------------
# protocol1 — CPP signature
# ---------------------------------------------------------------------------
class TestProtocol1Signature:
    def test_signature_artifact(self, ):
        p = _pipeline.build_pipeline()
        df_feat = p["df_feat"]
        assert DF_FEAT_COLS.issubset(df_feat.columns)
        assert len(df_feat) == _pipeline.N_FILTER
        assert df_feat["abs_auc"].is_monotonic_decreasing  # ranked best-first

    def test_signature_reproducible(self):
        # Reproducibility: the same seeded run yields the same top feature.
        a = _pipeline.build_pipeline()["df_feat"]
        b = _pipeline.build_pipeline()["df_feat"]
        assert a["feature"].iloc[0] == b["feature"].iloc[0]


# ---------------------------------------------------------------------------
# protocol2 — sequence analysis (logo + conservation)
# ---------------------------------------------------------------------------
class TestProtocol2SequenceAnalysis:
    def test_conservation_scalar_finite(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=15)
        df_parts = aa.SequenceFeature().get_df_parts(
            df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        aal = aa.AAlogo(logo_type="probability")
        df_logo_info = aal.get_df_logo_info(df_parts=df_parts, tmd_len=20)
        cons = aal.get_conservation(df_logo_info=df_logo_info, value_type="mean")
        assert np.isfinite(cons) and cons >= 0


# ---------------------------------------------------------------------------
# protocol3 — sampling (benchmark set)
# ---------------------------------------------------------------------------
class TestProtocol3Sampling:
    ARMS = {"ctrl_freq": {"method": "synthetic", "n": 6, "window_size": 9, "generator": "global_freq"},
            "ctrl_unif": {"method": "synthetic", "n": 6, "window_size": 9, "generator": "uniform"}}

    def test_benchmark_set_schema_and_counts(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        df_bench = aa.AAWindowSampler(random_state=0).sample_benchmark_set(
            df_seq=df_seq, arms=self.ARMS, seed=0)
        assert {"window", "label", "role", "arm"}.issubset(df_bench.columns)
        assert df_bench["arm"].value_counts().to_dict() == {"ctrl_freq": 6, "ctrl_unif": 6}

    def test_benchmark_set_reproducible(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        a = aa.AAWindowSampler(random_state=0).sample_benchmark_set(df_seq=df_seq, arms=self.ARMS, seed=0)
        b = aa.AAWindowSampler(random_state=0).sample_benchmark_set(df_seq=df_seq, arms=self.ARMS, seed=0)
        assert a["window"].to_list() == b["window"].to_list()


# ---------------------------------------------------------------------------
# protocol4 — prediction tasks across levels (residue / domain / protein)
# ---------------------------------------------------------------------------
class TestProtocol4PredictionLevels:
    @pytest.mark.parametrize("name", ["AA_CASPASE3", "DOM_GSEC", "SEQ_AMYLO"])
    def test_level_pipeline_trains(self, name):
        sf = aa.SequenceFeature()
        df_scales = _pipeline.small_scales()
        df_seq = aa.load_dataset(name=name, n=8)
        labels = df_seq["label"].to_list()
        if name == "DOM_GSEC":
            df_parts = sf.get_df_parts(df_seq=df_seq)
            split_kws = None
        else:  # residue / protein windows are short -> small Segment-only splits
            df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"], jmd_n_len=0, jmd_c_len=0)
            split_kws = sf.get_split_kws(split_types=["Segment"], n_split_max=4)
        df_feat = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                         verbose=False, random_state=0).run(labels=labels, n_filter=6, n_jobs=1)
        X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts,
                              df_scales=df_scales, n_jobs=1)
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            X, labels=labels, use_rfe=False, n_cv=2, n_rounds=2)
        assert len(df_feat) > 0
        assert len(tm.feat_importance) == np.asarray(X).shape[1]


# ---------------------------------------------------------------------------
# protocol5 — engineer features
# ---------------------------------------------------------------------------
class TestProtocol5EngineerFeatures:
    def test_feature_matrix_then_correlation_filter(self):
        p = _pipeline.build_pipeline()
        X = np.asarray(p["X"])
        assert X.shape == (len(p["labels"]), _pipeline.N_FILTER)
        mask = np.asarray(aa.NumericalFeature().filter_correlation(X, max_cor=0.7))
        # The de-correlated set keeps at least one and no more than all features.
        assert 1 <= mask.sum() <= X.shape[1]


# ---------------------------------------------------------------------------
# protocol6 — compositional vs positional signatures
# ---------------------------------------------------------------------------
class TestProtocol6CompositionalPositional:
    def test_both_strategies_yield_categorised_features(self):
        df_seq = aa.load_dataset(name="DOM_GSEC", n=12)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        df_scales = _pipeline.small_scales()
        skw_comp = sf.get_split_kws(split_types=["Segment"], n_split_max=4)
        skw_pos = sf.get_split_kws(split_types=["PeriodicPattern"])
        df_comp = aa.CPP(df_parts=df_parts, split_kws=skw_comp, df_scales=df_scales,
                         verbose=False, random_state=0).run(labels=labels, n_filter=8, n_jobs=1)
        df_pos = aa.CPP(df_parts=df_parts, split_kws=skw_pos, df_scales=df_scales,
                        verbose=False, random_state=0).run(labels=labels, n_filter=8, n_jobs=1)
        for df_feat in (df_comp, df_pos):
            assert len(df_feat) > 0
            assert df_feat["category"].notna().all()


# ---------------------------------------------------------------------------
# protocol7 — feature selection
# ---------------------------------------------------------------------------
class TestProtocol7FeatureSelection:
    def test_selection_reduces_feature_set(self):
        p = _pipeline.build_pipeline()
        X, labels = np.asarray(p["X"]), np.asarray(p["labels"])
        auc = np.asarray(aa.comp_auc_adjusted(X=X, labels=labels, n_jobs=1))
        # Keep the most discriminative half -> a strictly smaller, non-empty set.
        keep = np.argsort(np.abs(auc))[::-1][: max(1, X.shape[1] // 2)]
        assert 0 < len(keep) < X.shape[1]
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            X[:, keep], labels=labels.tolist(), use_rfe=False, n_cv=2, n_rounds=2)
        assert len(tm.feat_importance) == len(keep)


# ---------------------------------------------------------------------------
# protocol8 — prediction (fit + eval + PU path)
# ---------------------------------------------------------------------------
class TestProtocol8Prediction:
    def test_fit_and_eval_metrics(self):
        p = _pipeline.build_pipeline()
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            p["X"], labels=p["labels"], use_rfe=False, n_cv=2, n_rounds=2)
        df_eval = tm.eval(p["X"], labels=p["labels"], list_is_selected=[tm.is_selected_],
                          n_cv=2, list_metrics=["accuracy", "f1"])
        num = df_eval.select_dtypes("number").values
        assert num.size > 0
        assert ((num >= -1e-9) & (num <= 1 + 1e-9)).all()

    def test_predictions_reproducible(self):
        p = _pipeline.build_pipeline()
        kws = dict(use_rfe=False, n_cv=2, n_rounds=2)
        a = aa.TreeModel(verbose=False, random_state=0).fit(p["X"], labels=p["labels"], **kws)
        b = aa.TreeModel(verbose=False, random_state=0).fit(p["X"], labels=p["labels"], **kws)
        pred_a, _ = a.predict_proba(p["X"])
        pred_b, _ = b.predict_proba(p["X"])
        assert np.allclose(np.asarray(pred_a), np.asarray(pred_b))


# ---------------------------------------------------------------------------
# protocol9 — interpretability (core TreeModel path; ShapModel is pro)
# ---------------------------------------------------------------------------
class TestProtocol9Interpretability:
    def test_importances_rank_features(self):
        p = _pipeline.build_pipeline()
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            p["X"], labels=p["labels"], use_rfe=False, n_cv=2, n_rounds=2)
        df_feat = tm.add_feat_importance(df_feat=p["df_feat"])
        imp = df_feat["feat_importance"]
        assert (imp >= 0).all() and imp.sum() > 0
        assert len(df_feat) == len(p["df_feat"])


# ---------------------------------------------------------------------------
# protocol10 — validation (adjusted AUC + bootstrap CI)
# ---------------------------------------------------------------------------
class TestProtocol10Validation:
    def test_auc_and_bootstrap_ci_finite(self):
        p = _pipeline.build_pipeline()
        X, labels = np.asarray(p["X"]), np.asarray(p["labels"])
        auc = np.asarray(aa.comp_auc_adjusted(X=X, labels=labels, n_jobs=1))
        assert np.isfinite(auc).all()
        assert (np.abs(auc) <= 0.5 + 1e-9).all()  # adjusted AUC lives in [-0.5, 0.5]
        ci = aa.comp_bootstrap_ci(values=auc.ravel(), n_rounds=200, ci=0.95, seed=0)
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


# ---------------------------------------------------------------------------
# Degenerate-dataset negatives (clear error, not a cryptic crash)
# ---------------------------------------------------------------------------
class TestE2EDegenerate:
    def test_single_class_pipeline_rejected(self):
        # All-one-class labels cannot define a TEST-vs-REF contrast.
        p = _pipeline.build_pipeline()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=p["df_parts"], df_scales=p["df_scales"], verbose=False).run(
                labels=[1] * len(p["labels"]), n_filter=5, n_jobs=1)

    def test_too_few_samples_for_model_rejected(self):
        # One sample cannot train/evaluate a model; the error must be explicit.
        p = _pipeline.build_pipeline()
        X1 = np.asarray(p["X"])[:1]
        with pytest.raises(ValueError):
            aa.TreeModel(verbose=False, random_state=0).fit(
                X1, labels=[1], use_rfe=False, n_cv=2, n_rounds=2)
