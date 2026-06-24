"""This script tests the aaanalysis.pipe.explain_features() SHAP explanation golden pipeline (pro)."""
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import pytest

# explain_features is pro-gated: skip the whole module cleanly when SHAP is not installed.
pytest.importorskip("shap")

import aaanalysis as aa
import aaanalysis.pipe as aap
from aaanalysis.pipe._explain_features import (_resolve_model_classes,
                                               _most_confident_target_sample,
                                               _normalize_samples_names)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

aa.options["verbose"] = False


# Shared seeded fixture data (small DOM_GSEC slice; n=20 -> 40 rows, 20 per class)
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
labels = df_seq["label"].to_list()
df_feat = aa.load_features().head(8)
sf = aa.SequenceFeature(verbose=False)
df_parts = sf.get_df_parts(df_seq=df_seq)
entry0 = str(df_seq["entry"].iloc[0])


def _manual_explain(samples, names, random_state=1, label_target_class=1, list_model_classes=None):
    """The explicit primitive chain aap.explain_features is supposed to mirror byte-for-byte."""
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    sm = aa.ShapModel(list_model_classes=list_model_classes, random_state=random_state, verbose=False)
    sm.fit(X, labels=labels, label_target_class=label_target_class)
    return sm.add_feat_impact(df_feat=df_feat.copy(), samples=samples, names=names, df_seq=df_seq)


class TestExplainFeatures:
    """Positive and negative tests for aap.explain_features(), one parameter per test."""

    # Positive tests
    def test_returns_triple(self):
        result = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0, random_state=0)
        assert isinstance(result, tuple) and len(result) == 3
        df_shap, ax, evals = result
        assert isinstance(df_shap, pd.DataFrame)
        assert isinstance(ax, Axes)
        assert evals is None

    def test_evals_slot_is_none(self):
        _, _, evals = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0, random_state=0)
        assert evals is None

    def test_impact_column_added(self):
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=entry0, random_state=0)
        assert f"feat_impact_{entry0}" in df_shap.columns
        assert np.isfinite(df_shap[f"feat_impact_{entry0}"]).all()

    def test_plot_false_returns_none_axes(self):
        df_shap, ax, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0, plot=False, random_state=0)
        assert ax is None
        assert any(c.startswith("feat_impact_") for c in df_shap.columns)

    def test_samples_auto_select(self):
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, random_state=0, plot=False)
        # Exactly one impact column for the auto-selected sample
        imp_cols = [c for c in df_shap.columns if c.startswith("feat_impact_")]
        assert len(imp_cols) == 1

    def test_samples_position(self):
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=2, plot=False, random_state=0)
        entry2 = str(df_seq["entry"].iloc[2])
        assert f"feat_impact_{entry2}" in df_shap.columns

    def test_samples_entry_name(self):
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=entry0, plot=False, random_state=0)
        assert f"feat_impact_{entry0}" in df_shap.columns

    def test_samples_list(self):
        names = [str(df_seq["entry"].iloc[i]) for i in (0, 1)]
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=[0, 1], plot=False, random_state=0)
        for n in names:
            assert f"feat_impact_{n}" in df_shap.columns

    def test_list_model_classes_parameter(self):
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0,
                                             list_model_classes=[RandomForestClassifier], plot=False, random_state=0)
        assert any(c.startswith("feat_impact_") for c in df_shap.columns)

    def test_label_target_class_parameter(self):
        df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, label_target_class=0,
                                             plot=False, random_state=0)
        assert any(c.startswith("feat_impact_") for c in df_shap.columns)

    def test_name_test_ref_parameters(self):
        _, ax, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0,
                                        name_test="GSEC", name_ref="OTHER", random_state=0)
        assert isinstance(ax, Axes)

    def test_n_jobs_parameter(self):
        for n_jobs in [None, 1]:
            df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0,
                                                 n_jobs=n_jobs, plot=False, random_state=0)
            assert any(c.startswith("feat_impact_") for c in df_shap.columns)

    def test_verbose_parameter(self):
        for verbose in [True, False]:
            df_shap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0,
                                                 verbose=verbose, plot=False, random_state=0)
            assert any(c.startswith("feat_impact_") for c in df_shap.columns)

    # Negative tests
    def test_invalid_label_target_class(self):
        for ltc in [-1, "x", 1.5]:
            with pytest.raises(ValueError):
                aap.explain_features(df_feat.copy(), df_seq, labels, label_target_class=ltc, random_state=0)

    def test_invalid_random_state(self):
        for random_state in [-1, "invalid", 1.5]:
            with pytest.raises(ValueError):
                aap.explain_features(df_feat.copy(), df_seq, labels, samples=0, random_state=random_state)

    def test_invalid_plot(self):
        with pytest.raises(ValueError):
            aap.explain_features(df_feat.copy(), df_seq, labels, samples=0, plot="yes", random_state=0)

    def test_absent_target_class_raises(self):
        # No sample of label_target_class=2 -> auto-select cannot pick a sample
        with pytest.raises(ValueError):
            aap.explain_features(df_feat.copy(), df_seq, labels, label_target_class=2, plot=False, random_state=0)


class TestExplainFeaturesComplex:
    """Combinations, the byte-identical parity contract, and reproducibility."""

    def test_byte_identical_to_manual_chain(self):
        df_man = _manual_explain(samples=entry0, names=entry0, random_state=1)
        df_aap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=entry0,
                                            random_state=1, plot=False)
        col = f"feat_impact_{entry0}"
        assert np.array_equal(df_aap[col].to_numpy(), df_man[col].to_numpy())

    def test_byte_identical_with_explicit_models(self):
        models = [ExtraTreesClassifier]
        df_man = _manual_explain(samples=0, names=entry0, random_state=3, list_model_classes=models)
        df_aap, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, samples=0,
                                            list_model_classes=models, random_state=3, plot=False)
        col = f"feat_impact_{entry0}"
        assert np.array_equal(df_aap[col].to_numpy(), df_man[col].to_numpy())

    def test_random_state_reproducible(self):
        df1, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, random_state=5, plot=False)
        df2, _, _ = aap.explain_features(df_feat.copy(), df_seq, labels, random_state=5, plot=False)
        c1 = [c for c in df1.columns if c.startswith("feat_impact_")][0]
        c2 = [c for c in df2.columns if c.startswith("feat_impact_")][0]
        assert c1 == c2  # same auto-selected sample
        assert np.array_equal(df1[c1].to_numpy(), df2[c2].to_numpy())

    def test_does_not_mutate_input_df_feat(self):
        df_in = df_feat.copy()
        aap.explain_features(df_in, df_seq, labels, samples=0, plot=False, random_state=0)
        assert not any(c.startswith("feat_impact_") for c in df_in.columns)


class TestExplainFeaturesHelpers:
    """Unit tests for the private selection helpers."""

    def test_resolve_model_classes_default(self):
        assert _resolve_model_classes(None) == [RandomForestClassifier, ExtraTreesClassifier]

    def test_resolve_model_classes_single(self):
        assert _resolve_model_classes(RandomForestClassifier) == [RandomForestClassifier]

    def test_resolve_model_classes_passthrough(self):
        models = [RandomForestClassifier]
        assert _resolve_model_classes(models) is models

    def test_most_confident_target_sample_deterministic(self):
        X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
        p1 = _most_confident_target_sample(X, labels=labels, list_model_classes=None,
                                           label_target_class=1, random_state=0)
        p2 = _most_confident_target_sample(X, labels=labels, list_model_classes=None,
                                           label_target_class=1, random_state=0)
        assert p1 == p2
        assert np.asarray(labels)[p1] == 1  # selected a target-class sample

    def test_most_confident_target_sample_absent_class(self):
        X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
        with pytest.raises(ValueError):
            _most_confident_target_sample(X, labels=labels, list_model_classes=None,
                                          label_target_class=2, random_state=0)

    def test_normalize_samples_names_position(self):
        samples, names = _normalize_samples_names(0, df_seq=df_seq)
        assert samples == [0]
        assert names == [str(df_seq["entry"].iloc[0])]

    def test_normalize_samples_names_entry_and_list(self):
        samples, names = _normalize_samples_names([entry0, 1], df_seq=df_seq)
        assert samples == [entry0, 1]
        assert names == [entry0, str(df_seq["entry"].iloc[1])]
