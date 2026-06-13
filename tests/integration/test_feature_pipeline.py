"""This is a script to test the feature-engineering seams (real components, no mocks).

Integration tier (ADR-0031): each test wires two/three public classes together
and asserts the *seam contract* holds — structural/range assertions, never
frozen exact values (that is the regression anchor's job). Negatives here are
**composition failures** (only visible when components meet), not re-runs of
unit-level input validation. Properties are pipeline invariants / metamorphic
checks with small ``max_examples``.

Seams covered:
  1. load_scales -> AAclust.fit -> reduced scales -> CPP.run
  2. load_dataset -> SequenceFeature parts/splits -> CPP.run df_feat schema
  3. CPP df_feat -> TreeModel.add_feat_importance -> CPPPlot.ranking
  4. SequenceFeature.feature_matrix -> NumericalFeature.filter_correlation
  5. CPPGrid.run -> per-config df_feat
"""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
from tests import _pipeline

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

pytestmark = pytest.mark.integration

# df_feat columns CPP.run emits that downstream consumers rely on.
DF_FEAT_COLS = {"feature", "abs_auc", "mean_dif", "category"}


# ---------------------------------------------------------------------------
# Seam 1: load_scales -> AAclust -> reduced scales -> CPP
# ---------------------------------------------------------------------------
class TestAAclustScaleReductionToCPP:
    """A clustered, reduced scale set still drives a CPP run."""

    def test_reduced_scales_feed_cpp(self, pipeline):
        df_scales = _pipeline.small_scales()
        aac = aa.AAclust(verbose=False).fit(df_scales.T.values, n_clusters=5,
                                            names=list(df_scales.T.index))
        df_scales_red = df_scales[aac.medoid_names_]
        df_feat = aa.CPP(df_parts=pipeline["df_parts"], df_scales=df_scales_red,
                         verbose=False, random_state=0).run(labels=pipeline["labels"],
                                                            n_filter=10, n_jobs=1)
        assert len(df_feat) > 0
        assert DF_FEAT_COLS.issubset(df_feat.columns)

    @settings(max_examples=5, deadline=None)
    @given(n_clusters=some.integers(min_value=2, max_value=8))
    def test_reduced_count_at_most_input(self, n_clusters):
        # Property: the reduced (medoid) scale set is never larger than the input.
        df_scales = _pipeline.small_scales()
        aac = aa.AAclust(verbose=False).fit(df_scales.T.values, n_clusters=n_clusters,
                                            names=list(df_scales.T.index))
        assert len(aac.medoid_names_) == n_clusters
        assert len(aac.medoid_names_) <= df_scales.shape[1]

    def test_single_cluster_scale_still_runs(self, pipeline):
        # Composition edge: collapsing to one representative scale must not crash CPP.
        df_scales = _pipeline.small_scales()
        aac = aa.AAclust(verbose=False).fit(df_scales.T.values, n_clusters=1,
                                            names=list(df_scales.T.index))
        df_scales_red = df_scales[aac.medoid_names_]
        df_feat = aa.CPP(df_parts=pipeline["df_parts"], df_scales=df_scales_red,
                         verbose=False, random_state=0).run(labels=pipeline["labels"],
                                                            n_filter=5, n_jobs=1)
        assert len(df_feat) > 0


# ---------------------------------------------------------------------------
# Seam 2: load_dataset -> SequenceFeature -> CPP df_feat
# ---------------------------------------------------------------------------
class TestSequenceFeatureToCPP:
    """The parts/splits a SequenceFeature emits drive CPP and yield the df_feat schema."""

    def test_df_feat_schema(self, pipeline):
        df_feat = pipeline["df_feat"]
        assert DF_FEAT_COLS.issubset(df_feat.columns)
        assert len(df_feat) == _pipeline.N_FILTER
        # abs_auc is a discriminative magnitude in [0, 0.5].
        assert df_feat["abs_auc"].between(0, 0.5).all()

    def test_same_seed_same_df_feat(self, pipeline):
        # Reproducibility property across the parts->CPP seam.
        df_feat_a = _pipeline.build_df_feat(df_parts=pipeline["df_parts"],
                                            labels=pipeline["labels"],
                                            df_scales=pipeline["df_scales"])
        df_feat_b = _pipeline.build_df_feat(df_parts=pipeline["df_parts"],
                                            labels=pipeline["labels"],
                                            df_scales=pipeline["df_scales"])
        assert df_feat_a["feature"].to_list() == df_feat_b["feature"].to_list()

    def test_row_permutation_same_feature_set(self, pipeline):
        # Metamorphic: reordering samples must not change the SET of selected features.
        df_parts = pipeline["df_parts"]
        labels = np.asarray(pipeline["labels"])
        perm = np.argsort(labels, kind="stable")[::-1]  # deterministic reordering
        df_parts_perm = df_parts.iloc[perm].reset_index(drop=True)
        labels_perm = labels[perm].tolist()
        df_feat_perm = _pipeline.build_df_feat(df_parts=df_parts_perm, labels=labels_perm,
                                               df_scales=pipeline["df_scales"])
        assert set(df_feat_perm["feature"]) == set(pipeline["df_feat"]["feature"])

    def test_single_class_labels_rejected(self, pipeline):
        # Composition failure: CPP needs two groups to contrast; all-one-class must raise.
        labels_one = [1] * len(pipeline["labels"])
        with pytest.raises(ValueError):
            aa.CPP(df_parts=pipeline["df_parts"], df_scales=pipeline["df_scales"],
                   verbose=False).run(labels=labels_one, n_filter=5, n_jobs=1)


# ---------------------------------------------------------------------------
# Seam 3: CPP df_feat -> TreeModel.add_feat_importance -> CPPPlot.ranking
# ---------------------------------------------------------------------------
class TestCPPFeatImportanceToPlot:
    """A TreeModel injects feat_importance into df_feat, which CPPPlot then renders."""

    def test_ranking_consumes_importance_annotated_feat(self, pipeline):
        tm = aa.TreeModel(verbose=False, random_state=0).fit(
            pipeline["X"], labels=pipeline["labels"], use_rfe=False, n_cv=2, n_rounds=2)
        df_feat = tm.add_feat_importance(df_feat=pipeline["df_feat"])
        assert "feat_importance" in df_feat.columns
        fig, ax = aa.CPPPlot().ranking(df_feat=df_feat, n_top=5)
        assert fig is not None

    def test_plot_rejects_feat_without_importance(self, pipeline):
        # Composition failure: ranking needs the importance column the TreeModel adds.
        with pytest.raises(ValueError, match="feat_importance"):
            aa.CPPPlot().ranking(df_feat=pipeline["df_feat"], n_top=5)


# ---------------------------------------------------------------------------
# Seam 4: feature_matrix -> NumericalFeature.filter_correlation
# ---------------------------------------------------------------------------
class TestFeatureMatrixToNumericalFeature:
    """The X that SequenceFeature builds is consumable by NumericalFeature's filters."""

    def test_filter_correlation_mask_shape(self, pipeline):
        mask = aa.NumericalFeature().filter_correlation(pipeline["X"], max_cor=0.7)
        mask = np.asarray(mask)
        assert mask.dtype == bool
        assert mask.shape[0] == np.asarray(pipeline["X"]).shape[1]

    @settings(max_examples=5, deadline=None)
    @given(max_cor=some.floats(min_value=0.3, max_value=0.95))
    def test_mask_length_invariant(self, pipeline, max_cor):
        # Property: the boolean mask always aligns 1:1 with the feature axis of X.
        X = pipeline["X"]
        mask = np.asarray(aa.NumericalFeature().filter_correlation(X, max_cor=max_cor))
        assert mask.shape[0] == np.asarray(X).shape[1]
        assert mask.any()  # at least one feature survives


# ---------------------------------------------------------------------------
# Seam 5: CPPGrid.run -> per-config df_feat
# ---------------------------------------------------------------------------
class TestCPPGridSweep:
    """CPPGrid runs the full parts->splits->scales->CPP pipeline per configuration."""

    def test_grid_returns_one_feat_table_per_config(self, pipeline):
        cppg = aa.CPPGrid(df_seq=pipeline["df_seq"], labels=pipeline["labels"],
                          verbose=False, random_state=0, n_jobs=1)
        list_df_feat, df_params = cppg.run(params_cpp=dict(n_filter=[10, 15]),
                                           params_scales=_pipeline.small_scales())
        assert len(list_df_feat) == len(df_params) == 2
        for df_feat in list_df_feat:
            assert df_feat is not None and DF_FEAT_COLS.issubset(df_feat.columns)

    def test_n_filter_sweep_sizes(self, pipeline):
        # The two swept n_filter values produce feature tables of the matching sizes.
        cppg = aa.CPPGrid(df_seq=pipeline["df_seq"], labels=pipeline["labels"],
                          verbose=False, random_state=0, n_jobs=1)
        list_df_feat, _ = cppg.run(params_cpp=dict(n_filter=[8, 12]),
                                   params_scales=_pipeline.small_scales())
        sizes = sorted(len(df) for df in list_df_feat)
        assert sizes == [8, 12]
