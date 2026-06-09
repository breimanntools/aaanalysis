"""This is a script to test the CPP.simplify() method: interpretability-guided
scale swapping with three strategies (greedy / consolidate / swap_all) and a
configurable cross-validation-gate model (ml_model).

The CV gate makes the call non-trivial, so fixtures are kept small and shared
module-wide. Positive cases use hypothesis property-based testing (the house
standard); negative cases pin the tailored validation errors.
"""

import warnings

import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._simplify import (
    _build_base_matrix_,
    _merged_scale_corr_,
)

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


@pytest.fixture(scope="module")
def fitted():
    """A fitted CPP + a rated feature set (interpretability-tiered scales)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
        labels = df_seq["label"].to_list()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq)
        split_kws = sf.get_split_kws(
            n_split_min=1, n_split_max=2, split_types=["Segment"]
        )
        df_scales = aa.load_scales(top_explain_n=20)
        cpp = aa.CPP(
            df_parts=df_parts, df_scales=df_scales, split_kws=split_kws, verbose=False
        )
        df_feat = cpp.run(labels=labels, n_filter=10)
    return cpp, df_feat, labels


def _simplify(cpp, df_feat, labels, **kwargs):
    kwargs.setdefault("n_cv", 3)
    kwargs.setdefault("random_state", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cpp.simplify(df_feat=df_feat, labels=labels, **kwargs)


class TestSimplify:
    """Normal cases — one parameter per test, explored with hypothesis."""

    @settings(max_examples=5, deadline=None)
    @given(top_n=some.integers(min_value=1, max_value=6))
    def test_top_n(self, fitted, top_n):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=top_n)
        assert isinstance(out, pd.DataFrame) and 1 <= len(out) <= len(df_feat)

    @settings(max_examples=5, deadline=None)
    @given(max_interpretability=some.integers(min_value=1, max_value=10))
    def test_max_interpretability(self, fitted, max_interpretability):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, max_interpretability=max_interpretability)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=5, deadline=None)
    @given(min_cor=some.floats(min_value=0.5, max_value=1.0))
    def test_min_cor(self, fitted, min_cor):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, min_cor=min_cor)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=5, deadline=None)
    @given(tol=some.floats(min_value=0.0, max_value=0.2))
    def test_tol(self, fitted, tol):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, tol=tol)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(n_cv=some.integers(min_value=2, max_value=4))
    def test_n_cv(self, fitted, n_cv):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=2, n_cv=n_cv)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(strategy=some.sampled_from(["greedy", "consolidate", "swap_all"]))
    def test_strategy(self, fitted, strategy):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=5, strategy=strategy)
        assert list(out.columns) == list(ut.LIST_COLS_FEAT) and 1 <= len(out) <= len(
            df_feat
        )

    @settings(max_examples=3, deadline=None)
    @given(ml_model=some.sampled_from(["svm", "rf", "log_reg"]))
    def test_ml_model_presets(self, fitted, ml_model):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, ml_model=ml_model)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(metric=some.sampled_from(["balanced_accuracy", "accuracy", "f1"]))
    def test_metric(self, fitted, metric):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=2, metric=metric)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(on_unimprovable=some.sampled_from(["keep", "drop", "drop_if_perf_allows"]))
    def test_on_unimprovable(self, fitted, on_unimprovable):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, on_unimprovable=on_unimprovable)
        assert 1 <= len(out) <= len(df_feat)

    @settings(max_examples=2, deadline=None)
    @given(redundancy_tie_break=some.sampled_from(["interpretability", "performance"]))
    def test_redundancy_tie_break(self, fitted, redundancy_tie_break):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, top_n=5, redundancy_tie_break=redundancy_tie_break
        )
        assert isinstance(out, pd.DataFrame)

    def test_canonical_schema(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3)
        assert list(out.columns) == list(ut.LIST_COLS_FEAT)

    def test_return_details_tuple(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, return_details=True)
        assert isinstance(out, tuple) and len(out) == 2
        _, df_cand = out
        assert "candidate_scale" in df_cand.columns and "accepted" in df_cand.columns

    def test_ml_model_custom_instance(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, ml_model=SVC(kernel="linear"))
        assert isinstance(out, pd.DataFrame)

    def test_X_provided(self, fitted):
        cpp, df_feat, labels = fitted
        X = _build_base_matrix_(
            df_feat=df_feat, df_parts=cpp.df_parts, df_scales_self=cpp.df_scales
        )
        out = _simplify(cpp, df_feat, labels, top_n=2, X=X)
        assert isinstance(out, pd.DataFrame)

    # --- negative ---
    def test_invalid_strategy(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="strategy"):
            cpp.simplify(df_feat=df_feat, labels=labels, strategy="bogus")

    def test_invalid_on_unimprovable(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="on_unimprovable"):
            cpp.simplify(df_feat=df_feat, labels=labels, on_unimprovable="bogus")

    def test_invalid_tie_break(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="redundancy_tie_break"):
            cpp.simplify(df_feat=df_feat, labels=labels, redundancy_tie_break="bogus")

    def test_invalid_ml_model_string(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="ml_model"):
            cpp.simplify(df_feat=df_feat, labels=labels, ml_model="bogus")

    def test_invalid_ml_model_object(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="ml_model"):
            cpp.simplify(df_feat=df_feat, labels=labels, ml_model=42)

    def test_max_interpretability_and_top_n_mutually_exclusive(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="mutually exclusive"):
            cpp.simplify(
                df_feat=df_feat, labels=labels, max_interpretability=3, top_n=5
            )

    def test_n_cv_too_large(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="n_cv"):
            cpp.simplify(df_feat=df_feat, labels=labels, n_cv=999)

    def test_n_cv_too_small(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="n_cv"):
            cpp.simplify(df_feat=df_feat, labels=labels, n_cv=1)

    def test_min_cor_out_of_range(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="min_cor"):
            cpp.simplify(df_feat=df_feat, labels=labels, min_cor=1.5)

    def test_max_interpretability_out_of_range(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="max_interpretability"):
            cpp.simplify(df_feat=df_feat, labels=labels, max_interpretability=99)

    def test_top_n_below_one(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="top_n"):
            cpp.simplify(df_feat=df_feat, labels=labels, top_n=0)

    def test_metric_not_str(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="metric"):
            cpp.simplify(df_feat=df_feat, labels=labels, metric=123)

    def test_labels_wrong_length(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError):
            cpp.simplify(df_feat=df_feat, labels=labels[:5])


class TestSimplifyComplex:
    """Combinations and edge interactions."""

    def test_swapped_features_improve_interpretability(self, fitted):
        cpp, df_feat, labels = fitted
        _, df_cand = _simplify(cpp, df_feat, labels, top_n=5, return_details=True)
        accepted = df_cand[df_cand["accepted"]]
        if len(accepted):
            assert (
                accepted["interpretability_cand"] < accepted["interpretability_orig"]
            ).all()

    def test_redundancy_does_not_increase_count(self, fitted):
        cpp, df_feat, labels = fitted
        for strategy in ["greedy", "consolidate", "swap_all"]:
            out = _simplify(cpp, df_feat, labels, top_n=5, strategy=strategy)
            assert len(out) <= len(df_feat)

    def test_consolidate_reduces_or_equal_subcats(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=8, strategy="consolidate", tol=0.1)
        assert out["subcategory"].nunique() <= df_feat["subcategory"].nunique()

    def test_swap_all_drops_unimprovable(self, fitted):
        cpp, df_feat, labels = fitted
        # swap_all does no CV; on_unimprovable='drop' still prunes unswappable targets.
        out = _simplify(
            cpp,
            df_feat,
            labels,
            top_n=6,
            strategy="swap_all",
            min_cor=1.0,
            on_unimprovable="drop",
        )
        assert list(out.columns) == list(ut.LIST_COLS_FEAT) and 1 <= len(out) <= len(
            df_feat
        )

    def test_accepted_swaps_pass_max_std_test(self, fitted):
        cpp, df_feat, labels = fitted
        _, df_cand = _simplify(
            cpp, df_feat, labels, top_n=5, return_details=True, max_std_test=0.2
        )
        accepted = df_cand[df_cand["accepted"]]
        if len(accepted):
            assert (accepted["std_test"] <= 0.2).all()

    def test_reproducible_same_seed(self, fitted):
        cpp, df_feat, labels = fitted
        out1 = _simplify(cpp, df_feat, labels, top_n=5, random_state=0)
        out2 = _simplify(cpp, df_feat, labels, top_n=5, random_state=0)
        pd.testing.assert_frame_equal(
            out1.reset_index(drop=True), out2.reset_index(drop=True)
        )

    def test_drop_if_perf_allows_can_drop(self, fitted):
        cpp, df_feat, labels = fitted
        # min_cor=1.0 blocks swaps; generous tol lets the perf-allows drop accept.
        out = _simplify(
            cpp,
            df_feat,
            labels,
            top_n=3,
            min_cor=1.0,
            on_unimprovable="drop_if_perf_allows",
            tol=1.0,
        )
        assert 1 <= len(out) < len(df_feat)

    def test_consolidate_drop_if_perf_allows(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp,
            df_feat,
            labels,
            top_n=3,
            strategy="consolidate",
            min_cor=1.0,
            on_unimprovable="drop_if_perf_allows",
            tol=1.0,
        )
        assert 1 <= len(out) <= len(df_feat)

    def test_no_rated_features_warns_and_returns_unchanged(self, fitted):
        cpp, df_feat, labels = fitted
        df_fake = df_feat.copy()
        df_fake[ut.COL_FEATURE] = [
            ut.join_feat_id(
                part=ut.split_feat_id(feat_id=f)[0],
                split=ut.split_feat_id(feat_id=f)[1],
                scale_id="ZZZ_FAKE",
            )
            for f in df_fake[ut.COL_FEATURE]
        ]
        with pytest.warns(RuntimeWarning, match="no AAontology-rated"):
            out = cpp.simplify(df_feat=df_fake, labels=labels, n_cv=3, random_state=0)
        assert len(out) == len(df_fake)

    def test_drop_never_removes_last_feature(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, min_cor=1.0, on_unimprovable="drop")
        assert len(out) >= 1

    def test_merged_corr_includes_custom_scale(self, fitted):
        # _merged_scale_corr_ must cover a scale present only in df_scales_self.
        pool = aa.load_scales()
        df_scales_self = pool.iloc[:, :3].copy()
        df_scales_self["CUSTOM_X"] = pool.iloc[:, 10].to_numpy()
        df_feat = pd.DataFrame(
            {
                ut.COL_FEATURE: [
                    ut.join_feat_id(
                        part="TMD", split="Segment(1,1)", scale_id=pool.columns[0]
                    ),
                    ut.join_feat_id(
                        part="TMD", split="Segment(1,1)", scale_id="CUSTOM_X"
                    ),
                ]
            }
        )
        df_cor = _merged_scale_corr_(
            df_feat=df_feat, df_scales_pool=pool, df_scales_self=df_scales_self
        )
        assert "CUSTOM_X" in df_cor.columns and pool.columns[0] in df_cor.columns
