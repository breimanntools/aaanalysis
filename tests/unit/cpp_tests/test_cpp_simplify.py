"""This is a script to test the CPP.simplify() method: interpretability-guided
scale swapping with three strategies (greedy / consolidate / swap_all) and a
configurable cross-validation gate (ml_model / ml_metric / ml_th / ml_cv).

The CV gate makes the call non-trivial, so fixtures are kept small and shared
module-wide. Positive cases use hypothesis property-based testing (the house
standard); negative cases pin the tailored validation errors. The CV-gate model
is seeded from the CPP constructor's random_state.
"""

import warnings

import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._simplify import _merged_scale_corr_

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


@pytest.fixture(scope="module")
def fitted():
    """A fitted CPP (seeded) + a rated feature set (interpretability-tiered scales)."""
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
            df_parts=df_parts,
            df_scales=df_scales,
            split_kws=split_kws,
            random_state=0,
            verbose=False,
        )
        df_feat = cpp.run(labels=labels, n_filter=10)
    return cpp, df_feat, labels


def _simplify(cpp, df_feat, labels, **kwargs):
    kwargs.setdefault("ml_cv", 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cpp.simplify(df_feat=df_feat, labels=labels, **kwargs)


class TestSimplify:
    """Normal cases — one parameter per test, explored with hypothesis."""

    @settings(max_examples=5, deadline=None)
    @given(max_interpret_grade=some.integers(min_value=1, max_value=10))
    def test_max_interpret_grade(self, fitted, max_interpret_grade):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, max_interpret_grade=max_interpret_grade)
        assert isinstance(out, pd.DataFrame) and 1 <= len(out) <= len(df_feat)

    @settings(max_examples=5, deadline=None)
    @given(min_cor=some.floats(min_value=0.5, max_value=1.0))
    def test_min_cor(self, fitted, min_cor):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, min_cor=min_cor)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=5, deadline=None)
    @given(ml_th=some.floats(min_value=0.0, max_value=0.2))
    def test_ml_th(self, fitted, ml_th):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, ml_th=ml_th)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(ml_cv=some.integers(min_value=2, max_value=4))
    def test_ml_cv(self, fitted, ml_cv):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, ml_cv=ml_cv)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(strategy=some.sampled_from(["greedy", "consolidate", "swap_all"]))
    def test_strategy(self, fitted, strategy):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, strategy=strategy)
        assert list(out.columns) == list(ut.LIST_COLS_FEAT) and 1 <= len(out) <= len(
            df_feat
        )

    @settings(max_examples=3, deadline=None)
    @given(ml_model=some.sampled_from(["svm", "rf", "log_reg"]))
    def test_ml_model_presets(self, fitted, ml_model):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, ml_model=ml_model)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(ml_metric=some.sampled_from(["balanced_accuracy", "accuracy", "f1"]))
    def test_ml_metric(self, fitted, ml_metric):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, ml_metric=ml_metric)
        assert isinstance(out, pd.DataFrame)

    @settings(max_examples=3, deadline=None)
    @given(on_unimprovable=some.sampled_from(["keep", "drop", "drop_if_perf_allows"]))
    def test_on_unimprovable(self, fitted, on_unimprovable):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, on_unimprovable=on_unimprovable)
        assert 1 <= len(out) <= len(df_feat)

    @settings(max_examples=2, deadline=None)
    @given(redundancy_tie_break=some.sampled_from(["interpretability", "performance"]))
    def test_redundancy_tie_break(self, fitted, redundancy_tie_break):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, redundancy_tie_break=redundancy_tie_break)
        assert isinstance(out, pd.DataFrame)

    def test_canonical_schema(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels)
        assert list(out.columns) == list(ut.LIST_COLS_FEAT)

    def test_return_details_tuple(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, return_details=True)
        assert isinstance(out, tuple) and len(out) == 2
        _, df_cand = out
        assert "candidate_scale" in df_cand.columns and "accepted" in df_cand.columns

    def test_ml_model_custom_instance(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, ml_model=SVC(kernel="linear"))
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

    def test_ml_cv_too_large(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="ml_cv"):
            cpp.simplify(df_feat=df_feat, labels=labels, ml_cv=999)

    def test_ml_cv_too_small(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="ml_cv"):
            cpp.simplify(df_feat=df_feat, labels=labels, ml_cv=1)

    def test_min_cor_out_of_range(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="min_cor"):
            cpp.simplify(df_feat=df_feat, labels=labels, min_cor=1.5)

    def test_max_interpret_grade_out_of_range(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="max_interpret_grade"):
            cpp.simplify(df_feat=df_feat, labels=labels, max_interpret_grade=99)

    def test_ml_metric_not_str(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError, match="ml_metric"):
            cpp.simplify(df_feat=df_feat, labels=labels, ml_metric=123)

    def test_labels_wrong_length(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(ValueError):
            cpp.simplify(df_feat=df_feat, labels=labels[:5])


class TestSimplifyComplex:
    """Combinations and edge interactions."""

    def test_swapped_features_improve_interpretability(self, fitted):
        cpp, df_feat, labels = fitted
        _, df_cand = _simplify(cpp, df_feat, labels, return_details=True)
        accepted = df_cand[df_cand["accepted"]]
        if len(accepted):
            assert (
                accepted["interpretability_cand"] < accepted["interpretability_orig"]
            ).all()

    def test_redundancy_does_not_increase_count(self, fitted):
        cpp, df_feat, labels = fitted
        for strategy in ["greedy", "consolidate", "swap_all"]:
            out = _simplify(cpp, df_feat, labels, strategy=strategy)
            assert len(out) <= len(df_feat)

    def test_consolidate_reduces_or_equal_subcats(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, strategy="consolidate", ml_th=0.1)
        assert out["subcategory"].nunique() <= df_feat["subcategory"].nunique()

    def test_swap_all_drops_unimprovable(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp,
            df_feat,
            labels,
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
            cpp, df_feat, labels, return_details=True, max_std_test=0.2
        )
        accepted = df_cand[df_cand["accepted"]]
        if len(accepted):
            assert (accepted["std_test"] <= 0.2).all()

    def test_reproducible_same_seed(self, fitted):
        cpp, df_feat, labels = fitted
        out1 = _simplify(cpp, df_feat, labels)
        out2 = _simplify(cpp, df_feat, labels)
        pd.testing.assert_frame_equal(
            out1.reset_index(drop=True), out2.reset_index(drop=True)
        )

    def test_drop_if_perf_allows_can_drop(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp,
            df_feat,
            labels,
            min_cor=1.0,
            on_unimprovable="drop_if_perf_allows",
            ml_th=1.0,
        )
        assert 1 <= len(out) < len(df_feat)

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
        with pytest.warns(RuntimeWarning, match="no AAontology-graded"):
            out = cpp.simplify(df_feat=df_fake, labels=labels, ml_cv=3)
        assert len(out) == len(df_fake)

    def test_already_good_enough_no_warning(self, fitted):
        """Case 2: graded features all at/under the grade cut -> silent no-op, NOT the Case-1 warning."""
        import warnings as _w
        cpp, df_feat, labels = fitted
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)  # any RuntimeWarning would fail the test
            # grade 10 is the worst, so no feature is graded *worse* -> no targets (Case 2)
            out = cpp.simplify(df_feat=df_feat, labels=labels, max_interpret_grade=10, ml_cv=3)
        assert len(out) == len(df_feat)  # returned unchanged, silently (verbose=False)

    def test_drop_never_removes_last_feature(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, min_cor=1.0, on_unimprovable="drop")
        assert len(out) >= 1

    def test_merged_corr_includes_custom_scale(self, fitted):
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

    def test_originals_protected_dropped_only_if_swapped(self, fitted):
        # Redundancy reduction protects originals: an original feature disappears
        # only if it was itself swapped (never silently dropped).
        cpp, df_feat, labels = fitted
        for strategy in ["greedy", "consolidate", "swap_all"]:
            out, df_cand = _simplify(
                cpp, df_feat, labels, strategy=strategy, return_details=True
            )
            originals = set(df_feat[ut.COL_FEATURE])
            result = set(out[ut.COL_FEATURE])
            swapped_origs = set(df_cand[df_cand["accepted"]]["feature"])
            missing = originals - result
            assert missing <= swapped_origs, (
                f"{strategy}: originals dropped without being swapped: "
                f"{missing - swapped_origs}"
            )

    def test_top_tier_features_never_swapped_or_dropped(self, fitted):
        # grade==1 (best tier) features cannot be improved and are originals, so
        # they must survive every strategy unchanged -- the alpha-helix regression.
        from aaanalysis.feature_engineering._backend.cpp._simplify import (
            _load_candidate_pool_,
        )

        cpp, df_feat, labels = fitted
        _, _, _, dict_interp, _ = _load_candidate_pool_()
        top_tier = [
            f
            for f in df_feat[ut.COL_FEATURE]
            if dict_interp.get(ut.split_feat_id(feat_id=f)[2]) == 1
        ]
        assert top_tier, "fixture should contain grade==1 features"
        for strategy in ["greedy", "consolidate", "swap_all"]:
            out = _simplify(cpp, df_feat, labels, strategy=strategy)
            kept = set(out[ut.COL_FEATURE])
            assert all(
                f in kept for f in top_tier
            ), f"{strategy}: a top-tier (grade==1) feature was swapped or dropped"
