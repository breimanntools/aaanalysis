"""This is a script to test the CPP.simplify() method: interpretability-guided
scale swapping (issue: new simplify feature).

simplify swaps each feature's scale for a more interpretable correlated scale
(keeping PART-SPLIT), recomputes its stats, accepts a swap only behind CPP
filtering + a random-forest cross-validation gate, then redundancy-reduces the
set. The RF+CV gate makes the call non-trivial, so fixtures are kept small and
shared module-wide.
"""

import warnings

import pandas as pd
import pytest
from hypothesis import settings

import aaanalysis as aa
import aaanalysis.utils as ut

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
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws,
                     verbose=False)
        df_feat = cpp.run(labels=labels, n_filter=10)
    return cpp, df_feat, labels


def _simplify(cpp, df_feat, labels, **kwargs):
    kwargs.setdefault("n_cv", 3)
    kwargs.setdefault("random_state", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cpp.simplify(df_feat=df_feat, labels=labels, **kwargs)


class TestSimplify:
    """Normal cases — one parameter per test."""

    def test_default_returns_df_feat(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels)
        assert isinstance(out, pd.DataFrame) and len(out) >= 1

    def test_canonical_schema(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3)
        assert list(out.columns) == list(ut.LIST_COLS_FEAT)

    def test_top_n(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=2)
        assert len(out) <= len(df_feat)

    def test_max_interpretability(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, max_interpretability=3)
        assert isinstance(out, pd.DataFrame)

    def test_min_cor_high_limits_swaps(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, min_cor=0.99)
        assert isinstance(out, pd.DataFrame)

    def test_tol(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, tol=0.1)
        assert isinstance(out, pd.DataFrame)

    def test_n_cv(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=2, n_cv=4)
        assert isinstance(out, pd.DataFrame)

    def test_on_unimprovable_keep(self, fitted):
        cpp, df_feat, labels = fitted
        keep = _simplify(
            cpp, df_feat, labels, top_n=3, min_cor=1.0, on_unimprovable="keep"
        )
        drop = _simplify(
            cpp, df_feat, labels, top_n=3, min_cor=1.0, on_unimprovable="drop"
        )
        # min_cor=1.0 blocks all swaps; 'keep' retains at least as many as 'drop'.
        assert 1 <= len(drop) <= len(keep) <= len(df_feat)

    def test_on_unimprovable_drop(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, top_n=3, min_cor=0.99, on_unimprovable="drop"
        )
        assert len(out) <= len(df_feat)

    def test_on_unimprovable_drop_if_perf(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp,
            df_feat,
            labels,
            top_n=3,
            min_cor=0.99,
            on_unimprovable="drop_if_perf_allows",
        )
        assert isinstance(out, pd.DataFrame) and len(out) >= 1

    def test_redundancy_tie_break_performance(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(
            cpp, df_feat, labels, top_n=3, redundancy_tie_break="performance"
        )
        assert isinstance(out, pd.DataFrame)

    def test_metric_accuracy(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=2, metric="accuracy")
        assert isinstance(out, pd.DataFrame)

    def test_return_details_tuple(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=3, return_details=True)
        assert isinstance(out, tuple) and len(out) == 2
        df_out, df_cand = out
        assert "candidate_scale" in df_cand.columns and "accepted" in df_cand.columns

    def test_X_provided(self, fitted):
        cpp, df_feat, labels = fitted
        from aaanalysis.feature_engineering._backend.cpp._simplify import (
            _build_base_matrix_,
        )

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
        out, df_cand = _simplify(cpp, df_feat, labels, top_n=5, return_details=True)
        accepted = df_cand[df_cand["accepted"]]
        if len(accepted):  # every accepted swap must go to a strictly better rating
            assert (
                accepted["interpretability_cand"] < accepted["interpretability_orig"]
            ).all()

    def test_redundancy_does_not_increase_count(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=5)
        assert len(out) <= len(df_feat)

    def test_accepted_swaps_pass_max_std_test(self, fitted):
        cpp, df_feat, labels = fitted
        out, df_cand = _simplify(
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

    def test_consolidate_strategy_not_implemented(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(NotImplementedError, match="staged"):
            cpp.simplify(
                df_feat=df_feat,
                labels=labels,
                strategy="consolidate",
                top_n=2,
                n_cv=3,
                random_state=0,
            )

    def test_swap_all_strategy_not_implemented(self, fitted):
        cpp, df_feat, labels = fitted
        with pytest.raises(NotImplementedError, match="staged"):
            cpp.simplify(
                df_feat=df_feat,
                labels=labels,
                strategy="swap_all",
                top_n=2,
                n_cv=3,
                random_state=0,
            )

    def test_details_records_rejected_and_accepted(self, fitted):
        cpp, df_feat, labels = fitted
        _, df_cand = _simplify(cpp, df_feat, labels, top_n=5, return_details=True)
        assert set(df_cand["accepted"].unique()).issubset({True, False})

    def test_output_scales_in_pool_or_original(self, fitted):
        cpp, df_feat, labels = fitted
        out = _simplify(cpp, df_feat, labels, top_n=5)
        # every output feature id is a valid PART-SPLIT-SCALE
        for f in out[ut.COL_FEATURE]:
            part, split, scale = ut.split_feat_id(feat_id=f)
            assert part and split and scale

    def test_drop_never_removes_last_feature(self, fitted):
        cpp, df_feat, labels = fitted
        # Force everything unimprovable (min_cor=1.0) + drop → must keep >= 1 feature
        out = _simplify(cpp, df_feat, labels, min_cor=1.0, on_unimprovable="drop")
        assert len(out) >= 1
