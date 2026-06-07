"""This is a script to test the TreeModel.select_features() method."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")
# Note: @settings(deadline=None) below is intentional: TreeModel.select_features fits a forest
# per example; fit time is variable and is not the property under test (#83).


# Constants for testing
COL_FEAT_IMPORT = "feat_importance"
N_FEAT = 40

# Create valid X, labels, and a fitted TreeModel (shared, read-only across tests)
df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
valid_labels = df_seq["label"].to_list()
_df_feat = aa.load_features(name="DOM_GSEC").head(N_FEAT)
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=_df_feat["feature"], df_parts=df_parts)


def fitted_tm(use_rfe=False):
    """Return a freshly fitted TreeModel (seeded for reproducibility)."""
    tm = aa.TreeModel(random_state=0)
    if use_rfe:
        tm.fit(valid_X, labels=valid_labels, use_rfe=True, n_cv=2, n_rounds=3,
               n_feat_min=5, n_feat_max=20, step=5)
    else:
        tm.fit(valid_X, labels=valid_labels, use_rfe=False, n_cv=2, n_rounds=2)
    return tm


def get_df_feat():
    return aa.load_features(name="DOM_GSEC").head(N_FEAT)


# Module-level fitted models (reused where the fit itself is not under test)
TM = fitted_tm(use_rfe=False)
TM_RFE = fitted_tm(use_rfe=True)


# Main Test Classes
class TestSelectFeatures:
    """Normal and error cases, one parameter per test method."""

    # Positive Tests
    def test_returns_dataframe(self):
        result = TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=10)
        assert isinstance(result, pd.DataFrame)

    @given(n=st.integers(min_value=1, max_value=N_FEAT))
    @settings(max_examples=15, deadline=None)
    def test_top_k_returns_n_rows(self, n):
        result = TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=n)
        assert len(result) == n

    def test_top_k_keeps_highest_importance(self):
        df = get_df_feat()
        result = TM.select_features(df_feat=df, strategy="top_k", param=5)
        # The kept features must be exactly those at the 5 highest-importance positions
        expected_idx = np.argsort(-TM.feat_importance, kind="stable")[:5]
        expected_features = set(df.iloc[expected_idx]["feature"])
        assert set(result["feature"]) == expected_features

    @given(frac=st.floats(min_value=0.1, max_value=0.9))
    @settings(max_examples=10, deadline=None)
    def test_threshold_keeps_above_value(self, frac):
        thr = float(np.quantile(TM.feat_importance, frac))
        result = TM.select_features(df_feat=get_df_feat(), strategy="threshold", param=thr)
        n_expected = int(np.sum(TM.feat_importance >= thr))
        assert len(result) == n_expected

    def test_threshold_zero_keeps_all(self):
        result = TM.select_features(df_feat=get_df_feat(), strategy="threshold", param=0)
        assert len(result) == N_FEAT

    @given(min_freq=st.floats(min_value=0.1, max_value=1.0))
    @settings(max_examples=10, deadline=None)
    def test_frequency_rfe_returns_subset(self, min_freq):
        result = TM_RFE.select_features(df_feat=get_df_feat(), strategy="frequency", param=min_freq)
        assert 0 < len(result) <= N_FEAT

    def test_frequency_without_rfe_warns_and_keeps_all(self):
        with pytest.warns(RuntimeWarning):
            result = TM.select_features(df_feat=get_df_feat(), strategy="frequency", param=0.5)
        assert len(result) == N_FEAT

    def test_index_is_reset(self):
        result = TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=7)
        assert list(result.index) == list(range(len(result)))

    def test_columns_preserved(self):
        df = get_df_feat()
        result = TM.select_features(df_feat=df, strategy="top_k", param=10)
        assert list(result.columns) == list(df.columns)

    def test_reproducible_same_seed(self):
        a = fitted_tm().select_features(df_feat=get_df_feat(), strategy="top_k", param=12)
        b = fitted_tm().select_features(df_feat=get_df_feat(), strategy="top_k", param=12)
        pd.testing.assert_frame_equal(a, b)

    # Negative Tests
    def test_strategy_invalid(self):
        for strategy in ["bogus", "rfe", "", 123, None]:
            with pytest.raises(ValueError):
                TM.select_features(df_feat=get_df_feat(), strategy=strategy, param=10)

    def test_top_k_param_too_small(self):
        for param in [0, -1]:
            with pytest.raises(ValueError):
                TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=param)

    def test_top_k_param_too_large(self):
        with pytest.raises(ValueError):
            TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=N_FEAT + 1)

    def test_top_k_param_not_int(self):
        for param in [5.5, "5", None]:
            with pytest.raises(ValueError):
                TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=param)

    def test_threshold_param_negative(self):
        with pytest.raises(ValueError):
            TM.select_features(df_feat=get_df_feat(), strategy="threshold", param=-0.5)

    def test_threshold_param_too_high_selects_nothing(self):
        # Above the max importance -> empty selection -> backend invariant raises
        with pytest.raises(ValueError):
            TM.select_features(df_feat=get_df_feat(), strategy="threshold",
                               param=float(TM.feat_importance.max()) + 1.0)

    def test_frequency_param_out_of_range(self):
        for param in [0, -0.1, 1.5, 2]:
            with pytest.raises(ValueError):
                TM_RFE.select_features(df_feat=get_df_feat(), strategy="frequency", param=param)

    def test_param_dict_rejected(self):
        # dict is reserved for future multi-knob strategies; the three current ones reject it
        for strategy in ["top_k", "threshold", "frequency"]:
            with pytest.raises(ValueError):
                TM.select_features(df_feat=get_df_feat(), strategy=strategy, param={"n": 5})

    def test_unfitted_raises(self):
        tm = aa.TreeModel()
        with pytest.raises(ValueError):
            tm.select_features(df_feat=get_df_feat(), strategy="top_k", param=5)

    def test_df_feat_invalid_type(self):
        for df in [None, "", [], 123, pd.DataFrame()]:
            with pytest.raises(ValueError):
                TM.select_features(df_feat=df, strategy="top_k", param=5)


class TestSelectFeaturesComplex:
    """Combinations and edge interactions."""

    # Positive Tests
    def test_output_feeds_round_trip_columns(self):
        df = get_df_feat()
        result = TM.select_features(df_feat=df, strategy="threshold", param=0)
        # threshold=0 keeps all -> identical content, reset index
        pd.testing.assert_frame_equal(result, df.reset_index(drop=True))

    def test_top_k_one_feature(self):
        result = TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=1)
        assert len(result) == 1
        top_idx = int(np.argsort(-TM.feat_importance, kind="stable")[0])
        assert result.iloc[0]["feature"] == get_df_feat().iloc[top_idx]["feature"]

    def test_top_k_all_features(self):
        result = TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=N_FEAT)
        assert len(result) == N_FEAT

    def test_frequency_one_keeps_persistent(self):
        # min_freq=1.0 keeps only features selected in *every* RFE round (subset of any-round)
        strict = TM_RFE.select_features(df_feat=get_df_feat(), strategy="frequency", param=1.0)
        loose = TM_RFE.select_features(df_feat=get_df_feat(), strategy="frequency", param=0.1)
        assert len(strict) <= len(loose)

    def test_strategies_can_give_different_sets(self):
        a = TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=10)
        b = TM_RFE.select_features(df_feat=get_df_feat(), strategy="frequency", param=0.5)
        assert list(a["feature"]) != list(b["feature"]) or len(a) != len(b)

    # Negative Tests
    def test_df_feat_length_mismatch(self):
        df_short = get_df_feat().head(N_FEAT - 5)
        with pytest.raises(ValueError):
            TM.select_features(df_feat=df_short, strategy="top_k", param=5)

    def test_df_feat_length_too_long(self):
        df_long = aa.load_features(name="DOM_GSEC").head(N_FEAT + 10)
        with pytest.raises(ValueError):
            TM.select_features(df_feat=df_long, strategy="top_k", param=5)

    def test_strategy_and_param_both_invalid(self):
        with pytest.raises(ValueError):
            TM.select_features(df_feat=get_df_feat(), strategy="nope", param=-1)

    def test_top_k_param_none_with_valid_strategy(self):
        with pytest.raises(ValueError):
            TM.select_features(df_feat=get_df_feat(), strategy="top_k", param=None)

    def test_frequency_param_none(self):
        with pytest.raises(ValueError):
            TM_RFE.select_features(df_feat=get_df_feat(), strategy="frequency", param=None)
