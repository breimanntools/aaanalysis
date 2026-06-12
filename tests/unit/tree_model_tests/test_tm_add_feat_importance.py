"""This script tests the TreeModel.add_feat_importance() method."""
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Constants for testing
COL_FEAT_IMPORT = "feat_importance"
COL_FEAT_IMPORT_STD = "feat_importance_std"


def create_df_feat(drop=True):
    df_feat = aa.load_features(name="DOM_GSEC").head(50)
    if drop:
        df_feat = df_feat[[x for x in list(df_feat) if x not in [COL_FEAT_IMPORT, COL_FEAT_IMPORT_STD]]]
    return df_feat


# Create valid X
df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
_df_feat = aa.load_features().head(50)
valid_labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=_df_feat["feature"], df_parts=df_parts)

N_ROUNDS = 2
ARGS = dict(use_rfe=False, n_cv=2, n_rounds=N_ROUNDS)



# Main Test Classes
class TestAddFeatImportance:
    """Test the add_feat_importance method with positive and negative test cases for each parameter."""

    # Positive Tests
    def test_df_feat_valid(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat()
        result = tm.add_feat_importance(df_feat=df)
        assert isinstance(result, pd.DataFrame)
        assert COL_FEAT_IMPORT in result.columns
        assert COL_FEAT_IMPORT_STD in result.columns

    def test_drop_valid(self):
        for drop in [True, False]:
            tm = aa.TreeModel()
            tm.fit(valid_X, labels=valid_labels, **ARGS)
            df = create_df_feat(drop=not drop)
            result = tm.add_feat_importance(df_feat=df, drop=drop)
            assert isinstance(result, pd.DataFrame)
            assert COL_FEAT_IMPORT in result.columns
            assert COL_FEAT_IMPORT_STD in result.columns

    # Negative Tests
    def test_df_feat_invalid(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat(drop=False)
        # No dropping before
        with pytest.raises(ValueError):
            tm.add_feat_importance(df_feat=df)
        # Wrong input
        with pytest.raises(ValueError):
            tm.add_feat_importance(df_feat="")
        with pytest.raises(ValueError):
            tm.add_feat_importance(df_feat=pd.DataFrame())
        df = create_df_feat(drop=True)
        with pytest.raises(ValueError):
            df.columns = [x + "invalid" for x in list(df)]
            tm.add_feat_importance(df_feat=df)

    def test_drop_invalid(self):
       tm = aa.TreeModel()
       tm.fit(valid_X, labels=valid_labels, **ARGS)
       df = create_df_feat()
       with pytest.raises(ValueError):
           tm.add_feat_importance(df_feat=df, drop="asdf")
       with pytest.raises(ValueError):
           tm.add_feat_importance(df_feat=df, drop=[])

    def test_sort_valid(self):
        for sort in [True, False]:
            tm = aa.TreeModel()
            tm.fit(valid_X, labels=valid_labels, **ARGS)
            df = create_df_feat()
            result = tm.add_feat_importance(df_feat=df, sort=sort)
            assert isinstance(result, pd.DataFrame)
            assert COL_FEAT_IMPORT in result.columns
            assert COL_FEAT_IMPORT_STD in result.columns
            # Same features are retained regardless of order
            assert set(result["feature"]) == set(df["feature"])

    def test_sort_invalid(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat()
        for invalid_sort in ["asdf", [], 1, None]:
            with pytest.raises(ValueError):
                tm.add_feat_importance(df_feat=df, sort=invalid_sort)


class TestAddFeatImportanceComplex:
    """Test the add_feat_importance method with complex cases combining multiple parameters."""

    # Testing with valid DataFrame and varying drop parameter
    def test_valid_df_with_varying_drop(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        for drop in [True, False]:
            df = create_df_feat(drop=not drop)
            result = tm.add_feat_importance(df_feat=df, drop=drop)
            assert isinstance(result, pd.DataFrame)
            assert COL_FEAT_IMPORT in result.columns
            assert COL_FEAT_IMPORT_STD in result.columns

    # Testing with invalid DataFrame types
    def test_invalid_df_types(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        invalid_dfs = [None, "", [], 123]
        for df in invalid_dfs:
            with pytest.raises(ValueError):
                tm.add_feat_importance(df_feat=df)

    # sort=True yields a strictly non-increasing feat_importance with a reset index
    def test_sort_descending_and_reset_index(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat()
        result = tm.add_feat_importance(df_feat=df, sort=True)
        importances = result[COL_FEAT_IMPORT].to_list()
        assert importances == sorted(importances, reverse=True)
        # Index is reset to a clean 0..n-1 range
        assert list(result.index) == list(range(len(result)))
        # Top row carries the maximum importance
        assert result[COL_FEAT_IMPORT].iloc[0] == max(importances)

    # sort=False preserves the input feature order
    def test_sort_false_preserves_order(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat()
        result = tm.add_feat_importance(df_feat=df, sort=False)
        assert result["feature"].to_list() == df["feature"].to_list()

    # sort combines with drop (drop existing columns, then re-add and sort)
    def test_sort_with_drop(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat(drop=False)
        result = tm.add_feat_importance(df_feat=df, drop=True, sort=True)
        importances = result[COL_FEAT_IMPORT].to_list()
        assert importances == sorted(importances, reverse=True)
        assert set(result["feature"]) == set(df["feature"])

    # sort does not change which features/importances are present, only their order
    def test_sort_is_order_only(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat()
        unsorted = tm.add_feat_importance(df_feat=df, sort=False)
        result = tm.add_feat_importance(df_feat=df, sort=True)
        assert dict(zip(unsorted["feature"], unsorted[COL_FEAT_IMPORT])) == \
               dict(zip(result["feature"], result[COL_FEAT_IMPORT]))

    # Invalid sort values raise ValueError even when combined with a valid drop
    def test_sort_invalid_with_drop(self):
        tm = aa.TreeModel()
        tm.fit(valid_X, labels=valid_labels, **ARGS)
        df = create_df_feat()
        for invalid_sort in ["yes", {}, 0]:
            with pytest.raises(ValueError):
                tm.add_feat_importance(df_feat=df, drop=False, sort=invalid_sort)
