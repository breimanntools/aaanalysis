"""This script tests the TreeModel.add_feat_importance() method."""
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
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
