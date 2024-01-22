"""This script tests the ShapExplainer.add_feat_impact method."""
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import aaanalysis as aa

aa.options["verbose"] = False

def create_df_feat(drop=True):
    df_feat = aa.load_features(name="DOM_GSEC").head(50)
    if drop:
        df_feat = df_feat[[x for x in list(df_feat) if "FEAT_IMPACT" not in x]]
    return df_feat


def create_shap_values(n_samples, n_features):
    return np.random.rand(n_samples, n_features)

# Create valid X for testing
df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
_df_feat = aa.load_features().head(50)
valid_labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
valid_X = sf.feature_matrix(features=_df_feat["feature"], df_parts=df_parts)

valid_shap_values = aa.ShapExplainer().fit(valid_X, labels=valid_labels)

N_ROUNDS = 2
ARGS = dict(n_rounds=N_ROUNDS)


class TestAddFeatImpact:
    """Test the add_feat_impact method with positive test cases for each parameter."""

    # Positive tests
    def test_df_feat_valid(self):
        se = aa.ShapExplainer()
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)
        assert sum(["feat_impact" in x for x in list(df_feat)]) == len(df_seq)

    def test_drop_valid(self):
        for drop in [True, False]:
            se = aa.ShapExplainer()
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat(drop=not drop)
            df_feat = se.add_feat_impact(df_feat=df_feat, drop=drop)
            assert isinstance(df_feat, pd.DataFrame)
            assert sum(["feat_impact" in x for x in list(df_feat)]) == len(df_seq)

    def test_pos_valid(self):
        for pos in range(len(df_seq)):
            se = aa.ShapExplainer()
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, pos=pos, drop=True)
            assert isinstance(df_feat, pd.DataFrame)

    def test_names_valid(self):
        names = [f"P{i}" for i in range(len(df_seq))]
        se = aa.ShapExplainer()
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat, names=names)
        assert isinstance(df_feat, pd.DataFrame)

    def test_normalize_valid(self):
        for normalize in [True, False]:
            se = aa.ShapExplainer()
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, normalize=normalize)
            assert isinstance(df_feat, pd.DataFrame)

    def test_group_average_valid(self):
        for group_average in [True, False]:
            se = aa.ShapExplainer()
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, group_average=group_average)
            assert isinstance(df_feat, pd.DataFrame)

    def test_shap_feat_importance_valid(self):
        for shap_feat_importance in [True, False]:
            se = aa.ShapExplainer()
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, shap_feat_importance=shap_feat_importance, drop=True)
            assert isinstance(df_feat, pd.DataFrame)

    # Negative tests