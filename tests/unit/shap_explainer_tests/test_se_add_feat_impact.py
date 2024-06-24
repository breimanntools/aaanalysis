"""This script tests the ShapExplainer.add_feat_impact method."""
import pandas as pd
import numpy as np
import pytest
import random
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
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat)
        assert isinstance(df_feat, pd.DataFrame)
        assert sum(["feat_impact" in x for x in list(df_feat)]) == len(df_seq)

    def test_drop_valid(self):
        for drop in [True, False]:
            se = aa.ShapExplainer(verbose=False)
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat(drop=not drop)
            df_feat = se.add_feat_impact(df_feat=df_feat, drop=drop)
            assert isinstance(df_feat, pd.DataFrame)
            assert sum(["feat_impact" in x for x in list(df_feat)]) == len(df_seq)

    def test_sample_positions_valid(self):
        se = aa.ShapExplainer(verbose=False)
        sample_positions = random.sample(range(len(df_seq)), 3)
        for pos in sample_positions:
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, sample_positions=pos, drop=True)
            assert isinstance(df_feat, pd.DataFrame)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat, sample_positions=list(range(0, len(df_seq)-1)),
                                     drop=True)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat, sample_positions=sample_positions, drop=True)
        assert isinstance(df_feat, pd.DataFrame)

    def test_names_valid(self):
        names = [f"P{i}" for i in range(len(df_seq))]
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat, names=names, drop=True)
        assert isinstance(df_feat, pd.DataFrame)
        df_feat = se.add_feat_impact(df_feat=df_feat, names="test", sample_positions=1, drop=True)
        assert isinstance(df_feat, pd.DataFrame)

    def test_name_single_valid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        names = "SampleName"
        pos = 0
        df_feat = se.add_feat_impact(df_feat=df_feat, names=names, sample_positions=pos)
        assert isinstance(df_feat, pd.DataFrame)
        assert f"feat_impact_{names}" in list(df_feat)

    def test_name_multiple_valid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        names = ["SampleName1", "SampleName2"]
        pos = [0, 1]
        df_feat = se.add_feat_impact(df_feat=df_feat, names=names, sample_positions=pos)
        #assert isinstance(df_feat, pd.DataFrame)
        for name in names:
            assert f"feat_impact_{name}" in df_feat.columns

    def test_group_average_valid(self):
        se = aa.ShapExplainer(verbose=False)
        for group_average in [True, False]:
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, group_average=group_average)
            assert isinstance(df_feat, pd.DataFrame)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        df_feat = se.add_feat_impact(df_feat=df_feat, sample_positions=[1, 2, 3], group_average=True)
        assert isinstance(df_feat, pd.DataFrame)

    def test_normalize_valid(self):
        for normalize in [True, False]:
            se = aa.ShapExplainer(verbose=False)
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, normalize=normalize)
            assert isinstance(df_feat, pd.DataFrame)

    def test_shap_feat_importance_valid(self):
        for shap_feat_importance in [True, False]:
            se = aa.ShapExplainer(verbose=False)
            se.fit(valid_X, labels=valid_labels, **ARGS)
            df_feat = create_df_feat()
            df_feat = se.add_feat_impact(df_feat=df_feat, shap_feat_importance=shap_feat_importance, drop=True)
            assert isinstance(df_feat, pd.DataFrame)

    # Negative tests
    def test_df_feat_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=None)

    def test_drop_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        df_feat = create_df_feat()
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=df_feat, drop="not_a_boolean")
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=df_feat, drop=[])
        with pytest.raises(ValueError):
            df_feat = se.add_feat_impact(df_feat=df_feat, drop=True)
            df_feat = se.add_feat_impact(df_feat=df_feat, drop=False)

    def test_pos_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), sample_positions="not_a_valid_pos")
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), sample_positions=[1, None])
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), sample_positions=[1, "asdf"])

    def test_name_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), names=123)  # Invalid name type
        with pytest.raises(ValueError):
            names = ["Name"] * (len(valid_labels))
            se.add_feat_impact(df_feat=create_df_feat(), names=names)  # Duplicated names
        with pytest.raises(ValueError):
            names = [f"Name{i}" for i in range(len(valid_labels) - 1)]   # Not matching
            se.add_feat_impact(df_feat=create_df_feat(), names=names)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), sample_positions=[1, 2], names="Not matching")

    def test_group_average_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), group_average="not_a_boolean")
        # Only str is allowed if group_average is True
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), group_average=True, names=[])
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), group_average=True, names=["Group1"])
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), group_average=True, names=["Group1", 234])

    def test_normalize_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), normalize="not_a_boolean")
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), normalize=123)

    def test_shap_feat_importance_invalid(self):
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), shap_feat_importance="not_a_boolean")
        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=create_df_feat(), shap_feat_importance=123)

class TestAddFeatImpactComplex:
    """Test the add_feat_impact method with complex cases combining multiple parameters."""

    def test_complex_valid(self):
        """Complex test with valid parameter combinations."""
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)

        # Testing with a combination of valid parameters
        df_feat = create_df_feat(drop=True)
        names = ["Sample1", "Sample2"]
        pos = [0, 1]
        group_average = False
        normalize = True

        result = se.add_feat_impact(df_feat=df_feat, names=names, sample_positions=pos,
                                    group_average=group_average, normalize=normalize)
        assert isinstance(result, pd.DataFrame)
        for name in names:
            assert f"feat_impact_{name}" in result.columns

    def test_complex_invalid(self):
        """Complex test with invalid parameter combinations."""
        se = aa.ShapExplainer(verbose=False)
        se.fit(valid_X, labels=valid_labels, **ARGS)

        # Testing with a combination of invalid parameters
        df_feat = create_df_feat(drop=True)
        names = ["Sample1", 123]  # Invalid name
        pos = [0, 1]
        group_average = True  # Group average shouldn't be True for multiple names

        with pytest.raises(ValueError):
            se.add_feat_impact(df_feat=df_feat, names=names, sample_positions=pos, group_average=group_average)
