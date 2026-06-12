"""This is a script to test the SequenceFeature().get_feature_descriptions() method ."""
from hypothesis import given, settings, strategies as st
import pytest
import random
import pandas as pd
import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def get_random_features(n_feat=100):
    """Draw a random sample of feature ids from the default feature space."""
    sf = aa.SequenceFeature()
    features = sf.get_features()
    return random.sample(features, n_feat)


class TestGetFeatureDescriptions:
    """Class for testing get_feature_descriptions function in positive scenarios."""

    def test_valid_features(self):
        """Test valid 'features' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        result = sf.get_feature_descriptions(features=features)
        assert isinstance(result, list) and all(isinstance(des, str) for des in result)
        assert len(result) == len(features)

    def test_valid_df_cat(self):
        """Test valid 'df_cat' DataFrame input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        df_cat = aa.load_scales(name="scales_cat")
        result = sf.get_feature_descriptions(features=features, df_cat=df_cat)
        assert isinstance(result, list) and all(isinstance(des, str) for des in result)

    @settings(max_examples=5, deadline=None)
    @given(start=st.integers(min_value=1))
    def test_valid_start(self, start):
        """Test valid 'start' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_descriptions(features=features, start=start)
        assert isinstance(result, list) and all(isinstance(des, str) for des in result)

    @settings(max_examples=5, deadline=None)
    @given(tmd_len=st.integers(min_value=20, max_value=2000))
    def test_valid_tmd_len(self, tmd_len):
        """Test valid 'tmd_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_descriptions(features=features, tmd_len=tmd_len)
        assert isinstance(result, list) and all(isinstance(des, str) for des in result)

    @settings(max_examples=5, deadline=None)
    @given(jmd_c_len=st.integers(min_value=10, max_value=2000))
    def test_valid_jmd_c_len(self, jmd_c_len):
        """Test valid 'jmd_c_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_descriptions(features=features, jmd_c_len=jmd_c_len)
        assert isinstance(result, list) and all(isinstance(des, str) for des in result)

    @settings(max_examples=5, deadline=None)
    @given(jmd_n_len=st.integers(min_value=10, max_value=2000))
    def test_valid_jmd_n_len(self, jmd_n_len):
        """Test valid 'jmd_n_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        result = sf.get_feature_descriptions(features=features, jmd_n_len=jmd_n_len)
        assert isinstance(result, list) and all(isinstance(des, str) for des in result)

    # Negative Tests
    def test_invalid_features(self):
        """Negative test for invalid 'features' input."""
        sf = aa.SequenceFeature()
        invalid_features = [None, 123, "invalid_input", {}]
        for features in invalid_features:
            with pytest.raises(ValueError):
                sf.get_feature_descriptions(features=features)

    def test_invalid_df_cat(self):
        """Negative test for invalid 'df_cat' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        invalid_df_cats = [123, "invalid_input", [], {}]
        for df_cat in invalid_df_cats:
            with pytest.raises(ValueError):
                sf.get_feature_descriptions(features=features, df_cat=df_cat)

    @settings(max_examples=5, deadline=None)
    @given(start=st.one_of(st.none(), st.text(), st.floats()))
    def test_invalid_start(self, start):
        """Negative test for invalid 'start' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_descriptions(features=features, start=start)

    @settings(max_examples=5, deadline=None)
    @given(tmd_len=st.one_of(st.none(), st.text(), st.floats(), st.integers(max_value=0)))
    def test_invalid_tmd_len(self, tmd_len):
        """Negative test for invalid 'tmd_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_descriptions(features=features, tmd_len=tmd_len)

    @settings(max_examples=5, deadline=None)
    @given(jmd_c_len=st.one_of(st.none(), st.text(), st.floats(), st.integers(max_value=0)))
    def test_invalid_jmd_c_len(self, jmd_c_len):
        """Negative test for invalid 'jmd_c_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_descriptions(features=features, jmd_c_len=jmd_c_len)

    @settings(max_examples=5, deadline=None)
    @given(jmd_n_len=st.one_of(st.none(), st.text(), st.floats(), st.integers(max_value=0)))
    def test_invalid_jmd_n_len(self, jmd_n_len):
        """Negative test for invalid 'jmd_n_len' input."""
        sf = aa.SequenceFeature()
        features = get_random_features()
        with pytest.raises(ValueError):
            sf.get_feature_descriptions(features=features, jmd_n_len=jmd_n_len)

    # df_feat DataFrame accepted for 'features'
    def test_valid_features_df_feat(self):
        """A df_feat DataFrame is accepted and equals the list-of-ids form."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        df_feat = pd.DataFrame({"feature": features})
        assert sf.get_feature_descriptions(features=df_feat) == sf.get_feature_descriptions(features=features)

    def test_invalid_features_df_feat_missing_col(self):
        """A DataFrame without a 'feature' column raises ValueError."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=20)
        with pytest.raises(ValueError, match="feature"):
            sf.get_feature_descriptions(features=pd.DataFrame({"wrong": features}))


class TestGetFeatureDescriptionsComplex:
    """Class for testing get_feature_descriptions in complex scenarios (KPIs from issue #20)."""

    def test_all_three_split_types_covered(self):
        """Each split type (Segment, Pattern, PeriodicPattern) yields a readable phrase."""
        sf = aa.SequenceFeature()
        feats = ["TMD-Segment(2,4)-ANDN920101",
                 "JMD_N_TMD_N-Pattern(N,3,7,11)-CHAM820101",
                 "TMD-PeriodicPattern(N,i+3/4,1)-KLEP840101"]
        des = sf.get_feature_descriptions(features=feats)
        assert "segment 2 of 4" in des[0]
        assert "pattern (from N-terminus)" in des[1]
        assert "periodic pattern (steps 3/4 from N-terminus)" in des[2]

    def test_part_label_present(self):
        """The readable part label (not the raw part token) appears in the description."""
        sf = aa.SequenceFeature()
        des = sf.get_feature_descriptions(features=["TMD_C_JMD_C-Segment(1,1)-FAUJ880109"])[0]
        assert des.startswith("TMD-C+JMD-C")
        assert "tmd_c_jmd_c" not in des.lower()

    def test_scale_name_and_category_present(self):
        """The AAontology scale name, category, and subcategory all appear in the description."""
        sf = aa.SequenceFeature()
        df_cat = aa.load_scales(name="scales_cat")
        row = df_cat[df_cat[ut.COL_SCALE_ID] == "ANDN920101"].iloc[0]
        des = sf.get_feature_descriptions(features=["TMD-Segment(2,4)-ANDN920101"])[0]
        assert row[ut.COL_SCALE_NAME] in des
        assert row[ut.COL_CAT] in des
        assert row[ut.COL_SUBCAT] in des

    # KPI: determinism
    def test_deterministic_byte_identical(self):
        """Identical id + metadata produce a byte-identical string across repeated runs."""
        sf = aa.SequenceFeature()
        features = get_random_features(n_feat=50)
        first = sf.get_feature_descriptions(features=features)
        for _ in range(3):
            assert sf.get_feature_descriptions(features=features) == first

    def test_deterministic_order_independent(self):
        """A feature's description does not depend on its neighbours in the list."""
        sf = aa.SequenceFeature()
        feats = ["TMD-Segment(2,4)-ANDN920101", "JMD_N_TMD_N-Pattern(N,3,7,11)-CHAM820101"]
        des_forward = sf.get_feature_descriptions(features=feats)
        des_reverse = sf.get_feature_descriptions(features=list(reversed(feats)))
        assert des_forward[0] == des_reverse[1]
        assert des_forward[1] == des_reverse[0]

    # KPI: 100% non-empty coverage on a real DOM_GSEC df_feat
    def test_dom_gsec_full_coverage(self):
        """Every feature in the bundled DOM_GSEC df_feat gets a non-empty description."""
        sf = aa.SequenceFeature()
        df_feat = aa.load_features(name="DOM_GSEC")
        des = sf.get_feature_descriptions(features=df_feat)
        assert len(des) == len(df_feat)
        assert all(isinstance(d, str) and d.strip() != "" for d in des)

    def test_assignable_as_df_feat_column(self):
        """The description list can be assigned to the canonical df_feat column."""
        sf = aa.SequenceFeature()
        df_feat = aa.load_features(name="DOM_GSEC").head(20).copy()
        df_feat[ut.COL_FEAT_DES] = sf.get_feature_descriptions(features=df_feat)
        assert ut.COL_FEAT_DES in df_feat.columns
        assert df_feat[ut.COL_FEAT_DES].notna().all()

    # KPI: terminology consistency
    def test_part_label_vocabulary_subset(self):
        """Part labels used are a subset of the canonical DICT_PART_LABEL vocabulary."""
        sf = aa.SequenceFeature()
        df_feat = aa.load_features(name="DOM_GSEC")
        part_labels = set(ut.DICT_PART_LABEL.values())
        des = sf.get_feature_descriptions(features=df_feat)
        for d in des:
            part_label = d.split(",")[0]
            assert part_label in part_labels

    def test_category_matches_scales_cat(self):
        """Category tokens in descriptions match the categories in scales_cat.tsv."""
        sf = aa.SequenceFeature()
        df_cat = aa.load_scales(name="scales_cat")
        valid_cats = set(df_cat[ut.COL_CAT])
        df_feat = aa.load_features(name="DOM_GSEC")
        des = sf.get_feature_descriptions(features=df_feat)
        for d in des:
            # AAontology classification is the trailing '[category: subcategory]' bracket
            cat = d.rsplit("[", 1)[1].rstrip("]").split(": ")[0]
            assert cat in valid_cats

    def test_distinguishes_parts_that_get_feature_names_drops(self):
        """Two ids differing only in PART get distinct descriptions (get_feature_names would not)."""
        sf = aa.SequenceFeature()
        feat_tmd = "TMD-Segment(1,1)-ANDN920101"
        feat_jmd = "JMD_N-Segment(1,1)-ANDN920101"
        des = sf.get_feature_descriptions(features=[feat_tmd, feat_jmd])
        assert des[0] != des[1]
