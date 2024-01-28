"""This is a script to test the SequenceFeature().get_df_pos() method ."""
import pandas as pd
import pytest
import random
import aaanalysis as aa

# Utility function for DataFrame creation
def _get_df_feat(n_feat=10, n_samples=20, list_parts=None):
    """Create input for sf.get_df_feat()"""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(n_feat)
    features = df_feat["feature"].to_list()
    sf = aa.SequenceFeature()
    if list_parts is not None:
        if type(list_parts) is str:
            list_parts = [list_parts]
        list_feat_parts = list(set([x.split("-")[0].lower() for x in features]))
        list_parts += list_feat_parts
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts)
    else:
        df_parts = sf.get_df_parts(df_seq=df_seq)
    df_feat = sf.get_df_feat(features=features, labels=labels, df_parts=df_parts)
    return df_feat


class TestGetDfPos:
    """Class for testing get_df_pos function in normal scenarios."""

    # Positive tests
    def test_valid_df_feat(self):
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)

            sf = aa.SequenceFeature
            result = sf.get_df_pos(df_feat=df_feat)
            assert isinstance(result, pd.DataFrame)

    def test_valid_col_val(self):

        for i in range(5):
            col_val = random.choice(['abs_auc', 'abs_mean_dif', 'mean_dif', 'std_test', 'std_ref'])
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, col_val=col_val)
            assert isinstance(result, pd.DataFrame)

    def test_valid_col_cat(self):
        for i in range(5):
            col_cat = random.choice(['category', 'subcategory', 'scale_name'])
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, col_cat=col_cat)
            assert isinstance(result, pd.DataFrame)

    def test_valid_start(self):
        for i in range(5):
            start = random.randint(1, 100)
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, start=start)
            assert isinstance(result, pd.DataFrame)

    def test_valid_tmd_len(self):
        for i in range(5):
            tmd_len = random.randint(20, 100)
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, tmd_len=tmd_len)
            assert isinstance(result, pd.DataFrame)

    def test_valid_jmd_n_len(self):
        for i in range(5):
            jmd_n_len = random.randint(15, 100)
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, jmd_n_len=jmd_n_len)
            assert isinstance(result, pd.DataFrame)

    def test_valid_jmd_c_len(self):
        for i in range(5):
            jmd_c_len = random.randint(15, 100)
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, jmd_c_len=jmd_c_len)
            assert isinstance(result, pd.DataFrame)

    def test_valid_list_parts(self):
        for i in range(5):
            list_parts = random.choice(['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c',
                                        'ext_c', 'ext_n', 'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c',
                                        'ext_n_tmd_n', 'tmd_c_ext_c'])
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples, list_parts=list_parts)
            sf = aa.SequenceFeature()
            result = sf.get_df_pos(df_feat=df_feat, list_parts=list_parts)
            assert isinstance(result, pd.DataFrame)

    def test_valid_normalize(self):
        n_feat = random.randint(5, 100)
        n_samples = random.randint(5, 50)
        df_feat = _get_df_feat(n_feat=n_feat, n_samples=n_samples)
        sf = aa.SequenceFeature()
        result = sf.get_df_pos(df_feat=df_feat, normalize=True)
        assert isinstance(result, pd.DataFrame)
        result = sf.get_df_pos(df_feat=df_feat, normalize=False)
        assert isinstance(result, pd.DataFrame)

    # Negative tests
    def test_invalid_df_feat(self):
        with pytest.raises(ValueError):
            sf = aa.SequenceFeature()
            sf.get_df_pos(df_feat=None)

    def test_invalid_col_val(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, col_val="invalid_value")

    def test_invalid_col_cat(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, col_cat="invalid_category")

    def test_invalid_start(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, start=None)

    def test_invalid_tmd_len(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, tmd_len=0)

    def test_invalid_jmd_n_len(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, jmd_n_len=-1)

    def test_invalid_jmd_c_len(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, jmd_c_len=-1)

    def test_invalid_list_parts(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, list_parts="invalid_parts")

    def test_invalid_normalize(self):
        df_feat = _get_df_feat()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, normalize="not_a_boolean")

class TestGetDfPosComplex:
    """Class for testing get_df_pos function in complex scenarios."""

    # Complex Positive Tests
    def test_complex_case_positive_1(self):
        df_feat = _get_df_feat(n_feat=15, n_samples=30)
        sf = aa.SequenceFeature()
        result = sf.get_df_pos(df_feat=df_feat, col_val='mean_dif',
                               col_cat='category', start=5, tmd_len=25, jmd_n_len=15, jmd_c_len=15,
                               normalize=True)
        assert isinstance(result, pd.DataFrame) and not result.empty

    def test_complex_case_positive_2(self):
        df_feat = _get_df_feat(n_feat=20, n_samples=40, list_parts=['tmd', 'jmd_n', 'jmd_c'])
        sf = aa.SequenceFeature()
        result = sf.get_df_pos(df_feat=df_feat, col_val='abs_auc', col_cat='scale_name',
                               start=10, tmd_len=30, jmd_n_len=20, jmd_c_len=20, normalize=False)
        assert isinstance(result, pd.DataFrame) and not result.empty

    # Complex Negative Tests
    def test_complex_case_negative_1(self):
        df_feat = _get_df_feat(n_feat=10, n_samples=20)
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, col_val='invalid_value', col_cat='invalid_category',
                          start=-10, tmd_len=0, jmd_n_len=30, jmd_c_len=30, normalize=True)

    def test_complex_case_negative_2(self):
        df_feat = _get_df_feat(n_feat=5, n_samples=10)
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_df_pos(df_feat=df_feat, col_val='mean_dif', col_cat='category', start=100,
                          tmd_len=25, jmd_n_len=50, jmd_c_len=-10, normalize="not_a_boolean")