"""This is a script to test the SequenceFeature().feature_matrix() method ."""
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
import numpy as np
import random
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False


def _get_df_feat_input(n_feat=10, n_samples=20, list_parts=None):
    """Create input for sf.get_df_feat()"""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(n_feat)
    features = df_feat["feature"].to_list()
    sf = aa.SequenceFeature()
    if list_parts is not None:
        list_feat_parts = list(set([x.split("-")[0].lower() for x in features]))
        list_parts += list_feat_parts
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts)
    else:
        df_parts = sf.get_df_parts(df_seq=df_seq)
    return features, df_parts, labels


class TestFeatureMatrix:
    """Test class for the 'feature_matrix' method of the SequenceFeature class."""

    def test_valid_features(self):
        """Positive test for valid 'features' input."""
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            result = sf.feature_matrix(features=features, df_parts=df_parts)
            assert isinstance(result, np.ndarray)

    @settings(max_examples=5, deadline=None)
    @given(list_parts=st.lists(st.sampled_from(
        ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n', 'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c',
         'ext_n_tmd_n', 'tmd_c_ext_c']), min_size=1))
    def test_valid_df_parts(self, list_parts):
        """Test valid 'df_parts' DataFrame inputs."""
        features, df_parts, labels = _get_df_feat_input(n_feat=10, n_samples=50, list_parts=list_parts)
        sf = aa.SequenceFeature()
        result = sf.feature_matrix(features=features, df_parts=df_parts)
        assert isinstance(result, np.ndarray)

    def test_valid_df_scales(self):
        for i in range(5):
            n_feat = random.randint(5, 100)
            n_samples = random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            df_scales = aa.load_scales()
            sf = aa.SequenceFeature()
            result = sf.feature_matrix(features=features, df_parts=df_parts, df_scales=df_scales)
            assert isinstance(result, np.ndarray)
            scales = list(set([x.split("-")[2] for x in features]))
            result = sf.feature_matrix(features=features, df_parts=df_parts, df_scales=df_scales[scales])
            assert isinstance(result, np.ndarray)

    def test_n_jobs(self):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        result = sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=2)
        assert isinstance(result, np.ndarray)
        result = sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=None)
        assert isinstance(result, np.ndarray)
        result = sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=-1)
        assert isinstance(result, np.ndarray)

    def test_accept_gaps(self):
        features, df_parts, labels = _get_df_feat_input()
        sf = aa.SequenceFeature()
        result = sf.feature_matrix(features=features, df_parts=df_parts, accept_gaps=True)
        assert isinstance(result, np.ndarray)
        result = sf.feature_matrix(features=features, df_parts=df_parts, accept_gaps=False)
        assert isinstance(result, np.ndarray)

    # Negative tests
    def test_invalid_features(self):
        """Negative test for invalid 'features' input."""
        sf = aa.SequenceFeature()
        df_parts = _get_df_feat_input()[1]
        with pytest.raises(ValueError):
            sf.feature_matrix(features=None, df_parts=df_parts)
        with pytest.raises(ValueError):
            sf.feature_matrix(features="invalid_input", df_parts=df_parts)

    def test_invalid_df_parts(self):
        """Negative test for invalid 'df_parts' input."""
        sf = aa.SequenceFeature()
        features = _get_df_feat_input()[0]
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=None)
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts="invalid_input")

    def test_invalid_df_scales(self):
        """Negative test for invalid 'df_scales' input."""
        sf = aa.SequenceFeature()
        features, df_parts = _get_df_feat_input()[:2]
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=df_parts, df_scales="invalid_input")

    def test_invalid_accept_gaps(self):
        """Negative test for invalid 'accept_gaps' input."""
        sf = aa.SequenceFeature()
        features, df_parts = _get_df_feat_input()[:2]
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=df_parts, accept_gaps="not_a_boolean")

    def test_invalid_n_jobs(self):
        """Negative test for invalid 'n_jobs' input."""
        sf = aa.SequenceFeature()
        features, df_parts = _get_df_feat_input()[:2]
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=df_parts, n_jobs="invalid")
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=-2)

class TestFeatureMatrixComplex:
    """Complex positive tests for the 'feature_matrix' method."""

    def test_valid_combinations(self):
        """Test with valid combinations of parameters."""
        sf = aa.SequenceFeature()
        for i in range(3):
            n_feat, n_samples = random.randint(5, 100), random.randint(5, 50)
            features, df_parts, labels = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            df_scales = aa.load_scales() if random.choice([True, False]) else None
            accept_gaps = random.choice([True, False])
            n_jobs = random.choice([1, 2, None])

            result = sf.feature_matrix(features=features, df_parts=df_parts, df_scales=df_scales,
                                       accept_gaps=accept_gaps, n_jobs=n_jobs)
            assert isinstance(result, np.ndarray)

    def test_invalid_combinations(self):
        """Test with invalid combinations of parameters."""
        sf = aa.SequenceFeature()
        features, df_parts, _ = _get_df_feat_input()
        # Test with invalid 'features' and 'df_parts'
        with pytest.raises(ValueError):
            sf.feature_matrix(features=None, df_parts="invalid_input")
        # Test with invalid 'df_scales' and incorrect 'n_jobs'
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=df_parts, df_scales="invalid", n_jobs="invalid")
        # Test with invalid 'accept_gaps' type
        with pytest.raises(ValueError):
            sf.feature_matrix(features=features, df_parts=df_parts, accept_gaps="not_a_boolean")


class TestFeatureMatrixGoldenValues:
    """Hand-computed values + shape invariants (not just 'returns an ndarray')."""

    @staticmethod
    def _scales():
        import pandas as pd
        order = "ACDEFGHIKLMNPQRSTVWY"
        return pd.DataFrame({"S1": {aa: float(i + 1) for i, aa in enumerate(order)}})

    def test_golden_segment_whole_mean(self):
        """'AC' over Segment(1,1) -> mean(S1[A]=1, S1[C]=2) == 1.5 exactly."""
        import pandas as pd
        sf = aa.SequenceFeature()
        X = sf.feature_matrix(features=["TMD-Segment(1,1)-S1"],
                              df_parts=pd.DataFrame({"tmd": ["AC"]}),
                              df_scales=self._scales())
        assert np.asarray(X).shape == (1, 1)
        assert float(np.asarray(X)[0, 0]) == 1.5

    def test_golden_segment_second_half(self):
        """'ACDE' Segment(2,2) -> 'DE' -> mean(S1[D]=3, S1[E]=4) == 3.5."""
        import pandas as pd
        sf = aa.SequenceFeature()
        X = sf.feature_matrix(features=["TMD-Segment(2,2)-S1"],
                              df_parts=pd.DataFrame({"tmd": ["ACDE"]}),
                              df_scales=self._scales())
        assert float(np.asarray(X)[0, 0]) == 3.5

    def test_property_shape_rows_cols(self):
        """Invariant: result is (n_samples, n_features) for every input size."""
        for n_feat, n_samples in [(5, 10), (20, 8), (1, 30)]:
            features, df_parts, _ = _get_df_feat_input(n_feat=n_feat, n_samples=n_samples)
            sf = aa.SequenceFeature()
            X = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts))
            # DOM_GSEC returns 2*n rows (n per class); assert against df_parts length.
            assert X.shape == (len(df_parts), len(features))

    def test_property_parallel_equals_serial(self):
        """Invariant: n_jobs must not change the values (only the schedule)."""
        features, df_parts, _ = _get_df_feat_input(n_feat=15, n_samples=20)
        sf = aa.SequenceFeature()
        X1 = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=1))
        X2 = np.asarray(sf.feature_matrix(features=features, df_parts=df_parts, n_jobs=2))
        assert np.allclose(X1, X2, equal_nan=True)
