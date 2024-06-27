"""This is a script to test the filter_seq() function."""
import pandas as pd
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa
import pytest
import random

aa.options["verbose"] = False

# Set default deadline from 200 to 20000
settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")


# Normal Cases
class TestFilterSeq:
    """Test filter_seq function."""

    def test_valid_method(self):
        """Test a valid 'method' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for method in ["cd-hit", "mmseqs"]:
            df_clust = aa.filter_seq(df_seq=df_seq, method=method)
            assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_method(self):
        """Test an invalid 'method' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for method in [None, 0, "invalid", "cd_hit"]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, method=method)

    def test_valid_similarity_threshold(self):
        """Test a valid 'similarity_threshold' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for similarity_threshold in [0.8, 1]:
            df_clust = aa.filter_seq(df_seq=df_seq, similarity_threshold=similarity_threshold, method="cd-hit")
            assert isinstance(df_clust, pd.DataFrame)
        df_clust = aa.filter_seq(df_seq=df_seq, similarity_threshold=0.8, method="mmseqs")
        assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_similarity_threshold(self):
        """Test an invalid 'similarity_threshold' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for similarity_threshold in [-0.1, 0.3, 1.1, "invalid", None, []]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, similarity_threshold=similarity_threshold, method="cd-hit")
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, similarity_threshold=similarity_threshold, method="mmseqs")

    def test_valid_word_size(self):
        """Test a valid 'word_size' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        df_clust = aa.filter_seq(df_seq=df_seq, word_size=3, method="cd-hit")
        assert isinstance(df_clust, pd.DataFrame)
        df_clust = aa.filter_seq(df_seq=df_seq, word_size=6, method="mmseqs")
        assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_word_size(self):
        """Test an invalid 'word_size' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for word_size in [True, 1, -1, 0, "invalid"]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, word_size=word_size, method="cd-hit")
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, word_size=word_size, method="mmseqs")
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, word_size=6, method="cd-hit")
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, word_size=4, method="mmseqs")

    def test_valid_global_identity(self):
        """Test a valid 'global_identity' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        df_clust = aa.filter_seq(df_seq=df_seq, global_identity=False, method="cd-hit")
        assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_global_identity(self):
        """Test an invalid 'global_identity' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for global_identity in [0, "invalid", [], [True, False]]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, global_identity=global_identity, method="cd-hit")
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, global_identity=global_identity, method="mmseqs")

    def test_valid_coverage_long(self):
        """Test a valid 'coverage_long' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for coverage_long in [0.1, 1]:
            df_clust = aa.filter_seq(df_seq=df_seq, coverage_long=coverage_long, method="cd-hit")
            assert isinstance(df_clust, pd.DataFrame)
        df_clust = aa.filter_seq(df_seq=df_seq, coverage_long=0.1, method="mmseqs")
        assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_coverage_long(self):
        """Test an invalid 'coverage_long' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for coverage_long in [0, 1.1, "asdf", []]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, coverage_long=coverage_long, method="cd-hit")
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, coverage_long=coverage_long, method="mmseqs")

    def test_valid_coverage_short(self):
        """Test a valid 'coverage_short' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for coverage_short in [0.1, 1]:
            df_clust = aa.filter_seq(df_seq=df_seq, coverage_short=coverage_short, method="cd-hit")
            assert isinstance(df_clust, pd.DataFrame)
        df_clust = aa.filter_seq(df_seq=df_seq, coverage_short=1, method="mmseqs")
        assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_coverage_short(self):
        """Test an invalid 'coverage_short' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for coverage_short in [0, 1.1, "asdf", []]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, coverage_short=coverage_short, method="cd-hit")
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, coverage_short=coverage_short, method="mmseqs")

    def test_valid_n_jobs(self):
        """Test a valid 'n_jobs' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for n_jobs in [-1, 5]:
            df_clust = aa.filter_seq(df_seq=df_seq, n_jobs=n_jobs, method="cd-hit")
            assert isinstance(df_clust, pd.DataFrame)
            df_clust = aa.filter_seq(df_seq=df_seq, n_jobs=n_jobs, method="mmseqs")
            assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_n_jobs(self):
        """Test an invalid 'n_jobs' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for n_jobs in ["invalid", 0, -2, [], [1, 2]]:
            with pytest.raises(ValueError):
                df_clust = aa.filter_seq(df_seq=df_seq, n_jobs=n_jobs, method="cd-hit")
                assert isinstance(df_clust, pd.DataFrame)
                df_clust = aa.filter_seq(df_seq=df_seq, n_jobs=n_jobs, method="mmseqs")
                assert isinstance(df_clust, pd.DataFrame)

    def test_valid_sort_clusters(self):
        """Test a valid 'sort_clusters' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        df_clust = aa.filter_seq(df_seq=df_seq, sort_clusters=True, method="cd-hit")
        assert isinstance(df_clust, pd.DataFrame)
        df_clust = aa.filter_seq(df_seq=df_seq, sort_clusters=True, method="mmseqs")
        assert isinstance(df_clust, pd.DataFrame)

    def test_invalid_sort_clusters(self):
        """Test an invalid 'sort_clusters' parameter."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        for sort_cluster in [None, 0, "invalid"]:
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, sort_clusters=sort_cluster, method="cd-hit")
            with pytest.raises(ValueError):
                aa.filter_seq(df_seq=df_seq, sort_clusters=sort_cluster, method="mmseqs")


# Complex Cases
class TestFilterSeqComplex:
    """Test filter_seq function for complex cases."""

    @settings(max_examples=4)
    @given(method=st.sampled_from(['cd-hit', 'mmseqs']),
           similarity_threshold=st.floats(min_value=0.8, max_value=0.9),
           coverage_long=st.floats(min_value=0.4, max_value=1.0),
           coverage_short=st.floats(min_value=0.4, max_value=1.0))
    def test_valid_combination(self, method, similarity_threshold,
                               coverage_long, coverage_short):
        """Test valid combinations of parameters."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        df_clust = aa.filter_seq(df_seq=df_seq, method=method,
                                 similarity_threshold=similarity_threshold,
                                 coverage_long=coverage_long, coverage_short=coverage_short)
        assert isinstance(df_clust, pd.DataFrame)

    @settings(max_examples=10)
    @given(method=st.sampled_from(['cd-hit', 'mmseqs']),
           similarity_threshold=st.floats(min_value=0.4, max_value=1.0),
           global_identity=st.booleans(),
           coverage_long=st.floats(min_value=0.1, max_value=1.0),
           coverage_short=st.floats(min_value=0.1, max_value=1.0))
    def test_invalid_combination(self, method, similarity_threshold, global_identity,
                                 coverage_long, coverage_short):
        """Test invalid combinations of parameters."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, method="invalid", similarity_threshold=similarity_threshold,
                          global_identity=global_identity,
                          coverage_long=coverage_long, coverage_short=coverage_short)
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, method=method, similarity_threshold=-0.1,
                          global_identity=global_identity,
                          coverage_long=coverage_long, coverage_short=coverage_short)
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, method=method, similarity_threshold=similarity_threshold,
                          word_size=1, global_identity=global_identity,
                          coverage_long=coverage_long, coverage_short=coverage_short)
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, method=method, similarity_threshold=similarity_threshold,
                          global_identity="invalid",
                          coverage_long=coverage_long, coverage_short=coverage_short)
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, method=method, similarity_threshold=similarity_threshold,
                          global_identity=global_identity,
                          coverage_long=-0.1, coverage_short=coverage_short)
        with pytest.raises(ValueError):
            aa.filter_seq(df_seq=df_seq, method=method, similarity_threshold=similarity_threshold,
                          global_identity=global_identity,
                          coverage_long=coverage_long, coverage_short=-0.1)
