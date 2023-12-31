"""
This is a script for testing the initialization of the CPP class.
"""
import pytest
import random
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
import pandas as pd

STR_SEGMENT = "Segment"
STR_PATTERN = "Pattern"
STR_PERIODIC_PATTERN = "PeriodicPattner"

def get_df_parts():
    """"""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    return df_parts

class TestCheckCPP:
    """Test aa.AAclust class individual parameters"""

    # Positive tests
    @settings(max_examples=10, deadline=1000)
    @given(val=some.integers(min_value=2, max_value=10))
    def test_valid_segment_split_kws(self, val):
        df_parts = get_df_parts()
        i = random.randint(0, 5)
        split_kws = dict(Segment=dict(n_split_min=val, n_split_max=val+i))
        aa.CPP(df_parts=df_parts, split_kws=split_kws)

    @settings(max_examples=10, deadline=1000)
    @given(val=some.integers(min_value=2, max_value=10))
    def test_valid_pattern_split_kws(self, val):
        i = random.randint(0, 5)
        j = random.randint(0, 5)
        n_min, n_max, len_max = val, val+i, val+i
        steps = [i, i+1, i+j+1]
        split_kws = dict(Pattern=dict(steps=steps, n_min=n_min, n_max=n_max, len_max=len_max))
        df_parts = get_df_parts()
        aa.CPP(df_parts=df_parts, split_kws=split_kws)

    @settings(max_examples=10, deadline=1000)
    @given(val=some.integers(min_value=2, max_value=10))
    def test_valid_periodic_pattern_split_kws(self, val):
        j = random.randint(0, 5)
        # Only two steps are allowed
        steps = [val, val+j]
        split_kws = dict(PeriodicPattern=dict(steps=steps))
        df_parts = get_df_parts()
        aa.CPP(df_parts=df_parts, split_kws=split_kws)

    def test_valid_df_parts(self):
        # Only dataset with sufficient long sequences
        all_data_set_names = [x for x in aa.load_dataset()["Dataset"].to_list() if "AA" not in x and "AMYLO" not in x]
        for name in all_data_set_names:
            df_seq = aa.load_dataset(name=name, n=10, min_len=50)
            sf = aa.SequenceFeature()
            df_parts = sf.get_df_parts(df_seq=df_seq)
            if len(df_seq) > 20:
                assert isinstance(aa.CPP(df_parts=df_parts, accept_gaps=True).df_parts, pd.DataFrame)

    def test_valid_df_scales(self):
        df_parts = get_df_parts()
        list_name = ["scales", "scales_raw"]
        for name in list_name:
            df_scales = aa.load_scales(name=name)
            assert isinstance(aa.CPP(df_parts=df_parts, df_scales=df_scales).df_scales, pd.DataFrame)

    def test_valid_df_cat(self):
        df_parts = get_df_parts()
        df_cat = aa.load_scales(name="scales_cat")
        df_scales = aa.load_scales()
        assert isinstance(aa.CPP(df_parts=df_parts, df_cat=df_cat).df_cat, pd.DataFrame)
        df_scales = df_scales[list(df_scales)[0:100]]
        # check if df_cat and df_scales adjusted
        cpp = aa.CPP(df_parts=df_parts, df_cat=df_cat, df_scales=df_scales, verbose=False)
        _df_cat = cpp.df_cat
        _df_scales = cpp.df_scales
        assert len(_df_cat) ==  len(list(_df_scales))

    def test_valid_accept_gaps(self):
        df_parts = get_df_parts()
        for accept_gaps in [True, False]:
            assert aa.CPP(df_parts=df_parts, accept_gaps=accept_gaps)._accept_gaps == accept_gaps

    def test_valid_verbose(self):
        df_parts = get_df_parts()
        for verbose in [True, False]:
            assert aa.CPP(df_parts=df_parts, verbose=verbose)._verbose == verbose


    # Negative tests
    def test_invalid_segment(self):
        df_parts = get_df_parts()
        i = random.randint(0, 5)
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(Segment=dict(n_split=i, n_split_max=i+1)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(Segment=dict(n_split=i)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(Segment=dict(n_split=str(i), n_split_max=i+1)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(Segment=dict(n_split=None, n_split_max=i+1)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(Segment=dict(n_spl=i, n_split_max=i+1)))

    @settings(max_examples=10, deadline=1000)
    @given(val=some.integers(min_value=2, max_value=10))
    def test_invalid_segment_split_kws(self, val):
        df_parts = get_df_parts()
        i = random.randint(0, 5)
        split_kws = dict(Segment=dict(n_split_min=val+i, n_split_max=val))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws)

    def test_invalid_pattern(self):
        i = random.randint(0, 5)
        j = random.randint(0, 5)
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts,
                   split_kws=dict(Pattern=dict(steps=[str(1), 1], n_min=i, n_max=i + j, len_max=i - 2)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts,
                   split_kws=dict(Pattern=dict(steps=None, n_min=i, n_max=i + j, len_max=i - 2)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts,
                   split_kws=dict(Pattern=dict(steps=[1, 1], n_max=i + j, len_max=i - 2)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts,
                   split_kws=dict(Pattern=dict(steps=[1, 1], n_min=i, n_max=i + j)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts,
                   split_kws=dict(Pattern=dict(steps=[1, 1], n_min=i, len_max=i - 2)))

    @settings(max_examples=10, deadline=1000)
    @given(val=some.integers(min_value=2, max_value=10))
    def test_invalid_pattern_split_kws(self, val):
        i = random.randint(0, 5)
        j = random.randint(0, 5)
        n_min, n_max, len_max = val, val+i, val+i
        steps = [i, i+1, i+j+1]
        split_kws = dict(Pattern=dict(steps=steps, n_min=n_min, n_max=n_max, len_max=0))
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws)

    def test_invalid_periodic_pattern(self):
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            # Not matching with df_parts
            aa.CPP(df_parts=df_parts, split_kws=dict(PeriodicPattern=dict(steps=[1000000, 100000000])))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(PeriodicPattern=dict(steps=None)))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(PeriodicPattern=dict(steps=[1, str(2)])))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(PeriodicPattern=dict(steps=[1, None])))
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=dict(PeriodicPattern=dict(stepss=[1, "invalid"])))

    @settings(max_examples=10, deadline=1000)
    @given(val=some.integers(min_value=2, max_value=10))
    def test_invalid_periodic_pattern_split_kws(self, val):
        j = random.randint(0, 5)
        steps = [val+j+1, val-1,]
        split_kws = dict(PeriodicPattern=dict(steps=steps))
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws)

    def test_invalid_df_parts(self):
        all_data_set_names = ["SEQ_AMYLO", "AA_CASPASE3"]
        for name in all_data_set_names:
            df_seq = aa.load_dataset(name=name, n=10, aa_window_size=5)
            sf = aa.SequenceFeature()
            df_parts = sf.get_df_parts(df_seq=df_seq)
            with pytest.raises(ValueError):
                aa.CPP(df_parts=df_parts, accept_gaps=True)

    def test_invalid_df_scales(self):
        df_parts = get_df_parts()
        list_name = ["scales", "scales_raw", "scales_pc"]
        for name in list_name:
            df_scales = aa.load_scales(name=name)
            df_scales.columns = ["wrong"] + list(df_scales)[1:]
            with pytest.raises(ValueError):
                aa.CPP(df_parts=df_parts, df_scales=df_scales)

    def test_invalid_df_cat(self):
        df_parts = get_df_parts()
        df_cat = aa.load_scales(name="scales_cat")
        df_scales = aa.load_scales()
        # check if df_cat and df_scales adjusted
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, df_cat=df_cat.head(2), df_scales=df_scales, verbose=False)

    def test_invalid_accept_gaps(self):
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, accept_gaps="invalid")

    def test_invalid_verbose(self):
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, verbose="invalid")

class TestCheckCPPComplex:
    """Complex Test cases for CPP class"""

    # Positive Complex Cases
    def test_valid_all_parameters_combined(self):
        df_parts = get_df_parts()
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")
        split_kws = {'Segment': {'n_split_min': 2, 'n_split_max': 5}}
        cpp_instance = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales,
                              df_cat=df_cat, accept_gaps=True, verbose=True)
        assert isinstance(cpp_instance, aa.CPP)

    def test_valid_randomized_parameters(self):
        df_parts = get_df_parts()
        split_kws = {'Segment': {'n_split_min': random.randint(2, 5), 'n_split_max': random.randint(5, 10)}}
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")
        cpp_instance = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat, accept_gaps=random.choice([True, False]), verbose=random.choice([True, False]))
        assert isinstance(cpp_instance, aa.CPP)

    # Negative Complex Cases
    def test_invalid_conflicting_parameters(self):
        df_parts = get_df_parts()
        # Creating a conflicting split_kws
        split_kws = {'Segment': {'n_split_min': 10, 'n_split_max': 2}}
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat, accept_gaps=True)

    def test_invalid_extreme_values(self):
        # Generate extreme DataFrames
        import numpy as np
        df_parts = pd.DataFrame(np.random.rand(10000, 1000))
        df_scales = pd.DataFrame(np.random.rand(10000, 1000))
        df_cat = pd.DataFrame(np.random.rand(1000, 500))
        split_kws = {'Segment': {'n_split_min': 1000, 'n_split_max': 2000}}
        with pytest.raises(ValueError):
            aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat, accept_gaps=True, verbose=True)
