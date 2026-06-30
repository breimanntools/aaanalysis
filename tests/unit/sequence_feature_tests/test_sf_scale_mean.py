"""This is a script to test the SequenceFeature().scale_mean() method."""
from hypothesis import given, settings
import hypothesis.strategies as st
import warnings
import pytest
import numpy as np
import pandas as pd
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False

LIST_PARTS = ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
              'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']


def _get_input(n_samples=10):
    """Load a small DOM_GSEC fixture and the default scales."""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    df_scales = aa.load_scales()
    return df_seq, df_scales


def _manual_scale_X(df_parts, df_scales):
    """Cell-27 comprehension over the concatenated jmd_n + tmd + jmd_c span."""
    seqs = (df_parts["jmd_n"] + df_parts["tmd"] + df_parts["jmd_c"]).to_list()
    return np.array([df_scales.loc[[a for a in s if a in df_scales.index]].mean(axis=0).values
                     for s in seqs])


class TestScaleMean:
    """Test class for the 'scale_mean' method of the SequenceFeature class."""

    # Positive tests
    def test_valid_df_seq(self):
        """Positive test for valid 'df_seq' input (several sizes)."""
        for n in [3, 5, 10]:
            df_seq, df_scales = _get_input(n_samples=n)
            sf = aa.SequenceFeature()
            X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
            assert isinstance(X, np.ndarray)
            assert X.shape == (len(df_seq), df_scales.shape[1])

    def test_valid_df_scales(self):
        """Positive test for valid 'df_scales' input, incl. a column subset and the default."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        # Full scales
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert X.shape[1] == df_scales.shape[1]
        # Subset of scales
        sub = df_scales.iloc[:, :5]
        X_sub = sf.scale_mean(df_seq=df_seq, df_scales=sub)
        assert X_sub.shape == (len(df_seq), 5)
        # Default (df_scales=None) loads the bundled scales
        X_def = sf.scale_mean(df_seq=df_seq)
        assert isinstance(X_def, np.ndarray)
        assert X_def.shape[0] == len(df_seq)

    @settings(max_examples=6, deadline=None)
    @given(list_parts=st.lists(st.sampled_from(LIST_PARTS), min_size=1, max_size=3, unique=True))
    def test_valid_list_parts(self, list_parts):
        """Positive test for valid 'list_parts' inputs (single, str, multiple)."""
        df_seq, df_scales = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales, list_parts=list_parts)
        assert isinstance(X, np.ndarray)
        assert X.shape == (len(df_seq), df_scales.shape[1])
        # A single part may also be passed as a bare string
        X_str = sf.scale_mean(df_seq=df_seq, df_scales=df_scales, list_parts=list_parts[0])
        assert X_str.shape == (len(df_seq), df_scales.shape[1])

    def test_valid_list_parts_none_is_whole_span(self):
        """list_parts=None equals the explicit jmd_n + tmd + jmd_c part list."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        X_none = sf.scale_mean(df_seq=df_seq, df_scales=df_scales, list_parts=None)
        X_explicit = sf.scale_mean(df_seq=df_seq, df_scales=df_scales,
                                   list_parts=["jmd_n", "tmd", "jmd_c"])
        assert np.allclose(X_none, X_explicit, equal_nan=True)

    def test_valid_return_df(self):
        """Positive test for valid 'return_df' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales, return_df=False)
        assert isinstance(X, np.ndarray)
        df = sf.scale_mean(df_seq=df_seq, df_scales=df_scales, return_df=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(df_scales.columns)
        assert df.shape == (len(df_seq), df_scales.shape[1])
        # Values agree between the array and DataFrame forms
        assert np.allclose(np.asarray(df.values, dtype=float), X, equal_nan=True)

    # Negative tests
    def test_invalid_df_seq(self):
        """Negative test for invalid 'df_seq' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=None, df_scales=df_scales)
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq="invalid_input", df_scales=df_scales)
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=pd.DataFrame({"wrong": ["A"]}), df_scales=df_scales)

    def test_invalid_df_scales(self):
        """Negative test for invalid 'df_scales' input."""
        df_seq, _ = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=df_seq, df_scales="invalid_input")

    def test_invalid_list_parts(self):
        """Negative test for invalid 'list_parts' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=df_seq, df_scales=df_scales, list_parts=["not_a_part"])
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=df_seq, df_scales=df_scales, list_parts=[])

    def test_invalid_return_df(self):
        """Negative test for invalid 'return_df' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=df_seq, df_scales=df_scales, return_df="not_a_boolean")


class TestScaleMeanComplex:
    """Complex positive / negative tests for the 'scale_mean' method."""

    def test_valid_combinations(self):
        """Test with valid combinations of parameters."""
        df_seq, df_scales = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        for list_parts in [None, "tmd", ["jmd_n", "jmd_c"], ["tmd_jmd"]]:
            for return_df in [True, False]:
                out = sf.scale_mean(df_seq=df_seq, df_scales=df_scales,
                                    list_parts=list_parts, return_df=return_df)
                if return_df:
                    assert isinstance(out, pd.DataFrame)
                else:
                    assert isinstance(out, np.ndarray)
                assert np.asarray(out).shape == (len(df_seq), df_scales.shape[1])

    def test_invalid_combinations(self):
        """Test with invalid combinations of parameters."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=None, df_scales="invalid")
        with pytest.raises(ValueError):
            sf.scale_mean(df_seq=df_seq, df_scales=df_scales, list_parts="bad_part", return_df="bad")


class TestScaleMeanGoldenValues:
    """Hand-computed values, the notebook-comprehension match, and edge cases."""

    def test_golden_matches_notebook_comprehension(self):
        """Output equals the gamma-secretase notebook (cell 27) comprehension on DOM_GSEC."""
        aa.options["verbose"] = False
        df_seq = aa.load_dataset(name="DOM_GSEC")
        df_scales = aa.load_scales()
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        X_manual = _manual_scale_X(df_parts=df_parts, df_scales=df_scales)
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert X.shape == X_manual.shape
        assert np.allclose(X, X_manual, equal_nan=True)

    def test_golden_hand_computed_mean(self):
        """'AC' (jmd empty) over a single scale -> mean(S1[A]=1, S1[C]=2) == 1.5 exactly."""
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: float(i + 1) for i, a in enumerate(order)}})
        # Part-based df_seq with jmd_n/jmd_c so the span is exactly the TMD 'AC'
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AC"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert X.shape == (1, 1)
        assert float(X[0, 0]) == 1.5

    def test_golden_frequency_weighted(self):
        """Residues count by frequency: 'AAC' -> mean(1, 1, 2) == 4/3."""
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: float(i + 1) for i, a in enumerate(order)}})
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AAC"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert np.isclose(float(X[0, 0]), 4.0 / 3.0)

    def test_non_canonical_residues_dropped(self):
        """Non-canonical residues ('X') and gaps are ignored before averaging."""
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: float(i + 1) for i, a in enumerate(order)}})
        # 'AXC' should equal 'AC' because 'X' is not in the scale index
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["AXC", "AC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert float(X[0, 0]) == float(X[1, 0]) == 1.5

    def test_edge_all_non_canonical_is_nan(self):
        """An all-non-canonical / empty span yields an all-NaN row and warns when verbose."""
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: float(i + 1) for i, a in enumerate(order)},
                                  "S2": {a: float(i) for i, a in enumerate(order)}})
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["XXXX", "AC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature(verbose=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert np.all(np.isnan(X[0]))         # all-non-canonical -> NaN row
        assert np.all(np.isfinite(X[1]))      # canonical row stays finite
        assert any("no scored residue" in str(r.message) for r in records)

    def test_edge_all_non_canonical_silent_when_not_verbose(self):
        """No warning is emitted for the NaN row when verbose=False."""
        order = "ACDEFGHIKLMNPQRSTVWY"
        df_scales = pd.DataFrame({"S1": {a: float(i + 1) for i, a in enumerate(order)}})
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["XXXX"], "jmd_c": [""]})
        sf = aa.SequenceFeature(verbose=False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.scale_mean(df_seq=df_seq, df_scales=df_scales)
        assert np.all(np.isnan(X[0]))
        assert not any("no scored residue" in str(r.message) for r in records)

    def test_property_shape_invariant(self):
        """Invariant: result is (n_samples, n_scales) for every input size."""
        df_scales = aa.load_scales()
        for n in [3, 7, 12]:
            df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
            sf = aa.SequenceFeature()
            X = np.asarray(sf.scale_mean(df_seq=df_seq, df_scales=df_scales))
            assert X.shape == (len(df_seq), df_scales.shape[1])
