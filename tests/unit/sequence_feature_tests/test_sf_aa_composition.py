"""This is a script to test the SequenceFeature().aa_composition() method."""
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

CANON = list(aa.utils.LIST_CANONICAL_AA)   # 'A', 'C', 'D', ..., 'Y'

LIST_PARTS = ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
              'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']


def _get_input(n_samples=10):
    """Load a small DOM_GSEC fixture."""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    return df_seq


def _manual_aac(df_parts):
    """Reference AAC over the concatenated jmd_n + tmd + jmd_c span (per-sequence comprehension)."""
    seqs = (df_parts["jmd_n"] + df_parts["tmd"] + df_parts["jmd_c"]).to_list()
    rows = []
    for s in seqs:
        kept = [a for a in s if a in CANON]
        if len(kept) == 0:
            rows.append([np.nan] * 20)
        else:
            rows.append([kept.count(a) / len(kept) for a in CANON])
    return np.array(rows)


class TestAAComposition:
    """Test class for the 'aa_composition' method of the SequenceFeature class."""

    # Positive tests
    def test_valid_df_seq(self):
        """Positive test for valid 'df_seq' input (several sizes)."""
        for n in [3, 5, 10]:
            df_seq = _get_input(n_samples=n)
            sf = aa.SequenceFeature()
            X = sf.aa_composition(df_seq=df_seq)
            assert isinstance(X, np.ndarray)
            assert X.shape == (len(df_seq), 20)

    def test_valid_df_seq_part_based(self):
        """Positive test for a part-based df_seq input."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["ACDE", "AAAA"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert X.shape == (2, 20)

    @settings(max_examples=6, deadline=None)
    @given(list_parts=st.lists(st.sampled_from(LIST_PARTS), min_size=1, max_size=3, unique=True))
    def test_valid_list_parts(self, list_parts):
        """Positive test for valid 'list_parts' inputs (single, str, multiple)."""
        df_seq = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq, list_parts=list_parts)
        assert isinstance(X, np.ndarray)
        assert X.shape == (len(df_seq), 20)
        # A single part may also be passed as a bare string
        X_str = sf.aa_composition(df_seq=df_seq, list_parts=list_parts[0])
        assert X_str.shape == (len(df_seq), 20)

    def test_valid_list_parts_none_is_whole_span(self):
        """list_parts=None equals the explicit jmd_n + tmd + jmd_c part list."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        X_none = sf.aa_composition(df_seq=df_seq, list_parts=None)
        X_explicit = sf.aa_composition(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        assert np.allclose(X_none, X_explicit, equal_nan=True)

    def test_valid_list_parts_single_str(self):
        """A single part as a bare string works and matches a one-element list."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        X_str = sf.aa_composition(df_seq=df_seq, list_parts="tmd")
        X_list = sf.aa_composition(df_seq=df_seq, list_parts=["tmd"])
        assert np.allclose(X_str, X_list, equal_nan=True)

    def test_valid_return_df(self):
        """Positive test for valid 'return_df' input."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq, return_df=False)
        assert isinstance(X, np.ndarray)
        df = sf.aa_composition(df_seq=df_seq, return_df=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CANON
        assert df.shape == (len(df_seq), 20)
        # Values agree between the array and DataFrame forms
        assert np.allclose(np.asarray(df.values, dtype=float), X, equal_nan=True)

    def test_valid_return_df_index(self):
        """return_df=True uses the df_parts index (protein entries) as rows."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        df = sf.aa_composition(df_seq=df_seq, return_df=True)
        assert len(df.index) == len(df_seq)

    # Negative tests
    def test_invalid_df_seq(self):
        """Negative test for invalid 'df_seq' input."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=None)
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq="invalid_input")
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=pd.DataFrame({"wrong": ["A"]}))

    def test_invalid_df_seq_empty(self):
        """Negative test for an empty DataFrame df_seq."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=pd.DataFrame())

    def test_invalid_list_parts(self):
        """Negative test for invalid 'list_parts' input."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=df_seq, list_parts=["not_a_part"])
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=df_seq, list_parts=[])

    def test_invalid_list_parts_type(self):
        """Negative test for a non-str/list list_parts."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=df_seq, list_parts=123)

    def test_invalid_return_df(self):
        """Negative test for invalid 'return_df' input."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=df_seq, return_df="not_a_boolean")
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=df_seq, return_df=None)


class TestAACompositionComplex:
    """Complex positive / negative tests for the 'aa_composition' method."""

    def test_valid_combinations(self):
        """Test with valid combinations of parameters."""
        df_seq = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        for list_parts in [None, "tmd", ["jmd_n", "jmd_c"], ["tmd_jmd"]]:
            for return_df in [True, False]:
                out = sf.aa_composition(df_seq=df_seq, list_parts=list_parts, return_df=return_df)
                if return_df:
                    assert isinstance(out, pd.DataFrame)
                else:
                    assert isinstance(out, np.ndarray)
                assert np.asarray(out).shape == (len(df_seq), 20)

    def test_invalid_combinations(self):
        """Test with invalid combinations of parameters."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=None, return_df="bad")
        with pytest.raises(ValueError):
            sf.aa_composition(df_seq=df_seq, list_parts="bad_part", return_df="bad")

    def test_rows_sum_to_one(self):
        """Every finite row of the composition matrix sums to 1."""
        df_seq = _get_input(n_samples=12)
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        finite = ~np.isnan(X).any(axis=1)
        assert finite.any()
        assert np.allclose(X[finite].sum(axis=1), 1.0)

    def test_values_nonnegative(self):
        """All fractions are in [0, 1]."""
        df_seq = _get_input(n_samples=12)
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        finite = ~np.isnan(X).any(axis=1)
        assert np.all(X[finite] >= 0.0) and np.all(X[finite] <= 1.0)


class TestAACompositionGoldenValues:
    """Hand-computed values, the reference-comprehension match, and edge cases."""

    def test_byte_identical_reference(self):
        """Byte-identical to a hand-coded per-residue count on a tiny known sequence.

        span 'ACDA': A -> 2/4, C -> 1/4, D -> 1/4, all others 0. Assert exact equality.
        """
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["ACDA"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        expected = np.zeros((1, 20))
        expected[0, CANON.index("A")] = 2 / 4
        expected[0, CANON.index("C")] = 1 / 4
        expected[0, CANON.index("D")] = 1 / 4
        assert np.array_equal(X, expected)          # byte-identical, no tolerance
        assert X[0].sum() == 1.0

    def test_golden_matches_reference_comprehension(self):
        """Output equals the reference per-residue comprehension on DOM_GSEC."""
        aa.options["verbose"] = False
        df_seq = aa.load_dataset(name="DOM_GSEC")
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        X_manual = _manual_aac(df_parts=df_parts)
        X = sf.aa_composition(df_seq=df_seq)
        assert X.shape == X_manual.shape
        assert np.allclose(X, X_manual, equal_nan=True)

    def test_byte_identical_frequency_weighted(self):
        """Residues count by frequency: 'AAC' -> A -> 2/3, C -> 1/3, exact."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AAC"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        expected = np.zeros((1, 20))
        expected[0, CANON.index("A")] = 2 / 3
        expected[0, CANON.index("C")] = 1 / 3
        assert np.array_equal(X, expected)

    def test_golden_two_residues(self):
        """A two-residue span 'AC' -> 0.5 at A and 0.5 at C."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AC"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert X[0, CANON.index("A")] == 0.5 and X[0, CANON.index("C")] == 0.5

    def test_verbose_default_is_silent(self):
        """The default SequenceFeature (options['verbose']=False) does not warn on an all-NaN row."""
        aa.options["verbose"] = False
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["XXXX"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.aa_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))
        assert not any("no canonical residue" in str(r.message) for r in records)

    def test_golden_homopolymer(self):
        """A homopolymer 'AAAA' -> fraction 1.0 at 'A', 0 elsewhere."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AAAA"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert X[0, CANON.index("A")] == 1.0
        assert X[0].sum() == 1.0

    def test_golden_single_residue(self):
        """A single-residue span still has a valid AAC (fraction 1.0 at that residue)."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["W"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert X[0, CANON.index("W")] == 1.0
        assert np.isclose(X[0].sum(), 1.0)

    def test_non_canonical_residues_dropped(self):
        """Non-canonical residues ('X') and gaps are ignored before counting."""
        # 'AXC' should equal 'AC' because 'X'/'-' are non-canonical
        df_seq = pd.DataFrame({"entry": ["E1", "E2", "E3"], "jmd_n": ["", "", ""],
                               "tmd": ["AXC", "A-C", "AC"], "jmd_c": ["", "", ""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert np.array_equal(X[0], X[2])
        assert np.array_equal(X[1], X[2])
        assert X[2, CANON.index("A")] == 0.5 and X[2, CANON.index("C")] == 0.5

    def test_non_latin1_residues_dropped(self):
        """Codepoints above 255 (e.g. 'Ā') and lowercase are non-canonical and dropped, not crash."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["AĀC", "AcC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert X[0, CANON.index("A")] == 0.5 and X[0, CANON.index("C")] == 0.5
        assert np.array_equal(X[0], X[1])

    def test_column_order_is_canonical(self):
        """Columns follow ut.LIST_CANONICAL_AA order in the labeled frame."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["ACDEFGHIKLMNPQRSTVWY"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        df = sf.aa_composition(df_seq=df_seq, return_df=True)
        assert list(df.columns) == CANON
        # each canonical AA appears once -> uniform 1/20
        assert np.allclose(df.values, 1 / 20)

    def test_edge_all_non_canonical_is_nan(self):
        """An all-non-canonical / empty span yields an all-NaN row and warns when verbose."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["XXXX", "AC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature(verbose=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.aa_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))         # all-non-canonical -> NaN row
        assert np.all(np.isfinite(X[1]))      # canonical row stays finite
        assert any("no canonical residue" in str(r.message) for r in records)

    def test_edge_all_non_canonical_silent_when_not_verbose(self):
        """No warning is emitted for the NaN row when verbose=False."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["XXXX"], "jmd_c": [""]})
        sf = aa.SequenceFeature(verbose=False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.aa_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))
        assert not any("no canonical residue" in str(r.message) for r in records)

    def test_edge_no_division_by_zero(self):
        """An all-non-canonical span (n_kept=0) does not emit a RuntimeWarning on the 0/0 division."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["XXXX"], "jmd_c": [""]})
        sf = aa.SequenceFeature(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            X = sf.aa_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))

    def test_property_shape_invariant(self):
        """Invariant: result is (n_samples, 20) for every input size."""
        for n in [3, 7, 12]:
            df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
            sf = aa.SequenceFeature()
            X = np.asarray(sf.aa_composition(df_seq=df_seq))
            assert X.shape == (len(df_seq), 20)

    @settings(max_examples=15, deadline=None)
    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1, max_size=30))
    def test_property_finite_rows_sum_to_one(self, seq):
        """Property: any all-canonical span produces a row summing to 1 with counts matching."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": [seq], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.aa_composition(df_seq=df_seq)
        assert np.isclose(X[0].sum(), 1.0)
        expected = np.array([seq.count(a) / len(seq) for a in CANON])
        assert np.allclose(X[0], expected)
