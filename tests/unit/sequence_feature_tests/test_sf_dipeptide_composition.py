"""This is a script to test the SequenceFeature().dipeptide_composition() method."""
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

CANON = list(aa.utils.LIST_CANONICAL_AA)                 # 'A', 'C', 'D', ..., 'Y'
PAIRS = [a + b for a in CANON for b in CANON]            # 'AA', 'AC', ..., 'YY' (400)

LIST_PARTS = ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
              'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']


def _pair_code(x, y):
    """Column index of the ordered pair xy (first residue x, second residue y)."""
    return CANON.index(x) * 20 + CANON.index(y)


def _get_input(n_samples=10):
    """Load a small DOM_GSEC fixture."""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    return df_seq


def _manual_dpc(df_parts):
    """Reference DPC over the concatenated jmd_n + tmd + jmd_c span (drop-then-pair comprehension)."""
    seqs = (df_parts["jmd_n"] + df_parts["tmd"] + df_parts["jmd_c"]).to_list()
    rows = []
    for s in seqs:
        kept = [a for a in s if a in CANON]
        if len(kept) < 2:
            rows.append([np.nan] * 400)
            continue
        counts = np.zeros(400)
        for i in range(len(kept) - 1):
            counts[_pair_code(kept[i], kept[i + 1])] += 1
        rows.append(list(counts / counts.sum()))
    return np.array(rows)


class TestDipeptideComposition:
    """Test class for the 'dipeptide_composition' method of the SequenceFeature class."""

    # Positive tests
    def test_valid_df_seq(self):
        """Positive test for valid 'df_seq' input (several sizes)."""
        for n in [3, 5, 10]:
            df_seq = _get_input(n_samples=n)
            sf = aa.SequenceFeature()
            X = sf.dipeptide_composition(df_seq=df_seq)
            assert isinstance(X, np.ndarray)
            assert X.shape == (len(df_seq), 400)

    def test_valid_df_seq_part_based(self):
        """Positive test for a part-based df_seq input."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["ACDE", "AAAA"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert X.shape == (2, 400)

    @settings(max_examples=6, deadline=None)
    @given(list_parts=st.lists(st.sampled_from(LIST_PARTS), min_size=1, max_size=3, unique=True))
    def test_valid_list_parts(self, list_parts):
        """Positive test for valid 'list_parts' inputs (single, str, multiple)."""
        df_seq = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq, list_parts=list_parts)
        assert isinstance(X, np.ndarray)
        assert X.shape == (len(df_seq), 400)
        X_str = sf.dipeptide_composition(df_seq=df_seq, list_parts=list_parts[0])
        assert X_str.shape == (len(df_seq), 400)

    def test_valid_list_parts_none_is_whole_span(self):
        """list_parts=None equals the explicit jmd_n + tmd + jmd_c part list."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        X_none = sf.dipeptide_composition(df_seq=df_seq, list_parts=None)
        X_explicit = sf.dipeptide_composition(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        assert np.allclose(X_none, X_explicit, equal_nan=True)

    def test_valid_list_parts_single_str(self):
        """A single part as a bare string works and matches a one-element list."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        X_str = sf.dipeptide_composition(df_seq=df_seq, list_parts="tmd")
        X_list = sf.dipeptide_composition(df_seq=df_seq, list_parts=["tmd"])
        assert np.allclose(X_str, X_list, equal_nan=True)

    def test_valid_return_df(self):
        """Positive test for valid 'return_df' input."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq, return_df=False)
        assert isinstance(X, np.ndarray)
        df = sf.dipeptide_composition(df_seq=df_seq, return_df=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == PAIRS
        assert df.shape == (len(df_seq), 400)
        assert np.allclose(np.asarray(df.values, dtype=float), X, equal_nan=True)

    def test_valid_return_df_index(self):
        """return_df=True uses the df_parts index (protein entries) as rows."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        df = sf.dipeptide_composition(df_seq=df_seq, return_df=True)
        assert len(df.index) == len(df_seq)

    # Negative tests
    def test_invalid_df_seq(self):
        """Negative test for invalid 'df_seq' input."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=None)
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq="invalid_input")
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=pd.DataFrame({"wrong": ["A"]}))

    def test_invalid_df_seq_empty(self):
        """Negative test for an empty DataFrame df_seq."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=pd.DataFrame())

    def test_invalid_list_parts(self):
        """Negative test for invalid 'list_parts' input."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=df_seq, list_parts=["not_a_part"])
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=df_seq, list_parts=[])

    def test_invalid_list_parts_type(self):
        """Negative test for a non-str/list list_parts."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=df_seq, list_parts=123)

    def test_invalid_return_df(self):
        """Negative test for invalid 'return_df' input."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=df_seq, return_df="not_a_boolean")
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=df_seq, return_df=None)


class TestDipeptideCompositionComplex:
    """Complex positive / negative tests for the 'dipeptide_composition' method."""

    def test_valid_combinations(self):
        """Test with valid combinations of parameters."""
        df_seq = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        for list_parts in [None, "tmd", ["jmd_n", "jmd_c"], ["tmd_jmd"]]:
            for return_df in [True, False]:
                out = sf.dipeptide_composition(df_seq=df_seq, list_parts=list_parts, return_df=return_df)
                if return_df:
                    assert isinstance(out, pd.DataFrame)
                else:
                    assert isinstance(out, np.ndarray)
                assert np.asarray(out).shape == (len(df_seq), 400)

    def test_invalid_combinations(self):
        """Test with invalid combinations of parameters."""
        df_seq = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=None, return_df="bad")
        with pytest.raises(ValueError):
            sf.dipeptide_composition(df_seq=df_seq, list_parts="bad_part", return_df="bad")

    def test_rows_sum_to_one(self):
        """Every finite row of the composition matrix sums to 1."""
        df_seq = _get_input(n_samples=12)
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        finite = ~np.isnan(X).any(axis=1)
        assert finite.any()
        assert np.allclose(X[finite].sum(axis=1), 1.0)

    def test_values_nonnegative(self):
        """All fractions are in [0, 1]."""
        df_seq = _get_input(n_samples=12)
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        finite = ~np.isnan(X).any(axis=1)
        assert np.all(X[finite] >= 0.0) and np.all(X[finite] <= 1.0)


class TestDipeptideCompositionGoldenValues:
    """Hand-computed values, the reference-comprehension match, and edge cases."""

    def test_byte_identical_reference(self):
        """Byte-identical to a hand-coded adjacent-pair count on a tiny known sequence.

        span 'ACDA' -> adjacent pairs AC, CD, DA, each 1/3; all other 397 pairs 0.
        """
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["ACDA"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        expected = np.zeros((1, 400))
        expected[0, _pair_code("A", "C")] = 1 / 3
        expected[0, _pair_code("C", "D")] = 1 / 3
        expected[0, _pair_code("D", "A")] = 1 / 3
        assert np.array_equal(X, expected)          # byte-identical, no tolerance
        assert X[0].sum() == 1.0

    def test_byte_identical_repeated_pair(self):
        """span 'AAA' -> pair AA counted twice -> fraction 1.0 at 'AA', exact."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AAA"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        expected = np.zeros((1, 400))
        expected[0, _pair_code("A", "A")] = 1.0
        assert np.array_equal(X, expected)

    def test_ordered_pairs_directional(self):
        """Pairs are ordered: 'AC' and 'CA' are distinct columns."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["AC", "CA"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert X[0, _pair_code("A", "C")] == 1.0 and X[0, _pair_code("C", "A")] == 0.0
        assert X[1, _pair_code("C", "A")] == 1.0 and X[1, _pair_code("A", "C")] == 0.0

    def test_golden_matches_reference_comprehension(self):
        """Output equals the reference drop-then-pair comprehension on DOM_GSEC."""
        aa.options["verbose"] = False
        df_seq = aa.load_dataset(name="DOM_GSEC")
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd", "jmd_c"])
        X_manual = _manual_dpc(df_parts=df_parts)
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert X.shape == X_manual.shape
        assert np.allclose(X, X_manual, equal_nan=True)

    def test_non_canonical_dropped_then_paired(self):
        """Non-canonical residues are dropped BEFORE pairing, so an adjacency spans over them."""
        # 'AXC' -> drop X -> 'AC' -> single pair AC (not A? or ?C)
        df_seq = pd.DataFrame({"entry": ["E1", "E2", "E3"], "jmd_n": ["", "", ""],
                               "tmd": ["AXC", "A-C", "AC"], "jmd_c": ["", "", ""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert X[0, _pair_code("A", "C")] == 1.0
        assert np.array_equal(X[0], X[2])
        assert np.array_equal(X[1], X[2])

    def test_non_latin1_dropped(self):
        """Codepoints above 255 and lowercase are dropped before pairing, not crash."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["AĀC", "AcC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert X[0, _pair_code("A", "C")] == 1.0
        assert np.array_equal(X[0], X[1])

    def test_pairs_cross_part_boundary(self):
        """Concatenating multiple parts pairs the last residue of one with the first of the next."""
        # jmd_n 'A', tmd 'C' -> concatenated 'AC' -> one pair AC crossing the jmd_n|tmd boundary
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": ["A"], "tmd": ["C"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq, list_parts=["jmd_n", "tmd"])
        assert X[0, _pair_code("A", "C")] == 1.0
        assert X[0].sum() == 1.0

    def test_no_pair_across_different_sequences(self):
        """Pairs never form across two different sequences (same-sequence mask)."""
        # last residue of E1 is 'D', first of E2 is 'M'; 'DM' must NOT appear
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["AD", "MK"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert X[0, _pair_code("A", "D")] == 1.0
        assert X[1, _pair_code("M", "K")] == 1.0
        assert X[0, _pair_code("D", "M")] == 0.0
        assert X[1, _pair_code("D", "M")] == 0.0

    def test_column_order_pairs(self):
        """Columns follow the AA-major ordered-pair scheme 'AA','AC',...,'YY'."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["AC"], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        df = sf.dipeptide_composition(df_seq=df_seq, return_df=True)
        assert df.columns[0] == "AA" and df.columns[-1] == "YY"
        assert list(df.columns) == PAIRS
        assert df.iloc[0]["AC"] == 1.0

    def test_edge_single_residue_is_nan(self):
        """A single canonical residue has no adjacent pair -> all-NaN row, warns when verbose."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["A", "AC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature(verbose=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.dipeptide_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))
        assert np.all(np.isfinite(X[1]))
        assert any("fewer than two canonical residues" in str(r.message) for r in records)

    def test_edge_all_non_canonical_is_nan(self):
        """An all-non-canonical / empty span yields an all-NaN row."""
        df_seq = pd.DataFrame({"entry": ["E1", "E2"], "jmd_n": ["", ""],
                               "tmd": ["XXXX", "AC"], "jmd_c": ["", ""]})
        sf = aa.SequenceFeature(verbose=False)
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))
        assert np.all(np.isfinite(X[1]))

    def test_edge_single_residue_silent_when_not_verbose(self):
        """No warning is emitted for the NaN row when verbose=False."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["A"], "jmd_c": [""]})
        sf = aa.SequenceFeature(verbose=False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.dipeptide_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))
        assert not any("fewer than two canonical residues" in str(r.message) for r in records)

    def test_edge_no_division_by_zero(self):
        """An all-non-canonical span (n_pairs=0) does not emit a RuntimeWarning on the 0/0 division."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": ["XXXX"], "jmd_c": [""]})
        sf = aa.SequenceFeature(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            X = sf.dipeptide_composition(df_seq=df_seq)
        assert np.all(np.isnan(X[0]))

    def test_property_shape_invariant(self):
        """Invariant: result is (n_samples, 400) for every input size."""
        for n in [3, 7, 12]:
            df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
            sf = aa.SequenceFeature()
            X = np.asarray(sf.dipeptide_composition(df_seq=df_seq))
            assert X.shape == (len(df_seq), 400)

    @settings(max_examples=15, deadline=None)
    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=2, max_size=30))
    def test_property_finite_rows_sum_to_one(self, seq):
        """Property: any all-canonical span of length >= 2 gives a row summing to 1 matching the count."""
        df_seq = pd.DataFrame({"entry": ["E1"], "jmd_n": [""], "tmd": [seq], "jmd_c": [""]})
        sf = aa.SequenceFeature()
        X = sf.dipeptide_composition(df_seq=df_seq)
        assert np.isclose(X[0].sum(), 1.0)
        counts = np.zeros(400)
        for i in range(len(seq) - 1):
            counts[_pair_code(seq[i], seq[i + 1])] += 1
        assert np.allclose(X[0], counts / counts.sum())
