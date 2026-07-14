"""This is a script to test the SequenceFeature().acc() (scale auto-covariance) method."""
import inspect
from hypothesis import given, settings
import hypothesis.strategies as st
import warnings
import pytest
import numpy as np
import pandas as pd
import aaanalysis as aa

# Set default deadline from 200 to None
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False

ORDER = "ACDEFGHIKLMNPQRSTVWY"
LIST_PARTS = ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
              'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']


def _get_input(n_samples=10):
    """Load a small DOM_GSEC fixture and the default scales."""
    aa.options["verbose"] = False
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n_samples)
    df_scales = aa.load_scales()
    return df_seq, df_scales


def _index_scales(n_scales=1):
    """A tiny index-valued scale table (S1 = position+1, S2 = 2*position, ...)."""
    data = {}
    for j in range(n_scales):
        data[f"S{j + 1}"] = {a: float((j + 1) * (i + 1) if j == 0 else (j + 1) * i)
                             for i, a in enumerate(ORDER)}
    return pd.DataFrame(data)


def _ref_acc(df_parts, df_scales, n_lag):
    """Documented reference: mean-centered scale auto-covariance, lag-major columns.

    For each row the parts are concatenated into one span, residues absent from
    ``df_scales.index`` are dropped, and for scale ``j`` / lag ``k``
    ``AC(j, k) = sum_i (S[i, j] - mean_j)(S[i + k, j] - mean_j) / (N - k)`` (``mean_j`` the
    span's scale mean over the ``N`` scored residues). A lag with ``N - k < 1`` is ``NaN``.
    """
    parts = list(df_parts.columns)
    scales = list(df_scales.columns)
    rows = []
    for _, r in df_parts.iterrows():
        span = "".join(str(r[p]) for p in parts)
        kept = [a for a in span if a in df_scales.index]
        S = (df_scales.loc[kept, scales].to_numpy(dtype=float)
             if kept else np.zeros((0, len(scales))))
        N = S.shape[0]
        blocks = []
        for k in range(1, n_lag + 1):
            if N - k < 1:
                blocks.append(np.full(len(scales), np.nan))
            else:
                C = S - S.mean(axis=0)
                blocks.append((C[:-k] * C[k:]).sum(axis=0) / (N - k))
        rows.append(np.concatenate(blocks))
    return np.array(rows)


def _seq_from_tmds(list_tmd, jmd_n="", jmd_c=""):
    """Part-based df_seq where the whole span is exactly each given TMD string."""
    return pd.DataFrame({"entry": [f"E{i}" for i in range(len(list_tmd))],
                         "jmd_n": [jmd_n] * len(list_tmd),
                         "tmd": list(list_tmd),
                         "jmd_c": [jmd_c] * len(list_tmd)})


class TestAcc:
    """Test class for the 'acc' method of the SequenceFeature class."""

    # Positive tests
    def test_valid_df_seq(self):
        """Positive test for valid 'df_seq' input (several sizes)."""
        for n in [3, 5, 10]:
            df_seq, df_scales = _get_input(n_samples=n)
            sf = aa.SequenceFeature()
            X = sf.acc(df_seq=df_seq, df_scales=df_scales)
            assert isinstance(X, np.ndarray)
            assert X.shape == (len(df_seq), df_scales.shape[1])

    def test_valid_df_scales(self):
        """Positive test for valid 'df_scales' input, incl. a column subset and the default."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales)
        assert X.shape[1] == df_scales.shape[1]
        sub = df_scales.iloc[:, :5]
        X_sub = sf.acc(df_seq=df_seq, df_scales=sub)
        assert X_sub.shape == (len(df_seq), 5)
        X_def = sf.acc(df_seq=df_seq)
        assert isinstance(X_def, np.ndarray)
        assert X_def.shape[0] == len(df_seq)

    @settings(max_examples=6, deadline=None)
    @given(list_parts=st.lists(st.sampled_from(LIST_PARTS), min_size=1, max_size=3, unique=True))
    def test_valid_list_parts(self, list_parts):
        """Positive test for valid 'list_parts' inputs (single, str, multiple)."""
        df_seq, df_scales = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=list_parts)
        assert isinstance(X, np.ndarray)
        assert X.shape == (len(df_seq), df_scales.shape[1])
        X_str = sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=list_parts[0])
        assert X_str.shape == (len(df_seq), df_scales.shape[1])

    def test_valid_list_parts_none_is_whole_span(self):
        """list_parts=None equals the explicit jmd_n + tmd + jmd_c part list."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        X_none = sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=None)
        X_explicit = sf.acc(df_seq=df_seq, df_scales=df_scales,
                            list_parts=["jmd_n", "tmd", "jmd_c"])
        assert np.allclose(X_none, X_explicit, equal_nan=True)

    def test_valid_n_lag_shapes(self):
        """Output width is n_scales * n_lag for a range of lags."""
        df_seq, df_scales = _get_input(n_samples=6)
        sf = aa.SequenceFeature()
        for n_lag in [1, 2, 3, 5]:
            X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=n_lag)
            assert X.shape == (len(df_seq), df_scales.shape[1] * n_lag)

    def test_valid_n_lag_default_is_one(self):
        """The default n_lag=1 gives exactly n_scales columns."""
        df_seq, df_scales = _get_input(n_samples=5)
        sf = aa.SequenceFeature()
        X_default = sf.acc(df_seq=df_seq, df_scales=df_scales)
        X_one = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        assert X_default.shape[1] == df_scales.shape[1]
        assert np.allclose(X_default, X_one, equal_nan=True)

    def test_valid_return_df(self):
        """Positive test for valid 'return_df' input, incl. lag-major column names."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2, return_df=False)
        assert isinstance(X, np.ndarray)
        df = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2, return_df=True)
        assert isinstance(df, pd.DataFrame)
        expected_cols = [f"{s}_lag{k}" for k in (1, 2) for s in df_scales.columns]
        assert list(df.columns) == expected_cols
        assert df.shape == (len(df_seq), df_scales.shape[1] * 2)
        assert np.allclose(np.asarray(df.values, dtype=float), X, equal_nan=True)

    # Negative tests
    def test_invalid_df_seq(self):
        """Negative test for invalid 'df_seq' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.acc(df_seq=None, df_scales=df_scales)
        with pytest.raises(ValueError):
            sf.acc(df_seq="invalid_input", df_scales=df_scales)
        with pytest.raises(ValueError):
            sf.acc(df_seq=pd.DataFrame({"wrong": ["A"]}), df_scales=df_scales)

    def test_invalid_df_scales(self):
        """Negative test for invalid 'df_scales' input."""
        df_seq, _ = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.acc(df_seq=df_seq, df_scales="invalid_input")

    def test_invalid_list_parts(self):
        """Negative test for invalid 'list_parts' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=["not_a_part"])
        with pytest.raises(ValueError):
            sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=[])

    def test_invalid_n_lag(self):
        """Negative test for invalid 'n_lag' input (must be an integer >= 1)."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        for bad in [0, -1, 1.5, None]:
            with pytest.raises(ValueError):
                sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=bad)

    def test_invalid_return_df(self):
        """Negative test for invalid 'return_df' input."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.acc(df_seq=df_seq, df_scales=df_scales, return_df="not_a_boolean")


class TestAccComplex:
    """Complex positive / negative tests and edge cases for the 'acc' method."""

    def test_valid_combinations(self):
        """Test with valid combinations of parameters."""
        df_seq, df_scales = _get_input(n_samples=8)
        sf = aa.SequenceFeature()
        for list_parts in [None, "tmd", ["jmd_n", "jmd_c"], ["tmd_jmd"]]:
            for n_lag in [1, 3]:
                for return_df in [True, False]:
                    out = sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=list_parts,
                                 n_lag=n_lag, return_df=return_df)
                    if return_df:
                        assert isinstance(out, pd.DataFrame)
                    else:
                        assert isinstance(out, np.ndarray)
                    assert np.asarray(out).shape == (len(df_seq), df_scales.shape[1] * n_lag)

    def test_invalid_combinations(self):
        """Test with invalid combinations of parameters."""
        df_seq, df_scales = _get_input()
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.acc(df_seq=None, df_scales="invalid")
        with pytest.raises(ValueError):
            sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=0, return_df="bad")

    def test_multiple_parts_concatenated(self):
        """Multiple parts are concatenated per sequence before the covariance is taken."""
        df_seq, df_scales = _get_input(n_samples=6)
        sf = aa.SequenceFeature()
        X_multi = sf.acc(df_seq=df_seq, df_scales=df_scales, list_parts=["jmd_n", "tmd"], n_lag=2)
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["jmd_n", "tmd"])
        ref = _ref_acc(df_parts=df_parts, df_scales=df_scales, n_lag=2)
        assert X_multi.shape == ref.shape
        assert np.allclose(X_multi, ref, equal_nan=True)

    def test_short_span_lag_becomes_nan(self):
        """A span with exactly 2 scored residues: lag 1 finite, lags >= 2 all NaN."""
        df_scales = _index_scales(n_scales=1)
        df_seq = _seq_from_tmds(["AC"])  # N = 2
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=3)
        assert X.shape == (1, 3)
        assert np.isfinite(X[0, 0])       # lag 1 (N - 1 = 1)
        assert np.isnan(X[0, 1])          # lag 2 (N - 2 = 0)
        assert np.isnan(X[0, 2])          # lag 3 (N - 3 < 0)

    def test_single_residue_span_all_nan(self):
        """A span with a single scored residue (N = 1) is all-NaN for every lag."""
        df_scales = _index_scales(n_scales=2)
        df_seq = _seq_from_tmds(["A"])
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2)
        assert np.all(np.isnan(X[0]))

    def test_empty_scored_span_all_nan(self):
        """A span with no scored residue (all gaps, N = 0) is an all-NaN row."""
        df_scales = _index_scales(n_scales=2)
        df_seq = _seq_from_tmds(["----", "ACDE"])  # gap-only span has no scored residue
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        assert np.all(np.isnan(X[0]))
        assert np.all(np.isfinite(X[1]))

    def test_all_non_canonical_all_nan_and_warns(self):
        """An all-non-canonical span yields an all-NaN row and warns when verbose."""
        df_scales = _index_scales(n_scales=2)
        df_seq = _seq_from_tmds(["XXXX", "ACD"])
        sf = aa.SequenceFeature(verbose=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        assert np.all(np.isnan(X[0]))       # all-non-canonical -> NaN row
        assert np.all(np.isfinite(X[1]))    # scored row stays finite
        assert any("too short for lag 1" in str(r.message) for r in records)

    def test_all_non_canonical_silent_when_not_verbose(self):
        """No warning is emitted for the all-NaN row when verbose=False."""
        df_scales = _index_scales(n_scales=1)
        df_seq = _seq_from_tmds(["XXXX"])
        sf = aa.SequenceFeature(verbose=False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        assert np.all(np.isnan(X[0]))
        assert not any("too short for lag 1" in str(r.message) for r in records)

    def test_non_canonical_residues_dropped(self):
        """Non-canonical residues ('X') are dropped before the covariance (span crosses them)."""
        df_scales = _index_scales(n_scales=2)
        df_seq = _seq_from_tmds(["AXCD", "ACD"])  # 'X' dropped -> both reduce to 'ACD'
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2)
        assert np.allclose(X[0], X[1], equal_nan=True)

    def test_non_latin1_residues_dropped(self):
        """Codepoints above 255 and lowercase are treated as non-canonical and dropped."""
        df_scales = _index_scales(n_scales=1)
        df_seq = _seq_from_tmds(["AĀCD", "AcCD", "ACD"])
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2)
        assert np.allclose(X[0], X[2], equal_nan=True)
        assert np.allclose(X[1], X[2], equal_nan=True)

    def test_multi_char_scale_label_never_matches_residue(self):
        """A multi-character index label in df_scales must not be matched by a residue."""
        df_scales = _index_scales(n_scales=1)
        df_scales.loc["AC"] = 99.0  # multi-character label that must be ignored
        df_seq = _seq_from_tmds(["ACD"])
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2)
        ref = _ref_acc(df_parts=sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"]),
                       df_scales=df_scales.drop(index="AC"), n_lag=2)
        assert np.allclose(X, ref, equal_nan=True)

    def test_default_scales_shape(self):
        """df_scales=None uses the bundled scale set (586 scales) -> 586 * n_lag columns."""
        df_seq, df_scales = _get_input(n_samples=4)
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, n_lag=2)
        assert X.shape == (len(df_seq), df_scales.shape[1] * 2)


class TestAccGoldenValues:
    """Hand-computed values, reference-implementation match, and shape properties."""

    def test_golden_hand_computed_n_lag1(self):
        """Exact hand values for two spans at n_lag=1 (S1 = position + 1)."""
        df_scales = _index_scales(n_scales=1)
        # E1='ACD' -> S1 [1,2,3], mean 2, C [-1,0,1]; lag1 = ((-1)(0)+(0)(1))/2 = 0
        # E2='ADC' -> S1 [1,3,2], mean 2, C [-1,1,0]; lag1 = ((-1)(1)+(1)(0))/2 = -0.5
        df_seq = _seq_from_tmds(["ACD", "ADC"])
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        assert X.shape == (2, 1)
        assert X[0, 0] == 0.0
        assert X[1, 0] == -0.5

    def test_golden_hand_computed_n_lag3(self):
        """Exact hand values incl. NaN for the too-short lag at n_lag=3."""
        df_scales = _index_scales(n_scales=1)
        # E1='ACD' (N=3): lag1=0, lag2=((-1)(1))/1=-1, lag3 N-3=0 -> NaN
        # E2='ADC' (N=3): lag1=-0.5, lag2=((-1)(0))/1=0, lag3 -> NaN
        df_seq = _seq_from_tmds(["ACD", "ADC"])
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=3)
        assert X.shape == (2, 3)
        assert X[0, 0] == 0.0 and X[0, 1] == -1.0 and np.isnan(X[0, 2])
        assert X[1, 0] == -0.5 and X[1, 1] == 0.0 and np.isnan(X[1, 2])

    def test_golden_two_scales_lag_major(self):
        """Two-scale n_lag=2 output is lag-major (all scales lag1, then all scales lag2)."""
        df_scales = _index_scales(n_scales=2)  # S1 = pos+1, S2 = 2*pos
        # E1='ACD': S1 [1,2,3] -> lag1 0, lag2 -1 ; S2 [0,2,4] -> lag1 0, lag2 -4
        df_seq = _seq_from_tmds(["ACD"])
        sf = aa.SequenceFeature()
        df = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=2, return_df=True)
        assert list(df.columns) == ["S1_lag1", "S2_lag1", "S1_lag2", "S2_lag2"]
        row = df.iloc[0]
        assert row["S1_lag1"] == 0.0 and row["S2_lag1"] == 0.0
        assert row["S1_lag2"] == -1.0 and row["S2_lag2"] == -4.0

    def test_golden_matches_reference_n_lag1(self):
        """Byte-identical to the documented reference on DOM_GSEC for n_lag=1."""
        aa.options["verbose"] = False
        df_seq = aa.load_dataset(name="DOM_GSEC")
        df_scales = aa.load_scales()
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
        ref = _ref_acc(df_parts=df_parts, df_scales=df_scales, n_lag=1)
        assert X.shape == ref.shape
        assert np.allclose(X, ref, equal_nan=True, rtol=0, atol=1e-9)

    def test_golden_matches_reference_n_lag3(self):
        """Byte-identical to the documented reference on DOM_GSEC for n_lag=3."""
        aa.options["verbose"] = False
        df_seq = aa.load_dataset(name="DOM_GSEC")
        df_scales = aa.load_scales()
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=3)
        df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"])
        ref = _ref_acc(df_parts=df_parts, df_scales=df_scales, n_lag=3)
        assert X.shape == ref.shape
        assert np.allclose(X, ref, equal_nan=True, rtol=0, atol=1e-9)

    def test_return_df_matches_array(self):
        """The DataFrame and array forms carry the same values."""
        df_seq, df_scales = _get_input(n_samples=6)
        sf = aa.SequenceFeature()
        X = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=3)
        df = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=3, return_df=True)
        assert np.allclose(np.asarray(df.values, dtype=float), X, equal_nan=True)

    def test_lag1_equals_scale_composition_relationship(self):
        """Sanity: acc uses the same scored-residue set as scale_composition (same NaN rows)."""
        df_scales = _index_scales(n_scales=2)
        df_seq = _seq_from_tmds(["XXXX", "ACDE"])
        sf = aa.SequenceFeature()
        X_acc = sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=1)
        X_sc = sf.scale_composition(df_seq=df_seq, df_scales=df_scales)
        # The all-non-canonical span is NaN in both; the scored span is finite in both
        assert np.all(np.isnan(X_acc[0])) and np.all(np.isnan(X_sc[0]))
        assert np.all(np.isfinite(X_acc[1])) and np.all(np.isfinite(X_sc[1]))

    @settings(max_examples=8, deadline=None)
    @given(n_lag=st.integers(min_value=1, max_value=6))
    def test_property_shape_varying_n_lag(self, n_lag):
        """Invariant: result is (n_samples, n_scales * n_lag) for every n_lag."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        df_scales = aa.load_scales().iloc[:, :4]
        sf = aa.SequenceFeature()
        X = np.asarray(sf.acc(df_seq=df_seq, df_scales=df_scales, n_lag=n_lag))
        assert X.shape == (len(df_seq), df_scales.shape[1] * n_lag)

    def test_property_shape_invariant_sizes(self):
        """Invariant: result is (n_samples, n_scales) at the default n_lag for every size."""
        df_scales = aa.load_scales()
        for n in [3, 7, 12]:
            df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
            sf = aa.SequenceFeature()
            X = np.asarray(sf.acc(df_seq=df_seq, df_scales=df_scales))
            assert X.shape == (len(df_seq), df_scales.shape[1])

    def test_no_per_sequence_python_loop(self):
        """The hot path is vectorized: get_acc_ has no per-sequence Python loop / DataFrame.loc."""
        from aaanalysis.feature_engineering._backend.cpp.sequence_feature import get_acc_
        src = inspect.getsource(get_acc_)
        for token in ["iterrows", "itertuples", "in range(n_seq)", ".loc[", ".iloc["]:
            assert token not in src, f"unexpected per-sequence construct '{token}' in get_acc_"
