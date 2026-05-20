"""This is a script to test EmbeddingPreprocessor.build_pseudo_scales()."""
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# Helpers --------------------------------------------------------------
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _make_fixture(n=5, D=8, seed=0, alphabet=ALPHABET):
    """Build a deterministic (df_seq, embeddings) fixture."""
    rng = np.random.default_rng(seed)
    seqs = []
    for i in range(n):
        length = 10 + (i % 5)
        seqs.append("".join(rng.choice(list(alphabet), size=length)))
    df_seq = pd.DataFrame({
        "entry": [f"P{i}" for i in range(n)],
        "sequence": seqs,
    })
    embeddings = {
        f"P{i}": rng.standard_normal((len(seqs[i]), D)).astype("float32")
        for i in range(n)
    }
    return df_seq, embeddings


# Normal cases ---------------------------------------------------------
class TestBuildPseudoScales:
    """Positive and parameter-level negative tests for build_pseudo_scales."""

    # Positive cases
    def test_returns_dataframe(self):
        df_seq, embeddings = _make_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        assert isinstance(df, pd.DataFrame)

    def test_shape_is_20_by_D(self):
        df_seq, embeddings = _make_fixture(n=5, D=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        assert df.shape == (20, 8)

    @given(D=some.integers(min_value=1, max_value=64))
    def test_shape_tracks_D(self, D):
        df_seq, embeddings = _make_fixture(n=5, D=D)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        assert df.shape == (20, D)

    def test_index_is_canonical_aa_alphabetical(self):
        df_seq, embeddings = _make_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        assert list(df.index) == list(ALPHABET)

    def test_columns_are_dim_labels(self):
        df_seq, embeddings = _make_fixture(D=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        assert list(df.columns) == [f"dim_{i}" for i in range(5)]

    def test_emits_user_warning_about_dataset_dependence(self):
        df_seq, embeddings = _make_fixture()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 1
        assert "dataset-dependent" in str(user_warns[0].message).lower()

    def test_absent_aa_becomes_nan_row(self):
        # Build a corpus that contains only a handful of AAs — others should be NaN
        df_seq = pd.DataFrame({"entry": ["P0"], "sequence": ["ACG"]})
        embeddings = {"P0": np.ones((3, 4), dtype="float32")}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        present = {"A", "C", "G"}
        for aa_letter in ALPHABET:
            row = df.loc[aa_letter]
            if aa_letter in present:
                assert not row.isna().any(), f"{aa_letter} should have non-NaN values"
            else:
                assert row.isna().all(), f"{aa_letter} should be NaN (absent from corpus)"

    def test_present_aa_mean_matches_manual(self):
        # AA 'A' appears in two positions, embedding values 1.0 and 3.0 → mean 2.0
        df_seq = pd.DataFrame({"entry": ["P0"], "sequence": ["AGAG"]})
        emb = np.array([[1.0], [10.0], [3.0], [20.0]], dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings={"P0": emb})
        assert df.loc["A", "dim_0"] == pytest.approx(2.0)
        assert df.loc["G", "dim_0"] == pytest.approx(15.0)

    def test_handles_non_canonical_residues_by_skipping(self):
        # 'X' is a non-canonical residue — its embeddings should NOT affect any pseudo-scale row.
        df_seq = pd.DataFrame({"entry": ["P0"], "sequence": ["AXA"]})
        emb = np.array([[1.0], [999.0], [3.0]], dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings={"P0": emb})
        assert df.loc["A", "dim_0"] == pytest.approx(2.0)
        # All non-A rows that aren't in the corpus stay NaN — X is silently dropped
        for aa_letter in ALPHABET:
            if aa_letter != "A":
                assert pd.isna(df.loc[aa_letter, "dim_0"])

    def test_deterministic_on_rerun(self):
        df_seq, embeddings = _make_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df1 = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
            df2 = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        pd.testing.assert_frame_equal(df1, df2)

    # Negative cases
    def test_invalid_df_seq_none(self):
        _, embeddings = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=None, embeddings=embeddings)

    def test_invalid_embeddings_none(self):
        df_seq, _ = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=None)

    def test_invalid_embeddings_not_dict(self):
        df_seq, _ = _make_fixture()
        for bad in ["not a dict", 42, [1, 2, 3], np.zeros((3, 3))]:
            with pytest.raises(ValueError):
                aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=bad)

    def test_invalid_embeddings_missing_entry(self):
        df_seq, embeddings = _make_fixture()
        del embeddings["P3"]
        with pytest.raises(ValueError, match="missing"):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_embeddings_value_not_ndarray(self):
        df_seq, embeddings = _make_fixture()
        embeddings["P0"] = [[1.0, 2.0]] * 10  # nested list, not ndarray
        with pytest.raises(ValueError, match="np.ndarray"):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_embeddings_value_wrong_ndim(self):
        df_seq, embeddings = _make_fixture()
        embeddings["P0"] = np.zeros((10,), dtype="float32")  # 1D, expected 2D
        with pytest.raises(ValueError, match="2D"):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_embeddings_length_mismatch(self):
        df_seq, embeddings = _make_fixture()
        seq_len = len(df_seq.iloc[0]["sequence"])
        embeddings["P0"] = np.zeros((seq_len + 5, 8), dtype="float32")
        with pytest.raises(ValueError, match="sequence length"):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_embeddings_inconsistent_D(self):
        df_seq, embeddings = _make_fixture()
        seq_len = len(df_seq.iloc[1]["sequence"])
        embeddings["P1"] = np.zeros((seq_len, 16), dtype="float32")  # D=16 vs others' D=8
        with pytest.raises(ValueError, match="consistent embedding dimensionality"):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)


# Complex / interaction cases ------------------------------------------
class TestBuildPseudoScalesComplex:
    """Combinations and edge interactions for build_pseudo_scales."""

    def test_single_entry_single_aa(self):
        df_seq = pd.DataFrame({"entry": ["P0"], "sequence": ["W"]})
        emb = np.array([[5.0, -3.0]], dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings={"P0": emb})
        assert df.shape == (20, 2)
        assert df.loc["W", "dim_0"] == pytest.approx(5.0)
        assert df.loc["W", "dim_1"] == pytest.approx(-3.0)
        # All other rows are NaN
        for aa_letter in ALPHABET:
            if aa_letter != "W":
                assert df.loc[aa_letter].isna().all()

    def test_multiple_entries_aggregate_correctly(self):
        # Two entries, both contain 'A' at known positions
        df_seq = pd.DataFrame({"entry": ["P0", "P1"], "sequence": ["AA", "AA"]})
        embeddings = {
            "P0": np.array([[1.0], [2.0]], dtype="float32"),
            "P1": np.array([[3.0], [4.0]], dtype="float32"),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        # A occurs 4 times total: values [1,2,3,4], mean = 2.5
        assert df.loc["A", "dim_0"] == pytest.approx(2.5)

    def test_high_D_does_not_explode(self):
        # Stress a larger D
        df_seq, embeddings = _make_fixture(n=3, D=128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        assert df.shape == (20, 128)
        # Result should have no Infs from any normalization step
        assert not np.isinf(df.values).any()

    def test_zero_length_sequence_is_handled(self):
        df_seq = pd.DataFrame({"entry": ["P0", "P1"], "sequence": ["", "ACG"]})
        embeddings = {
            "P0": np.zeros((0, 4), dtype="float32"),
            "P1": np.ones((3, 4), dtype="float32"),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings=embeddings)
        # Only A, C, G should be filled (from P1); shape is still (20, 4)
        assert df.shape == (20, 4)
        assert not df.loc["A"].isna().any()

    def test_invalid_combined_none_inputs(self):
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=None, embeddings=None)

    def test_invalid_combined_empty_dict_and_valid_df_seq(self):
        df_seq, _ = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor.build_pseudo_scales(df_seq=df_seq, embeddings={})
