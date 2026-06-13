"""This is a script to test the data-handling seams (real components, no mocks).

Integration tier (ADR-0031).

Seams covered:
  11. to_fasta -> read_fasta round-trip
  12. SequencePreprocessor.encode_one_hot / encode_integer -> downstream consumer
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

pytestmark = pytest.mark.integration

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Seam 11: to_fasta -> read_fasta round-trip
# ---------------------------------------------------------------------------
class TestFastaRoundTrip:
    """A df_seq survives a write/read round-trip through the FASTA layer."""

    def test_round_trip_preserves_entries_and_sequences(self, tmp_path):
        df = aa.load_dataset(name="DOM_GSEC", n=4)[["entry", "sequence"]].copy()
        fp = str(tmp_path / "seqs.fasta")
        aa.to_fasta(df, file_path=fp)
        df_read = aa.read_fasta(fp)
        assert {"entry", "sequence"}.issubset(df_read.columns)
        assert sorted(df_read["entry"]) == sorted(df["entry"])
        assert set(df_read["sequence"]) == set(df["sequence"])

    def test_round_trip_is_stable_on_re_read(self, tmp_path):
        # Property: a second write/read of the read-back frame is a fixed point.
        df = aa.load_dataset(name="DOM_GSEC", n=3)[["entry", "sequence"]].copy()
        fp1, fp2 = str(tmp_path / "a.fasta"), str(tmp_path / "b.fasta")
        aa.to_fasta(df, file_path=fp1)
        df_read1 = aa.read_fasta(fp1)
        aa.to_fasta(df_read1, file_path=fp2)
        df_read2 = aa.read_fasta(fp2)
        pd.testing.assert_frame_equal(
            df_read1.sort_values("entry").reset_index(drop=True),
            df_read2.sort_values("entry").reset_index(drop=True))

    def test_missing_sequence_column_rejected(self, tmp_path):
        # Composition failure: to_fasta needs a df_seq carrying a sequence column.
        with pytest.raises(ValueError, match="sequence"):
            aa.to_fasta(pd.DataFrame({"entry": ["a"]}), file_path=str(tmp_path / "x.fasta"))


# ---------------------------------------------------------------------------
# Seam 12: SequencePreprocessor encoders -> downstream consumer
# ---------------------------------------------------------------------------
class TestEncoderToConsumer:
    """One-hot / integer encodings are well-formed matrices for downstream use."""

    def test_one_hot_feeds_clustering(self):
        seqs = aa.load_dataset(name="DOM_GSEC", n=6)["sequence"].str[:10].to_list()
        X, features = aa.SequencePreprocessor.encode_one_hot(list_seq=seqs)
        X = np.asarray(X)
        assert X.shape == (len(seqs), 10 * len(ALPHABET))
        assert len(features) == X.shape[1]
        # The encoded matrix is a valid feature matrix for AAclust.
        aac = aa.AAclust(verbose=False).fit(X, n_clusters=3)
        assert len(np.unique(aac.labels_)) == 3

    @settings(max_examples=5, deadline=None)
    @given(win=some.integers(min_value=4, max_value=12))
    def test_one_hot_width_invariant(self, win):
        # Property: one-hot width is always (sequence length) x (alphabet size).
        seqs = aa.load_dataset(name="DOM_GSEC", n=4)["sequence"].str[:win].to_list()
        X, _ = aa.SequencePreprocessor.encode_one_hot(list_seq=seqs)
        assert np.asarray(X).shape[1] == win * len(ALPHABET)

    def test_ragged_sequences_padded_to_common_width(self):
        # Composition contract: variable-length inputs are padded to one width.
        X, _ = aa.SequencePreprocessor.encode_one_hot(list_seq=["ACDE", "ACDEFGHI", "AC"])
        X = np.asarray(X)
        assert X.shape == (3, 8 * len(ALPHABET))  # widest sequence (8) sets the width

    def test_out_of_alphabet_char_rejected(self):
        # Composition failure: a residue outside the alphabet must raise clearly.
        with pytest.raises(ValueError, match="alphabet"):
            aa.SequencePreprocessor.encode_one_hot(list_seq=["ACDZ"])
