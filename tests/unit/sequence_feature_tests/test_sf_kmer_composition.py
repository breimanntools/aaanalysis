"""Tests for SequenceFeature.kmer_composition().

``sf.kmer_composition(df_seq, k=1)`` builds the no-positional-split k-mer-composition
baseline: the fraction of each of the ``20 ** k`` ordered overlapping k-mers of adjacent
canonical residues over the concatenated Parts of each sequence. ``k=1`` is amino-acid
composition (AAC, identical to :meth:`aa_composition`) and ``k=2`` dipeptide composition
(DPC, identical to :meth:`dipeptide_composition`); higher ``k`` captures longer local order.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False
warnings.filterwarnings("ignore")

AA = list(ut.LIST_CANONICAL_AA)


# I Helpers
def _df_seq(seqs):
    """Minimal df_seq (whole sequence = tmd, no flanks) for composition over the full span."""
    n = len(seqs)
    return pd.DataFrame({"entry": [f"P{i}" for i in range(n)], "sequence": seqs})


def _kmers(k):
    return ["".join(p) for p in itertools.product(AA, repeat=k)]


def _hand_kmer_fraction(seq, k):
    """Reference k-mer fraction vector over the canonical-only, gap-free span (hand loop)."""
    clean = [c for c in seq if c in AA]
    codes = _kmers(k)
    idx = {km: i for i, km in enumerate(codes)}
    counts = np.zeros(len(codes))
    for i in range(len(clean) - k + 1):
        counts[idx["".join(clean[i:i + k])]] += 1
    total = counts.sum()
    return counts / total if total else np.full(len(codes), np.nan)


# II Per-parameter tests
class TestKmerComposition:
    """Per-parameter positive and negative coverage."""

    # df_seq
    def test_valid_df_seq(self):
        sf = aa.SequenceFeature()
        X = sf.kmer_composition(df_seq=_df_seq(["ACDEFGHIK", "MKKLLA"]), k=1)
        assert X.shape == (2, 20)

    def test_invalid_df_seq_none(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.kmer_composition(df_seq=None, k=1)

    # k
    @pytest.mark.parametrize("k,ncol", [(1, 20), (2, 400), (3, 8000)])
    def test_valid_k_shapes(self, k, ncol):
        sf = aa.SequenceFeature()
        X = sf.kmer_composition(df_seq=_df_seq(["ACDEFGHIKLMNPQRSTVWY"]), k=k)
        assert X.shape == (1, ncol)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, False, 5, 10, "2", None])
    def test_invalid_k(self, bad):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.kmer_composition(df_seq=_df_seq(["ACDEFGHIK"]), k=bad)

    # list_parts
    def test_valid_list_parts_subset(self):
        sf = aa.SequenceFeature()
        df_seq = pd.DataFrame({"entry": ["P0"], "sequence": ["ACDEFGHIKLMNPQRSTVWY"],
                               "tmd_start": [5], "tmd_stop": [14]})
        X_tmd = sf.kmer_composition(df_seq=df_seq, k=1, list_parts="tmd")
        assert X_tmd.shape == (1, 20)

    # return_df
    def test_return_df_labels_and_index(self):
        sf = aa.SequenceFeature()
        df_seq = _df_seq(["ACDEFGHIK", "MKKLLA"])
        df = sf.kmer_composition(df_seq=df_seq, k=2, return_df=True)
        assert list(df.columns) == _kmers(2)
        assert list(df.index) == list(sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_jmd"]).index)

    def test_invalid_return_df(self):
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.kmer_composition(df_seq=_df_seq(["ACDEFGHIK"]), k=1, return_df="yes")


# III Equivalence + contract tests
class TestKmerCompositionContract:
    """Equivalence to the named baselines, shapes, dtype, normalization."""

    def test_k1_equals_aa_composition(self):
        sf = aa.SequenceFeature()
        df_seq = _df_seq(["ACDEFGHIKLMNPQRSTVWY", "MKKLLA", "AAAC", "WY", "A", "XZ"])
        assert np.array_equal(sf.kmer_composition(df_seq=df_seq, k=1),
                              sf.aa_composition(df_seq=df_seq), equal_nan=True)

    def test_k2_equals_dipeptide_composition(self):
        sf = aa.SequenceFeature()
        df_seq = _df_seq(["ACDEFGHIKLMNPQRSTVWY", "MKKLLA", "AAAC", "WY", "A", "XZ"])
        assert np.array_equal(sf.kmer_composition(df_seq=df_seq, k=2),
                              sf.dipeptide_composition(df_seq=df_seq), equal_nan=True)

    def test_rows_sum_to_one(self):
        sf = aa.SequenceFeature()
        X = sf.kmer_composition(df_seq=_df_seq(["ACDEFGHIKLMNPQRSTVWY", "MKKLLAAC"]), k=2)
        assert np.allclose(np.nansum(X, axis=1), 1.0)

    def test_dtype_float(self):
        sf = aa.SequenceFeature()
        X = sf.kmer_composition(df_seq=_df_seq(["ACDEFGHIK"]), k=3)
        assert X.dtype == np.float64


# IV Golden-value tests
class TestKmerCompositionGoldenValues:
    """Pin exact fractions against a hand-computed reference loop."""

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_golden_matches_hand_loop(self, k):
        sf = aa.SequenceFeature()
        seqs = ["ACDACDACD", "MKKLLAACDE", "WYWYWY"]
        X = sf.kmer_composition(df_seq=_df_seq(seqs), k=k)
        expected = np.vstack([_hand_kmer_fraction(s, k) for s in seqs])
        assert np.allclose(X, expected, equal_nan=True)

    def test_non_canonical_dropped(self):
        """Non-canonical residues (X, gaps, lowercase) are removed before forming k-mers."""
        sf = aa.SequenceFeature()
        X_clean = sf.kmer_composition(df_seq=_df_seq(["ACAC"]), k=2)
        X_noisy = sf.kmer_composition(df_seq=_df_seq(["AXCXAXC"]), k=2)  # same canonical chain ACAC
        assert np.array_equal(X_clean, X_noisy, equal_nan=True)

    def test_all_nan_row_when_span_shorter_than_k(self):
        """A span with fewer than k canonical residues yields an all-NaN row."""
        sf = aa.SequenceFeature()
        X = sf.kmer_composition(df_seq=_df_seq(["A", "AC"]), k=2)
        assert np.isnan(X[0]).all()          # 1 residue, no dipeptide
        assert np.isclose(np.nansum(X[1]), 1.0)  # 2 residues -> one dipeptide

    def test_kmer_crosses_part_boundary_on_gapfree_span(self):
        """k-mers form on the concatenated gap-free span, so adjacency spans dropped residues."""
        sf = aa.SequenceFeature()
        X = sf.kmer_composition(df_seq=_df_seq(["A--C"]), k=2, return_df=True)  # gaps dropped -> 'AC'
        assert X.loc[X.index[0], "AC"] == 1.0
