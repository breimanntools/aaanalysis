"""This is a script to test branch arcs of data_handling loaders/preprocessors.

These focused tests exercise option/None arms reachable only through the
public API (aa.load_scales / aa.load_features / aa.read_fasta /
aa.SequencePreprocessor / aa.EmbeddingPreprocessor) that the per-method
suites leave uncovered. Public surface only: no private backend calls.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helpers --------------------------------------------------------------
def _write_fasta(tmp_path, text, name="seq.fasta"):
    """Write FASTA text to a tmp file and return its path."""
    fp = tmp_path / name
    fp.write_text(text)
    return str(fp)


# I load_scales branch arcs --------------------------------------------
class TestLoadScalesBranch:
    """Branch arcs of aa.load_scales: top60_n string/int forms and option arms."""

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=1, max_value=60))
    def test_top60_n_str_aac_id(self, n):
        """top60_n as an 'AAC' id string is parsed and validated (L22-25)."""
        df = aa.load_scales(name="scales", top60_n=f"AAC{n:02d}")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 20

    def test_top60_n_str_without_aac_raises(self):
        """A non-'AAC' string for top60_n raises (L23)."""
        with pytest.raises(ValueError, match="should be int or 'AAC' id"):
            aa.load_scales(name="scales", top60_n="XYZ")

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=1, max_value=60))
    def test_top60_n_scales_cat_branch(self, n):
        """name='scales_cat' with top60_n returns the filtered df_cat (L260-263)."""
        df_cat = aa.load_scales(name="scales_cat", top60_n=n)
        assert isinstance(df_cat, pd.DataFrame)
        assert "scale_id" in df_cat.columns

    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=1, max_value=60))
    def test_top60_n_scales_raw_branch(self, n):
        """name='scales_raw' with top60_n loads + dtype-adjusts the matrix (L264-268)."""
        df = aa.load_scales(name="scales_raw", top60_n=n)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 20

    def test_scales_pc_unclassified_out_skips_scale_filter(self):
        """name='scales_pc' with unclassified_out reaches the non-scales filter arm (L303->306)."""
        df = aa.load_scales(name="scales_pc", unclassified_out=True)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 20

    def test_invalid_name_raises(self):
        """An invalid scale-set name raises before any branch (negative)."""
        with pytest.raises(ValueError, match="is not valid"):
            aa.load_scales(name="not_a_scale_set")


# II load_features (kept minimal; negative arm) ------------------------
class TestLoadFeaturesBranch:
    """Negative arm of aa.load_features."""

    def test_valid_name(self):
        df_feat = aa.load_features(name="DOM_GSEC")
        assert isinstance(df_feat, pd.DataFrame)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="should be one of"):
            aa.load_features(name="NOPE")


# III read_fasta branch arcs -------------------------------------------
class TestReadFastaBranch:
    """Branch arcs of aa.read_fasta: warnings + cols_info / col_db rebuilds."""

    def test_duplicate_entries_warns(self, tmp_path):
        """Duplicated ids trigger the uniqueness warning (L18-21)."""
        fp = _write_fasta(tmp_path, ">P1\nACDEF\n>P1\nGHIKL\n")
        with pytest.warns(UserWarning, match="should be unique"):
            df = aa.read_fasta(file_path=fp)
        assert isinstance(df, pd.DataFrame)

    def test_col_db_not_in_df_warns(self, tmp_path):
        """col_db absent from the parsed frame warns (L27-29)."""
        fp = _write_fasta(tmp_path, ">P1\nACDEF\n>P2\nGHIKL\n")
        with pytest.warns(UserWarning, match="not in 'df_seq'"):
            df = aa.read_fasta(file_path=fp, col_db="database")
        assert isinstance(df, pd.DataFrame)

    def test_cols_info_shorter_is_padded(self, tmp_path):
        """cols_info shorter than the info columns is padded with infoN (L37, L40, L44)."""
        fp = _write_fasta(tmp_path, ">P1|e1|e2\nACDEF\n>P2|f1|f2\nGHIKL\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.read_fasta(file_path=fp, cols_info=["onlyone"])
        assert "onlyone" in df.columns
        assert "info1" in df.columns

    def test_col_db_with_cols_info_rebuild(self, tmp_path):
        """col_db + cols_info rebuilds columns including the db column (L41-42)."""
        fp = _write_fasta(tmp_path, ">sp|P1|e1|e2\nACDEF\n>sp|P2|f1|f2\nGHIKL\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aa.read_fasta(file_path=fp, col_db="database", cols_info=["ca", "cb", "cc"])
        assert "database" in df.columns
        assert "ca" in df.columns

    def test_empty_fasta_file_no_entries(self, tmp_path):
        """A .fasta with no header entries flushes no entry (parse_fasta L29 false arm).

        check_is_fasta validates only the extension, so an empty .fasta passes
        validation; the parser flushes nothing and read_fasta then fails on the
        absent id column. The backend false arc is exercised before that.
        """
        fp = _write_fasta(tmp_path, "")
        with pytest.raises(KeyError):
            aa.read_fasta(file_path=fp)


# IV SequencePreprocessor branch arcs ----------------------------------
class TestSeqPreprocBranch:
    """Branch arcs of the SequencePreprocessor check helpers + sliding backend."""

    def test_encode_gap_in_alphabet_raises(self):
        """gap contained in alphabet raises (L34-35)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError, match="should not be contained in the 'alphabet'"):
            seqp.encode_one_hot(list_seq=["ACDE"], alphabet="ACDE-", gap="-")

    def test_get_aa_window_neither_pos_stop_nor_window_size(self):
        """Neither pos_stop nor window_size given raises (L56-57)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError, match="must be specified. Both are 'None'"):
            seqp.get_aa_window(seq="ACDEFG", pos_start=0)

    def test_get_aa_window_both_pos_stop_and_window_size(self):
        """Both pos_stop and window_size given raises (L58-59)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError, match="Both are given"):
            seqp.get_aa_window(seq="ACDEFG", pos_start=0, pos_stop=3, window_size=2)

    def test_get_aa_window_window_extends_beyond_seq_no_gap(self):
        """accept_gap=False with a window past the sequence end raises (L74-78)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError, match=r"window_size.* should be >="):
            seqp.get_aa_window(seq="ACDEF", pos_start=2, window_size=10, accept_gap=False)

    @settings(max_examples=5, deadline=None)
    @given(ws=some.integers(min_value=1, max_value=4))
    def test_get_aa_window_within_seq_no_gap_ok(self, ws):
        """accept_gap=False with a window that fits takes the no-raise arm (L77->exit)."""
        seqp = aa.SequencePreprocessor()
        window = seqp.get_aa_window(seq="ACDEFGHIK", pos_start=2, window_size=ws,
                                  accept_gap=False)
        assert isinstance(window, str)
        assert len(window) == ws

    def test_get_sliding_slide_start_gt_slide_stop_raises(self):
        """slide_start > slide_stop raises (L85-86)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError, match="should be smaller than 'slide_stop'"):
            seqp.get_sliding_aa_window(seq="ACDEFGHIK", slide_start=5, slide_stop=2, window_size=2)

    def test_get_sliding_window_extends_beyond_seq_no_gap(self):
        """accept_gap=False with slide_start + window_size past the end raises (L111-112)."""
        seqp = aa.SequencePreprocessor()
        with pytest.raises(ValueError, match=r"window_size.* should be >="):
            seqp.get_sliding_aa_window(seq="ACDEF", slide_start=2, window_size=10, accept_gap=False)

    @settings(max_examples=5, deadline=None)
    @given(start=some.integers(min_value=1, max_value=3))
    def test_get_sliding_slide_stop_none_index1(self, start):
        """slide_stop=None with index1=True takes the +1 backend arm (sliding L15-18)."""
        seqp = aa.SequencePreprocessor()
        windows = seqp.get_sliding_aa_window(seq="ACDEFGHIK", slide_start=start,
                                           window_size=3, index1=True)
        assert isinstance(windows, list)
        assert all(isinstance(w, str) for w in windows)


# V EmbeddingPreprocessor branch arcs ----------------------------------
class TestEmbedPreprocBranch:
    """Branch arcs of EmbeddingPreprocessor backends (encode sigmoid, std all-NaN)."""

    @settings(max_examples=5, deadline=None)
    @given(D=some.integers(min_value=2, max_value=6))
    def test_encode_sigmoid_method(self, D):
        """method='sigmoid' takes the sigmoid fit arm (encode.py L77)."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"entry": ["P0", "P1"], "sequence": ["ACDE", "FGHI"]})
        emb = {"P0": rng.standard_normal((4, D)) * 5,
               "P1": rng.standard_normal((4, D)) * 5}
        dict_num = aa.EmbeddingPreprocessor(verbose=False).encode(
            df_seq=df, embeddings=emb, method="sigmoid")
        assert set(dict_num) == {"P0", "P1"}
        for arr in dict_num.values():
            assert np.all((arr >= 0) & (arr <= 1))

    def test_build_scales_no_canonical_aa_std_all_nan(self):
        """A non-canonical-only corpus leaves counts all-zero so the std fill
        skips its body (build_pseudo_scales.py L62 false arm)."""
        df = pd.DataFrame({"entry": ["P0"], "sequence": ["XXXX"]})
        emb = {"P0": np.random.default_rng(0).standard_normal((4, 3))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            means, stds = aa.EmbeddingPreprocessor(verbose=False).build_scales(
                df_seq=df, dict_num=emb, return_std=True)
        assert means.shape == (20, 3)
        assert np.isnan(stds.values).all()
