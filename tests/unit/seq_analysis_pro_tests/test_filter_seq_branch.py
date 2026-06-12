"""This is a script to test branch arms of filter_seq() via the public API.

Targets backend option/edge arms (multi-member clusters, local vs. global
identity, verbose printing, word_size passthrough, cluster ordering) plus the
``run_command`` subprocess-failure path. All exercised only through
``aa.filter_seq`` (no private imports). Skipped on Windows (CD-HIT is not
installable there)."""
import platform
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

import aaanalysis as aa

is_windows = platform.system() == "Windows"
_HAS_CD_HIT = shutil.which("cd-hit") is not None
_HAS_MMSEQS = shutil.which("mmseqs") is not None

cd_hit_required = pytest.mark.skipif(not _HAS_CD_HIT, reason="cd-hit not on PATH")
mmseqs_required = pytest.mark.skipif(not _HAS_MMSEQS, reason="mmseqs not on PATH")

COL_SEQ = "sequence"
COL_ENTRY = "entry"


# I Helper Functions
def _df_multi_member():
    """s1/s2 are near-identical (cluster together: one rep + one member); s3 is
    distinct (its own singleton cluster)."""
    return pd.DataFrame({
        COL_ENTRY: ["s1", "s2", "s3"],
        COL_SEQ: [
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKM",   # 1 substitution vs. s1
            "WYWYWYWYWYWYWYWYWYWYWYWYWYWYWY",   # distinct
        ],
    })


# II Test Classes
@pytest.mark.skipif(is_windows, reason="Skipping tests on Windows")
class TestFilterSeqBranch:
    """Branch arms reachable through aa.filter_seq with one option per test."""

    # --- pre-flight tool guard (mocked PATH miss) ---
    def test_invalid_tool_not_installed(self):
        """check_is_tool raise arm: method binary absent from PATH."""
        df = _df_multi_member()
        with patch("aaanalysis.seq_analysis_pro._filter_seq.shutil.which",
                   return_value=None):
            with pytest.raises(ValueError, match="not installed or not in the PATH"):
                aa.filter_seq(df_seq=df, method="cd-hit")

    # --- gap / length guards (raise arms) ---
    def test_invalid_sequence_with_gaps(self):
        """check_seq_gaps raise arm (true branch)."""
        df = pd.DataFrame({
            COL_ENTRY: ["g1", "g2"],
            COL_SEQ: ["ACDEFG-IKLMN", "ACDEFGHIKLMN"],
        })
        with pytest.raises(ValueError, match="gaps"):
            aa.filter_seq(df_seq=df, method="cd-hit")

    def test_invalid_sequence_too_short(self):
        """check_seq_len raise arm (true branch)."""
        df = pd.DataFrame({COL_ENTRY: ["a", "b"], COL_SEQ: ["ACDEF", "ACDEFGHIKLMN"]})
        with pytest.raises(ValueError, match="Minimum required length"):
            aa.filter_seq(df_seq=df, method="cd-hit")

    # --- cd-hit: multi-member cluster (rep + non-rep parse arms) ---
    @cd_hit_required
    def test_valid_cd_hit_multi_member_cluster(self):
        """cd_hit parse: is_rep true arm AND non-rep else arm both hit."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="cd-hit",
                                 similarity_threshold=0.9, verbose=True)
        assert isinstance(df_clust, pd.DataFrame)
        reps = df_clust["is_representative"].to_list()
        assert 1 in reps and 0 in reps           # both parse arms exercised
        # the non-rep member carries an identity < 100
        member = df_clust[df_clust["is_representative"] == 0]
        assert (member["identity_with_rep"] < 100).all()

    @cd_hit_required
    def test_valid_cd_hit_local_identity_band(self):
        """cd_hit identity parse with '/'-format band (local identity arm)."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="cd-hit",
                                 similarity_threshold=0.9,
                                 global_identity=False, verbose=True)
        assert isinstance(df_clust, pd.DataFrame)
        member = df_clust[df_clust["is_representative"] == 0]
        assert len(member) >= 1

    @cd_hit_required
    def test_valid_cd_hit_cluster_order_size(self):
        """cluster_order='size' arm for cd-hit."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="cd-hit",
                                 similarity_threshold=0.9, cluster_order="size")
        assert isinstance(df_clust, pd.DataFrame)

    @cd_hit_required
    def test_valid_cd_hit_cluster_order_input(self):
        """cluster_order='input' reorder arm for cd-hit."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="cd-hit",
                                 similarity_threshold=0.9, cluster_order="input")
        assert df_clust[COL_ENTRY].to_list() == ["s1", "s2", "s3"]

    # --- run_command subprocess-failure arm (RuntimeError + temp cleanup) ---
    @cd_hit_required
    def test_invalid_cd_hit_command_failure(self):
        """run_command failure arm: cd-hit rejects word_size=5 with c=0.5.

        Exercises _utils.run_command returncode!=0 -> RuntimeError and the
        remove_temp(isdir) cleanup arm."""
        df = _df_multi_member()
        with pytest.raises(RuntimeError, match="failed"):
            aa.filter_seq(df_seq=df, method="cd-hit",
                          word_size=5, similarity_threshold=0.5, verbose=True)

    @cd_hit_required
    def test_invalid_cd_hit_command_failure_quiet(self):
        """run_command failure arm with verbose=False (the non-verbose branch)."""
        df = _df_multi_member()
        with pytest.raises(RuntimeError, match="failed"):
            aa.filter_seq(df_seq=df, method="cd-hit",
                          word_size=5, similarity_threshold=0.5, verbose=False)

    # --- mmseqs: multi-member cluster (rep/non-rep + comp_seq_sim_ arm) ---
    @mmseqs_required
    def test_valid_mmseqs_multi_member_cluster(self):
        """mmseq _get_df_clust: representative true arm AND non-rep arm
        (which calls comp_seq_sim_ for the member identity)."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6,
                                 verbose=True)
        assert isinstance(df_clust, pd.DataFrame)
        reps = df_clust["is_representative"].to_list()
        assert 1 in reps and 0 in reps
        member = df_clust[df_clust["is_representative"] == 0]
        assert (member["identity_with_rep"] < 100).all()

    @mmseqs_required
    def test_valid_mmseqs_word_size_passthrough(self):
        """mmseq word_size is not None arm (-k flag appended)."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6)
        assert isinstance(df_clust, pd.DataFrame)

    @mmseqs_required
    def test_valid_mmseqs_cluster_order_size(self):
        """cluster_order='size' sort arm for mmseqs (multi-member input)."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6,
                                 cluster_order="size")
        assert isinstance(df_clust, pd.DataFrame)
        # size order: the 2-member cluster precedes the singleton
        assert df_clust.iloc[0]["cluster"] == df_clust.iloc[1]["cluster"]

    @mmseqs_required
    def test_valid_mmseqs_cluster_order_input(self):
        """cluster_order='input' reorder arm for mmseqs."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6,
                                 cluster_order="input")
        assert df_clust[COL_ENTRY].to_list() == ["s1", "s2", "s3"]

    @mmseqs_required
    def test_valid_mmseqs_cluster_order_none(self):
        """cluster_order=None native-order arm for mmseqs."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6,
                                 cluster_order=None)
        assert isinstance(df_clust, pd.DataFrame)

    @mmseqs_required
    def test_valid_mmseqs_coverage_short_arm(self):
        """mmseq coverage_short elif arm (--cov-mode 1) when coverage_long unset."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6,
                                 coverage_short=0.8)
        assert isinstance(df_clust, pd.DataFrame)

    @mmseqs_required
    def test_valid_mmseqs_coverage_long_arm(self):
        """mmseq coverage_long if arm (--cov-mode 0)."""
        df = _df_multi_member()
        df_clust = aa.filter_seq(df_seq=df, method="mmseqs",
                                 similarity_threshold=0.5, word_size=6,
                                 coverage_long=0.8)
        assert isinstance(df_clust, pd.DataFrame)
