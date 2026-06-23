"""Branch-coverage tests for StructurePreprocessor reached purely through
the public ``aa.StructurePreprocessor`` API + local fixtures.

Targets un-hit branch arms in the frontend (``_struct_preproc.py``) and the
struct backend (``_pae_io``, ``_domain_io``, ``encode_domains``,
``encode_pae``, ``_alphafold`` skip path). Network arcs are out of scope.
"""
import json
import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

PDB_FIXTURES = Path(__file__).resolve().parents[3] / \
    "aaanalysis" / "_data" / "pdb_test"
AF_FIXTURE_SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"   # AF_TINY (L=30)


# I Helper Functions
def _part_based_df():
    """A part-based df_seq that passes check_df_seq but has NO 'sequence'
    column, so each method's tailored COL_SEQ guard fires."""
    return pd.DataFrame({
        "entry": ["P1"],
        "jmd_n": ["AAAA"],
        "tmd": ["CDEFGHIK"],
        "jmd_c": ["LLLL"],
    })


def _df_af():
    return pd.DataFrame({"entry": ["AF_TINY"], "sequence": [AF_FIXTURE_SEQ]})


def _pae_folder(json_payload, name="AF_TINY.json"):
    td = tempfile.TemporaryDirectory()
    (Path(td.name) / name).write_text(json.dumps(json_payload))
    return td


# II Test Classes
class TestSeqColumnGuards:
    """Each method's own 'sequence column required' raise (part-based df_seq
    passes check_df_seq but lacks COL_SEQ)."""

    def test_get_dssp_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="for get_dssp"):
            stp.get_dssp(df_seq=_part_based_df(), pdb_folder=str(PDB_FIXTURES))

    def test_encode_dssp_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="for encode_dssp"):
            stp.encode_dssp(df_seq=_part_based_df(),
                            pdb_folder=str(PDB_FIXTURES), features=["ss3"])

    def test_encode_pdb_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="for encode_pdb"):
            stp.encode_pdb(df_seq=_part_based_df(),
                           pdb_folder=str(PDB_FIXTURES), features=["bfactor"])

    def test_encode_pae_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="for encode_pae"):
            stp.encode_pae(df_seq=_part_based_df(),
                           pae_folder=str(PDB_FIXTURES),
                           features=["pae_row_mean"])

    def test_get_domains_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="for get_domains"):
            stp.get_domains(df_seq=_part_based_df(),
                            pdb_folder=str(PDB_FIXTURES))

    def test_encode_domains_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="for encode_domains"):
            stp.encode_domains(df_seq=_part_based_df(),
                               domain_folder=str(PDB_FIXTURES),
                               features=["domain_boundary"])

    def test_build_scales_seq_guard(self):
        stp = aa.StructurePreprocessor(verbose=False)
        d = {"P1": np.zeros((8, 1))}
        with pytest.raises(ValueError, match="for build_scales"):
            stp.build_scales(df_seq=_part_based_df(), dict_num=d, features=["bfactor"])


class TestPaeIoBranches:
    """_pae_io._coerce_to_matrix layouts + shape guards via encode_pae."""

    def test_pae_bare_list_payload(self):
        # payload is NOT a dict -> isinstance(payload, dict) False arm.
        L = len(AF_FIXTURE_SEQ)
        mat = (1.0 + np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
               ).astype(float)
        td = _pae_folder(mat.tolist())
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_row_mean"])
        assert d["AF_TINY"].shape == (L, 1)
        assert not np.isnan(d["AF_TINY"]).all()
        td.cleanup()

    def test_pae_pae_key_payload(self):
        # dict with the 'pae' key (second key in the resolver loop).
        L = len(AF_FIXTURE_SEQ)
        mat = (1.0 + np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
               ).astype(float)
        td = _pae_folder({"pae": mat.tolist()})
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_row_mean"])
        assert d["AF_TINY"].shape == (L, 1)
        td.cleanup()

    def test_pae_not_2d_warns_and_marks_failed(self):
        # 1-D payload -> ndim != 2 -> RuntimeError -> warned, pae_ok False.
        td = _pae_folder({"predicted_aligned_error": [1.0, 2.0, 3.0]})
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning, match="PAE load failed"):
            d, df_out = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                       features=["pae_row_mean"],
                                       return_df=True)
        assert bool(df_out.iloc[0]["pae_ok"]) is False
        assert np.isnan(d["AF_TINY"]).all()
        td.cleanup()

    def test_pae_non_square_warns_and_marks_failed(self):
        # Square-vs-L mismatch first triggers; build a genuinely non-square 2D.
        td = _pae_folder({"predicted_aligned_error": [[1.0, 2.0, 3.0],
                                                      [4.0, 5.0, 6.0]]})
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning, match="PAE load failed"):
            d, df_out = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                       features=["pae_row_mean"],
                                       return_df=True)
        assert bool(df_out.iloc[0]["pae_ok"]) is False
        td.cleanup()


class TestEncodePaeDistalEmpty:
    """encode_pae local/distal masks: local_window >= L makes distal empty."""

    def test_distal_mask_empty_when_window_covers_all(self):
        L = len(AF_FIXTURE_SEQ)
        mat = (1.0 + np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
               ).astype(float)
        td = _pae_folder({"predicted_aligned_error": mat.tolist()})
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_distal_mean"],
                           local_window=L + 5)
        # Every residue is "local"; distal mean is all NaN.
        assert np.isnan(d["AF_TINY"]).all()
        td.cleanup()


class TestEncodeDomainsBranches:
    """encode_domains backend arcs via crafted chopping files."""

    def _domain_folder(self, entry, text, ext=".txt"):
        td = tempfile.TemporaryDirectory()
        (Path(td.name) / f"{entry}{ext}").write_text(text)
        return td

    def test_chopping_trailing_comma_empty_domain(self):
        # "1-5," -> split on ',' yields an empty piece -> `if not dom_str`.
        td = self._domain_folder("AF_TINY", "1-5,10-15")
        stp = aa.StructurePreprocessor(verbose=False)
        td2 = self._domain_folder("AF_TINY", "1-5,")
        d = stp.encode_domains(df_seq=_df_af(), domain_folder=td2.name,
                               features=["domain_boundary"])
        assert d["AF_TINY"].shape[0] == len(AF_FIXTURE_SEQ)
        td.cleanup(); td2.cleanup()

    def test_overlapping_segments_keep_first_assignment(self):
        # Two domains over the same residues -> _residue_to_domain_index
        # `if out[i] == -1` False arm (already assigned).
        td = self._domain_folder("AF_TINY", "1-10,5-15")
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_domains(df_seq=_df_af(), domain_folder=td.name,
                               features=["n_domains_in_protein"])
        assert d["AF_TINY"].shape[0] == len(AF_FIXTURE_SEQ)
        td.cleanup()

    def test_endpoint_beyond_L_skipped(self):
        # Domain end far beyond L -> boundary endpoint idx >= L -> the
        # `if 0 <= idx < L` False arm in encode_domain_boundary.
        td = self._domain_folder("AF_TINY", "1-9999")
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_domains(df_seq=_df_af(), domain_folder=td.name,
                               features=["domain_boundary"])
        assert d["AF_TINY"].shape[0] == len(AF_FIXTURE_SEQ)
        td.cleanup()


class TestDomainIoBranches:
    """_domain_io parse + TSV reader error arcs via get_domains/encode_domains."""

    def _folder(self, entry, text, ext):
        td = tempfile.TemporaryDirectory()
        (Path(td.name) / f"{entry}{ext}").write_text(text)
        return td

    def test_segment_start_lt_one_raises_failure(self):
        # "0-5" -> start < 1 -> RuntimeError in parser -> row marked failed.
        td = self._folder("AF_TINY", "0-5", ".txt")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning):
            d, df_out = stp.encode_domains(df_seq=_df_af(),
                                           domain_folder=td.name,
                                           features=["domain_boundary"],
                                           return_df=True)
        assert bool(df_out.iloc[0]["domain_ok"]) is False
        td.cleanup()

    def test_tsv_missing_chopping_column(self):
        td = self._folder("AF_TINY", "id\tfoo\nAF_TINY\t1-5\n", ".tsv")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning):
            d, df_out = stp.encode_domains(df_seq=_df_af(),
                                           domain_folder=td.name,
                                           features=["domain_boundary"],
                                           return_df=True)
        assert bool(df_out.iloc[0]["domain_ok"]) is False
        td.cleanup()

    def test_tsv_header_no_data_row(self):
        td = self._folder("AF_TINY", "chopping\n", ".tsv")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning):
            d, df_out = stp.encode_domains(df_seq=_df_af(),
                                           domain_folder=td.name,
                                           features=["domain_boundary"],
                                           return_df=True)
        assert bool(df_out.iloc[0]["domain_ok"]) is False
        td.cleanup()

    def test_tsv_row_fewer_cols_than_header(self):
        # chopping is the 2nd column but the data row has only 1 field.
        td = self._folder("AF_TINY", "id\tchopping\nonlyone\n", ".tsv")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning):
            d, df_out = stp.encode_domains(df_seq=_df_af(),
                                           domain_folder=td.name,
                                           features=["domain_boundary"],
                                           return_df=True)
        assert bool(df_out.iloc[0]["domain_ok"]) is False
        td.cleanup()

    def test_tsv_blank_lines_only_is_empty(self):
        # .tsv whose only content is blank lines -> no non-empty lines ->
        # 'domain file ... is empty' RuntimeError -> row marked failed.
        td = self._folder("AF_TINY", "\n   \n\n", ".tsv")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.warns(UserWarning):
            d, df_out = stp.encode_domains(df_seq=_df_af(),
                                           domain_folder=td.name,
                                           features=["domain_boundary"],
                                           return_df=True)
        assert bool(df_out.iloc[0]["domain_ok"]) is False
        td.cleanup()

    def test_tsv_valid_chopping_column(self):
        td = self._folder("AF_TINY", "chopping\n1-10,11-20\n", ".tsv")
        stp = aa.StructurePreprocessor(verbose=False)
        d = stp.encode_domains(df_seq=_df_af(), domain_folder=td.name,
                               features=["n_domains_in_protein"])
        assert d["AF_TINY"].shape[0] == len(AF_FIXTURE_SEQ)
        td.cleanup()


class TestEncodePdbDepthNoMsms:
    """encode_pdb 'depth' dispatch: msms unavailable -> RuntimeError path
    (the _extras.check_msms_available raise + the L1016 dispatch arm)."""

    def test_depth_raises_when_msms_missing(self):
        if shutil.which("msms") is not None:
            pytest.skip("msms present; this exercises the missing-tool arm")
        td = tempfile.TemporaryDirectory()
        shutil.copy(PDB_FIXTURES / "AF_TINY.pdb", Path(td.name) / "AF_TINY.pdb")
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(RuntimeError, match="msms"):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=td.name,
                           features=["depth"])
        td.cleanup()


_HETATM_ONLY_PDB = (
    "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 30.00      O\n"
    "HETATM    2  O   HOH A   2       3.000   0.000   0.000  1.00 30.00      O\n"
    "HETATM    3  O   HOH A   3       6.000   0.000   0.000  1.00 30.00      O\n"
    "END\n"
)


class TestEncodePdbNoAminoAcidChains:
    """encode_pdb encoders hit their 'if not chains' empty-structure arm when
    the PDB has no standard amino-acid residues (HETATM/water only)."""

    @pytest.mark.parametrize("feat", [
        "bfactor", "plddt", "chi1_sincos", "ca_centroid_dist",
        "ca_centroid_dist_norm", "contact_count_8A", "hse", "disulfide",
    ])
    def test_no_aa_chains_returns_nan(self, feat, tmp_path):
        (tmp_path / "AF_TINY.pdb").write_text(_HETATM_ONLY_PDB)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(tmp_path),
                               features=[feat])
        # Empty-structure path yields an all-NaN per-residue block.
        assert d["AF_TINY"].shape[0] == len(AF_FIXTURE_SEQ)
        assert np.isnan(d["AF_TINY"]).all()


class TestFetchAlphafoldSkipVerbose:
    """_alphafold skip-existing verbose print (no network)."""

    def test_skip_existing_verbose(self, tmp_path):
        (tmp_path / "P1.pdb").write_text("x")
        (tmp_path / "AF-P1-F1-predicted_aligned_error_v4.json").write_text("{}")
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEFGHIK"]})
        stp = aa.StructurePreprocessor(verbose=True)
        BACKEND = ("aaanalysis.data_handling_pro._backend.struct_preproc."
                   "_alphafold")
        with patch(f"{BACKEND}.requests.get") as mg:
            out = stp.fetch_alphafold(df_seq=df, out_folder=str(tmp_path),
                                      skip_existing=True)
        mg.assert_not_called()
        assert bool(out.iloc[0]["skipped"]) is True
