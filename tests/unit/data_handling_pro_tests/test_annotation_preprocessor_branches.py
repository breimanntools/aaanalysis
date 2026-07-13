"""This is a script to test edge branches of the AnnotationPreprocessor frontend
(``data_handling_pro/_annot_preproc.py``) not exercised by the main
test_annotation_preprocessor.py: the fetch_uniprot validation + dispatch (with a
mocked backend), encode return_df / out-of-coverage / unselected-feature paths,
build_scales validation + return_std, to_df_seq + build_cat dim-name overrides,
and the _check_df_annot / _resolve_dim_names helpers.

fetch_uniprot's network call is mocked at ``fetch_and_map`` (no network).
"""
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

MODULE = "aaanalysis.data_handling_pro._annot_preproc"


# I Helper Functions
def _anp():
    return aa.AnnotationPreprocessor(verbose=False)


def _df_seq(seq="ACDEFGHIKL", entry="P1"):
    return pd.DataFrame({ut.COL_ENTRY: [entry], ut.COL_SEQ: [seq]})


def _ingest_annot(annp, rows):
    return annp.ingest(pd.DataFrame(rows))


def _annot_two_keys(annp):
    """df_annot with two auto-registered functional keys at single positions."""
    return _ingest_annot(annp, {
        ut.COL_PROTEIN_ID: ["P1", "P1"],
        ut.COL_START: [3, 5],
        ut.COL_FEATURE_TYPE: ["hot1", "hot2"],
    })


# II Test Classes
class TestFetchUniprot:
    """fetch_uniprot: dispatch (mocked) + validation + evidence modes."""

    def _mock_df_annot(self):
        return pd.DataFrame(
            [["P1", 3, 3, "S", "phospho", "PTMs", "UniProt", "", 1.0, None]],
            columns=ut.COLS_ANNOT)

    def test_valid_manual_evidence(self):
        annp = _anp()
        with patch(f"{MODULE}.fetch_and_map",
                   return_value=self._mock_df_annot()) as m:
            out = annp.fetch_uniprot(df_seq=_df_seq(), evidence="manual")
        assert isinstance(out, pd.DataFrame)
        # manual -> a concrete ECO allow-set (not None)
        assert m.call_args.kwargs["evidence_codes"] is not None

    def test_valid_experimental_evidence(self):
        annp = _anp()
        with patch(f"{MODULE}.fetch_and_map",
                   return_value=self._mock_df_annot()) as m:
            annp.fetch_uniprot(df_seq=_df_seq(), evidence="experimental")
        assert m.call_args.kwargs["evidence_codes"] is not None

    def test_valid_all_evidence_no_filter(self):
        annp = _anp()
        with patch(f"{MODULE}.fetch_and_map",
                   return_value=self._mock_df_annot()) as m:
            annp.fetch_uniprot(df_seq=_df_seq(), evidence="all")
        assert m.call_args.kwargs["evidence_codes"] is None

    def test_valid_features_forwarded(self):
        annp = _anp()
        with patch(f"{MODULE}.fetch_and_map",
                   return_value=self._mock_df_annot()) as m:
            annp.fetch_uniprot(df_seq=_df_seq(), features=["phospho"])
        assert m.call_args.kwargs["allowed_features"] == ["phospho"]

    def test_invalid_evidence_value(self):
        annp = _anp()
        with pytest.raises(ValueError, match="evidence"):
            annp.fetch_uniprot(df_seq=_df_seq(), evidence="bogus")

    def test_invalid_timeout(self):
        annp = _anp()
        with pytest.raises(ValueError, match="timeout"):
            annp.fetch_uniprot(df_seq=_df_seq(), timeout=0)

    def test_invalid_unknown_feature(self):
        annp = _anp()
        with pytest.raises(ValueError):
            annp.fetch_uniprot(df_seq=_df_seq(), features=["__nope__"])


class TestIngestValidation:
    """ingest: non-DataFrame + missing columns."""

    def test_invalid_not_dataframe(self):
        with pytest.raises(ValueError, match="df_user"):
            _anp().ingest(df_user="not-a-df")

    def test_invalid_missing_columns(self):
        with pytest.raises(ValueError, match="missing required columns"):
            _anp().ingest(pd.DataFrame({ut.COL_PROTEIN_ID: ["P1"]}))


class TestEncodeBranches:
    """encode: schema guards + unselected feature + out-of-coverage + return_df."""

    def test_invalid_missing_sequence_column(self):
        annp = _anp()
        annot = _annot_two_keys(annp)
        df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1"]})  # no sequence
        with pytest.raises(ValueError, match="sequence"):
            annp.encode(df_seq=df_seq, df_annot=annot, features=["hot1"])

    def test_invalid_df_annot_not_dataframe(self):
        annp = _anp()
        with pytest.raises(ValueError, match="df_annot"):
            annp.encode(df_seq=_df_seq(), df_annot="nope", features=["hot1"])

    def test_invalid_df_annot_missing_columns(self):
        annp = _anp()
        _annot_two_keys(annp)  # register keys
        bad = pd.DataFrame({ut.COL_PROTEIN_ID: ["P1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            annp.encode(df_seq=_df_seq(), df_annot=bad, features=["hot1"])

    def test_valid_unselected_feature_skipped(self):
        annp = _anp()
        annot = _annot_two_keys(annp)  # has hot1 + hot2
        out = annp.encode(df_seq=_df_seq(), df_annot=annot, features=["hot1"])
        # hot2 rows are skipped; only hot1 position (3) is set.
        arr = out["P1"]
        assert arr.shape == (10, 1)
        assert arr[2, 0] == 1.0

    def test_valid_out_of_coverage_skipped(self):
        annp = _anp()
        annot = _ingest_annot(annp, {
            ut.COL_PROTEIN_ID: ["P1"],
            ut.COL_START: [999],  # beyond sequence length
            ut.COL_FEATURE_TYPE: ["hot1"]})
        out = annp.encode(df_seq=_df_seq(), df_annot=annot, features=["hot1"])
        assert np.all(out["P1"] == 0.0)  # nothing in coverage

    def test_valid_return_df(self):
        annp = _anp()
        annot = _annot_two_keys(annp)
        dict_num, df_out = annp.encode(df_seq=_df_seq(), df_annot=annot,
                                     features=["hot1"], return_df=True)
        assert "encode_ok" in df_out.columns
        assert bool(df_out["encode_ok"].iloc[0])
        assert "P1" in dict_num


class TestBuildScalesBranches:
    """build_scales: validation branches + return_std."""

    def _good(self, annp):
        df_seq = _df_seq()
        annot = _annot_two_keys(annp)
        dict_num = annp.encode(df_seq=df_seq, df_annot=annot, features=["hot1"])
        return df_seq, dict_num

    def test_invalid_none_args(self):
        with pytest.raises(ValueError, match="should both be provided"):
            _anp().build_scales(df_seq=None, dict_num=None, features=["hot1"])

    def test_invalid_missing_sequence_column(self):
        annp = _anp()
        _, dict_num = self._good(annp)
        df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1"]})
        with pytest.raises(ValueError, match="sequence"):
            annp.build_scales(df_seq=df_seq, dict_num=dict_num, features=["hot1"])

    def test_invalid_missing_entries(self):
        annp = _anp()
        df_seq, _ = self._good(annp)
        with pytest.raises(ValueError, match="missing entries"):
            annp.build_scales(df_seq=df_seq, dict_num={}, features=["hot1"])

    def test_invalid_dict_num_not_2d(self):
        annp = _anp()
        df_seq, _ = self._good(annp)
        with pytest.raises(ValueError, match="2-D np.ndarray"):
            annp.build_scales(df_seq=df_seq, dict_num={"P1": [1, 2, 3]},
                            features=["hot1"])

    def test_invalid_dict_num_wrong_length(self):
        annp = _anp()
        df_seq, _ = self._good(annp)
        bad = {"P1": np.zeros((3, 1))}  # seq is length 10
        with pytest.raises(ValueError, match=r"shape\[0\]"):
            annp.build_scales(df_seq=df_seq, dict_num=bad, features=["hot1"])

    def test_invalid_dict_num_wrong_dims(self):
        annp = _anp()
        df_seq, _ = self._good(annp)
        bad = {"P1": np.zeros((10, 3))}  # D should be 1
        with pytest.raises(ValueError, match=r"shape\[1\]"):
            annp.build_scales(df_seq=df_seq, dict_num=bad, features=["hot1"])

    def test_valid_returns_df_scales(self):
        annp = _anp()
        df_seq, dict_num = self._good(annp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = annp.build_scales(df_seq=df_seq, dict_num=dict_num,
                                        features=["hot1"])
        assert df_scales.shape == (20, 1)

    def test_valid_return_std(self):
        annp = _anp()
        df_seq, dict_num = self._good(annp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales, df_stds = annp.build_scales(
                df_seq=df_seq, dict_num=dict_num, features=["hot1"],
                return_std=True)
        assert df_scales.shape == (20, 1)
        assert df_stds.shape == (20, 1)

    def test_valid_noncanonical_aa_skipped(self):
        annp = _anp()
        annot = _ingest_annot(annp, {
            ut.COL_PROTEIN_ID: ["P1"], ut.COL_START: [1],
            ut.COL_FEATURE_TYPE: ["hot1"]})
        df_seq = _df_seq(seq="XCDEFGHIKL")  # leading non-canonical X
        dict_num = annp.encode(df_seq=df_seq, df_annot=annot, features=["hot1"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = annp.build_scales(df_seq=df_seq, dict_num=dict_num,
                                        features=["hot1"])
        assert "X" not in df_scales.index  # X not a canonical row

    def test_valid_all_nan_row_skipped(self):
        annp = _anp()
        df_seq, dict_num = self._good(annp)
        nan_dict = {"P1": np.full((10, 1), np.nan)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = annp.build_scales(df_seq=df_seq, dict_num=nan_dict,
                                        features=["hot1"])
        assert np.isnan(df_scales.to_numpy()).all()


class TestToDfSeqBranches:
    """to_df_seq: missing-sequence guard (other branches in main test)."""

    def test_invalid_missing_sequence_column(self):
        annp = _anp()
        annot = _annot_two_keys(annp)
        df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1"]})
        with pytest.raises(ValueError, match="sequence"):
            annp.to_df_seq(df_seq=df_seq, df_annot=annot, feature_type="hot1")


class TestPartBasedSequenceGuards:
    """The per-method 'sequence' guards ARE reachable: a part-based df_seq
    (['jmd_n','tmd','jmd_c']) passes check_df_seq but has no 'sequence' column,
    so encode/build_scales/to_df_seq fire their tailored guard (not check_df_seq).
    Matching the guard's specific message proves the guard line is hit.
    """

    @staticmethod
    def _part_based():
        return pd.DataFrame({
            ut.COL_ENTRY: ["P1"],
            "jmd_n": ["AAA"], "tmd": ["CCCCCCC"], "jmd_c": ["AAA"],
        })

    def test_encode_guard_hit(self):
        annp = _anp()
        annot = _annot_two_keys(annp)
        with pytest.raises(ValueError, match="for encode"):
            annp.encode(df_seq=self._part_based(), df_annot=annot, features=["hot1"])

    def test_build_scales_guard_hit(self):
        annp = _anp()
        _annot_two_keys(annp)
        with pytest.raises(ValueError, match="for.*build_scales|build_scales"):
            annp.build_scales(df_seq=self._part_based(),
                            dict_num={"P1": np.zeros((3, 1))}, features=["hot1"])

    def test_to_df_seq_guard_hit(self):
        annp = _anp()
        annot = _annot_two_keys(annp)
        with pytest.raises(ValueError, match="for to_df_seq"):
            annp.to_df_seq(df_seq=self._part_based(), df_annot=annot,
                         feature_type="hot1")


class TestResolveDimNames:
    """build_cat dim_names_override -> _resolve_dim_names branches."""

    def test_valid_override_applied(self):
        annp = _anp()
        _annot_two_keys(annp)  # register hot1
        df_cat = annp.build_cat(features=["hot1"], dim_names_override=["X0"])
        assert list(df_cat[ut.COL_SCALE_ID]) == ["X0"]

    def test_invalid_override_not_list(self):
        annp = _anp()
        _annot_two_keys(annp)
        with pytest.raises(ValueError, match="dim_names_override"):
            annp.build_cat(features=["hot1"], dim_names_override="X0")

    def test_invalid_override_wrong_length(self):
        annp = _anp()
        _annot_two_keys(annp)
        with pytest.raises(ValueError, match="dim_names_override"):
            annp.build_cat(features=["hot1"], dim_names_override=["a", "b"])

    def test_invalid_override_non_str_item(self):
        annp = _anp()
        _annot_two_keys(annp)
        with pytest.raises(ValueError, match="should be str"):
            annp.build_cat(features=["hot1"], dim_names_override=[123])


class TestAnnotComplex:
    """Cross-cutting combinations."""

    def test_complex_ingest_encode_build_scales_pipeline(self):
        annp = _anp()
        df_seq = _df_seq()
        annot = _annot_two_keys(annp)
        dict_num = annp.encode(df_seq=df_seq, df_annot=annot,
                             features=["hot1", "hot2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scales = annp.build_scales(df_seq=df_seq, dict_num=dict_num,
                                        features=["hot1", "hot2"])
        assert df_scales.shape == (20, 2)

    def test_complex_encode_two_features_layout(self):
        annp = _anp()
        annot = _annot_two_keys(annp)
        out = annp.encode(df_seq=_df_seq(), df_annot=annot,
                        features=["hot1", "hot2"])
        arr = out["P1"]
        assert arr.shape == (10, 2)
        assert arr[2, 0] == 1.0   # hot1 at pos 3, col 0
        assert arr[4, 1] == 1.0   # hot2 at pos 5, col 1

    def test_complex_build_cat_categories(self):
        annp = _anp()
        _annot_two_keys(annp)
        df_cat = annp.build_cat(features=["hot1", "hot2"])
        assert df_cat.shape[0] == 2
        assert set(df_cat[ut.COL_CAT]) == {ut.LIST_CAT[-1]}  # Functional sites

    def test_complex_fetch_uniprot_entries_passed(self):
        annp = _anp()
        df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1", "P2"],
                               ut.COL_SEQ: ["ACDE", "FGHI"]})
        empty = pd.DataFrame(columns=ut.COLS_ANNOT)
        with patch(f"{MODULE}.fetch_and_map", return_value=empty) as m:
            annp.fetch_uniprot(df_seq=df_seq, evidence="all")
        assert m.call_args.kwargs["entries"] == ["P1", "P2"]

    def test_complex_return_df_mismatch_marks_not_ok(self):
        annp = _anp()
        # annotate pos 1 with aa 'Z' (wrong vs sequence 'A') -> drop + flag.
        annot = _ingest_annot(annp, {
            ut.COL_PROTEIN_ID: ["P1"], ut.COL_START: [1],
            ut.COL_FEATURE_TYPE: ["hot1"], ut.COL_AA: ["Z"]})
        dict_num, df_out = annp.encode(df_seq=_df_seq(), df_annot=annot,
                                     features=["hot1"], on_mismatch="drop",
                                     return_df=True)
        assert not bool(df_out["encode_ok"].iloc[0])
