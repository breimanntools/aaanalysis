"""This is a script to test NumericalFeature.from_pssm()."""
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

DATA_DIR = pathlib.Path(__file__).parent / "data_pssm"
DICT_SEQ = {"P1": "ACDKW", "P2": "MKLVFGHE", "P3": "RSTNQY"}
ORDER_PSIBLAST = "ARNDCQEGHILKMFPSTWYV"
INVALID_BOOLS = [None, 1, 0, "True", [], 1.0]
INVALID_VALUES = [None, "log-odds", "freq", "LOG_ODDS", 1, [], "percentages"]

# Hand-derived from the first row of P1.pssm, whose PSI-BLAST columns (ARNDCQEGHILKMFPSTWYV)
# hold log-odds j-10 and percentages 5*j for header index j, read in canonical ACDEFGHIKLMNPQRSTVWY order.
GOLD_P1_ROW0_LOG_ODDS = [-10, -6, -7, -4, 3, -3, -2, -1, 1, 0, 2, -8, 4, -5, -9, 5, 6, 9, 7, 8]
GOLD_P1_ROW0_FREQ = [0, 20, 15, 30, 65, 35, 40, 45, 55, 50, 60, 10, 70, 25, 5, 75, 80, 95, 85, 90]
# Second row of P1.pssm, PSI-BLAST order: 7 5 6 2 6 0 1 6 -3 -1 -3 1 8 -3 0 1 7 -2 2 -1
GOLD_P1_ROW1_LOG_ODDS = [7, 6, 2, 1, -3, 6, -3, -1, 1, -3, 8, 6, 0, 0, 5, 1, 7, -1, -2, 2]


# Helper Functions
def _df_seq(entries=None, dict_seq=None):
    dict_seq = DICT_SEQ if dict_seq is None else dict_seq
    entries = list(dict_seq) if entries is None else entries
    return pd.DataFrame({"entry": entries, "sequence": [dict_seq[e] for e in entries]})


def _dict_paths(entries=None):
    entries = list(DICT_SEQ) if entries is None else entries
    return {e: str(DATA_DIR / f"{e}.pssm") for e in entries}


def write_pssm(path, seq, seed=0, with_freq=True, header=ORDER_PSIBLAST):
    """Write a synthetic PSSM in PSI-BLAST ASCII layout (header, two 20-column blocks, K/Lambda footer)."""
    rng = np.random.default_rng(seed)
    lines = ["",
             "Last position-specific scoring matrix computed, weighted observed percentages rounded down, "
             "information per position, and relative weight of gapless real matches to pseudocounts",
             "           " + "".join(f"{a:>4}" for a in header)
             + ("".join(f"{a:>4}" for a in header) if with_freq else "")]
    for i, r in enumerate(seq):
        row = f"{i + 1:5d} {r} " + "".join(f"{int(s):4d}" for s in rng.integers(-4, 9, 20))
        if with_freq:
            row += "  " + "".join(f"{int(p):4d}" for p in rng.integers(0, 40, 20)) + "  0.45 0.12"
        lines.append(row)
    lines += ["", "                      K         Lambda",
              "Standard Ungapped    0.1330     0.3176", "Standard Gapped      0.0410     0.2670",
              "PSI Ungapped         0.1420     0.3176", "PSI Gapped           0.0410     0.2670", ""]
    pathlib.Path(path).write_text("\n".join(lines))


# Test Classes
class TestFromPssm:
    """Normal cases, one parameter per test."""

    # Positive tests: pssm
    @settings(max_examples=5, deadline=None)
    @given(normalize=some.booleans())
    def test_pssm_directory(self, normalize):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), normalize=normalize)
        assert list(dict_num) == ["P1", "P2", "P3"]
        for entry, arr in dict_num.items():
            assert arr.shape == (len(DICT_SEQ[entry]), 20)
            assert arr.dtype == np.float64

    def test_pssm_directory_uppercase_extension(self, tmp_path):
        write_pssm(tmp_path / "U1.PSSM", "ACD", seed=1)
        write_pssm(tmp_path / "U2.pssm", "ACDK", seed=2)
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(tmp_path))
        assert list(dict_num) == ["U1", "U2"]
        assert dict_num["U1"].shape == (3, 20)

    @settings(max_examples=5, deadline=None)
    @given(entry=some.sampled_from(list(DICT_SEQ)))
    def test_pssm_single_file(self, entry):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR / f"{entry}.pssm"))
        assert list(dict_num) == [entry]
        assert dict_num[entry].shape == (len(DICT_SEQ[entry]), 20)

    def test_pssm_single_file_pathlib(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=DATA_DIR / "P1.pssm", df_seq=_df_seq(["P1"]))
        assert list(dict_num) == ["P1"]

    def test_pssm_pathlib_directory(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=DATA_DIR)
        assert set(dict_num) == set(DICT_SEQ)

    @settings(max_examples=5, deadline=None)
    @given(entries=some.lists(some.sampled_from(list(DICT_SEQ)), min_size=1, max_size=3, unique=True))
    def test_pssm_dict_of_paths(self, entries):
        dict_num = aa.NumericalFeature.from_pssm(pssm=_dict_paths(entries))
        assert list(dict_num) == entries

    @settings(max_examples=5, deadline=None)
    @given(n_rows=some.integers(min_value=1, max_value=30), seed=some.integers(min_value=0, max_value=100))
    def test_pssm_dict_of_arrays(self, n_rows, seed):
        arr = np.random.default_rng(seed).integers(-8, 9, (n_rows, 20)).astype(float)
        dict_num = aa.NumericalFeature.from_pssm(pssm={"X1": arr}, normalize=False)
        np.testing.assert_array_equal(dict_num["X1"], arr)

    def test_pssm_dict_of_nested_lists(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm={"X1": [[1] * 20, [2] * 20]}, normalize=False)
        assert dict_num["X1"].shape == (2, 20)

    # Positive tests: df_seq
    @settings(max_examples=5, deadline=None)
    @given(entries=some.lists(some.sampled_from(list(DICT_SEQ)), min_size=1, max_size=3, unique=True))
    def test_df_seq_matching(self, entries):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=_df_seq(entries))
        assert set(entries).issubset(dict_num)

    def test_df_seq_none(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=None)
        assert len(dict_num) == 3

    # Positive tests: values
    @pytest.mark.parametrize("values", ["log_odds", "frequencies"])
    def test_values(self, values):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), values=values, normalize=False)
        arr = dict_num["P2"]
        assert arr.shape == (8, 20)
        if values == "frequencies":
            assert (arr >= 0).all()

    # Positive tests: normalize
    @settings(max_examples=5, deadline=None)
    @given(values=some.sampled_from(["log_odds", "frequencies"]))
    def test_normalize_in_unit_range(self, values):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), values=values, normalize=True)
        for arr in dict_num.values():
            assert np.isfinite(arr).all()
            assert (arr >= 0).all() and (arr <= 1).all()

    @settings(max_examples=5, deadline=None)
    @given(values=some.sampled_from(["log_odds", "frequencies"]))
    def test_normalize_false_keeps_raw_integers(self, values):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), values=values, normalize=False)
        for arr in dict_num.values():
            np.testing.assert_array_equal(arr, np.round(arr))

    # Positive tests: return_scales
    @settings(max_examples=5, deadline=None)
    @given(values=some.sampled_from(["log_odds", "frequencies"]))
    def test_return_scales(self, values):
        dict_num, df_scales, df_cat = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), values=values,
                                                                   return_scales=True)
        assert isinstance(dict_num, dict)
        assert df_scales.shape == (20, 20)
        assert list(df_scales.index) == ut.LIST_CANONICAL_AA
        assert list(df_cat["scale_id"]) == list(df_scales.columns)

    def test_return_scales_false(self):
        result = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), return_scales=False)
        assert isinstance(result, dict)

    # Negative tests: pssm
    def test_invalid_pssm_type(self):
        for pssm in [None, 1, 2.5, [], ["P1.pssm"], pd.DataFrame()]:
            with pytest.raises(ValueError, match="'pssm'"):
                aa.NumericalFeature.from_pssm(pssm)

    def test_invalid_pssm_missing_directory(self):
        with pytest.raises(ValueError, match="should be an existing '.pssm' file, a directory"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR / "does_not_exist"))

    def test_invalid_pssm_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="at least one '.pssm' file"):
                aa.NumericalFeature.from_pssm(pssm=tmp)

    def test_invalid_pssm_empty_dict(self):
        with pytest.raises(ValueError, match="empty dict"):
            aa.NumericalFeature.from_pssm(pssm={})

    def test_invalid_pssm_missing_file(self):
        with pytest.raises(ValueError, match="existing PSSM file"):
            aa.NumericalFeature.from_pssm(pssm={"P1": str(DATA_DIR / "P9.pssm")})

    @settings(max_examples=5, deadline=None)
    @given(width=some.integers(min_value=1, max_value=40).filter(lambda w: w != 20))
    def test_invalid_pssm_width(self, width):
        with pytest.raises(ValueError, match=r"\(L, 20\)"):
            aa.NumericalFeature.from_pssm(pssm={"X1": np.zeros((5, width))})

    def test_invalid_pssm_array_ndim_or_empty(self):
        for arr in [np.zeros(20), np.zeros((2, 3, 20)), np.zeros((0, 20))]:
            with pytest.raises(ValueError, match=r"\(L, 20\)"):
                aa.NumericalFeature.from_pssm(pssm={"X1": arr})

    def test_invalid_pssm_array_not_finite(self):
        for bad in [np.nan, np.inf, -np.inf]:
            arr = np.zeros((3, 20))
            arr[1, 4] = bad
            with pytest.raises(ValueError, match=r"'pssm\['X1'\]'.*row 2, column 'F'.*NaN or infinite"):
                aa.NumericalFeature.from_pssm(pssm={"X1": arr})

    def test_invalid_pssm_array_non_numeric(self):
        with pytest.raises(ValueError, match="only numbers"):
            aa.NumericalFeature.from_pssm(pssm={"X1": [["a"] * 20]})

    def test_invalid_pssm_value_type(self):
        for src in [5, None, {"a": 1}]:
            with pytest.raises(ValueError, match="file path"):
                aa.NumericalFeature.from_pssm(pssm={"X1": src})

    def test_invalid_pssm_key(self):
        with pytest.raises(ValueError, match="entry names"):
            aa.NumericalFeature.from_pssm(pssm={1: np.zeros((3, 20))})

    def test_invalid_pssm_unparseable_file(self, tmp_path):
        (tmp_path / "bad.pssm").write_text("this is not a PSSM\n1 2 3\n")
        with pytest.raises(ValueError, match="could not be parsed"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path))

    def test_invalid_pssm_single_file_wrong_extension(self, tmp_path):
        path = tmp_path / "P1.txt"
        write_pssm(path, "ACD")
        with pytest.raises(ValueError, match="should be a PSI-BLAST ASCII '.pssm' file"):
            aa.NumericalFeature.from_pssm(pssm=str(path))

    def test_invalid_pssm_file_not_finite(self, tmp_path):
        for bad in ["nan", "inf", "-inf"]:
            write_pssm(tmp_path / "B1.pssm", "ACD")
            lines = (tmp_path / "B1.pssm").read_text().splitlines()
            lines[3] = lines[3].replace(lines[3].split()[2], bad, 1)
            (tmp_path / "B1.pssm").write_text("\n".join(lines))
            with pytest.raises(ValueError, match=r"'pssm\['B1'\]'.*row 1.*NaN or infinite"):
                aa.NumericalFeature.from_pssm(pssm=str(tmp_path / "B1.pssm"))

    def test_invalid_pssm_file_malformed_number(self, tmp_path):
        write_pssm(tmp_path / "B2.pssm", "ACD")
        lines = (tmp_path / "B2.pssm").read_text().splitlines()
        lines[4] = lines[4].replace(lines[4].split()[2], "n/a", 1)
        (tmp_path / "B2.pssm").write_text("\n".join(lines))
        with pytest.raises(ValueError, match=r"'pssm\['B2'\]' \(field 'n/a' on line 5 of file .*should be a number"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path / "B2.pssm"))

    def test_invalid_pssm_frequencies_out_of_range(self, tmp_path):
        for bad, i_col in [(101, 3), (-1, 3)]:
            arr = np.full((2, 20), 50.0)
            arr[1, i_col] = bad
            with pytest.raises(ValueError, match=r"weighted observed percentage in \[0, 100\]"):
                aa.NumericalFeature.from_pssm(pssm={"X1": arr}, values="frequencies")

    def test_invalid_pssm_file_frequencies_out_of_range(self, tmp_path):
        write_pssm(tmp_path / "B3.pssm", "ACD")
        lines = (tmp_path / "B3.pssm").read_text().splitlines()
        tokens = lines[3].split()
        tokens[2 + 20] = "150"
        lines[3] = "    1 A  " + " ".join(tokens[2:])
        (tmp_path / "B3.pssm").write_text("\n".join(lines))
        with pytest.raises(ValueError, match=r"'pssm\['B3'\]' \(150.0 at row 1.*\[0, 100\]"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path / "B3.pssm"), values="frequencies")

    def test_invalid_pssm_header(self, tmp_path):
        write_pssm(tmp_path / "bad.pssm", "ACD", header="AANDCQEGHILKMFPSTWYV")
        with pytest.raises(ValueError, match="column header .* should list"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path))

    # Negative tests: df_seq
    def test_invalid_df_seq_type(self):
        for df_seq in [1, "df_seq", [], pd.DataFrame({"sequence": ["ACDKW"]})]:
            with pytest.raises(ValueError, match="df_seq"):
                aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=df_seq)

    def test_invalid_df_seq_length_mismatch(self):
        df_seq = _df_seq(dict_seq={"P1": "ACDKWA", "P2": DICT_SEQ["P2"]})
        with pytest.raises(ValueError, match="row count differs"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=df_seq)

    def test_invalid_df_seq_residue_mismatch(self):
        df_seq = _df_seq(dict_seq={"P1": "ACDKY"})
        with pytest.raises(ValueError, match="residue column differs"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=df_seq)

    def test_invalid_df_seq_missing_entry(self):
        df_seq = _df_seq(dict_seq={"P1": "ACDKW", "P7": "ACD"})
        with pytest.raises(ValueError, match="missing in 'pssm'"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=df_seq)

    def test_invalid_df_seq_without_sequence_column(self):
        df_seq = pd.DataFrame({"entry": ["P1"], "jmd_n": ["A"], "tmd": ["CDK"], "jmd_c": ["W"]})
        with pytest.raises(ValueError, match="'sequence' column"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=df_seq)

    # Negative tests: values / normalize / return_scales
    def test_invalid_values(self):
        for values in INVALID_VALUES:
            with pytest.raises(ValueError, match="'values'"):
                aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), values=values)

    def test_invalid_values_frequencies_block_missing(self, tmp_path):
        write_pssm(tmp_path / "P1.pssm", "ACDKW", with_freq=False)
        with pytest.raises(ValueError, match="no weighted observed percentage block"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path), values="frequencies")

    def test_invalid_normalize(self):
        for normalize in INVALID_BOOLS:
            with pytest.raises(ValueError, match="'normalize'"):
                aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), normalize=normalize)

    def test_invalid_return_scales(self):
        for return_scales in INVALID_BOOLS:
            with pytest.raises(ValueError, match="'return_scales'"):
                aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), return_scales=return_scales)

    def test_keyword_only_options(self):
        with pytest.raises(TypeError):
            aa.NumericalFeature.from_pssm(str(DATA_DIR), None)  # noqa


class TestFromPssmComplex:
    """Combinations of parameters and edge interactions."""

    # Positive tests
    def test_log_odds_file_without_frequency_block(self, tmp_path):
        write_pssm(tmp_path / "P1.pssm", "ACDKW", with_freq=False)
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(tmp_path), df_seq=_df_seq(["P1"]))
        assert dict_num["P1"].shape == (5, 20)

    @settings(max_examples=5, deadline=None)
    @given(values=some.sampled_from(["log_odds", "frequencies"]), normalize=some.booleans())
    def test_mixed_paths_and_arrays(self, values, normalize):
        arr = np.full((4, 20), 50.0)
        pssm = {"P1": str(DATA_DIR / "P1.pssm"), "X1": arr}
        dict_seq = {"P1": "ACDKW", "X1": "AAAA"}
        dict_num = aa.NumericalFeature.from_pssm(pssm, df_seq=_df_seq(dict_seq=dict_seq), values=values,
                                                 normalize=normalize)
        assert dict_num["X1"].shape == (4, 20)
        if normalize and values == "frequencies":
            np.testing.assert_allclose(dict_num["X1"], 0.5)

    def test_residue_x_and_lowercase_tolerated(self, tmp_path):
        write_pssm(tmp_path / "Q1.pssm", "ACXKW", seed=3)
        df_seq = _df_seq(dict_seq={"Q1": "aCDkX"})
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(tmp_path), df_seq=df_seq)
        assert dict_num["Q1"].shape == (5, 20)

    def test_extra_pssm_entries_not_in_df_seq(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=_df_seq(["P2"]), return_scales=False)
        assert set(dict_num) == set(DICT_SEQ)

    def test_frequencies_return_scales_description(self):
        _, df_scales, df_cat = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), values="frequencies",
                                                            normalize=False, return_scales=True)
        assert df_cat["scale_description"].str.contains("weighted observed percentage").all()
        assert df_scales.columns[0] == "PSSM_A"

    def test_same_input_same_output(self):
        a = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=_df_seq())
        b = aa.NumericalFeature.from_pssm(pssm=_dict_paths(), df_seq=_df_seq())
        for entry in DICT_SEQ:
            np.testing.assert_array_equal(a[entry], b[entry])

    def test_end_to_end_get_parts_run_num(self, tmp_path):
        """from_pssm -> NumericalFeature.get_parts -> CPP.run_num yields a finite df_feat."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=3)
        for i, (entry, seq) in enumerate(zip(df_seq["entry"], df_seq["sequence"])):
            write_pssm(tmp_path / f"{entry}.pssm", seq, seed=i)
        nf = aa.NumericalFeature()
        dict_num, df_scales, df_cat = nf.from_pssm(pssm=str(tmp_path), df_seq=df_seq, return_scales=True)
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, df_cat=df_cat, verbose=False)
        df_feat = cpp.run_num(dict_num_parts=dict_num_parts, labels=df_seq["label"].to_list(), n_jobs=1)
        assert len(df_feat) >= 1
        assert df_feat["feature"].notna().all()
        assert np.isfinite(df_feat["abs_auc"]).all()
        assert all(f.split("-")[-1].startswith("PSSM_") for f in df_feat["feature"])
        X = nf.feature_matrix(features=df_feat, dict_num_parts=dict_num_parts, df_parts=df_parts,
                              df_scales=df_scales)
        assert X.shape == (len(df_seq), len(df_feat))

    # Negative tests
    def test_all_mismatches_reported_together(self):
        df_seq = _df_seq(dict_seq={"P1": "ACDKY", "P2": "MKL", "P7": "AC"})
        with pytest.raises(ValueError) as exc:
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=df_seq)
        msg = str(exc.value)
        assert "P7" in msg and "P2 (PSSM rows=8, len(sequence)=3)" in msg and "['P1']" in msg

    def test_one_broken_file_in_directory(self, tmp_path):
        write_pssm(tmp_path / "A1.pssm", "ACD")
        (tmp_path / "A2.pssm").write_text("")
        with pytest.raises(ValueError, match="A2.pssm"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path))

    def test_mixed_dict_with_bad_array(self):
        pssm = {"P1": str(DATA_DIR / "P1.pssm"), "X1": np.zeros((3, 21))}
        with pytest.raises(ValueError, match=r"'pssm\['X1'\]' \(shape \(3, 21\)\)"):
            aa.NumericalFeature.from_pssm(pssm, return_scales=True)

    def test_mismatch_raises_even_with_return_scales(self):
        with pytest.raises(ValueError, match="should match 'df_seq'"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=_df_seq(dict_seq={"P3": "RSTNQ"}),
                                          return_scales=True, values="frequencies")

    def test_frequencies_missing_with_df_seq(self, tmp_path):
        write_pssm(tmp_path / "P1.pssm", "ACDKW", with_freq=False)
        with pytest.raises(ValueError, match=r"'values' \('frequencies'\) should be 'log_odds'"):
            aa.NumericalFeature.from_pssm(pssm=str(tmp_path), df_seq=_df_seq(["P1"]), values="frequencies",
                                          normalize=False)

    def test_array_length_mismatch_with_df_seq(self):
        with pytest.raises(ValueError, match="row count differs"):
            aa.NumericalFeature.from_pssm(pssm={"P1": np.zeros((4, 20))}, df_seq=_df_seq(["P1"]),
                                          normalize=False)

    def test_invalid_values_with_valid_df_seq(self):
        with pytest.raises(ValueError, match="'values'"):
            aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), df_seq=_df_seq(), values="scores")


class TestFromPssmGoldenValues:
    """Hand-derived values, including the PSI-BLAST -> canonical column permutation."""

    def test_psiblast_order_constant(self):
        assert "".join(ut.LIST_PSSM_AA_ORDER) == ORDER_PSIBLAST
        assert sorted(ut.LIST_PSSM_AA_ORDER) == sorted(ut.LIST_CANONICAL_AA)

    def test_log_odds_permutation_row0(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=_dict_paths(["P1"]), normalize=False)
        np.testing.assert_array_equal(dict_num["P1"][0], GOLD_P1_ROW0_LOG_ODDS)

    def test_log_odds_permutation_row1(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=_dict_paths(["P1"]), normalize=False)
        np.testing.assert_array_equal(dict_num["P1"][1], GOLD_P1_ROW1_LOG_ODDS)

    def test_frequencies_permutation_row0(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=_dict_paths(["P1"]), values="frequencies", normalize=False)
        np.testing.assert_array_equal(dict_num["P1"][0], GOLD_P1_ROW0_FREQ)

    def test_sigmoid_normalization_row0(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=_dict_paths(["P1"]), normalize=True)
        row = dict_num["P1"][0]
        i_l, i_a = ut.LIST_CANONICAL_AA.index("L"), ut.LIST_CANONICAL_AA.index("A")
        assert row[i_l] == pytest.approx(0.5)                           # L has log-odds 0
        assert row[i_a] == pytest.approx(1 / (1 + np.exp(10)))         # A has log-odds -10
        np.testing.assert_allclose(row, 1 / (1 + np.exp(-np.array(GOLD_P1_ROW0_LOG_ODDS))))

    def test_percentage_normalization_row0(self):
        dict_num = aa.NumericalFeature.from_pssm(pssm=_dict_paths(["P1"]), values="frequencies", normalize=True)
        np.testing.assert_allclose(dict_num["P1"][0], np.array(GOLD_P1_ROW0_FREQ) / 100)

    def test_scale_names_and_categories(self):
        _, df_scales, df_cat = aa.NumericalFeature.from_pssm(pssm=str(DATA_DIR), return_scales=True)
        assert list(df_scales.columns) == [f"PSSM_{a}" for a in ut.LIST_CANONICAL_AA]
        np.testing.assert_array_equal(df_scales.values, np.eye(20))
        dict_cat = dict(zip(df_cat["scale_id"], df_cat["category"]))
        assert dict_cat["PSSM_K"] == "Positive"
        assert dict_cat["PSSM_Y"] == "Aromatic"
        assert df_cat.set_index("scale_id").loc["PSSM_W", "scale_name"] == "PSSM W"

    def test_extreme_log_odds_normalize_stable(self, recwarn):
        arr = np.array([[-1e6, -800.0, -50.0, 0.0, 50.0, 800.0, 1e6] + [0.0] * 13])
        with np.errstate(over="raise", under="raise"):
            dict_num = aa.NumericalFeature.from_pssm(pssm={"X1": arr}, normalize=True)
        row = dict_num["X1"][0]
        assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
        assert np.isfinite(row).all()
        assert (row >= 0).all() and (row <= 1).all()
        np.testing.assert_allclose(row[:7], [0.0, 0.0, 1 / (1 + np.exp(50)), 0.5,
                                             1 / (1 + np.exp(-50)), 1.0, 1.0], atol=1e-12)

    def test_array_input_not_permuted(self):
        arr = np.tile(np.arange(20, dtype=float), (2, 1))
        dict_num = aa.NumericalFeature.from_pssm(pssm={"X1": arr}, values="frequencies", normalize=True)
        np.testing.assert_allclose(dict_num["X1"][1], np.arange(20) / 100)
