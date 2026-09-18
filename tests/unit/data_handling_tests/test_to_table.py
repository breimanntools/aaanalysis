"""This is a script to test the to_table() function."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as some

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

STEM_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
COLS_FEAT_REQUIRED = ["feature", "category", "subcategory", "scale_name",
                      "scale_description", "abs_auc", "abs_mean_dif", "mean_dif",
                      "std_test", "std_ref", "p_val_mann_whitney", "p_val_fdr_bh",
                      "positions"]


def _make_df_feat(n=3):
    """Return a tiny frame satisfying the required df_feat columns."""
    return pd.DataFrame({"feature": [f"TMD-Segment(1,{i + 1})-KLEP840101" for i in range(n)],
                         "category": ["Energy"] * n,
                         "subcategory": ["Charge"] * n,
                         "scale_name": ["Charge"] * n,
                         "scale_description": ["Net charge."] * n,
                         "abs_auc": np.linspace(0.1, 0.4, n),
                         "abs_mean_dif": np.linspace(0.01, 0.2, n),
                         "mean_dif": np.linspace(-0.2, 0.2, n),
                         "std_test": np.linspace(0.05, 0.15, n),
                         "std_ref": np.linspace(0.05, 0.15, n),
                         "p_val_mann_whitney": np.linspace(0.0, 0.05, n),
                         "p_val_fdr_bh": np.linspace(0.0, 0.05, n),
                         "positions": [f"{i + 1},{i + 2}" for i in range(n)]})


def _read_metadata(file_path):
    """Return the sidecar record written next to 'file_path'."""
    path_meta = os.path.splitext(file_path)[0] + ".meta.json"
    with open(path_meta, "r", encoding="utf-8") as file_meta:
        return json.load(file_meta)


class TestToTable:
    """Test the 'to_table' function by testing each parameter individually."""

    # Positive tests
    @settings(max_examples=5, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(n=some.integers(min_value=1, max_value=8))
    def test_df_valid(self, tmp_path, n):
        """Test 'df' with frames of varying length."""
        df_feat = _make_df_feat(n=n)
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path)
        assert dict_metadata["n_rows"] == n
        assert os.path.isfile(file_path)

    @settings(max_examples=5, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(stem=some.text(alphabet=STEM_ALPHABET, min_size=1, max_size=12))
    def test_file_path_valid(self, tmp_path, stem):
        """Test 'file_path' with varying file stems."""
        file_path = str(tmp_path / f"{stem}.csv")
        aa.to_table(df=_make_df_feat(), file_path=file_path)
        assert os.path.isfile(file_path)

    @pytest.mark.parametrize("name", ["df_feat", "df_seq", "df_parts", "X"])
    def test_name_valid(self, tmp_path, name):
        """Test 'name' with the schemas the exported frames are documented against."""
        dict_df = {"df_feat": _make_df_feat(),
                   "df_seq": pd.DataFrame({"entry": ["P1", "P2"], "sequence": ["AAC", "ACD"]}),
                   "df_parts": pd.DataFrame({"tmd": ["AAC"], "jmd_n": ["ACD"]}),
                   "X": pd.DataFrame(np.arange(6.0).reshape(2, 3), columns=["f1", "f2", "f3"])}
        file_path = str(tmp_path / "table.csv")
        dict_metadata = aa.to_table(df=dict_df[name], file_path=file_path, name=name)
        assert dict_metadata["name"] == name

    @pytest.mark.parametrize("sep,suffix", [(",", ".csv"), (";", ".csv"), ("\t", ".tsv"),
                                            (None, ".csv"), (None, ".tsv")])
    def test_sep_valid(self, tmp_path, sep, suffix):
        """Test 'sep' with every separator each table extension accepts."""
        file_path = str(tmp_path / f"df_feat{suffix}")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path, sep=sep)
        expected = sep if sep is not None else ("," if suffix == ".csv" else "\t")
        assert dict_metadata["sep"] == expected

    @settings(max_examples=5, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(random_state=some.integers(min_value=0, max_value=1000))
    def test_random_state_valid(self, tmp_path, random_state):
        """Test 'random_state' is recorded in the provenance part of the sidecar."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path,
                                    random_state=random_state)
        assert dict_metadata["provenance"]["random_state"] == random_state

    def test_random_state_none_valid(self, tmp_path):
        """Test 'random_state' None records that no seed was in effect."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path, random_state=None)
        assert dict_metadata["provenance"]["random_state"] is None
        assert dict_metadata["provenance"]["deterministic"] is False

    def test_returns_dict_metadata(self, tmp_path):
        """Test the returned record carries the documented top-level keys."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path)
        keys = {"schema_version", "name", "description", "file_name", "file_name_metadata",
                "sep", "n_rows", "n_cols", "columns", "columns_undocumented",
                "categories", "parts", "provenance"}
        assert isinstance(dict_metadata, dict)
        assert keys.issubset(set(dict_metadata))

    def test_sidecar_written(self, tmp_path):
        """Test the JSON sidecar is written next to the table and matches the return value."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path)
        assert os.path.isfile(str(tmp_path / "df_feat.meta.json"))
        assert _read_metadata(file_path) == dict_metadata

    def test_schema_version_valid(self, tmp_path):
        """Test the sidecar carries a non-empty schema-version field (KPI)."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path)
        assert isinstance(dict_metadata["schema_version"], str)
        assert dict_metadata["schema_version"].strip() != ""

    def test_column_descriptions_complete(self, tmp_path):
        """Test every df_feat column has a non-empty description (KPI)."""
        df_feat = aa.load_features()
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path, name="df_feat")
        assert set(dict_metadata["columns"]) == set(df_feat.columns)
        assert set(COLS_FEAT_REQUIRED).issubset(set(dict_metadata["columns"]))
        assert all(r["description"].strip() != "" for r in dict_metadata["columns"].values())
        assert dict_metadata["columns_undocumented"] == []

    def test_roundtrip_csv_lossless(self, tmp_path):
        """Test the CSV round-trip is lossless on the DOM_GSEC feature output (KPI)."""
        df_feat = aa.load_features(name="DOM_GSEC")
        file_path = str(tmp_path / "df_feat.csv")
        aa.to_table(df=df_feat, file_path=file_path, name="df_feat")
        df_read = pd.read_csv(file_path)
        assert df_read.equals(df_feat)
        assert list(df_read.dtypes) == list(df_feat.dtypes)

    def test_roundtrip_with_sidecar_dtypes(self, tmp_path):
        """Test the documented dtype-hinted load-back restores the frame exactly."""
        df_feat = aa.load_features()
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path)
        dtypes = {c: r["dtype_pandas"] for c, r in dict_metadata["columns"].items()}
        assert pd.read_csv(file_path, dtype=dtypes).equals(df_feat)

    def test_provenance_recorded(self, tmp_path):
        """Test the package version and an input fingerprint are recorded."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path)
        assert dict_metadata["provenance"]["aaanalysis_version"] == aa.__version__
        assert dict_metadata["provenance"]["input_hash"].startswith("sha256:")

    def test_index_not_written(self, tmp_path):
        """Test the row index is not written as an extra column."""
        df_feat = _make_df_feat()
        file_path = str(tmp_path / "df_feat.csv")
        aa.to_table(df=df_feat, file_path=file_path)
        with open(file_path, "r", encoding="utf-8") as file_table:
            header = file_table.readline().strip()
        assert header.split(",") == list(df_feat.columns)

    def test_vocabularies_recorded(self, tmp_path):
        """Test the AAontology category and sequence part vocabularies are recorded."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path)
        assert "Energy" in dict_metadata["categories"]
        assert "tmd" in dict_metadata["parts"] and "jmd_n" in dict_metadata["parts"]

    # Negative tests
    @pytest.mark.parametrize("df", [None, "df_feat", 42, ["feature"], {"feature": [1]}])
    def test_df_invalid(self, tmp_path, df):
        """Test 'df' rejects non-DataFrame input."""
        with pytest.raises(ValueError):
            aa.to_table(df=df, file_path=str(tmp_path / "df_feat.csv"))

    def test_df_empty_invalid(self, tmp_path):
        """Test 'df' rejects a frame without the schema's required columns."""
        with pytest.raises(ValueError):
            aa.to_table(df=pd.DataFrame(), file_path=str(tmp_path / "df_feat.csv"))

    @pytest.mark.parametrize("file_path", [None, 42, 1.5, ["out.csv"]])
    def test_file_path_invalid(self, file_path):
        """Test 'file_path' rejects non-string input."""
        with pytest.raises(ValueError):
            aa.to_table(df=_make_df_feat(), file_path=file_path)

    @pytest.mark.parametrize("suffix", [".txt", ".parquet", ".json", ".xlsx", ""])
    def test_file_path_extension_invalid(self, tmp_path, suffix):
        """Test 'file_path' rejects an unsupported table extension."""
        with pytest.raises(ValueError, match="should end with one of"):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / f"df_feat{suffix}"))

    @pytest.mark.parametrize("name", ["", "feat", "DF_FEAT", "df_features", 42, None])
    def test_name_invalid(self, tmp_path, name):
        """Test 'name' rejects anything that is not a known schema name."""
        with pytest.raises(ValueError):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_feat.csv"), name=name)

    @pytest.mark.parametrize("sep", ["|", " ", "::", "\t"])
    def test_sep_invalid_for_csv(self, tmp_path, sep):
        """Test 'sep' rejects separators a '.csv' file does not accept."""
        with pytest.raises(ValueError, match="is not valid for a"):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_feat.csv"), sep=sep)

    @pytest.mark.parametrize("sep", [42, 1.5, [","]])
    def test_sep_type_invalid(self, tmp_path, sep):
        """Test 'sep' rejects non-string input."""
        with pytest.raises(ValueError):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_feat.csv"), sep=sep)

    @pytest.mark.parametrize("random_state", [-1, -10, 1.5, "0", [0]])
    def test_random_state_invalid(self, tmp_path, random_state):
        """Test 'random_state' rejects anything but a non-negative integer or None."""
        with pytest.raises(ValueError):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_feat.csv"),
                        random_state=random_state)

    def test_missing_directory_invalid(self, tmp_path):
        """Test a 'file_path' in a non-existing directory raises an actionable error."""
        file_path = str(tmp_path / "missing" / "df_feat.csv")
        with pytest.raises(ValueError, match="does not exist"):
            aa.to_table(df=_make_df_feat(), file_path=file_path)

    def test_schema_mismatch_invalid(self, tmp_path):
        """Test a frame not matching the named schema names the missing columns."""
        with pytest.raises(ValueError, match="required columns"):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_seq.csv"),
                        name="df_seq")

    def test_missing_required_column_invalid(self, tmp_path):
        """Test dropping a required df_feat column is rejected by name."""
        df_feat = _make_df_feat().drop(columns=["abs_auc"])
        with pytest.raises(ValueError, match="abs_auc"):
            aa.to_table(df=df_feat, file_path=str(tmp_path / "df_feat.csv"), name="df_feat")

    def test_no_sidecar_on_invalid_input(self, tmp_path):
        """Test nothing is written when validation fails."""
        file_path = str(tmp_path / "df_feat.txt")
        with pytest.raises(ValueError):
            aa.to_table(df=_make_df_feat(), file_path=file_path)
        assert os.listdir(str(tmp_path)) == []


class TestToTableComplex:
    """Test the 'to_table' function with combined parameters and edge interactions."""

    # Positive tests
    def test_tsv_all_parameters(self, tmp_path):
        """Test every parameter combined on a tab-separated export."""
        df_feat = aa.load_features()
        file_path = str(tmp_path / "df_feat.tsv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path, name="df_feat",
                                    sep="\t", random_state=7)
        assert pd.read_csv(file_path, sep="\t").equals(df_feat)
        assert dict_metadata["sep"] == "\t"
        assert dict_metadata["provenance"]["random_state"] == 7
        assert dict_metadata["file_name"] == "df_feat.tsv"
        assert dict_metadata["file_name_metadata"] == "df_feat.meta.json"

    def test_semicolon_csv_roundtrip(self, tmp_path):
        """Test a semicolon-separated CSV round-trips losslessly."""
        df_feat = aa.load_features()
        file_path = str(tmp_path / "df_feat.csv")
        aa.to_table(df=df_feat, file_path=file_path, sep=";", name="df_feat")
        assert pd.read_csv(file_path, sep=";").equals(df_feat)

    def test_df_seq_roundtrip(self, tmp_path):
        """Test a sequence table round-trips under its own schema."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        file_path = str(tmp_path / "df_seq.csv")
        dict_metadata = aa.to_table(df=df_seq, file_path=file_path, name="df_seq")
        assert pd.read_csv(file_path).equals(df_seq)
        assert dict_metadata["columns"]["entry"]["description"].strip() != ""

    def test_feature_matrix_roundtrip(self, tmp_path):
        """Test the feature matrix wrapped as a DataFrame exports under the 'X' schema."""
        X = pd.DataFrame(np.arange(12.0).reshape(3, 4),
                         columns=[f"TMD-Segment(1,{i})-KLEP840101" for i in range(4)])
        file_path = str(tmp_path / "X.csv")
        dict_metadata = aa.to_table(df=X, file_path=file_path, name="X", random_state=0)
        assert pd.read_csv(file_path).equals(X)
        assert all(r["description"].strip() != "" for r in dict_metadata["columns"].values())

    def test_extra_column_documented_as_outside_contract(self, tmp_path):
        """Test a column outside the schema still gets a description and is flagged."""
        df_feat = _make_df_feat()
        df_feat["my_own_score"] = [0.1, 0.2, 0.3]
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path, name="df_feat")
        assert dict_metadata["columns_undocumented"] == ["my_own_score"]
        assert dict_metadata["columns"]["my_own_score"]["description"].strip() != ""

    def test_optional_df_feat_column_documented(self, tmp_path):
        """Test a post-hoc df_feat column is described from the feature contract."""
        df_feat = _make_df_feat()
        df_feat["feat_importance"] = [1.0, 2.0, 3.0]
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path, name="df_feat")
        assert dict_metadata["columns_undocumented"] == []
        assert "importance" in dict_metadata["columns"]["feat_importance"]["description"].lower()

    def test_filtered_frame_roundtrip(self, tmp_path):
        """Test a filtered frame round-trips after resetting its index."""
        df_feat = aa.load_features()
        df_feat_top = df_feat[df_feat["abs_auc"] > df_feat["abs_auc"].median()]
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat_top, file_path=file_path, name="df_feat")
        assert pd.read_csv(file_path).equals(df_feat_top.reset_index(drop=True))
        assert dict_metadata["n_rows"] == len(df_feat_top)

    def test_nan_values_roundtrip(self, tmp_path):
        """Test NaN values in an optional column survive the round-trip."""
        df_feat = _make_df_feat()
        df_feat["abs_auc_ci_low"] = [0.1, np.nan, 0.3]
        file_path = str(tmp_path / "df_feat.csv")
        aa.to_table(df=df_feat, file_path=file_path, name="df_feat")
        df_read = pd.read_csv(file_path)
        assert df_read.equals(df_feat)
        assert bool(df_read["abs_auc_ci_low"].isna().iloc[1])

    def test_overwrite_existing_export(self, tmp_path):
        """Test re-exporting to the same path replaces both artifacts."""
        file_path = str(tmp_path / "df_feat.csv")
        aa.to_table(df=_make_df_feat(n=2), file_path=file_path)
        dict_metadata = aa.to_table(df=_make_df_feat(n=5), file_path=file_path)
        assert dict_metadata["n_rows"] == 5
        assert _read_metadata(file_path)["n_rows"] == 5
        assert len(pd.read_csv(file_path)) == 5

    def test_same_frame_same_input_hash(self, tmp_path):
        """Test two exports of the same frame record the same input fingerprint."""
        df_feat = _make_df_feat()
        meta_a = aa.to_table(df=df_feat, file_path=str(tmp_path / "a.csv"), random_state=1)
        meta_b = aa.to_table(df=df_feat, file_path=str(tmp_path / "b.csv"), random_state=1)
        assert meta_a["provenance"]["input_hash"] == meta_b["provenance"]["input_hash"]
        assert meta_a["file_name"] != meta_b["file_name"]

    def test_sidecar_is_json_serializable(self, tmp_path):
        """Test the sidecar contains only JSON types and reloads unchanged."""
        df_feat = aa.load_features()
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=df_feat, file_path=file_path, name="df_feat")
        assert json.loads(json.dumps(dict_metadata)) == dict_metadata

    def test_allowed_values_exported_for_category(self, tmp_path):
        """Test the AAontology category column exports its allowed values."""
        file_path = str(tmp_path / "df_feat.csv")
        dict_metadata = aa.to_table(df=_make_df_feat(), file_path=file_path, name="df_feat")
        assert "Energy" in dict_metadata["columns"]["category"]["allowed_values"]
        assert dict_metadata["columns"]["subcategory"]["description"].strip() != ""

    # Negative tests
    def test_tsv_with_semicolon_invalid(self, tmp_path):
        """Test a '.tsv' path rejects a semicolon separator."""
        with pytest.raises(ValueError, match="is not valid for a"):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_feat.tsv"), sep=";")

    def test_csv_with_tab_invalid(self, tmp_path):
        """Test a '.csv' path rejects a tab separator."""
        with pytest.raises(ValueError, match="is not valid for a"):
            aa.to_table(df=_make_df_feat(), file_path=str(tmp_path / "df_feat.csv"), sep="\t")

    def test_df_seq_with_feat_name_invalid(self, tmp_path):
        """Test a sequence table exported under the df_feat schema is rejected."""
        df_seq = aa.load_dataset(name="DOM_GSEC", n=5)
        with pytest.raises(ValueError, match="required columns"):
            aa.to_table(df=df_seq, file_path=str(tmp_path / "t.csv"), name="df_feat")

    def test_df_parts_with_seq_name_invalid(self, tmp_path):
        """Test a parts table exported under the df_seq schema is rejected."""
        df_parts = pd.DataFrame({"tmd": ["AAC"], "jmd_n": ["ACD"]})
        with pytest.raises(ValueError, match="entry"):
            aa.to_table(df=df_parts, file_path=str(tmp_path / "t.csv"), name="df_seq")

    def test_missing_directory_with_tsv_invalid(self, tmp_path):
        """Test the missing-directory guard also applies to a '.tsv' export."""
        file_path = str(tmp_path / "sub" / "df_feat.tsv")
        with pytest.raises(ValueError, match="does not exist"):
            aa.to_table(df=_make_df_feat(), file_path=file_path, sep="\t", name="df_feat")

    def test_invalid_name_and_valid_frame_invalid(self, tmp_path):
        """Test an unknown schema name is rejected before the frame is inspected."""
        with pytest.raises(ValueError, match="should be one of"):
            aa.to_table(df=aa.load_features(), file_path=str(tmp_path / "t.csv"),
                        name="df_feature", sep=",", random_state=0)

    def test_extension_checked_before_schema(self, tmp_path):
        """Test the extension guard fires even when the frame matches no schema."""
        with pytest.raises(ValueError, match="should end with one of"):
            aa.to_table(df=pd.DataFrame({"a": [1]}), file_path=str(tmp_path / "t.txt"),
                        name="df_feat")
