from hypothesis import given, settings, strategies as st
import pandas as pd
import pytest
import aaanalysis as aa
import warnings

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")

FILE_IN = "valid_path.fasta"
FILE_DB_IN = "valid_path_db.fasta"
COL_DB = "database"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def creat_mock_file():
    """"""
    df_seq = aa.load_dataset(name="SEQ_AMYLO", n=10)
    df_seq["pred"] = range(len(df_seq))
    aa.to_fasta(df_seq=df_seq, file_path=FILE_IN, cols_info=["pred"])
    df_seq[COL_DB] = "sp"
    aa.to_fasta(df_seq=df_seq, file_path=FILE_DB_IN, cols_info=["pred"], col_db=COL_DB)


class TestReadFasta:
    """Test the aa.read_fasta function by testing each parameter individually."""

    # Property-based testing for positive cases
    def test_file_path(self):
        """Test the 'file_path' parameter with valid and invalid paths."""
        creat_mock_file()
        df = aa.read_fasta(file_path=FILE_IN)
        assert isinstance(df, pd.DataFrame)  # Expecting a DataFrame to be returned

    @given(col_id=st.text(min_size=1, alphabet=ALPHABET))
    def test_col_id(self, col_id):
        """Test valid 'col_id' parameter."""
        creat_mock_file()
        df = aa.read_fasta(file_path=FILE_IN, col_id=col_id)
        assert col_id in df.columns

    @given(col_seq=st.text(min_size=1, alphabet=ALPHABET))
    def test_col_seq(self, col_seq):
        """Test valid 'col_seq' parameter."""
        creat_mock_file()
        df = aa.read_fasta(file_path=FILE_IN, col_seq=col_seq)
        assert col_seq in df.columns

    @given(cols_info=st.lists(st.text(min_size=1, alphabet=ALPHABET), min_size=1, max_size=1))
    def test_cols_info(self, cols_info):
        """Test valid 'cols_info' parameter."""
        creat_mock_file()
        df = aa.read_fasta(file_path=FILE_IN, cols_info=cols_info)
        for col in cols_info:
            assert col in df.columns

    def test_col_db(self):
        """Test valid 'col_db' parameter."""
        creat_mock_file()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df = aa.read_fasta(file_path=FILE_DB_IN, col_db=COL_DB)
            assert COL_DB in df.columns

    @given(sep=st.text(min_size=1, max_size=1, alphabet=",|;-"))
    def test_sep(self, sep):
        """Test valid 'sep' parameter."""
        creat_mock_file()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df = aa.read_fasta(file_path=FILE_IN, sep=sep)
            assert isinstance(df, pd.DataFrame)

    # Property-based testing for negative cases
    def test_invalid_col_id(self):
        """Test invalid 'col_id' parameter."""
        creat_mock_file()
        for col_id in [None, [], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta(file_path=FILE_IN, col_id=col_id)

    def test_invalid_col_seq(self):
        """Test invalid 'col_seq' parameter."""
        creat_mock_file()
        for col_seq in [None, [], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta(file_path=FILE_IN, col_seq=col_seq)

    def test_invalid_col_db(self):
        """Test invalid 'col_db' parameter."""
        creat_mock_file()
        for col_db in [[], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta(file_path=FILE_DB_IN, col_db=col_db)

    def test_invalid_cols_info(self):
        """Test invalid 'cols_info' parameter."""
        creat_mock_file()
        with pytest.raises(ValueError):
            aa.read_fasta(file_path=FILE_IN, cols_info={})
        with pytest.raises(ValueError):
            aa.read_fasta(file_path=FILE_IN, cols_info=34)
        with pytest.raises(ValueError):
            aa.read_fasta(file_path=FILE_IN, cols_info=[None])

    def test_invalid_sep(self):
        """Test invalid 'sep' parameter."""
        creat_mock_file()
        for sep in [[], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta(file_path=FILE_IN, sep=sep)


class TestReadFastaComplex:
    """Test aa.read_fasta function with complex scenarios"""

    @given(col_id=st.text(min_size=1, alphabet=ALPHABET),
           col_seq=st.text(min_size=1, alphabet=ALPHABET),
           sep=st.text(min_size=1, max_size=1, alphabet=",|;-"))
    def test_combination_valid_inputs(self, col_id, col_seq, sep):
        """Test valid combinations of parameters."""
        creat_mock_file()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df = aa.read_fasta(file_path=FILE_IN, col_id=col_id, col_seq=col_seq, sep=sep)
                assert isinstance(df, pd.DataFrame)
                assert col_id in df.columns
                assert col_seq in df.columns
        except Exception as e:
            assert isinstance(e, (FileNotFoundError, ValueError))

    @given(col_id=st.text(max_size=0, alphabet=ALPHABET),
           col_seq=st.text(min_size=1, alphabet=ALPHABET),
           sep=st.integers())
    def test_combination_invalid(self, col_id, col_seq, sep):
        """Test invalid 'col_id' in combination with other parameters."""
        creat_mock_file()
        with pytest.raises(ValueError):
            aa.read_fasta(file_path=FILE_IN, col_id=col_id, col_seq=col_seq, sep=sep)


