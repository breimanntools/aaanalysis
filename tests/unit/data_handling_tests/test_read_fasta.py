from hypothesis import given, settings, strategies as st
import pandas as pd
import pytest
import aaanalysis as aa
import warnings

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


class TestReadFasta:
    """Test the aa.read_fasta function by testing each parameter individually."""

    # Property-based testing for positive cases
    @given(file_path=st.text())
    def test_file_path(self, file_path):
        """Test the 'file_path' parameter with valid and invalid paths."""
        try:
            df = aa.read_fasta(file_path)
            assert isinstance(df, pd.DataFrame)  # Expecting a DataFrame to be returned
        except Exception as e:
            assert isinstance(e, (FileNotFoundError, ValueError))

    @given(col_id=st.text(min_size=1))
    def test_col_id(self, col_id):
        """Test valid 'col_id' parameter."""
        df = aa.read_fasta("valid_path.fasta", col_id=col_id)
        assert col_id in df.columns

    @given(col_seq=st.text(min_size=1))
    def test_col_seq(self, col_seq):
        """Test valid 'col_seq' parameter."""
        df = aa.read_fasta("valid_path.fasta", col_seq=col_seq)
        assert col_seq in df.columns

    @given(cols_info=st.lists(st.text(min_size=1), min_size=1, max_size=1))
    def test_cols_info(self, cols_info):
        """Test valid 'cols_info' parameter."""
        df = aa.read_fasta("valid_path.fasta", cols_info=cols_info)
        for col in cols_info:
            assert col in df.columns

    @given(col_db=st.one_of(st.none(), st.text(min_size=1)))
    def test_col_db(self, col_db):
        """Test valid 'col_db' parameter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df = aa.read_fasta("valid_path_db.fasta", col_db=col_db)
            if col_db:
                assert col_db in df.columns

    @given(sep=st.text(min_size=1, max_size=1))
    def test_sep(self, sep):
        """Test valid 'sep' parameter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df = aa.read_fasta("valid_path.fasta", sep=sep)
            assert isinstance(df, pd.DataFrame)

    # Property-based testing for negative cases
    def test_invalid_col_id(self):
        """Test invalid 'col_id' parameter."""
        for col_id in [None, [], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta("valid_path.fasta", col_id=col_id)

    def test_invalid_col_seq(self):
        """Test invalid 'col_seq' parameter."""
        for col_seq in [None, [], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta("valid_path.fasta", col_seq=col_seq)

    def test_invalid_col_db(self):
        """Test invalid 'col_db' parameter."""
        for col_db in [[], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta("valid_path_db.fasta", col_db=col_db)

    def test_invalid_cols_info(self):
        """Test invalid 'cols_info' parameter."""
        with pytest.raises(ValueError):
            aa.read_fasta("valid_path.fasta", cols_info={})
        with pytest.raises(ValueError):
            aa.read_fasta("valid_path.fasta", cols_info=34)
        with pytest.raises(ValueError):
            aa.read_fasta("valid_path.fasta", cols_info=[None])

    def test_invalid_sep(self):
        """Test invalid 'sep' parameter."""
        for sep in [[], {}, 34]:
            with pytest.raises(ValueError):
                aa.read_fasta("valid_path.fasta", sep=sep)


class TestReadFastaComplex:
    """Test aa.read_fasta function with complex scenarios"""

    @given(file_path=st.text(), col_id=st.text(min_size=1), col_seq=st.text(min_size=1), sep=st.text(min_size=1, max_size=1))
    def test_combination_valid_inputs(self, file_path, col_id, col_seq, sep):
        """Test valid combinations of parameters."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df = aa.read_fasta(file_path, col_id=col_id, col_seq=col_seq, sep=sep)
                assert isinstance(df, pd.DataFrame)
                assert col_id in df.columns
                assert col_seq in df.columns
        except Exception as e:
            assert isinstance(e, (FileNotFoundError, ValueError))

    @given(file_path=st.text(), col_id=st.text(max_size=0), col_seq=st.text(min_size=1), sep=st.text(min_size=1, max_size=1))
    def test_combination_invalid(self, file_path, col_id, col_seq, sep):
        """Test invalid 'col_id' in combination with other parameters."""
        with pytest.raises(ValueError):
            aa.read_fasta(file_path, col_id=col_id, col_seq=col_seq, sep=sep)


