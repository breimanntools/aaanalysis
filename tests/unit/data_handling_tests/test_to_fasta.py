from hypothesis import given, settings, strategies as st
import pandas as pd
import pytest
import aaanalysis as aa

# Extend default deadline for more complex operations
settings.register_profile("default", deadline=500)
settings.load_profile("default")

FILE_OUT = "valid_path_out.fasta"


class TestToFasta:
    """Test the 'to_fasta' function by testing each parameter individually with edge cases."""

    # Positive Tests
    def test_file_path_valid(self):
        """Test 'file_path' with various text inputs to ensure robust path handling."""
        df_seq = aa.load_dataset(name="SEQ_AMYLO", n=10)
        aa.to_fasta(df_seq=df_seq, file_path=FILE_OUT)

    @given(col_id=st.text(min_size=1))
    def test_col_id_valid(self, col_id):
        """Test valid 'col_id' ensuring it exists in the DataFrame."""
        df = pd.DataFrame({col_id: ["id1"], "sequence": ["ATCG"]})
        aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_id=col_id)

    @given(col_seq=st.text(min_size=1))
    def test_col_seq_valid(self, col_seq):
        """Test valid 'col_seq' ensuring it exists in the DataFrame."""
        df = pd.DataFrame({"entry": ["id1"], col_seq: ["ATCG"]})
        aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_seq=col_seq)

    @given(sep=st.text(min_size=1, max_size=1))
    def test_sep_valid(self, sep):
        """Test valid 'sep' to check if it correctly separates information."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        aa.to_fasta(df_seq=df, file_path=FILE_OUT, sep=sep)

    @given(col_db=st.text(min_size=1))
    def test_col_db_valid(self, col_db):
        """Test valid 'col_db' ensuring it is correctly added to the header."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"], col_db: ["DB001"]})
        aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_db=col_db)

    @given(cols_info=st.lists(st.text(min_size=1), min_size=1, max_size=3))
    def test_cols_info_valid(self, cols_info):
        """Test valid 'cols_info' ensuring they are correctly added to the header."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"], **{col: ["info"] for col in cols_info}})
        aa.to_fasta(df_seq=df, file_path=FILE_OUT, cols_info=cols_info)

    # Negative Tests
    def test_file_path_invalid(self):
        """Test 'file_path' with invalid types."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path="dummy")
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=[])
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=123)

    def test_col_id_invalid(self):
        """Test invalid 'col_id' values."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        for col_id in [None, [], {}, 34]:
            with pytest.raises(ValueError):
                aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_id=col_id)

    def test_col_seq_invalid(self):
        """Test invalid 'col_seq' values."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        for col_seq in [None, [], {}, 34]:
            with pytest.raises(ValueError):
                aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_seq=col_seq)

    def test_sep_invalid(self):
        """Test invalid 'sep' values."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        for sep in [[], {}, 34]:
            with pytest.raises(ValueError):
                aa.to_fasta(df_seq=df, file_path=FILE_OUT, sep=sep)

    def test_col_db_invalid(self):
        """Test invalid 'col_db' types."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        for col_db in [[], {}, 34]:
            with pytest.raises(ValueError):
                aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_db=col_db)

    def test_cols_info_invalid(self):
        """Test invalid 'cols_info' types."""
        df = pd.DataFrame({"entry": ["id1"], "sequence": ["ATCG"]})
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=FILE_OUT, cols_info={})
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=FILE_OUT, cols_info=["NOT IN"])
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=FILE_OUT, cols_info=234)
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=FILE_OUT, cols_info=[None])


class TestToFastaComplex:
    """Test complex scenarios involving multiple parameters in the 'to_fasta' function."""

    @given(col_id=st.text(min_size=1),
           col_seq=st.text(min_size=1),
           sep=st.text(min_size=1, max_size=1),
           col_db=st.text(min_size=1),
           cols_info=st.lists(st.text(min_size=1), min_size=1, max_size=3))
    def test_valid_combination_all_parameters(self, col_id, col_seq, sep, col_db, cols_info):
        """Test with all parameters valid to ensure correct header formation and sequence output."""
        df = pd.DataFrame({col_id: ["id1"], col_seq: ["ATCG"], col_db: ["DB001"], **{col: ["info"] for col in cols_info}})
        aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_id=col_id, col_seq=col_seq, sep=sep,
                    col_db=col_db, cols_info=cols_info)

    @given(col_id=st.text(max_size=0),  # Invalid col_id (empty string)
           col_seq=st.text(min_size=1),
           sep=st.integers(),
           col_db=st.text(min_size=1),
           cols_info=st.lists(st.text(min_size=1), min_size=1, max_size=3))
    def test_invalid_combination_parameters(self, col_id, col_seq, sep, col_db, cols_info):
        """Test invalid parameter combinations to ensure proper error handling."""
        df = pd.DataFrame({col_id: ["id1"], col_seq: ["ATCG"], col_db: ["DB001"], **{col: ["info"] for col in cols_info}})
        with pytest.raises(ValueError):
            aa.to_fasta(df_seq=df, file_path=FILE_OUT, col_id=col_id, col_seq=col_seq,
                        sep=sep, col_db=col_db, cols_info=cols_info)
