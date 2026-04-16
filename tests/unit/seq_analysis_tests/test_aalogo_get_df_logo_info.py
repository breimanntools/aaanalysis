"""
This is a script to test the AAlogo.get_df_logo_info method.
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")

aa.options["verbose"] = False


# Helper
def get_df_parts(n=50):
    """Load default df_parts from DOM_GSEC dataset."""
    sf = aa.SequenceFeature()
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return sf.get_df_parts(df_seq=df_seq)


def get_labels(n=50):
    """Load labels from DOM_GSEC dataset."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    return df_seq["label"].values


# ===========================================================================
# I Test get_df_logo_info: df_parts
# ===========================================================================
class TestGetDfLogoInfoDfParts:
    """Test get_df_logo_info 'df_parts' parameter."""

    def test_valid_df_parts(self):
        """Test valid 'df_parts' returns a Series with index named 'pos'."""
        df_parts = get_df_parts()
        result = aa.AAlogo().get_df_logo_info(df_parts=df_parts)
        assert isinstance(result, pd.Series)
        assert result.index.name == "pos"

    def test_invalid_df_parts_none(self):
        """Test that df_parts=None raises ValueError."""
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=None)

    def test_invalid_df_parts_empty(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=pd.DataFrame())

    def test_invalid_df_parts_no_valid_columns(self):
        """Test that df_parts with no valid part columns raises ValueError."""
        df_bad = pd.DataFrame({"invalid_col": ["ACDE", "FGHI"]})
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_bad)

    def test_invalid_df_parts_wrong_type(self):
        """Test that non-DataFrame raises ValueError."""
        for df_parts in ["invalid", 1, [1, 2, 3]]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo_info(df_parts=df_parts)


# ===========================================================================
# II Test get_df_logo_info: labels and label_test
# ===========================================================================
class TestGetDfLogoInfoLabels:
    """Test get_df_logo_info 'labels' and 'label_test' parameters."""

    def test_valid_labels(self):
        """Test valid 'labels' filters correctly."""
        df_parts = get_df_parts()
        labels = get_labels()
        for label_test in [0, 1]:
            result = aa.AAlogo().get_df_logo_info(
                df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(result, pd.Series)

    def test_valid_labels_none(self):
        """Test that labels=None uses all samples."""
        df_parts = get_df_parts()
        result = aa.AAlogo().get_df_logo_info(df_parts=df_parts, labels=None)
        assert isinstance(result, pd.Series)

    def test_invalid_labels_wrong_length(self):
        """Test that labels with wrong length raises ValueError."""
        df_parts = get_df_parts()   # n=50
        for labels in [np.array([1, 0]), np.array([1] * 3)]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo_info(df_parts=df_parts, labels=labels)

    def test_invalid_labels_wrong_type(self):
        """Test that non-array labels raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_parts, labels="invalid")

    def test_valid_label_test(self):
        """Test valid integer label_test values."""
        df_parts = get_df_parts()
        labels = get_labels()
        for label_test in [0, 1]:
            result = aa.AAlogo().get_df_logo_info(
                df_parts=df_parts, labels=labels, label_test=label_test)
            assert isinstance(result, pd.Series)

    def test_invalid_label_test_type(self):
        """Test that non-integer label_test raises ValueError."""
        df_parts = get_df_parts()
        labels = get_labels()
        for label_test in [None, 1.5, "invalid"]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo_info(
                    df_parts=df_parts, labels=labels, label_test=label_test)

    def test_label_test_not_in_labels_raises(self):
        """Test that label_test absent from labels raises ValueError (empty-filter guard)."""
        df_parts = get_df_parts()
        labels = get_labels()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(
                df_parts=df_parts, labels=labels, label_test=99)


# ===========================================================================
# III Test get_df_logo_info: tmd_len
# ===========================================================================
class TestGetDfLogoInfoTmdLen:
    """Test get_df_logo_info 'tmd_len' parameter."""

    def test_valid_tmd_len_none(self):
        """Test that tmd_len=None uses the maximum TMD length."""
        df_parts = get_df_parts()
        result = aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len=None)
        assert isinstance(result, pd.Series)

    def test_valid_tmd_len_values(self):
        """Test valid integer tmd_len values up to the maximum in df_parts."""
        df_parts = get_df_parts()
        max_tmd_len = df_parts["tmd"].apply(len).max()
        for tmd_len in [1, max_tmd_len // 2, max_tmd_len]:
            result = aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len=tmd_len)
            assert isinstance(result, pd.Series)

    def test_invalid_tmd_len_zero(self):
        """Test that tmd_len=0 raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len=0)

    def test_invalid_tmd_len_negative(self):
        """Test that negative tmd_len raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len=-1)

    def test_invalid_tmd_len_float(self):
        """Test that float tmd_len raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len=1.5)

    def test_invalid_tmd_len_wrong_type(self):
        """Test that non-numeric tmd_len raises ValueError."""
        df_parts = get_df_parts()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len="invalid")

    def test_invalid_tmd_len_exceeds_max(self):
        """Test that tmd_len exceeding the maximum TMD length raises ValueError."""
        df_parts = get_df_parts()
        max_tmd_len = df_parts["tmd"].apply(len).max()
        with pytest.raises(ValueError):
            aa.AAlogo().get_df_logo_info(df_parts=df_parts, tmd_len=max_tmd_len + 1)


# ===========================================================================
# IV Test get_df_logo_info: start_n
# ===========================================================================
class TestGetDfLogoInfoStartN:
    """Test get_df_logo_info 'start_n' parameter."""

    def test_valid_start_n(self):
        """Test valid 'start_n' values."""
        df_parts = get_df_parts()
        for start_n in [True, False]:
            result = aa.AAlogo().get_df_logo_info(df_parts=df_parts, start_n=start_n)
            assert isinstance(result, pd.Series)

    def test_invalid_start_n(self):
        """Test that non-bool start_n raises ValueError."""
        df_parts = get_df_parts()
        for start_n in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo_info(df_parts=df_parts, start_n=start_n)


# ===========================================================================
# V Test get_df_logo_info: characters_to_ignore
# ===========================================================================
class TestGetDfLogoInfoCharactersToIgnore:
    """Test get_df_logo_info 'characters_to_ignore' parameter."""

    def test_valid_characters_to_ignore(self):
        """Test valid 'characters_to_ignore' string values."""
        df_parts = get_df_parts()
        for chars in [".-", ".", "-", ""]:
            result = aa.AAlogo().get_df_logo_info(
                df_parts=df_parts, characters_to_ignore=chars)
            assert isinstance(result, pd.Series)

    def test_invalid_characters_to_ignore(self):
        """Test that non-string characters_to_ignore raises ValueError."""
        df_parts = get_df_parts()
        for chars in [None, 123, [], [".", "-"]]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo_info(
                    df_parts=df_parts, characters_to_ignore=chars)


# ===========================================================================
# VI Test get_df_logo_info: pseudocount
# ===========================================================================
class TestGetDfLogoInfoPseudocount:
    """Test get_df_logo_info 'pseudocount' parameter."""

    def test_valid_pseudocount(self):
        """Test valid 'pseudocount' values."""
        df_parts = get_df_parts()
        for pseudocount in [0.0, 0.1, 0.5, 1.0]:
            result = aa.AAlogo().get_df_logo_info(
                df_parts=df_parts, pseudocount=pseudocount)
            assert isinstance(result, pd.Series)

    def test_invalid_pseudocount(self):
        """Test that negative or non-numeric pseudocount raises ValueError."""
        df_parts = get_df_parts()
        for pseudocount in [-0.1, -1, "invalid", None]:
            with pytest.raises(ValueError):
                aa.AAlogo().get_df_logo_info(
                    df_parts=df_parts, pseudocount=pseudocount)


# ===========================================================================
# VII Test get_df_logo_info: behavioral
# ===========================================================================
class TestGetDfLogoInfoBehavior:
    """Test get_df_logo_info behavioral properties."""

    def test_result_non_negative(self):
        """Test that all per-position information content values are >= 0."""
        df_parts = get_df_parts()
        result = aa.AAlogo().get_df_logo_info(df_parts=df_parts)
        assert (result >= 0).all()

    def test_logo_type_does_not_affect_result(self):
        """Test that logo_type has no effect: get_df_logo_info always uses information encoding."""
        df_parts = get_df_parts()
        results = {lt: aa.AAlogo(logo_type=lt).get_df_logo_info(df_parts=df_parts)
                   for lt in ["probability", "weight", "counts", "information"]}
        ref = results["probability"]
        for logo_type, result in results.items():
            assert result.equals(ref), (
                f"logo_type='{logo_type}' produced different result than 'probability'")

    def test_logo_type_attribute_not_mutated(self):
        """Test that calling get_df_logo_info does not change _logo_type."""
        df_parts = get_df_parts()
        for logo_type in ["probability", "weight", "counts"]:
            aalogo = aa.AAlogo(logo_type=logo_type)
            aalogo.get_df_logo_info(df_parts=df_parts)
            assert aalogo._logo_type == logo_type

    def test_result_equals_information_logo_sum(self):
        """Test that result equals get_df_logo(logo_type='information').sum(axis=1)."""
        import numpy as np
        df_parts = get_df_parts()
        df_logo_info = aa.AAlogo().get_df_logo_info(df_parts=df_parts)
        df_logo_manual = (aa.AAlogo(logo_type="information")
                          .get_df_logo(df_parts=df_parts)
                          .sum(axis=1))
        assert np.allclose(df_logo_info.values, df_logo_manual.values)


# ===========================================================================
# VIII Complex: valid parameter combinations
# ===========================================================================
class TestGetDfLogoInfoComplex:
    """Test get_df_logo_info with valid parameter combinations."""

    @settings(max_examples=5)
    @given(
        logo_type=st.sampled_from(["probability", "weight", "counts", "information"]),
        tmd_len=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
        start_n=st.booleans(),
        pseudocount=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_valid_combinations(self, logo_type, tmd_len, start_n, pseudocount):
        """Test valid parameter combinations return a Series with index named 'pos'."""
        df_parts = get_df_parts()
        aalogo = aa.AAlogo(logo_type=logo_type)
        result = aalogo.get_df_logo_info(
            df_parts=df_parts, tmd_len=tmd_len,
            start_n=start_n, pseudocount=pseudocount)
        assert isinstance(result, pd.Series)
        assert result.index.name == "pos"
        assert (result >= 0).all()
        # logo_type must not be mutated
        assert aalogo._logo_type == logo_type
